"""
download_delhi_data.py — Download Sentinel-2 + OSM Roads for Delhi
===================================================================
Uses Microsoft Planetary Computer (free, no account needed) for
Sentinel-2 L2A imagery, and Overpass API for OSM road vectors.

Target: Majnu ka Tila area, Delhi, India

Usage:
    pip install pystac-client planetary-computer rasterio requests shapely

    python download_delhi_data.py
    python download_delhi_data.py --bbox 77.20 28.68 77.24 28.70
"""

import argparse
import json
import os

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.features import rasterize
from shapely.geometry import shape, box, mapping
import requests


# 1. Download Sentinel-2 from Planetary Computer

def download_sentinel2(bbox, output_dir, max_cloud=10):
    """
    Query Planetary Computer STAC for Sentinel-2 L2A and download RGB bands.

    Args:
        bbox: (west, south, east, north) in WGS84
        output_dir: where to save the GeoTIFF
        max_cloud: max cloud cover percentage
    """
    from pystac_client import Client
    import planetary_computer

    print("=" * 60)
    print("Step 1: Downloading Sentinel-2 imagery from Planetary Computer")
    print(f"  AOI: {bbox}")
    print("=" * 60)

    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        query={"eo:cloud_cover": {"lt": max_cloud}},
        sortby=["-properties.datetime"],
        max_items=5,
    )

    items = list(search.items())
    if not items:
        print("ERROR: No Sentinel-2 scenes found. Try increasing --max_cloud.")
        return None

    # Pick the most recent low-cloud scene
    item = items[0]
    print(f"  Selected: {item.id}")
    print(f"  Date: {item.properties['datetime']}")
    print(f"  Cloud cover: {item.properties['eo:cloud_cover']:.1f}%")

    # Download B04 (Red), B03 (Green), B02 (Blue) — all 10m
    band_names = ["B04", "B03", "B02"]
    bands = []
    out_transform = None
    out_crs = None

    for band_name in band_names:
        asset = item.assets[band_name]
        href = asset.href
        print(f"  Reading {band_name}...")

        with rasterio.open(href) as src:
            # Reproject AOI bbox into raster's CRS (UTM)
            from pyproj import Transformer
            from rasterio.mask import mask as rio_mask

            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            west, south = transformer.transform(bbox[0], bbox[1])
            east, north = transformer.transform(bbox[2], bbox[3])
            aoi_utm = box(west, south, east, north)

            data, transform = rio_mask(src, [mapping(aoi_utm)], crop=True)
            bands.append(data[0])  # (H, W)
            out_transform = transform
            out_crs = src.crs
            profile = src.profile.copy()

    # Stack into 3-band GeoTIFF
    stacked = np.stack(bands, axis=0)  # (3, H, W)
    h, w = stacked.shape[1], stacked.shape[2]

    out_path = os.path.join(output_dir, "sentinel2_rgb.tif")
    profile.update(
        driver="GTiff",
        count=3,
        height=h,
        width=w,
        transform=out_transform,
        crs=out_crs,
        compress="lzw",
    )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(stacked)

    print(f"  Saved: {out_path} ({w}x{h} px, 3 bands, CRS={out_crs})")
    return out_path


# 2. Download OSM Roads

def download_osm_roads(bbox, output_dir):
    """
    Download road network from OpenStreetMap using Overpass API.

    Args:
        bbox: (west, south, east, north) in WGS84
        output_dir: where to save the GeoJSON
    """
    print("\n" + "=" * 60)
    print("Step 2: Downloading OSM roads from Overpass API")
    print("=" * 60)

    south, west, north, east = bbox[1], bbox[0], bbox[3], bbox[2]

    query = f"""
    [out:json][timeout:120];
    (
      way["highway"~"primary|secondary|tertiary|residential|trunk|motorway|unclassified"]({south},{west},{north},{east});
    );
    out body;
    >;
    out skel qt;
    """

    url = "https://overpass-api.de/api/interpreter"
    print("  Querying Overpass API...")
    response = requests.post(url, data={"data": query})

    if response.status_code != 200:
        print(f"  ERROR: Overpass API returned {response.status_code}")
        return None

    data = response.json()

    # Two-pass: Overpass returns ways before nodes, so collect nodes first
    nodes = {}
    raw_ways = []

    for element in data["elements"]:
        if element["type"] == "node":
            nodes[element["id"]] = (element["lon"], element["lat"])
        elif element["type"] == "way":
            raw_ways.append(element)

    # Now resolve way coordinates
    ways = []
    for element in raw_ways:
        coords = [nodes[nid] for nid in element.get("nodes", []) if nid in nodes]
        if len(coords) >= 2:
            ways.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords,
                },
                "properties": element.get("tags", {}),
            })

    geojson = {
        "type": "FeatureCollection",
        "features": ways,
    }

    out_path = os.path.join(output_dir, "roads.geojson")
    with open(out_path, "w") as f:
        json.dump(geojson, f)

    print(f"  Found {len(ways)} road segments")
    print(f"  Saved: {out_path}")
    return out_path


# 3. Rasterize Roads to Mask (aligned with imagery)

def rasterize_roads(image_path, roads_path, output_dir, buffer_meters=5):
    """
    Create a binary road mask aligned to the Sentinel-2 image.

    Args:
        image_path: path to Sentinel-2 GeoTIFF
        roads_path: path to roads GeoJSON
        output_dir: where to save mask
        buffer_meters: road buffer width
    """
    print("\n" + "=" * 60)
    print("Step 3: Rasterizing roads to binary mask")
    print("=" * 60)

    with open(roads_path) as f:
        roads_geojson = json.load(f)

    with rasterio.open(image_path) as src:
        crs = src.crs
        transform = src.transform
        width = src.width
        height = src.height

    print(f"  Image CRS: {crs}")
    print(f"  Image size: {width}x{height}")
    print(f"  Buffer: {buffer_meters}m")

    # Buffer road geometries
    from shapely.ops import transform as shapely_transform
    from pyproj import Transformer

    # Transform roads to UTM for buffering in meters
    # Delhi => UTM 43N (EPSG:32643)
    utm_crs = "EPSG:32643"
    to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    to_img_crs = Transformer.from_crs(utm_crs, crs, always_xy=True)

    buffered_shapes = []
    for feat in roads_geojson["features"]:
        try:
            geom = shape(feat["geometry"])
            # To UTM
            geom_utm = shapely_transform(to_utm.transform, geom)
            # Buffer
            geom_buffered = geom_utm.buffer(buffer_meters)
            # To image CRS
            geom_img = shapely_transform(to_img_crs.transform, geom_buffered)
            buffered_shapes.append(geom_img)
        except Exception:
            continue

    print(f"  Buffered {len(buffered_shapes)} road geometries")

    # Rasterize
    if buffered_shapes:
        mask = rasterize(
            [(geom, 1) for geom in buffered_shapes],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )
    else:
        mask = np.zeros((height, width), dtype=np.uint8)

    road_pct = mask.sum() / mask.size * 100
    print(f"  Road pixels: {mask.sum():,} ({road_pct:.1f}%)")

    out_path = os.path.join(output_dir, "road_mask.tif")
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": width,
        "height": height,
        "count": 1,
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask, 1)

    print(f"  Saved: {out_path}")
    return out_path


# 4. Main

def main():
    parser = argparse.ArgumentParser(
        description="Download Sentinel-2 + OSM roads for Delhi"
    )
    parser.add_argument("--bbox", nargs=4, type=float,
                        default=[77.20, 28.68, 77.24, 28.70],
                        help="Bounding box: west south east north (WGS84)")
    parser.add_argument("--output_dir", type=str, default="data/delhi",
                        help="Output directory")
    parser.add_argument("--max_cloud", type=int, default=10,
                        help="Max cloud cover (%)")
    parser.add_argument("--buffer", type=float, default=5,
                        help="Road buffer width in meters")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    bbox = tuple(args.bbox)

    print(f"Target: Majnu ka Tila area, Delhi")
    print(f"Bbox: {bbox}")
    print(f"Output: {args.output_dir}\n")

    # Step 1: Sentinel-2
    image_path = download_sentinel2(bbox, args.output_dir, args.max_cloud)
    if image_path is None:
        return

    # Step 2: OSM Roads
    roads_path = download_osm_roads(bbox, args.output_dir)
    if roads_path is None:
        return

    # Step 3: Rasterize
    mask_path = rasterize_roads(image_path, roads_path, args.output_dir, args.buffer)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"DATA READY!")
    print(f"{'=' * 60}")
    print(f"  Image: {image_path}")
    print(f"  Mask:  {mask_path}")
    print(f"\nNext steps:")
    print(f"  1. Tile:  python tile_dataset.py --image {image_path} --mask {mask_path} --out_dir data/delhi/train")
    print(f"  2. Train: python train_torchgeo.py --data_dir data/delhi")


if __name__ == "__main__":
    main()
