"""
download_india_data.py — Download Sentinel-2 + OSM Roads for Indian Cities
===========================================================================
Downloads training data for multiple cities across India.

Uses:
  - Microsoft Planetary Computer (free) for Sentinel-2 L2A
  - Overpass API (free) for OSM road vectors

Supported cities: delhi, mumbai, chennai, kolkata, goa, portblair

Usage:
    # Download all 6 cities:
    python download_india_data.py --cities all

    # Download specific cities:
    python download_india_data.py --cities delhi mumbai chennai

    # Download one city:
    python download_india_data.py --cities goa

    # Download ANY custom area (anywhere in India):
    python download_india_data.py --bbox 77.20 28.68 77.24 28.70 --name majnu_ka_tila

    # Download custom area + predefined cities:
    python download_india_data.py --cities delhi mumbai --bbox 73.85 18.50 73.90 18.55 --name pune_area
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape, box, mapping
import requests

  
# City Bounding Boxes (west[x min], south[y min], east[x max], north[y max]) — WGS84  

CITY_CONFIGS = {
    "delhi": {
        "name": "Delhi",
        "bbox": (76.95, 28.45, 77.35, 28.80),
        "utm_crs": "EPSG:32643",
    },
    "mumbai": {
        "name": "Mumbai",
        "bbox": (72.77, 18.89, 73.05, 19.27),
        "utm_crs": "EPSG:32643",
    },
    "chennai": {
        "name": "Chennai",
        "bbox": (80.15, 12.92, 80.32, 13.15),
        "utm_crs": "EPSG:32644",
    },
    "kolkata": {
        "name": "Kolkata",
        "bbox": (88.25, 22.45, 88.45, 22.65),
        "utm_crs": "EPSG:32645",
    },
    "goa": {
        "name": "Goa",
        "bbox": (73.70, 15.30, 74.00, 15.58),
        "utm_crs": "EPSG:32643",
    },
    "portblair": {
        "name": "Port Blair",
        "bbox": (92.70, 11.62, 92.80, 11.72),
        "utm_crs": "EPSG:32646",
    },
}


def get_utm_crs(lon):
    """Auto-detect UTM zone EPSG code from longitude."""
    zone = int((lon + 180) / 6) + 1
    return f"EPSG:326{zone:02d}"

  
# 1. Download Sentinel-2  

def download_sentinel2(city_key, bbox, output_dir, max_cloud=15):
    """Download Sentinel-2 RGB from Planetary Computer."""
    from pystac_client import Client
    import planetary_computer
    from pyproj import Transformer
    from rasterio.mask import mask as rio_mask

    print(f"\n  [Sentinel-2] Querying for {CITY_CONFIGS[city_key]['name']}...")

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
        print(f"  ERROR: No scenes found for {city_key}. Try --max_cloud higher.")
        return None

    item = items[0]
    print(f"  Scene: {item.id}")
    print(f"  Date: {item.properties['datetime'][:10]}")
    print(f"  Cloud: {item.properties['eo:cloud_cover']:.1f}%")

    # Download RGB bands
    band_names = ["B04", "B03", "B02"]
    bands = []
    out_transform = None
    out_crs = None

    for band_name in band_names:
        asset = item.assets[band_name]
        print(f"  Downloading {band_name}...", end=" ", flush=True)

        with rasterio.open(asset.href) as src:
            # Reproject bbox to raster's CRS
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            west, south = transformer.transform(bbox[0], bbox[1])
            east, north = transformer.transform(bbox[2], bbox[3])
            aoi_utm = box(west, south, east, north)

            data, transform = rio_mask(src, [mapping(aoi_utm)], crop=True)
            bands.append(data[0])
            out_transform = transform
            out_crs = src.crs
            profile = src.profile.copy()
            print(f"OK ({data.shape[1]}x{data.shape[2]})")

    stacked = np.stack(bands, axis=0)
    h, w = stacked.shape[1], stacked.shape[2]

    out_path = os.path.join(output_dir, "sentinel2_rgb.tif")
    profile.update(
        driver="GTiff", count=3, height=h, width=w,
        transform=out_transform, crs=out_crs, compress="lzw",
    )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(stacked)

    print(f"  Saved: {out_path} ({w}x{h} px)")
    return out_path

  
# 2. Download OSM Roads  

def download_osm_roads(city_key, bbox, output_dir):
    """Download road network from OSM Overpass API."""
    print(f"\n  [OSM] Downloading roads for {CITY_CONFIGS[city_key]['name']}...")

    south, west, north, east = bbox[1], bbox[0], bbox[3], bbox[2]

    query = f"""[out:json][timeout:180];
(
  way["highway"~"primary|secondary|tertiary|residential|trunk|motorway|unclassified|service"]({south},{west},{north},{east});
);
out body;
>;
out skel qt;
"""

    response = requests.post(
        "https://overpass-api.de/api/interpreter",
        data={"data": query},
        timeout=200,
    )

    if response.status_code != 200:
        print(f"  ERROR: Overpass returned {response.status_code}")
        return None

    data = response.json()

    # Two-pass parsing (ways reference nodes that come later)
    nodes = {}
    raw_ways = []

    for element in data["elements"]:
        if element["type"] == "node":
            nodes[element["id"]] = (element["lon"], element["lat"])
        elif element["type"] == "way":
            raw_ways.append(element)

    ways = []
    for element in raw_ways:
        coords = [nodes[nid] for nid in element.get("nodes", []) if nid in nodes]
        if len(coords) >= 2:
            ways.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": element.get("tags", {}),
            })

    geojson = {"type": "FeatureCollection", "features": ways}

    out_path = os.path.join(output_dir, "roads.geojson")
    with open(out_path, "w") as f:
        json.dump(geojson, f)

    print(f"  Found {len(ways)} road segments")
    return out_path

  
# 3. Rasterize Roads  

def rasterize_roads(city_key, image_path, roads_path, output_dir, buffer_meters=5):
    """Create binary road mask aligned to Sentinel-2 image."""
    from shapely.ops import transform as shapely_transform
    from pyproj import Transformer

    print(f"\n  [Rasterize] Creating road mask...")

    with open(roads_path) as f:
        roads = json.load(f)

    with rasterio.open(image_path) as src:
        crs = src.crs
        transform = src.transform
        width = src.width
        height = src.height

    utm_crs = CITY_CONFIGS[city_key]["utm_crs"]
    to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    to_img_crs = Transformer.from_crs(utm_crs, crs, always_xy=True)

    buffered = []
    for feat in roads["features"]:
        try:
            geom = shape(feat["geometry"])
            geom_utm = shapely_transform(to_utm.transform, geom)
            geom_buf = geom_utm.buffer(buffer_meters)
            geom_img = shapely_transform(to_img_crs.transform, geom_buf)
            buffered.append(geom_img)
        except Exception:
            continue

    if buffered:
        mask = rasterize(
            [(g, 1) for g in buffered],
            out_shape=(height, width),
            transform=transform,
            fill=0, dtype=np.uint8,
        )
    else:
        mask = np.zeros((height, width), dtype=np.uint8)

    road_pct = mask.sum() / mask.size * 100
    print(f"  Road pixels: {mask.sum():,} ({road_pct:.1f}%)")

    out_path = os.path.join(output_dir, "road_mask.tif")
    with rasterio.open(out_path, "w", driver="GTiff", dtype="uint8",
                       width=width, height=height, count=1,
                       crs=crs, transform=transform, compress="lzw") as dst:
        dst.write(mask, 1)

    print(f"  Saved: {out_path}")
    return out_path

  
# 4. Tile into Patches  

def tile_city(image_path, mask_path, out_dir, tile_size=256, stride=128,
              min_road_ratio=0.001, keep_empty_prob=0.1):
    """Tile a city's image+mask into training patches."""
    from rasterio.windows import Window

    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)

    with rasterio.open(image_path) as img_src, rasterio.open(mask_path) as msk_src:
        count = 0
        skipped = 0

        for y in range(0, img_src.height - tile_size + 1, stride):
            for x in range(0, img_src.width - tile_size + 1, stride):
                window = Window(x, y, tile_size, tile_size)

                img_tile = img_src.read(window=window)
                msk_tile = msk_src.read(1, window=window)

                road_ratio = msk_tile.sum() / msk_tile.size
                if road_ratio < min_road_ratio:
                    if np.random.rand() > keep_empty_prob:
                        skipped += 1
                        continue

                tile_transform = rasterio.windows.transform(window, img_src.transform)
                tile_name = f"tile_{count:05d}.tif"

                # Save image tile
                img_profile = img_src.profile.copy()
                img_profile.update(
                    width=tile_size, height=tile_size,
                    transform=tile_transform, driver="GTiff", compress="lzw",
                )
                with rasterio.open(
                    os.path.join(out_dir, "images", tile_name), "w", **img_profile
                ) as dst:
                    dst.write(img_tile)

                # Save mask tile
                msk_profile = msk_src.profile.copy()
                msk_profile.update(
                    width=tile_size, height=tile_size,
                    transform=tile_transform, driver="GTiff", compress="lzw",
                )
                with rasterio.open(
                    os.path.join(out_dir, "masks", tile_name), "w", **msk_profile
                ) as dst:
                    dst.write(msk_tile, 1)

                count += 1

    print(f"  Tiles: {count} saved, {skipped} skipped (low road content)")
    return count

  
# 5. Main  

def process_city(city_key, base_dir, max_cloud, buffer_m, tile_size, stride):
    """Download, rasterize, and tile data for one city."""
    config = CITY_CONFIGS[city_key]
    bbox = config["bbox"]
    city_dir = os.path.join(base_dir, city_key)
    os.makedirs(city_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  {config['name']} — bbox: {bbox}")
    print(f"{'=' * 60}")

    # Step 1: Sentinel-2
    image_path = download_sentinel2(city_key, bbox, city_dir, max_cloud)
    if image_path is None:
        return 0

    # Step 2: OSM Roads
    roads_path = download_osm_roads(city_key, bbox, city_dir)
    if roads_path is None:
        return 0

    # Step 3: Rasterize
    mask_path = rasterize_roads(city_key, image_path, roads_path, city_dir, buffer_m)

    # Step 4: Tile
    print(f"\n  [Tile] Cutting into {tile_size}x{tile_size} patches...")
    tile_dir = os.path.join(city_dir, "tiles")
    n_tiles = tile_city(image_path, mask_path, tile_dir, tile_size, stride)

    return n_tiles


def merge_tiles(base_dir, cities, out_dir, val_split=0.15):
    """Merge tiled patches from all cities into train/val splits."""
    import shutil
    import random

    train_img = os.path.join(out_dir, "train", "images")
    train_msk = os.path.join(out_dir, "train", "masks")
    val_img = os.path.join(out_dir, "val", "images")
    val_msk = os.path.join(out_dir, "val", "masks")

    for d in [train_img, train_msk, val_img, val_msk]:
        os.makedirs(d, exist_ok=True)

    all_tiles = []
    for city_key in cities:
        tile_dir = os.path.join(base_dir, city_key, "tiles")
        img_dir = os.path.join(tile_dir, "images")
        msk_dir = os.path.join(tile_dir, "masks")

        if not os.path.isdir(img_dir):
            continue

        for name in os.listdir(img_dir):
            all_tiles.append((
                os.path.join(img_dir, name),
                os.path.join(msk_dir, name),
                city_key,
            ))

    random.shuffle(all_tiles)
    n_val = max(1, int(len(all_tiles) * val_split))

    print(f"\nMerging {len(all_tiles)} tiles → {len(all_tiles) - n_val} train + {n_val} val")

    for i, (img_src, msk_src, city) in enumerate(all_tiles):
        new_name = f"{city}_{os.path.basename(img_src)}"

        if i < n_val:
            shutil.copy2(img_src, os.path.join(val_img, new_name))
            shutil.copy2(msk_src, os.path.join(val_msk, new_name))
        else:
            shutil.copy2(img_src, os.path.join(train_img, new_name))
            shutil.copy2(msk_src, os.path.join(train_msk, new_name))

    print(f"Train: {len(all_tiles) - n_val} tiles")
    print(f"Val:   {n_val} tiles")
    print(f"Saved to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Sentinel-2 + OSM roads for Indian cities",
        epilog="\n".join([
            "Examples:",
            "  python download_india_data.py --cities all",
            "  python download_india_data.py --cities delhi mumbai",
            "  python download_india_data.py --bbox 77.20 28.68 77.24 28.70 --name my_area",
            "  python download_india_data.py --cities delhi --bbox 73.85 18.50 73.90 18.55 --name pune",
        ]),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cities", nargs="+", default=["all"],
                        help="Cities to download: " + ", ".join(CITY_CONFIGS.keys()) + ", or 'all'")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"),
                        help="Custom bounding box (WGS84): west south east north. "
                             "Use bboxfinder.com to get coordinates.")
    parser.add_argument("--name", type=str, default=None,
                        help="Name for custom bbox area (used as folder name). Required with --bbox.")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Base data directory")
    parser.add_argument("--max_cloud", type=int, default=15,
                        help="Max cloud cover (%%)")
    parser.add_argument("--buffer", type=float, default=5,
                        help="Road buffer width (meters)")
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip download, just merge existing tiles")
    args = parser.parse_args()

    # Handle custom bbox
    if args.bbox:
        if not args.name:
            print("ERROR: --name is required when using --bbox")
            print("Example: --bbox 77.20 28.68 77.24 28.70 --name majnu_ka_tila")
            sys.exit(1)

        custom_key = args.name.lower().replace(" ", "_")
        bbox = tuple(args.bbox)
        center_lon = (bbox[0] + bbox[2]) / 2

        # Register custom area as a city config
        CITY_CONFIGS[custom_key] = {
            "name": args.name,
            "bbox": bbox,
            "utm_crs": get_utm_crs(center_lon),
        }
        print(f"Custom area registered: {args.name}")
        print(f"  bbox: {bbox}")
        print(f"  UTM CRS: {CITY_CONFIGS[custom_key]['utm_crs']}")

    # Resolve city list
    if args.bbox and "all" not in args.cities:
        # Add custom area to the city list
        cities = [c.lower() for c in args.cities if c != "all"]
        custom_key = args.name.lower().replace(" ", "_")
        if custom_key not in cities:
            cities.append(custom_key)
    elif args.bbox and "all" in args.cities and len(args.cities) == 1:
        # --bbox with default --cities all → only download custom area
        custom_key = args.name.lower().replace(" ", "_")
        cities = [custom_key]
    elif "all" in args.cities:
        cities = list(CITY_CONFIGS.keys())
    else:
        cities = [c.lower() for c in args.cities]

    for c in cities:
        if c not in CITY_CONFIGS:
            print(f"Unknown city: {c}. Available: {list(CITY_CONFIGS.keys())}")
            sys.exit(1)

    print(f"Cities: {', '.join(CITY_CONFIGS[c]['name'] for c in cities)}")
    print(f"Output: {args.data_dir}/")

    # Process each city
    if not args.skip_download:
        total_tiles = 0
        for city_key in cities:
            try:
                n = process_city(
                    city_key, args.data_dir, args.max_cloud,
                    args.buffer, args.tile_size, args.stride,
                )
                total_tiles += n
            except Exception as e:
                print(f"\n  ERROR processing {city_key}: {e}")
                continue

            # Brief pause between cities to be nice to APIs
            time.sleep(2)

        print(f"\n{'=' * 60}")
        print(f"Total tiles across all cities: {total_tiles}")
        print(f"{'=' * 60}")

    # Merge all city tiles into train/val
    merged_dir = os.path.join(args.data_dir, "india_combined")
    merge_tiles(args.data_dir, cities, merged_dir)

    print(f"\n{'=' * 60}")
    print(f"DONE! Ready to train:")
    print(f"  python train_torchgeo.py --data_dir {merged_dir} --epochs 50")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
