"""
tile_dataset.py — Cut aligned image + mask GeoTIFFs into 256x256 patches
=========================================================================
Reads a full-scene Sentinel-2 clipped image and its corresponding
binary road mask, and tiles them into aligned patches for ML training.

Usage:
    python tile_dataset.py --image image_clipped.tif --mask mask.tif \\
                           --out_dir dataset/train --tile_size 256 --stride 128
"""

import argparse
import os
import numpy as np
import rasterio
from rasterio.windows import Window


def tile_pair(image_path: str, mask_path: str, out_dir: str,
              tile_size: int = 256, stride: int = 256,
              min_road_ratio: float = 0.001, keep_empty_prob: float = 0.1):
    """
    Tile an aligned image and mask into patches.

    Args:
        image_path:      Path to 3-band Sentinel-2 GeoTIFF
        mask_path:        Path to 1-band binary mask GeoTIFF
        out_dir:          Output directory (will create images/ and masks/ subdirs)
        tile_size:        Size of square tiles in pixels
        stride:           Step size between tiles (< tile_size for overlap)
        min_road_ratio:   Minimum fraction of road pixels to keep a tile
        keep_empty_prob:  Probability of keeping tiles below min_road_ratio
    """
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)

    with rasterio.open(image_path) as img_src, rasterio.open(mask_path) as msk_src:
        # --- Verify alignment ---
        assert img_src.crs == msk_src.crs, (
            f"CRS mismatch: image={img_src.crs}, mask={msk_src.crs}")
        assert img_src.width == msk_src.width, (
            f"Width mismatch: image={img_src.width}, mask={msk_src.width}")
        assert img_src.height == msk_src.height, (
            f"Height mismatch: image={img_src.height}, mask={msk_src.height}")
        assert img_src.transform == msk_src.transform, "Transform mismatch"

        print(f"Image size: {img_src.width} × {img_src.height} px")
        print(f"Tile size:  {tile_size} × {tile_size}, stride: {stride}")

        count = 0
        skipped = 0

        for y in range(0, img_src.height - tile_size + 1, stride):
            for x in range(0, img_src.width - tile_size + 1, stride):
                window = Window(x, y, tile_size, tile_size)

                img_tile = img_src.read(window=window)      # (3, H, W)
                msk_tile = msk_src.read(1, window=window)    # (H, W)

                # Skip mostly-empty tiles (keep a small fraction for negatives)
                road_ratio = msk_tile.sum() / msk_tile.size
                if road_ratio < min_road_ratio:
                    if np.random.rand() > keep_empty_prob:
                        skipped += 1
                        continue

                # Compute tile transform
                tile_transform = rasterio.windows.transform(window, img_src.transform)

                # Save image tile
                img_profile = img_src.profile.copy()
                img_profile.update(
                    width=tile_size, height=tile_size,
                    transform=tile_transform,
                    driver="GTiff",
                    compress="lzw",
                )
                tile_name = f"tile_{count:05d}.tif"
                with rasterio.open(
                    os.path.join(out_dir, "images", tile_name), "w", **img_profile
                ) as dst:
                    dst.write(img_tile)

                # Save mask tile
                msk_profile = msk_src.profile.copy()
                msk_profile.update(
                    width=tile_size, height=tile_size, count=1,
                    transform=tile_transform,
                    driver="GTiff",
                    compress="lzw",
                )
                with rasterio.open(
                    os.path.join(out_dir, "masks", tile_name), "w", **msk_profile
                ) as dst:
                    dst.write(msk_tile[np.newaxis, :, :])

                count += 1

    print(f"\n✅ Created {count} tile pairs in '{out_dir}'")
    print(f"   Skipped {skipped} mostly-empty tiles")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tile image+mask into patches")
    parser.add_argument("--image", required=True, help="Path to clipped Sentinel-2 GeoTIFF")
    parser.add_argument("--mask", required=True, help="Path to binary mask GeoTIFF")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128,
                        help="Stride between tiles (use < tile_size for overlap)")
    args = parser.parse_args()

    tile_pair(args.image, args.mask, args.out_dir,
              tile_size=args.tile_size, stride=args.stride)
