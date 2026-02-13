# End-to-End Road Detection Pipeline

## Sentinel-2 Imagery + OpenStreetMap Ground Truth — India

> **Tools:** QGIS (data prep) · Python/PyTorch (ML) · ArcGIS Online (viewing only)
> **Author guidance:** Written as mentor-to-intern reference. Every step is free-tools-only and reproducible.

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Data Acquisition](#2-data-acquisition)
3. [Data Inspection (QGIS)](#3-data-inspection-qgis)
4. [Data Preparation (QGIS)](#4-data-preparation-qgis)
5. [Dataset Structure for ML](#5-dataset-structure-for-ml)
6. [Model Selection & Justification](#6-model-selection--justification)
7. [Python Training Pipeline](#7-python-training-pipeline)
8. [Evaluation](#8-evaluation)
9. [Common Mistakes & How to Avoid Them](#9-common-mistakes--how-to-avoid-them)
10. [Reproducibility Checklist](#10-reproducibility-checklist)

---

## 1. Pipeline Overview

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Sentinel-2  │    │     OSM      │    │    QGIS      │    │   Python     │    │  Evaluation  │
│  Download    │───▶│  Road Vectors│───▶│  Clip, CRS,  │───▶│  U-Net Train │───▶│  IoU / Dice  │
│  (B4,B3,B2)  │    │  (GeoJSON)   │    │  Rasterize   │    │  & Predict   │    │  Visual QC   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

**Key principle:** The satellite image and the road mask must be _pixel-aligned_ — same CRS, same extent, same resolution. Every downstream error traces back to this.

---

## 2. Data Acquisition

### 2.1 Sentinel-2 Imagery

| Source                  | URL                                      | Notes                                                  |
| ----------------------- | ---------------------------------------- | ------------------------------------------------------ |
| **Copernicus Browser**  | https://browser.dataspace.copernicus.eu/ | Free account, download L2A (atmospherically corrected) |
| **USGS EarthExplorer**  | https://earthexplorer.usgs.gov/          | Alternative mirror                                     |
| **Google Earth Engine** | https://code.earthengine.google.com/     | Programmatic access, export GeoTIFF                    |

**What to download:**

- Product: **S2 L2A** (Surface Reflectance — atmospheric correction already applied)
- Bands needed: **B4 (Red, 10 m), B3 (Green, 10 m), B2 (Blue, 10 m)**
- Cloud cover: **< 10 %** for your Area of Interest (AOI)
- Each band is a separate `.jp2` file inside the `GRANULE/…/IMG_DATA/R10m/` folder

**Tip:** You can also download a single stacked GeoTIFF from Google Earth Engine:

```javascript
// Google Earth Engine script (paste in code.earthengine.google.com)
var aoi = ee.Geometry.Rectangle([72.5, 23.0, 72.7, 23.2]); // example: Ahmedabad
var s2 = ee
  .ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
  .filterBounds(aoi)
  .filterDate("2024-01-01", "2024-03-31")
  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
  .median()
  .select(["B4", "B3", "B2"])
  .clip(aoi);

Export.image.toDrive({
  image: s2,
  description: "S2_RGB_AOI",
  region: aoi,
  scale: 10,
  crs: "EPSG:32643", // UTM 43N for western India
  maxPixels: 1e10,
});
```

### 2.2 OpenStreetMap Road Vectors

| Source                   | Method                                                                         |
| ------------------------ | ------------------------------------------------------------------------------ |
| **Overpass Turbo**       | https://overpass-turbo.eu/ — visual query builder                              |
| **Geofabrik**            | https://download.geofabrik.de/asia/india.html — bulk `.shp` downloads by state |
| **QGIS QuickOSM plugin** | In-QGIS download (see Step 3)                                                  |

**Overpass Turbo query for roads:**

```
[out:json][timeout:120];
area["name"="Ahmedabad"]["admin_level"="6"]->.a;
(
  way["highway"](area.a);
);
out body;
>;
out skel qt;
```

Export as **GeoJSON**. This gives you all road types (`highway=*`).

---

## 3. Data Inspection (QGIS)

Before any processing, you must _visually verify_ your data.

### 3.1 Open & Inspect Sentinel-2 Bands

1. **Open QGIS** → `Layer` → `Add Layer` → `Add Raster Layer`
2. Browse to each band file (e.g., `T43QBC_…_B04_10m.jp2`)
3. Or drag-and-drop the `.jp2` files into the QGIS canvas
4. Right-click any band → `Properties` → **Information** tab:
   - Note the **CRS** (e.g., `EPSG:32643` — UTM Zone 43N)
   - Note the **Pixel Size** (should be `10, -10` for 10 m bands)
   - Note the **Extent** (bounding box coordinates)

### 3.2 Create an RGB Composite (Optional, for Visual QC)

If you have 3 separate band files:

1. `Raster` → `Miscellaneous` → `Build Virtual Raster (Catalog)`
2. Select B4, B3, B2 → check **"Place each input file into a separate band"**
3. This gives you a 3-band virtual raster — set band rendering to RGB

If you exported a stacked GeoTIFF from GEE, it's already 3-band.

### 3.3 Open & Inspect OSM Road Vectors

1. `Layer` → `Add Layer` → `Add Vector Layer` → select your `.geojson`
2. Right-click → `Properties` → **Information** → note the **CRS** (likely `EPSG:4326` WGS 84)
3. Open the **Attribute Table** — check the `highway` field for road types:
   - `motorway`, `trunk`, `primary`, `secondary`, `tertiary`, `residential`, `service`, etc.
4. Visualize: the roads should overlay on the imagery _approximately_ correctly (exact alignment comes after reprojection)

### 3.4 Visual Sanity Check on ArcGIS Online

1. Upload your GeoJSON to ArcGIS Online as a hosted feature layer (free with a public account)
2. Layer over Esri's basemap to verify roads are in the right geographic location
3. **Do not use ArcGIS Online for any processing** — viewing and QC only

---

## 4. Data Preparation (QGIS)

> **Goal:** Produce pixel-aligned pairs: `image.tif` (3-band, uint16) + `mask.tif` (1-band, binary 0/1)

### 4.1 Define Your AOI

Create a polygon for your Area of Interest:

1. `Layer` → `Create Layer` → `New Shapefile Layer`
   - Geometry type: **Polygon**
   - CRS: choose the **same UTM zone as your Sentinel-2 data** (e.g., `EPSG:32643`)
2. Toggle editing → draw a rectangle/polygon around your study area
3. Save edits → save layer

### 4.2 Reproject Everything to a Common CRS

> ⚠ **Critical:** All layers MUST share the same CRS before clipping or rasterizing.

The target CRS should be the **UTM zone of your Sentinel-2 tile** (e.g., `EPSG:32643` for western India).

**Reproject OSM vectors (they come in EPSG:4326):**

1. Right-click the road vector layer → `Export` → `Save Features As…`
2. Format: **GeoJSON** or **GeoPackage**
3. CRS: select the target UTM CRS (e.g., `EPSG:32643`)
4. Save → a new reprojected layer is added

**Or use Processing Toolbox:**
`Processing` → `Toolbox` → search `Reproject layer` → set target CRS → Run

### 4.3 Clip Sentinel-2 Imagery to AOI

1. `Raster` → `Extraction` → `Clip Raster by Mask Layer`
   - Input layer: your Sentinel-2 raster (or virtual raster)
   - Mask layer: your AOI polygon
   - ✅ Check **"Match the extent of the clipped raster to the extent of the mask layer"**
   - ✅ Check **"Keep resolution of input raster"**
   - Target CRS: same UTM CRS
   - Output: `image_clipped.tif`
2. Run

### 4.4 Clip Road Vectors to AOI

1. `Vector` → `Geoprocessing Tools` → `Clip`
   - Input layer: reprojected road vectors
   - Overlay layer: your AOI polygon
   - Output: `roads_clipped.geojson`
2. Run

### 4.5 Buffer Road Lines (Important!)

Roads in OSM are **lines (1 pixel wide)**. A 1-pixel-wide target in a 10 m resolution image is nearly impossible to learn. You must buffer them to a realistic width.

Recommended buffer widths:
| Road Type | Real Width | Buffer (one side) |
|-----------|-----------|-------------------|
| motorway | ~30 m | 15 m |
| primary | ~15 m | 7–8 m |
| secondary | ~10 m | 5 m |
| residential | ~6 m | 3 m |
| all (simplified) | — | **5 m** (uniform) |

**Steps:**

1. `Vector` → `Geoprocessing Tools` → `Buffer`
   - Input: `roads_clipped.geojson`
   - Distance: **5** (meters, since CRS is UTM)
   - End cap style: **Flat** (better for roads)
   - ✅ **Dissolve result** (merges overlapping buffers)
   - Output: `roads_buffered.geojson`
2. Run

**Variable-width buffer (advanced):**
If you want different widths per road class, use a **data-defined override** on the Distance field:

```
CASE
  WHEN "highway" = 'motorway' THEN 15
  WHEN "highway" IN ('primary', 'trunk') THEN 8
  WHEN "highway" IN ('secondary', 'tertiary') THEN 5
  ELSE 3
END
```

### 4.6 Rasterize Road Vectors into a Binary Mask

> This is the most critical step. The mask must be _exactly_ aligned to the image.

1. `Raster` → `Conversion` → `Rasterize (Vector to Raster)`
   - Input layer: `roads_buffered.geojson`
   - Field to use for burn-in value: use **"A fixed value to burn"** → set to **1**
   - Output raster size units: **Georeferenced units**
   - Width/Horizontal resolution: **10** (same as Sentinel-2)
   - Height/Vertical resolution: **10** (same as Sentinel-2)
   - Output extent: **click `…` → "Calculate from Layer" → select `image_clipped.tif`**
   - Output data type: **Byte**
   - Pre-initialize with value: **0** (background)
   - ✅ Assign a specified NoData value: **255** (or leave unchecked)
   - Output file: `mask.tif`
2. Run

### 4.7 Verify Alignment (MANDATORY)

**Method 1 — Visual overlay in QGIS:**

1. Load both `image_clipped.tif` and `mask.tif`
2. Set mask to semi-transparent (Symbology → Opacity 50 %)
3. Roads in the mask should sit exactly on top of visible roads in imagery

**Method 2 — Check metadata:**
Right-click each layer → Properties → Information:

| Property    | `image_clipped.tif` | `mask.tif`    | Must Match? |
| ----------- | ------------------- | ------------- | ----------- |
| CRS         | EPSG:32643          | EPSG:32643    | ✅ Yes      |
| Pixel size  | 10, -10             | 10, -10       | ✅ Yes      |
| Width (px)  | N                   | N             | ✅ Yes      |
| Height (px) | M                   | M             | ✅ Yes      |
| Extent      | x1,y1 – x2,y2       | x1,y1 – x2,y2 | ✅ Yes      |

**Method 3 — Python verification:**

```python
import rasterio

with rasterio.open('image_clipped.tif') as img, rasterio.open('mask.tif') as msk:
    assert img.crs == msk.crs, f"CRS mismatch: {img.crs} vs {msk.crs}"
    assert img.width == msk.width, f"Width mismatch: {img.width} vs {msk.width}"
    assert img.height == msk.height, f"Height mismatch: {img.height} vs {msk.height}"
    assert img.transform == msk.transform, f"Transform mismatch"
    print("✅ Image and mask are perfectly aligned.")
```

### 4.8 Tile Into Patches (Python)

Sentinel-2 scenes are large (10980 × 10980 px). You must cut them into small tiles for training.

```python
"""tile_dataset.py — Cut aligned image + mask into 256×256 patches."""
import os, numpy as np, rasterio
from rasterio.windows import Window

def tile_pair(image_path, mask_path, out_dir, tile_size=256, stride=256):
    """Tile an image and its mask into aligned patches."""
    os.makedirs(f"{out_dir}/images", exist_ok=True)
    os.makedirs(f"{out_dir}/masks",  exist_ok=True)

    with rasterio.open(image_path) as img_src, rasterio.open(mask_path) as msk_src:
        # Verify alignment
        assert img_src.width == msk_src.width and img_src.height == msk_src.height

        count = 0
        for y in range(0, img_src.height - tile_size + 1, stride):
            for x in range(0, img_src.width - tile_size + 1, stride):
                window = Window(x, y, tile_size, tile_size)

                img_tile = img_src.read(window=window)          # (3, 256, 256)
                msk_tile = msk_src.read(1, window=window)       # (256, 256)

                # Skip tiles with no road pixels (optional: keep ~10% of empty tiles)
                road_ratio = msk_tile.sum() / msk_tile.size
                if road_ratio < 0.001:
                    if np.random.rand() > 0.1:  # keep 10% of empty tiles
                        continue

                # Save tiles
                profile = img_src.profile.copy()
                profile.update(width=tile_size, height=tile_size,
                               transform=rasterio.windows.transform(window, img_src.transform))

                with rasterio.open(f"{out_dir}/images/tile_{count:05d}.tif", 'w', **profile) as dst:
                    dst.write(img_tile)

                msk_profile = msk_src.profile.copy()
                msk_profile.update(width=tile_size, height=tile_size, count=1,
                                   transform=rasterio.windows.transform(window, msk_src.transform))

                with rasterio.open(f"{out_dir}/masks/tile_{count:05d}.tif", 'w', **msk_profile) as dst:
                    dst.write(msk_tile[np.newaxis, :, :])

                count += 1

    print(f"Created {count} tile pairs in {out_dir}")

if __name__ == "__main__":
    tile_pair("image_clipped.tif", "mask.tif", "dataset/train", tile_size=256, stride=128)
```

---

## 5. Dataset Structure for ML

```
dataset/
├── train/
│   ├── images/          # 3-band GeoTIFFs, 256×256
│   │   ├── tile_00000.tif
│   │   ├── tile_00001.tif
│   │   └── ...
│   └── masks/           # 1-band binary GeoTIFFs, 256×256
│       ├── tile_00000.tif
│       ├── tile_00001.tif
│       └── ...
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

**Split strategy:** 70 / 15 / 15 (train / val / test) — split _spatially_ (different geographic areas), **not** randomly from the same image. Random splitting causes data leakage because adjacent tiles are highly correlated.

---

## 6. Model Selection & Justification

### Recommendation: **U-Net**

| Factor                       | U-Net                                      | DeepLabV3+                                        |
| ---------------------------- | ------------------------------------------ | ------------------------------------------------- |
| Input resolution             | Works well at any                          | Designed for higher-res                           |
| Feature size (roads at 10 m) | ✅ Skip connections preserve thin features | Dilated convolutions may "skip over" thin targets |
| Training data needed         | ✅ Trains well with 500–2000 tiles         | Needs more data                                   |
| GPU memory                   | ✅ Lightweight                             | Heavier backbone                                  |
| Implementation complexity    | ✅ Simple                                  | More complex                                      |
| Internship-friendly          | ✅ Easy to explain & debug                 | Harder to justify design choices                  |

**Why U-Net wins here:**

1. **Roads are thin, linear features** at 10 m resolution (often just 1–3 pixels wide even after buffering). U-Net's **skip connections** directly propagate fine spatial details from the encoder to the decoder, which is critical for thin-object segmentation.

2. **Small dataset** — you likely have one or a few Sentinel-2 tiles, producing hundreds to low thousands of patches. U-Net trains effectively on small datasets; DeepLabV3+ with a ResNet backbone is data-hungry.

3. **Interpretability** — for an internship deliverable, a clean U-Net is easier to explain, debug, and present.

### Architecture Details

```
Input (3, 256, 256) — RGB, float32, normalized
    ↓
Encoder: 4 downsampling blocks (conv-bn-relu × 2 + maxpool)
    Channels: 64 → 128 → 256 → 512
    ↓
Bottleneck: 1024 channels
    ↓
Decoder: 4 upsampling blocks (upsample + concat skip + conv-bn-relu × 2)
    Channels: 512 → 256 → 128 → 64
    ↓
Output: 1×1 conv → sigmoid → (1, 256, 256) probability map
```

---

## 7. Python Training Pipeline

> See the companion files: `dataset.py` and `train_road_segmentation.py`

### 7.1 Dependencies

```bash
pip install torch torchvision rasterio numpy matplotlib scikit-learn tqdm
```

### 7.2 Key Design Decisions

| Decision      | Choice                                                  | Rationale                                                                 |
| ------------- | ------------------------------------------------------- | ------------------------------------------------------------------------- |
| Normalization | Per-channel min-max to [0, 1] using dataset-level stats | Sentinel-2 values are uint16 reflectance; ImageNet stats don't apply      |
| Loss function | BCE + Dice (combined)                                   | BCE alone struggles with class imbalance; Dice directly optimizes overlap |
| Optimizer     | Adam, lr=1e-4                                           | Reliable default for segmentation                                         |
| Augmentation  | Random flip, rotation, brightness jitter                | Small dataset → augmentation is essential                                 |
| Batch size    | 8–16                                                    | Fits in ~6 GB VRAM with 256² tiles                                        |

### 7.3 File Reference

| File                         | Purpose                                                        |
| ---------------------------- | -------------------------------------------------------------- |
| `dataset.py`                 | PyTorch Dataset class — loads tile pairs, normalizes, augments |
| `train_road_segmentation.py` | Full training loop, U-Net model, loss, metrics, checkpointing  |
| `tile_dataset.py`            | Tiling script (shown in Section 4.8)                           |

---

## 8. Evaluation

### 8.1 Metrics

| Metric            | Formula                 | What It Tells You                                      |
| ----------------- | ----------------------- | ------------------------------------------------------ |
| **IoU (Jaccard)** | TP / (TP + FP + FN)     | How well predicted and true road pixels overlap        |
| **Dice (F1)**     | 2·TP / (2·TP + FP + FN) | Harmonic mean of precision & recall for the road class |
| **Precision**     | TP / (TP + FP)          | Of all pixels you called "road", how many are correct  |
| **Recall**        | TP / (TP + FN)          | Of all true road pixels, how many did you find         |

**Target:** IoU ≥ 0.35–0.45 is a _reasonable_ result for 10 m Sentinel-2 imagery with OSM labels (which are noisy). IoU ≥ 0.5 is good.

### 8.2 Visual Evaluation

Always produce side-by-side visualizations:

```
[Input RGB]  |  [Ground Truth Mask]  |  [Predicted Mask]  |  [Overlay]
```

This is more informative than numbers alone. Include these in your report.

---

## 9. Common Mistakes & How to Avoid Them

### 🔴 CRS / Alignment Errors

| Mistake                     | Consequence             | Fix                                         |
| --------------------------- | ----------------------- | ------------------------------------------- |
| Image in UTM, mask in WGS84 | Mask shifted by km      | Reproject _before_ rasterizing              |
| Rasterize with wrong extent | Mask offset by pixels   | Set output extent from the image layer      |
| Different pixel size        | Mask stretched/squeezed | Set rasterize resolution = image resolution |

**Rule of thumb:** After _every_ processing step, check CRS + extent + pixel size.

### 🔴 Forgetting to Buffer Roads

- OSM roads are lines → 1 pixel in the mask → model learns nothing
- **Always buffer** to at least 3–5 m per side

### 🔴 Random Train/Val Split of Adjacent Tiles

- Adjacent 256 × 256 patches from the same scene are ~90 % identical
- **Split geographically** — use different parts of the AOI or different tiles for train vs. val

### 🔴 Wrong Normalization

- Sentinel-2 L2A values are uint16 surface reflectance (typically 0–10000)
- **Do NOT** use ImageNet mean/std (those are for 0–255 uint8 photos)
- Normalize per-channel: `(x - min) / (max - min)` or `x / 10000.0`

### 🔴 Ignoring Class Imbalance

- In most tiles, > 90 % of pixels are _not_ road
- Using only BCE loss → model predicts "no road" everywhere → 90 % accuracy but 0 % IoU
- Use **Dice loss** or **BCE + Dice** combined loss

### 🔴 Not Checking the Mask Values

- Your mask file should contain exactly **0** (background) and **1** (road)
- If it contains 0 and 255, or float values, the loss function breaks silently
- **Always print** `np.unique(mask)` on a few tiles before training

### 🔴 Training Without Augmentation

- With < 2000 tiles, the model will overfit within 10 epochs
- Use at minimum: horizontal flip, vertical flip, 90° rotations

---

## 10. Reproducibility Checklist

- [ ] Document the exact Sentinel-2 product ID you used
- [ ] Document the Overpass Turbo query or Geofabrik download date
- [ ] Save the QGIS project file (`.qgz`) with all layers
- [ ] Version-control the tiling and training scripts
- [ ] Pin Python dependencies (`pip freeze > requirements.txt`)
- [ ] Save the trained model weights (`.pth`)
- [ ] Record train/val/test IoU and Dice in a results table
- [ ] Include side-by-side prediction visualizations in the report
- [ ] Note the GPU used and training time
- [ ] Spatial train/val/test split boundaries documented on a map

---

_This guide accompanies `dataset.py` and `train_road_segmentation.py` in the same directory._
