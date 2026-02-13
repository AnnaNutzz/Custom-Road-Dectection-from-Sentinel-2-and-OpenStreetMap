# How to Run — Road Detection Pipeline

## Prerequisites

- Python 3.9+
- ~4 GB disk space (for venv + data)
- Internet connection (for data download)
- GPU optional (works on CPU, just slower)

---

## Step 0: Setup

```powershell
cd "d:\Internship related\GJ-Map Solutions\road-detection"
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
pip install pystac-client planetary-computer pyproj shapely albumentations
```

---

## Step 1: Download Data

### All 6 Indian Cities (Recommended)

```powershell
# Download Delhi, Mumbai, Chennai, Kolkata, Goa, Port Blair
# Downloads Sentinel-2 + OSM for each, tiles, and merges into train/val
python download_india_data.py --cities all
```

| Argument          | Default | Description                                        |
| ----------------- | ------- | -------------------------------------------------- |
| `--cities`        | `all`   | Cities: delhi mumbai chennai kolkata goa portblair |
| `--bbox`          |         | Custom area: `--bbox WEST SOUTH EAST NORTH`        |
| `--name`          |         | Name for custom area (required with --bbox)        |
| `--data_dir`      | `data`  | Base output directory                              |
| `--max_cloud`     | `15`    | Max cloud cover %                                  |
| `--buffer`        | `5`     | Road buffer width (meters)                         |
| `--tile_size`     | `256`   | Tile size in pixels                                |
| `--stride`        | `128`   | Tile stride (128 = 50% overlap)                    |
| `--skip_download` |         | Skip download, just re-merge existing tiles        |

```powershell
# Download specific cities only:
python download_india_data.py --cities delhi mumbai goa

# Custom area (anywhere in India — use bboxfinder.com for coordinates):
python download_india_data.py --bbox 73.85 18.50 73.95 18.55 --name pune

# Custom area + predefined cities:
python download_india_data.py --cities delhi mumbai --bbox 73.85 18.50 73.95 18.55 --name pune

# Re-merge tiles without re-downloading:
python download_india_data.py --cities all --skip_download
```

### Single City (Delhi Only)

```powershell
python download_delhi_data.py
python download_delhi_data.py --bbox 77.15 28.60 77.35 28.75  # larger area
```

---

## Step 2: Train

```powershell
# Quick test (no data needed)
python train_torchgeo.py --dry_run

# Train on all cities combined (ResNet-34)
python train_torchgeo.py --data_dir data/india_combined --epochs 50

# Train faster (EfficientNet-B0)
python train_torchgeo.py --data_dir data/india_combined --encoder efficientnet-b0 --epochs 50

# Train on single city
python train_torchgeo.py --data_dir data/delhi/tiles --epochs 50
```

| Argument           | Default           | Description                             |
| ------------------ | ----------------- | --------------------------------------- |
| `--data_dir`       | `data/mass_roads` | Dataset root (train/ and val/ subdirs)  |
| `--epochs`         | `50`              | Training epochs                         |
| `--batch_size`     | `8`               | Batch size (8 fits ~4GB VRAM)           |
| `--lr`             | `1e-4`            | Learning rate                           |
| `--save_dir`       | `checkpoints`     | Output directory                        |
| `--encoder`        | `resnet34`        | `resnet34` or `efficientnet-b0`         |
| `--workers`        | `0`               | DataLoader workers (0 safe for Windows) |
| `--device`         | `auto`            | `cuda`, `cpu`, or `auto`                |
| `--dry_run`        |                   | Test with synthetic data                |
| `--finetune`       |                   | Path to .pth to fine-tune               |
| `--freeze_encoder` |                   | Freeze encoder during fine-tuning       |

### Outputs (in `checkpoints/`)

- `best_model.pth` — best weights by validation IoU
- `training_curves.png` — loss and metric plots
- `sample_predictions.png` — side-by-side predictions

---

## Step 3: Predict

```powershell
# Predict on a full satellite image:
python predict.py --model checkpoints/best_model.pth --input data/delhi/sentinel2_rgb.tif

# Predict on a folder of tiles:
python predict.py --model checkpoints/best_model.pth --input data/delhi/tiles/images/

# Custom output directory:
python predict.py --model checkpoints/best_model.pth --input data/delhi/sentinel2_rgb.tif --output_dir predictions/delhi_custom
```

| Argument       | Default                      | Description                      |
| -------------- | ---------------------------- | -------------------------------- |
| `--model`      | `checkpoints/best_model.pth` | Path to trained model            |
| `--input`      | _(required)_                 | Image file or directory of tiles |
| `--output_dir` | `predictions`                | Output directory                 |
| `--tile_size`  | `256`                        | Sliding window tile size         |
| `--threshold`  | `0.5`                        | Road probability threshold       |
| `--device`     | `auto`                       | `cuda`, `cpu`, or `auto`         |

### Outputs (per image)

- `*_road_mask.tif` — GeoTIFF binary mask (0=background, 255=road)
- `*_prediction.png` — 4-panel viz: satellite | probability heatmap | binary mask | red overlay

---

## Step 4: Evaluate

Training prints: **IoU**, **Dice**, **Precision**, **Recall**

Target metrics:

- IoU ≥ 0.30 = learning (10m resolution)
- IoU ≥ 0.40 = good
- IoU ≥ 0.50 = excellent

---

## Quick Reference

```powershell
.\venv\Scripts\activate
python download_india_data.py --cities all
python train_torchgeo.py --data_dir data/india_combined --epochs 50
python predict.py --model checkpoints/best_model.pth --input data/delhi/sentinel2_rgb.tif
```
