# Road Detection from Satellite Imagery

Detect roads in Sentinel-2 satellite imagery using deep learning — trained on Indian roads across 6 cities.

## Overview

End-to-end pipeline for binary road segmentation:

1. **Data Acquisition** — Sentinel-2 imagery (Planetary Computer) + OSM road labels (Overpass API)
2. **Preprocessing** — Rasterize road vectors, tile into 256×256 patches
3. **Training** — U-Net with ResNet-34 encoder (ImageNet pretrained), Dice loss
4. **Prediction** — Road mask generation with sliding window for full satellite images
5. **Evaluation** — IoU, Dice, Precision, Recall + visual predictions

## Architecture

```
Sentinel-2 RGB (10m) ──┐
                        ├──► U-Net + ResNet-34 ──► Road Mask
OSM Road Labels ────────┘
```

| Component   | Choice                | Why                                       |
| ----------- | --------------------- | ----------------------------------------- |
| Imagery     | Sentinel-2 L2A        | Free, 10m resolution, global coverage     |
| Labels      | OpenStreetMap         | Free, crowd-sourced road network          |
| Model       | U-Net                 | Proven for segmentation                   |
| Encoder     | ResNet-34 / EffNet-B0 | ResNet (Accuracy) or EfficientNet (Speed) |
| Loss        | Dice Loss             | Handles road/background class imbalance   |
| Framework   | PyTorch + smp         | Industry standard                         |
| Data Source | Planetary Computer    | Free STAC API, no account needed          |

## Quick Start

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate          # Linux/Mac
# .\venv\Scripts\activate         # Windows
pip install -r requirements.txt
pip install pystac-client planetary-computer pyproj shapely

# 2. Download all 6 Indian cities (Delhi, Mumbai, Chennai, Kolkata, Goa, Port Blair)
python download_india_data.py --cities all

# 3. Train on combined dataset (ResNet-34 default)
python train_torchgeo.py --data_dir data/india_combined --epochs 50

# Train faster on CPU (EfficientNet-B0)
python train_torchgeo.py --data_dir data/india_combined --encoder efficientnet-b0 --epochs 50

# 4. Predict road masks
python predict.py --model checkpoints/best_model.pth --input data/delhi/sentinel2_rgb.tif

# Quick test (no data needed)
python train_torchgeo.py --dry_run
```

## Project Structure

```
road-detection/
├── README.md                       # This file
├── HOW_TO_RUN.md                   # Detailed step-by-step guide
├── The_Everything.md               # Full technical documentation
├── requirements.txt                # Python dependencies
│
├── download_india_data.py          # Step 1: Multi-city data download
├── download_delhi_data.py          # Single-city download (Delhi only)
├── tile_dataset.py                 # Tile image+mask into patches
├── train_torchgeo.py               # Step 2: U-Net + ResNet-34 training
├── predict.py                      # Step 3: Generate road mask predictions
│
├── data/                           # Downloaded + processed data
│   ├── delhi/                      # Per-city data
│   ├── mumbai/
│   ├── chennai/
│   ├── kolkata/
│   ├── goa/
│   ├── portblair/
│   └── india_combined/             # Merged train/val split
│       ├── train/images/ & masks/
│       └── val/images/ & masks/
│
├── predictions/                    # Road mask outputs
│   ├── delhi/
│   ├── mumbai/
│   └── ...                         # Per-city prediction masks + visualizations
│
└── checkpoints/                    # Training outputs
    ├── best_model.pth
    ├── training_curves.png
    └── sample_predictions.png
```

## Fine-Tuning

Fine-tune an existing model on new data:

```bash
# Train decoder only (faster)
python train_torchgeo.py --data_dir data/delhi --finetune model.pth --freeze_encoder

# Full fine-tune
python train_torchgeo.py --data_dir data/delhi --finetune model.pth
```

## Data Sources

| Source                                                                                      | What                   | Access           |
| ------------------------------------------------------------------------------------------- | ---------------------- | ---------------- |
| [Planetary Computer](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a)        | Sentinel-2 L2A imagery | Free, no account |
| [Overpass API](https://overpass-api.de/)                                                    | OSM road vectors       | Free             |
| [Massachusetts Roads](https://www.kaggle.com/datasets/balraj98/massachusetts-roads-dataset) | Pre-made dataset       | Kaggle (free)    |

## Training Cities

| City       | Area               | Road Types                 |
| ---------- | ------------------ | -------------------------- |
| Delhi      | Metro, North India | Urban grid, highways       |
| Mumbai     | Coastal metro      | Dense urban, coastal roads |
| Chennai    | South India metro  | Planned grid, arterials    |
| Kolkata    | East India metro   | Dense urban, bridges       |
| Goa        | Small state        | Rural, coastal, narrow     |
| Port Blair | Island             | Sparse, island roads       |

```bash
# Download specific cities:
python download_india_data.py --cities delhi mumbai goa

# Download all:
python download_india_data.py --cities all

# Custom area (anywhere in India):
python download_india_data.py --bbox 73.85 18.50 73.95 18.55 --name pune
```

## Prediction

```bash
# Predict on a satellite image:
python predict.py --model checkpoints/best_model.pth --input data/delhi/sentinel2_rgb.tif

# Predict on a folder of tiles:
python predict.py --model checkpoints/best_model.pth --input data/delhi/tiles/images/

# Adjust threshold (lower = more roads detected):
python predict.py --model checkpoints/best_model.pth --input data/delhi/sentinel2_rgb.tif --threshold 0.4
```

Outputs per image: GeoTIFF mask (`.tif`) + 4-panel visualization (`.png`).

## Requirements

- Python 3.9+
- PyTorch 2.0+
- See `requirements.txt` for full list

## Results

Trained on 6 Indian cities (547 train + 96 val tiles, 50 epochs, CPU):

| Metric    | Value  |
| --------- | ------ |
| Best IoU  | 0.3239 |
| Best Dice | 0.4880 |
| Precision | 0.3795 |
| Recall    | 0.6950 |

### Prediction Results

| City       | Road Coverage |
| ---------- | ------------- |
| Delhi      | 33.7%         |
| Mumbai     | 8.9%          |
| Chennai    | 25.8%         |
| Kolkata    | 28.6%         |
| Goa        | 0.9%          |
| Port Blair | 0.5%          |

## Acknowledgments

- [TorchGeo](https://github.com/microsoft/torchgeo) — geospatial deep learning
- [segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) — U-Net implementation
- [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) — free Sentinel-2 access
- [OpenStreetMap](https://www.openstreetmap.org/) — road labels
