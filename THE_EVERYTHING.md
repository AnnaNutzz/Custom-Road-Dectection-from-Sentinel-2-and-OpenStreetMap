# What Exactly Does this Program Do? - Pipeline

It downloads satellite images of Indian cities using OpenStreetMap to know where the roads are, and trains a deep learning model to automatically detect roads in new satellite images.

```
DOWNLOAD                    TRAIN                       USE
────────                    ─────                       ───
Sentinel-2 satellite    →   U-Net learns:               Give it any new
images (free, 10m)          "this image patch            satellite image →
        +                    has roads HERE"             it predicts
OSM road maps           →   by comparing its            road locations
(free, crowd-sourced)       guesses against              pixel by pixel
                            OSM ground truth
```

## Summary

> The model looks at satellite image patches and learns to predict which pixels are roads by comparing its predictions against OSM-derived ground truth masks, using Dice Loss to handle the class imbalance (roads are small \% of pixels). The ResNet-34 encoder gives it a head start because it already understands edges and textures from ImageNet.

# What is the input to this program?

## What is the input exactly?

Image Tile → 256x256 pixel RGB satellite image patch \
Mask Tile → 256x256 pixel binary mask (1 if road, 0 if not)

Basically,

- Give: city names/bounding box
- Script: downloads Sentinel-2 satellite images and OSM road maps for the given city/bounding box
- Model receives: Image Tiles and Mask Tiles
- Model outputs: Predicted Road Masks

## Where is the data stored?

The raw downloaded images are stored in `data/<city_name>/images/`.
For training, we combine them into a single dataset:

```
data/india_combined/
├── train/
│   ├── images/  ← Satellite tiles (input X) - 547 files
│   └── masks/   ← Road masks (label Y)      - 547 files
└── val/
    ├── images/  ← Validation tiles          - 96 files
    └── masks/
```

## Example Usage:

We can achieve needed output by training the model on the training data from OSM and Sentinel-2 of specific cities. For example, we can train the model on the training data of Delhi and then use the trained model to detect roads in the satellite images of Delhi.

```powershell
python download_delhi_data.py                # Download data for Delhi only
# Train the model on the Delhi data
python train_torchgeo.py --data_dir data/delhi --epochs 50

python download_india_data.py --cities all   # Download data for all 6 cities
# Train the model on the combined data
python train_torchgeo.py --data_dir data/india_combined --epochs 50

# Custom area (anywhere in India — get coordinates from http://bboxfinder.com/):
python download_india_data.py --bbox 73.85 18.50 73.95 18.55 --name pune
python train_torchgeo.py --data_dir data/pune --epochs 50
```

# Custom Area

```powershell
# Step 1: Get coordinates from http://bboxfinder.com/ or https://browser.dataspace.copernicus.eu/ (west south east north)
# Step 2: Download + tile
python download_india_data.py --bbox 73.85 18.50 73.95 18.55 --name pune

# Step 3: Train
python train_torchgeo.py --data_dir data/pune --epochs 50

# Step 4: Predict
python predict.py --model checkpoints/best_model.pth --input data/pune/sentinel2_rgb.tif
```

# Training

## What is it Training on exactly? - the training data

Image Tile → Sentinel-2 256x256 pixel RGB satellite image patch in GeoTIFF format 10m resolution \
Mask Tile → OpenStreetMap road vectors converted to 256x256 pixel binary mask (1 if road, 0 if not)

## Where is it Training exactly?

```
data/india_combined/
├── train/
│   ├── images/  ← Satellite tiles (input X)
│   └── masks/   ← Road masks (label Y)
└── val/
    ├── images/
    └── masks/

         ↓  loads via SimpleRoadDataset

    U-Net + ResNet-34 encoder
         ↓
    DiceLoss(predicted_mask, ground_truth_mask)
         ↓
    Adam optimizer updates weights
         ↓
    Repeat for 50 epochs
         ↓
    checkpoints/best_model.pth
```

## How does it Train?

```
Input Image (3, 256, 256)
       ↓
┌────────────────────────┐
│  ResNet-34 Encoder/    │  ← Pretrained on ImageNet (already knows edges, textures)
|  EfficientNet-B0       │  ← Pretrained on ImageNet (already knows edges, textures)
│  (Downsampling path)   │
│  256→128→64→32→16      │
└──────────┬─────────────┘
           ↓
┌──────────────────────┐
│    U-Net Decoder     │  ← Learns to upsample back to full resolution
│  (Upsampling path)   │
│  16→32→64→128→256    │
└──────────┬───────────┘
           ↓
Output Mask (1, 256, 256)  ← Per-pixel probability: "is this pixel a road?"
```

```powershell
# For each batch of 8 tiles:
images = batch of 8 satellite tiles     # shape: (8, 3, 256, 256)
masks  = batch of 8 road masks          # shape: (8, 1, 256, 256)  values: 0 or 1

# Forward pass — model predicts which pixels are roads
predictions = model(images)             # shape: (8, 1, 256, 256)  values: raw scores

# Model Architecture & Comparison

We support two backbones for the U-Net architecture. You can switch between them using `--encoder`.
```

| Feature            | ResNet-34 (Default)                          | EfficientNet-B0 (Recommended)                    |
| ------------------ | -------------------------------------------- | ------------------------------------------------ |
| **Parameters**     | ~24.4 Million                                | ~6.3 Million (4x smaller)                        |
| **Training Speed** | Slow (~10-15 mins/epoch on CPU)              | **Fast** (~3 mins/epoch on CPU)                  |
| **Stability**      | High memory usage, caused crashes on Windows | **Stable**, runs smoothly on CPU                 |
| **Best IoU**       | ~0.31 (Historical)                           | ~0.26 (Current Baseline)                         |
| **Use Case**       | Absolute maximum accuracy if you have a GPU  | **CPU training, fast iteration, large datasets** |

```
> [!NOTE]
> The lower IoU of EfficientNet-B0 (0.26) vs ResNet-34 (0.31) is largely due to **stronger augmentations** added to prevent overfitting. The 0.26 score is likely more "real" and robust than the previous 0.31.

## Loss — how wrong is the prediction?
loss = DiceLoss(predictions, masks)     # single number, e.g. 0.72

# Backward pass — compute gradients
loss.backward()                         # calculates how to adjust each weight

# Update weights
optimizer.step()                        # nudges weights to reduce the loss
```

Roads are rare pixels (~2-5% of image). Normal loss would learn "predict all background" and get 95% accuracy. Dice Loss fixes this:

```powershell
Dice = 2 × |predicted ∩ ground_truth| / (|predicted| + |ground_truth|)
Only gets a good score if it correctly finds the road pixels.
```

## The epochs

```
Epoch 1:
  ├─ Train: loop through ALL training tiles in batches of 8
  │    → compute loss, update weights (model learns)
  │
  ├─ Validate: loop through ALL val tiles
  │    → compute IoU, Dice (how good is it? but NO weight updates)
  │
  └─ Save if best IoU so far → best_model.pth

Epoch 2: repeat
...
Epoch 50: repeat

After 50 epochs, the model has seen every tile 50 times.
```

# To Improve further

- More epochs — try 100 or 150
- Increase road buffer — `--buffer 10` instead of 5m makes roads 2 pixels wide instead of sub-pixel
- Lower learning rate — `--lr 5e-5` for more careful learning
- **Strong Augmentations**: `albumentations` adds color jitter, elastic transforms, and noise (install with `pip install albumentations`)
- **Regularization**:
  - Weight decay (`1e-4`)
  - Early stopping (patience=15)
  - `scSE` decoder attention
- **Encoder Selection**:
  - `python train_torchgeo.py --encoder efficientnet-b0` (lighter model)
  - `python train_torchgeo.py --encoder resnet34` (default)

- Fine-tune — freeze encoder first, train decoder, then unfreeze:

```powershell
python train_torchgeo.py --data_dir data/india_combined --finetune checkpoints/best_model.pth --freeze_encoder --epochs 30
python train_torchgeo.py --data_dir data/india_combined --finetune checkpoints/best_model.pth --epochs 30
```

# Metrics used

**IoU (Intersection over Union)** - _Overlap between predicted and real roads_
= overlap / total_area
= 0.0 to 1.0
= 1.0 means perfect match

**Dice Score** - _Similar to IoU, but gives more weight to the smaller region_
= 2 × overlap / (predicted_pixels + ground_truth_pixels)
= 0.0 to 1.0
= 1.0 means perfect match

**F1 Score** - _Harmonic mean of precision and recall_
= 2 × (precision × recall) / (precision + recall)
= 0.0 to 1.0
= 1.0 means perfect match

**Precision** - _Of all pixels we predicted as road, what % actually are roads?_
= true_positives / (true_positives + false_positives)
= of all pixels we predicted as road, what % actually are roads?

**Recall** - _Of all actual roads, what % did we find?_
= true_positives / (true_positives + false_negatives)
= of all actual roads, what % did we find?

# Tech Stack

```powershell
Python 3.10+ (tested on 3.14.2)
PyTorch
TorchGeo
requests
rasterio
NumPy
shapely
Matplotlib
```

| What       | Why                                                                                                                              |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Sentinel-2 | Free satellite photos from ESA, covers all of India at 10m detail                                                                |
| OSM        | Free, Free "answer key" — humans have already mapped roads, we use that as training labels, crowd-sourced, road network data     |
| U-Net      | Deep learning model for semantic segmentation and neural network architecture designed specifically for "find objects in images" |
| ResNet-34  | A pre-trained "brain" that already understands edges, textures, shapes — so the model doesn't start from zero                    |
| Dice Loss  | A training objective that focuses on finding rare pixels (roads are only ~2-5% of an image)                                      |
| IoU        | Evaluation metric for segmentation tasks                                                                                         |
| F1 Score   | Evaluation metric for segmentation tasks                                                                                         |
| Precision  | Evaluation metric for segmentation tasks                                                                                         |
| Recall     | Evaluation metric for segmentation tasks                                                                                         |

# Prediction (predict.py)

After training, use `predict.py` to generate road masks for any satellite image.

## Usage

```powershell
# Predict on a single satellite image:
python predict.py --model checkpoints/best_model.pth --input data/delhi/sentinel2_rgb.tif

# Predict on a folder of tiles:
python predict.py --model checkpoints/best_model.pth --input data/delhi/tiles/images/

# Lower threshold to detect more roads:
python predict.py --model checkpoints/best_model.pth --input data/delhi/sentinel2_rgb.tif --threshold 0.4
```

Using ResNet-34 (Default)

```powershell
python train_torchgeo.py --data_dir data/india_combined --encoder resnet34 --epochs 50
```

How to use other encoders: _The script supports these options: resnet18, resnet34, resnet50, efficientnet-b0, efficientnet-b1._

```powershell
# Faster, lighter model
python train_torchgeo.py --data_dir data/india_combined --encoder resnet18

# Larger, more powerful model
python train_torchgeo.py --data_dir data/india_combined --encoder resnet50

# EfficientNet-b1
python train_torchgeo.py --data_dir data/india_combined --encoder efficientnet-b1
```

## How it works

For large images (bigger than 256x256), it uses a **sliding window**:

1. Slides a 256x256 window across the image with overlap
2. Predicts road probability for each window
3. Averages overlapping predictions
4. Applies threshold to create binary mask

## Output per image

- `*_road_mask.tif` — GeoTIFF binary mask (0=background, 255=road), openable in QGIS
- `*_prediction.png` — 4-panel visualization: satellite | probability heatmap | binary mask | red overlay

## Prediction Results (trained model, IoU=0.3239)

| City       | Image Size   | Road Coverage |
| ---------- | ------------ | ------------- |
| Delhi      | 2941x3950 px | 33.7%         |
| Mumbai     | 2998x4164 px | 8.9%          |
| Chennai    | 1852x2539 px | 25.8%         |
| Kolkata    | 2037x1527 px | 28.6%         |
| Goa        | 3237x3081 px | 0.9%          |
| Port Blair | 1092x1105 px | 0.5%          |

# File Structure

```
road-detection/
├── download_india_data.py          # Step 1: Download data
├── train_torchgeo.py               # Step 2: Train model
├── predict.py                      # Step 3: Generate predictions
├── download_delhi_data.py          # Single-city download
├── tile_dataset.py                 # Manual tiling utility
│
├── data/
│   ├── delhi/
│   │   ├── sentinel2_rgb.tif
│   │   ├── road_mask.tif
│   │   ├── roads.geojson
│   │   └── tiles/images/ & masks/
│   ├── mumbai/  (same structure)
│   ├── chennai/ (same structure)
│   ├── kolkata/ (same structure)
│   ├── goa/     (same structure)
│   ├── portblair/ (same structure)
│   └── india_combined/
│       ├── train/
│       │   ├── images/  (~85% of all tiles)
│       │   └── masks/
│       └── val/
│           ├── images/  (~15% of all tiles)
│           └── masks/
│
├── predictions/
│   ├── delhi/
│   │   ├── sentinel2_rgb_road_mask.tif
│   │   └── sentinel2_rgb_prediction.png
│   ├── mumbai/
│   └── ...
│
└── checkpoints/
    ├── best_model.pth
    ├── training_curves.png
    └── sample_predictions.png
```

# OUTPUTS

## Expected Ouput for Training

```powershell

Cities: Delhi, Mumbai, Chennai, Kolkata, Goa, Port Blair

============================================================
  Delhi — bbox: (76.95, 28.45, 77.35, 28.8)
============================================================

  [Sentinel-2] Querying for Delhi...
  Scene: S2C_MSIL2A_20251224T053241_R105_T43RGM_20251224T091910
  Date: 2025-12-24
  Cloud: 0.0%
  Downloading B04... OK (3882x3882)
  Downloading B03... OK (3882x3882)
  Downloading B02... OK (3882x3882)
  Saved: data/delhi/sentinel2_rgb.tif (3882x3882 px)

  [OSM] Downloading roads for Delhi...
  Found 12,450 road segments

  [Rasterize] Creating road mask...
  Road pixels: 245,000 (1.6%)
  Saved: data/delhi/road_mask.tif

  [Tile] Cutting into 256x256 patches...
  Tiles: 340 saved, 120 skipped (low road content)

  ... (repeats for each city) ...

Merging 1,200 tiles → 1,020 train + 180 val

DONE! Ready to train:
  python train_torchgeo.py --data_dir data/india_combined --epochs 50
```

## Expected Output for Epochs

### Using ResNet-34

```powershell
(venv) PS D:\Internship related\GJ-Map Solutions\road-detection> python train_torchgeo.py --data_dir data/india_combined --epochs 50
Using device: cpu
  Found 547 tiles in data/india_combined\train
  Found 96 tiles in data/india_combined\val
Train: 547 tiles
Val:   96 tiles
Parameters: 24,436,369 total, 24,436,369 trainable

============================================================
Training: U-Net + ResNet-34 | 50 epochs | lr=0.0001
============================================================

D:\Internship related\GJ-Map Solutions\road-detection\venv\Lib\site-packages\torch\utils\data\dataloader.py:1118: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  super().__init__(loader)
Epoch   1/50 | Train Loss: 0.6706 | Val Loss: 0.6666 | IoU: 0.2226 | Dice: 0.3625 | Prec: 0.2282 | Rec: 0.8888 | 219.4s
  >>> New best model saved (IoU=0.2226)
Epoch   2/50 | Train Loss: 0.6313 | Val Loss: 0.6430 | IoU: 0.2466 | Dice: 0.3943 | Prec: 0.2615 | Rec: 0.8154 | 218.3s
  >>> New best model saved (IoU=0.2466)
Epoch   3/50 | Train Loss: 0.6179 | Val Loss: 0.6436 | IoU: 0.2357 | Dice: 0.3800 | Prec: 0.2412 | Rec: 0.9008 | 216.9s
Epoch   4/50 | Train Loss: 0.6076 | Val Loss: 0.6204 | IoU: 0.2570 | Dice: 0.4074 | Prec: 0.2744 | Rec: 0.8017 | 218.5s
  >>> New best model saved (IoU=0.2570)
Epoch   5/50 | Train Loss: 0.5965 | Val Loss: 0.6210 | IoU: 0.2497 | Dice: 0.3983 | Prec: 0.2598 | Rec: 0.8592 | 221.2s
Epoch   6/50 | Train Loss: 0.5857 | Val Loss: 0.6048 | IoU: 0.2636 | Dice: 0.4157 | Prec: 0.2840 | Rec: 0.7877 | 216.0s
  >>> New best model saved (IoU=0.2636)
Epoch   7/50 | Train Loss: 0.5778 | Val Loss: 0.6013 | IoU: 0.2641 | Dice: 0.4165 | Prec: 0.2830 | Rec: 0.8015 | 215.1s
  >>> New best model saved (IoU=0.2641)
Epoch   8/50 | Train Loss: 0.5742 | Val Loss: 0.5947 | IoU: 0.2679 | Dice: 0.4211 | Prec: 0.2878 | Rec: 0.7930 | 218.4s
  >>> New best model saved (IoU=0.2679)
Epoch   9/50 | Train Loss: 0.5694 | Val Loss: 0.5882 | IoU: 0.2726 | Dice: 0.4269 | Prec: 0.2958 | Rec: 0.7746 | 219.4s
  >>> New best model saved (IoU=0.2726)
Epoch  10/50 | Train Loss: 0.5672 | Val Loss: 0.5864 | IoU: 0.2740 | Dice: 0.4288 | Prec: 0.3020 | Rec: 0.7565 | 216.6s
  >>> New best model saved (IoU=0.2740)
Epoch  11/50 | Train Loss: 0.5611 | Val Loss: 0.5800 | IoU: 0.2789 | Dice: 0.4347 | Prec: 0.3133 | Rec: 0.7299 | 215.9s
  >>> New best model saved (IoU=0.2789)
Epoch  12/50 | Train Loss: 0.5546 | Val Loss: 0.5750 | IoU: 0.2814 | Dice: 0.4375 | Prec: 0.3349 | Rec: 0.6553 | 215.9s
  >>> New best model saved (IoU=0.2814)
Epoch  13/50 | Train Loss: 0.5531 | Val Loss: 0.5693 | IoU: 0.2849 | Dice: 0.4420 | Prec: 0.3237 | Rec: 0.7124 | 216.1s
  >>> New best model saved (IoU=0.2849)
Epoch  14/50 | Train Loss: 0.5497 | Val Loss: 0.5663 | IoU: 0.2873 | Dice: 0.4449 | Prec: 0.3320 | Rec: 0.6901 | 217.3s
  >>> New best model saved (IoU=0.2873)
Epoch  15/50 | Train Loss: 0.5480 | Val Loss: 0.5661 | IoU: 0.2862 | Dice: 0.4437 | Prec: 0.3204 | Rec: 0.7377 | 215.0s
Epoch  16/50 | Train Loss: 0.5416 | Val Loss: 0.5695 | IoU: 0.2831 | Dice: 0.4390 | Prec: 0.3579 | Rec: 0.5957 | 218.6s
Epoch  17/50 | Train Loss: 0.5385 | Val Loss: 0.5599 | IoU: 0.2901 | Dice: 0.4482 | Prec: 0.3356 | Rec: 0.6964 | 218.6s
  >>> New best model saved (IoU=0.2901)
Epoch  18/50 | Train Loss: 0.5379 | Val Loss: 0.5563 | IoU: 0.2927 | Dice: 0.4515 | Prec: 0.3376 | Rec: 0.6998 | 218.6s
  >>> New best model saved (IoU=0.2927)
Epoch  19/50 | Train Loss: 0.5340 | Val Loss: 0.5552 | IoU: 0.2924 | Dice: 0.4512 | Prec: 0.3283 | Rec: 0.7360 | 214.3s
Epoch  20/50 | Train Loss: 0.5333 | Val Loss: 0.5542 | IoU: 0.2935 | Dice: 0.4521 | Prec: 0.3522 | Rec: 0.6516 | 215.7s
  >>> New best model saved (IoU=0.2935)
Epoch  21/50 | Train Loss: 0.5324 | Val Loss: 0.5555 | IoU: 0.2915 | Dice: 0.4503 | Prec: 0.3293 | Rec: 0.7261 | 217.0s
Epoch  22/50 | Train Loss: 0.5271 | Val Loss: 0.5484 | IoU: 0.2984 | Dice: 0.4583 | Prec: 0.3421 | Rec: 0.7090 | 217.2s
  >>> New best model saved (IoU=0.2984)
Epoch  23/50 | Train Loss: 0.5240 | Val Loss: 0.5471 | IoU: 0.2984 | Dice: 0.4584 | Prec: 0.3422 | Rec: 0.7086 | 217.2s
  >>> New best model saved (IoU=0.2984)
Epoch  24/50 | Train Loss: 0.5228 | Val Loss: 0.5450 | IoU: 0.3001 | Dice: 0.4603 | Prec: 0.3477 | Rec: 0.6985 | 220.8s
  >>> New best model saved (IoU=0.3001)
Epoch  25/50 | Train Loss: 0.5199 | Val Loss: 0.5412 | IoU: 0.3039 | Dice: 0.4648 | Prec: 0.3531 | Rec: 0.6955 | 216.4s
  >>> New best model saved (IoU=0.3039)
Epoch  26/50 | Train Loss: 0.5192 | Val Loss: 0.5382 | IoU: 0.3058 | Dice: 0.4670 | Prec: 0.3569 | Rec: 0.6910 | 216.5s
  >>> New best model saved (IoU=0.3058)
Epoch  27/50 | Train Loss: 0.5177 | Val Loss: 0.5373 | IoU: 0.3065 | Dice: 0.4678 | Prec: 0.3645 | Rec: 0.6667 | 215.2s
  >>> New best model saved (IoU=0.3065)
Epoch  28/50 | Train Loss: 0.5156 | Val Loss: 0.5372 | IoU: 0.3068 | Dice: 0.4679 | Prec: 0.3749 | Rec: 0.6350 | 215.8s
  >>> New best model saved (IoU=0.3068)
Epoch  29/50 | Train Loss: 0.5160 | Val Loss: 0.5367 | IoU: 0.3066 | Dice: 0.4677 | Prec: 0.3709 | Rec: 0.6501 | 217.8s
Epoch  30/50 | Train Loss: 0.5127 | Val Loss: 0.5323 | IoU: 0.3098 | Dice: 0.4718 | Prec: 0.3627 | Rec: 0.6903 | 216.9s
  >>> New best model saved (IoU=0.3098)
Epoch  31/50 | Train Loss: 0.5125 | Val Loss: 0.5330 | IoU: 0.3094 | Dice: 0.4712 | Prec: 0.3646 | Rec: 0.6819 | 216.6s
Epoch  32/50 | Train Loss: 0.5109 | Val Loss: 0.5317 | IoU: 0.3102 | Dice: 0.4722 | Prec: 0.3614 | Rec: 0.6959 | 217.4s
  >>> New best model saved (IoU=0.3102)
Epoch  33/50 | Train Loss: 0.5087 | Val Loss: 0.5307 | IoU: 0.3109 | Dice: 0.4731 | Prec: 0.3664 | Rec: 0.6816 | 216.2s
  >>> New best model saved (IoU=0.3109)
Epoch  34/50 | Train Loss: 0.5069 | Val Loss: 0.5297 | IoU: 0.3123 | Dice: 0.4745 | Prec: 0.3735 | Rec: 0.6644 | 262.1s
  >>> New best model saved (IoU=0.3123)
Epoch  35/50 | Train Loss: 0.5054 | Val Loss: 0.5277 | IoU: 0.3132 | Dice: 0.4757 | Prec: 0.3655 | Rec: 0.6946 | 273.0s
  >>> New best model saved (IoU=0.3132)
Epoch  36/50 | Train Loss: 0.5043 | Val Loss: 0.5303 | IoU: 0.3113 | Dice: 0.4735 | Prec: 0.3695 | Rec: 0.6732 | 263.6s
Epoch  37/50 | Train Loss: 0.5027 | Val Loss: 0.5231 | IoU: 0.3172 | Dice: 0.4802 | Prec: 0.3767 | Rec: 0.6743 | 258.9s
  >>> New best model saved (IoU=0.3172)
Epoch  38/50 | Train Loss: 0.5154 | Val Loss: 0.5353 | IoU: 0.3067 | Dice: 0.4678 | Prec: 0.3908 | Rec: 0.5918 | 261.6s
Epoch  39/50 | Train Loss: 0.5074 | Val Loss: 0.5287 | IoU: 0.3124 | Dice: 0.4745 | Prec: 0.3916 | Rec: 0.6157 | 272.0s
Epoch  40/50 | Train Loss: 0.5071 | Val Loss: 0.5262 | IoU: 0.3142 | Dice: 0.4769 | Prec: 0.3665 | Rec: 0.6935 | 239.4s
Epoch  41/50 | Train Loss: 0.5015 | Val Loss: 0.5226 | IoU: 0.3173 | Dice: 0.4804 | Prec: 0.3723 | Rec: 0.6893 | 259.5s
  >>> New best model saved (IoU=0.3173)
Epoch  42/50 | Train Loss: 0.5001 | Val Loss: 0.5203 | IoU: 0.3199 | Dice: 0.4833 | Prec: 0.3888 | Rec: 0.6492 | 532.3s
  >>> New best model saved (IoU=0.3199)
Epoch  43/50 | Train Loss: 0.4989 | Val Loss: 0.5193 | IoU: 0.3204 | Dice: 0.4838 | Prec: 0.3901 | Rec: 0.6501 | 544.4s
  >>> New best model saved (IoU=0.3204)
Epoch  44/50 | Train Loss: 0.5012 | Val Loss: 0.5189 | IoU: 0.3211 | Dice: 0.4845 | Prec: 0.3924 | Rec: 0.6467 | 488.8s
  >>> New best model saved (IoU=0.3211)
Epoch  45/50 | Train Loss: 0.4984 | Val Loss: 0.5203 | IoU: 0.3192 | Dice: 0.4826 | Prec: 0.3725 | Rec: 0.6987 | 429.8s
Epoch  46/50 | Train Loss: 0.4945 | Val Loss: 0.5207 | IoU: 0.3186 | Dice: 0.4820 | Prec: 0.3774 | Rec: 0.6805 | 416.7s
Epoch  47/50 | Train Loss: 0.4957 | Val Loss: 0.5167 | IoU: 0.3227 | Dice: 0.4864 | Prec: 0.3969 | Rec: 0.6422 | 452.2s
  >>> New best model saved (IoU=0.3227)
Epoch  48/50 | Train Loss: 0.4912 | Val Loss: 0.5167 | IoU: 0.3226 | Dice: 0.4861 | Prec: 0.3991 | Rec: 0.6349 | 454.0s
Epoch  49/50 | Train Loss: 0.4921 | Val Loss: 0.5149 | IoU: 0.3238 | Dice: 0.4879 | Prec: 0.3850 | Rec: 0.6787 | 335.0s
  >>> New best model saved (IoU=0.3238)
Epoch  50/50 | Train Loss: 0.4918 | Val Loss: 0.5144 | IoU: 0.3239 | Dice: 0.4880 | Prec: 0.3795 | Rec: 0.6950 | 295.6s
  >>> New best model saved (IoU=0.3239)

============================================================
Training complete. Best IoU: 0.3239
============================================================
Saved training curves to checkpoints/training_curves.png
Saved predictions to checkpoints\sample_predictions.png
```

### Using EfficientNet-B0

```powershell
PS D:\Internship related\GJ-Map Solutions\road-detection> .\venv\Scripts\python.exe train_torchgeo.py --data_dir data/india_combined --encoder efficientnet-b0 --epochs 50 --workers 0
Using device: cpu
  Found 547 tiles in data/india_combined\train
D:\Internship related\GJ-Map Solutions\road-detection\venv\Lib\site-packages\albumentations\core\validation.py:114: UserWarning: ShiftScaleRotate is a special case of Affine transform. Please use Affine transform instead.
  original_init(self, **validated_kwargs)
D:\Internship related\GJ-Map Solutions\road-detection\train_torchgeo.py:109: UserWarning: Argument(s) 'var_limit' are not valid for transform GaussNoise
  A.GaussNoise(var_limit=(5e-4, 1e-3), p=0.3),
  Found 97 tiles in data/india_combined\val
Train: 547 tiles
Val:   97 tiles
Encoder: efficientnet-b0
Parameters: 6,303,492 total, 6,303,492 trainable

============================================================
Training: U-Net + efficientnet-b0 | 50 epochs | lr=0.0001
  Weight decay: 0.0001 | Patience: 15
  Augmentations: albumentations
  Decoder attention: scSE
============================================================

D:\Internship related\GJ-Map Solutions\road-detection\venv\Lib\site-packages\torch\utils\data\dataloader.py:775: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  super().__init__(loader)
Epoch   1/50 | Train Loss: 0.7164 | Val Loss: 0.7207 | IoU: 0.1626 | Dice: 0.2743 | Prec: 0.1628 | Rec: 0.9949 | LR: 9.76e-05 | 240.8s
  >>> New best model saved (IoU=0.1626)
Epoch   2/50 | Train Loss: 0.6731 | Val Loss: 0.6819 | IoU: 0.2018 | Dice: 0.3322 | Prec: 0.2038 | Rec: 0.9287 | LR: 9.05e-05 | 196.5s
  >>> New best model saved (IoU=0.2018)
Epoch   3/50 | Train Loss: 0.6515 | Val Loss: 0.6630 | IoU: 0.2156 | Dice: 0.3503 | Prec: 0.2207 | Rec: 0.8603 | LR: 7.94e-05 | 221.7s
  >>> New best model saved (IoU=0.2156)
Epoch   4/50 | Train Loss: 0.6387 | Val Loss: 0.6481 | IoU: 0.2283 | Dice: 0.3660 | Prec: 0.2450 | Rec: 0.7354 | LR: 6.55e-05 | 169.2s
  >>> New best model saved (IoU=0.2283)
Epoch   5/50 | Train Loss: 0.6379 | Val Loss: 0.6453 | IoU: 0.2256 | Dice: 0.3628 | Prec: 0.2423 | Rec: 0.7893 | LR: 5.00e-05 | 170.6s
Epoch   6/50 | Train Loss: 0.6265 | Val Loss: 0.6349 | IoU: 0.2350 | Dice: 0.3740 | Prec: 0.2723 | Rec: 0.7195 | LR: 3.45e-05 | 169.2s
  >>> New best model saved (IoU=0.2350)
Epoch   7/50 | Train Loss: 0.6228 | Val Loss: 0.6305 | IoU: 0.2370 | Dice: 0.3769 | Prec: 0.2652 | Rec: 0.7309 | LR: 2.06e-05 | 216.9s
  >>> New best model saved (IoU=0.2370)
Epoch   8/50 | Train Loss: 0.6197 | Val Loss: 0.6294 | IoU: 0.2373 | Dice: 0.3770 | Prec: 0.2714 | Rec: 0.7386 | LR: 9.55e-06 | 235.9s
  >>> New best model saved (IoU=0.2373)
Epoch   9/50 | Train Loss: 0.6173 | Val Loss: 0.6277 | IoU: 0.2392 | Dice: 0.3790 | Prec: 0.3075 | Rec: 0.7124 | LR: 2.45e-06 | 252.1s
  >>> New best model saved (IoU=0.2392)
Epoch  10/50 | Train Loss: 0.6185 | Val Loss: 0.6286 | IoU: 0.2381 | Dice: 0.3777 | Prec: 0.2859 | Rec: 0.7309 | LR: 1.00e-04 | 174.5s
Epoch  11/50 | Train Loss: 0.6150 | Val Loss: 0.6252 | IoU: 0.2415 | Dice: 0.3818 | Prec: 0.2984 | Rec: 0.6777 | LR: 9.94e-05 | 179.9s
  >>> New best model saved (IoU=0.2415)
Epoch  12/50 | Train Loss: 0.6167 | Val Loss: 0.6224 | IoU: 0.2413 | Dice: 0.3818 | Prec: 0.2753 | Rec: 0.7112 | LR: 9.76e-05 | 265.8s
Epoch  13/50 | Train Loss: 0.6134 | Val Loss: 0.6260 | IoU: 0.2367 | Dice: 0.3759 | Prec: 0.2806 | Rec: 0.7629 | LR: 9.46e-05 | 341.7s
Epoch  14/50 | Train Loss: 0.6119 | Val Loss: 0.6172 | IoU: 0.2443 | Dice: 0.3856 | Prec: 0.2874 | Rec: 0.6880 | LR: 9.05e-05 | 350.7s
  >>> New best model saved (IoU=0.2443)
Epoch  15/50 | Train Loss: 0.6049 | Val Loss: 0.6161 | IoU: 0.2458 | Dice: 0.3871 | Prec: 0.3400 | Rec: 0.6761 | LR: 8.54e-05 | 351.6s
  >>> New best model saved (IoU=0.2458)
Epoch  16/50 | Train Loss: 0.6035 | Val Loss: 0.6152 | IoU: 0.2459 | Dice: 0.3872 | Prec: 0.3475 | Rec: 0.6919 | LR: 7.94e-05 | 332.0s
  >>> New best model saved (IoU=0.2459)
Epoch  17/50 | Train Loss: 0.6053 | Val Loss: 0.6155 | IoU: 0.2450 | Dice: 0.3868 | Prec: 0.2788 | Rec: 0.7326 | LR: 7.27e-05 | 176.3s
Epoch  18/50 | Train Loss: 0.6012 | Val Loss: 0.6129 | IoU: 0.2478 | Dice: 0.3896 | Prec: 0.3031 | Rec: 0.6672 | LR: 6.55e-05 | 183.1s
  >>> New best model saved (IoU=0.2478)
Epoch  19/50 | Train Loss: 0.5992 | Val Loss: 0.6126 | IoU: 0.2479 | Dice: 0.3897 | Prec: 0.2960 | Rec: 0.6766 | LR: 5.78e-05 | 240.8s
  >>> New best model saved (IoU=0.2479)
Epoch  20/50 | Train Loss: 0.6045 | Val Loss: 0.6104 | IoU: 0.2493 | Dice: 0.3916 | Prec: 0.3043 | Rec: 0.6766 | LR: 5.00e-05 | 210.7s
  >>> New best model saved (IoU=0.2493)
Epoch  21/50 | Train Loss: 0.6043 | Val Loss: 0.6105 | IoU: 0.2489 | Dice: 0.3914 | Prec: 0.2878 | Rec: 0.7050 | LR: 4.22e-05 | 216.0s
Epoch  22/50 | Train Loss: 0.5971 | Val Loss: 0.6101 | IoU: 0.2496 | Dice: 0.3919 | Prec: 0.3273 | Rec: 0.6833 | LR: 3.45e-05 | 227.4s
  >>> New best model saved (IoU=0.2496)
Epoch  23/50 | Train Loss: 0.5991 | Val Loss: 0.6092 | IoU: 0.2500 | Dice: 0.3926 | Prec: 0.3026 | Rec: 0.6853 | LR: 2.73e-05 | 213.9s
  >>> New best model saved (IoU=0.2500)
Epoch  24/50 | Train Loss: 0.5965 | Val Loss: 0.6089 | IoU: 0.2507 | Dice: 0.3932 | Prec: 0.3439 | Rec: 0.6692 | LR: 2.06e-05 | 224.2s
  >>> New best model saved (IoU=0.2507)
Epoch  25/50 | Train Loss: 0.5990 | Val Loss: 0.6081 | IoU: 0.2511 | Dice: 0.3939 | Prec: 0.3204 | Rec: 0.6791 | LR: 1.46e-05 | 227.4s
  >>> New best model saved (IoU=0.2511)
Epoch  26/50 | Train Loss: 0.5964 | Val Loss: 0.6079 | IoU: 0.2510 | Dice: 0.3939 | Prec: 0.2955 | Rec: 0.6895 | LR: 9.55e-06 | 198.6s
Epoch  27/50 | Train Loss: 0.5984 | Val Loss: 0.6073 | IoU: 0.2515 | Dice: 0.3947 | Prec: 0.2963 | Rec: 0.6910 | LR: 5.45e-06 | 191.5s
  >>> New best model saved (IoU=0.2515)
Epoch  28/50 | Train Loss: 0.5980 | Val Loss: 0.6075 | IoU: 0.2513 | Dice: 0.3943 | Prec: 0.2952 | Rec: 0.6913 | LR: 2.45e-06 | 194.5s
Epoch  29/50 | Train Loss: 0.5967 | Val Loss: 0.6072 | IoU: 0.2516 | Dice: 0.3946 | Prec: 0.2973 | Rec: 0.6833 | LR: 6.16e-07 | 194.8s
  >>> New best model saved (IoU=0.2516)
Epoch  30/50 | Train Loss: 0.5947 | Val Loss: 0.6074 | IoU: 0.2516 | Dice: 0.3947 | Prec: 0.3005 | Rec: 0.6807 | LR: 1.00e-04 | 190.2s
  >>> New best model saved (IoU=0.2516)
Epoch  31/50 | Train Loss: 0.5963 | Val Loss: 0.6073 | IoU: 0.2517 | Dice: 0.3943 | Prec: 0.3601 | Rec: 0.6599 | LR: 9.98e-05 | 190.4s
  >>> New best model saved (IoU=0.2517)
Epoch  32/50 | Train Loss: 0.5976 | Val Loss: 0.6070 | IoU: 0.2521 | Dice: 0.3949 | Prec: 0.3367 | Rec: 0.6490 | LR: 9.94e-05 | 191.0s
  >>> New best model saved (IoU=0.2521)
Epoch  33/50 | Train Loss: 0.5884 | Val Loss: 0.6082 | IoU: 0.2506 | Dice: 0.3931 | Prec: 0.3422 | Rec: 0.6824 | LR: 9.86e-05 | 189.6s
Epoch  34/50 | Train Loss: 0.5961 | Val Loss: 0.6065 | IoU: 0.2518 | Dice: 0.3947 | Prec: 0.3368 | Rec: 0.6701 | LR: 9.76e-05 | 190.5s
Epoch  35/50 | Train Loss: 0.5942 | Val Loss: 0.6057 | IoU: 0.2524 | Dice: 0.3954 | Prec: 0.3217 | Rec: 0.6842 | LR: 9.62e-05 | 191.0s
  >>> New best model saved (IoU=0.2524)
Epoch  36/50 | Train Loss: 0.5903 | Val Loss: 0.6052 | IoU: 0.2529 | Dice: 0.3957 | Prec: 0.3669 | Rec: 0.6339 | LR: 9.46e-05 | 192.5s
  >>> New best model saved (IoU=0.2529)
Epoch  37/50 | Train Loss: 0.5975 | Val Loss: 0.6046 | IoU: 0.2531 | Dice: 0.3963 | Prec: 0.3313 | Rec: 0.6861 | LR: 9.26e-05 | 190.7s
  >>> New best model saved (IoU=0.2531)
Epoch  38/50 | Train Loss: 0.5916 | Val Loss: 0.6048 | IoU: 0.2529 | Dice: 0.3961 | Prec: 0.2918 | Rec: 0.6859 | LR: 9.05e-05 | 190.7s
Epoch  39/50 | Train Loss: 0.5896 | Val Loss: 0.6025 | IoU: 0.2549 | Dice: 0.3984 | Prec: 0.2910 | Rec: 0.6717 | LR: 8.80e-05 | 190.7s
  >>> New best model saved (IoU=0.2549)
Epoch  40/50 | Train Loss: 0.5932 | Val Loss: 0.6031 | IoU: 0.2544 | Dice: 0.3976 | Prec: 0.2848 | Rec: 0.6679 | LR: 8.54e-05 | 190.5s
Epoch  41/50 | Train Loss: 0.5938 | Val Loss: 0.6019 | IoU: 0.2556 | Dice: 0.3991 | Prec: 0.3679 | Rec: 0.6480 | LR: 8.25e-05 | 189.8s
  >>> New best model saved (IoU=0.2556)
Epoch  42/50 | Train Loss: 0.5852 | Val Loss: 0.5998 | IoU: 0.2572 | Dice: 0.4012 | Prec: 0.3327 | Rec: 0.6478 | LR: 7.94e-05 | 190.4s
  >>> New best model saved (IoU=0.2572)
Epoch  43/50 | Train Loss: 0.5868 | Val Loss: 0.6001 | IoU: 0.2569 | Dice: 0.4009 | Prec: 0.3191 | Rec: 0.6689 | LR: 7.61e-05 | 189.9s
Epoch  44/50 | Train Loss: 0.5840 | Val Loss: 0.6000 | IoU: 0.2571 | Dice: 0.4010 | Prec: 0.3098 | Rec: 0.6354 | LR: 7.27e-05 | 189.4s
Epoch  45/50 | Train Loss: 0.5830 | Val Loss: 0.5978 | IoU: 0.2586 | Dice: 0.4030 | Prec: 0.3091 | Rec: 0.6666 | LR: 6.91e-05 | 189.5s
  >>> New best model saved (IoU=0.2586)
Epoch  46/50 | Train Loss: 0.5887 | Val Loss: 0.5983 | IoU: 0.2581 | Dice: 0.4025 | Prec: 0.3007 | Rec: 0.6831 | LR: 6.55e-05 | 191.1s
Epoch  47/50 | Train Loss: 0.5892 | Val Loss: 0.5995 | IoU: 0.2572 | Dice: 0.4012 | Prec: 0.3040 | Rec: 0.6899 | LR: 6.17e-05 | 189.5s
Epoch  48/50 | Train Loss: 0.5891 | Val Loss: 0.5979 | IoU: 0.2586 | Dice: 0.4028 | Prec: 0.2965 | Rec: 0.6422 | LR: 5.78e-05 | 190.0s
  >>> New best model saved (IoU=0.2586)
Epoch  49/50 | Train Loss: 0.5828 | Val Loss: 0.5961 | IoU: 0.2601 | Dice: 0.4046 | Prec: 0.3234 | Rec: 0.6428 | LR: 5.39e-05 | 190.4s
  >>> New best model saved (IoU=0.2601)
Epoch  50/50 | Train Loss: 0.5900 | Val Loss: 0.5968 | IoU: 0.2595 | Dice: 0.4039 | Prec: 0.3733 | Rec: 0.6460 | LR: 5.00e-05 | 189.7s

============================================================
Training complete. Best IoU: 0.2601
============================================================
Saved training curves to checkpoints/training_curves.png
Saved predictions to checkpoints\sample_predictions.png
```
