"""
train_torchgeo.py — Road Segmentation with TorchGeo + smp
==========================================================
Uses:
  - TorchGeo's RasterDataset + RandomGeoSampler (auto patching)
  - segmentation_models_pytorch U-Net + ResNet-34 (ImageNet pretrained)
  - DiceLoss for class imbalance
  - Standard PyTorch training loop

Usage:
    # Train on Massachusetts Roads dataset:
    python train_torchgeo.py --data_dir data/mass_roads --epochs 50

    # Fine-tune an existing model:
    python train_torchgeo.py --data_dir data/mass_roads --finetune model.pth

    # Quick test with synthetic data:
    python train_torchgeo.py --dry_run
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

import segmentation_models_pytorch as smp

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("WARNING: albumentations not installed. Using basic augmentations.")
    print("  Install with: pip install albumentations")

try:
    from torchgeo.datasets import RasterDataset
    from torchgeo.samplers import RandomGeoSampler, GridGeoSampler
    HAS_TORCHGEO_GEO = True
except ImportError:
    HAS_TORCHGEO_GEO = False

 
# 1. TorchGeo GeoDataset Setup 

class RoadImageDataset(RasterDataset):
    """TorchGeo RasterDataset for satellite images."""
    filename_glob = "*.tif"
    is_image = True


class RoadMaskDataset(RasterDataset):
    """TorchGeo RasterDataset for road masks."""
    filename_glob = "*.tif"
    is_image = False

 
# 2. Simple File-based Dataset (Fallback / Massachusetts Roads) 

class SimpleRoadDataset(Dataset):
    """
    Simple file-based dataset for pre-tiled image-mask pairs.
    Works with Massachusetts Roads or any dataset with images/ and masks/ subdirs.

    Supports both GeoTIFF (.tif) and PNG (.png) formats.
    Uses albumentations for strong augmentations to combat overfitting.
    """

    def __init__(self, root_dir: str, augment: bool = False, patch_size: int = 256):
        self.image_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks")
        self.augment = augment
        self.patch_size = patch_size

        # Support multiple formats
        exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
        self.tile_names = sorted([
            f for f in os.listdir(self.image_dir)
            if os.path.splitext(f)[1].lower() in exts
        ])
        print(f"  Found {len(self.tile_names)} tiles in {root_dir}")

        # Build augmentation pipeline
        if self.augment and HAS_ALBUMENTATIONS:
            self.transform = A.Compose([
                # Spatial — safe for segmentation masks
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.15, rotate_limit=15,
                    border_mode=0, p=0.5,
                ),
                A.ElasticTransform(
                    alpha=30, sigma=5, border_mode=0, p=0.3,
                ),
                # Pixel-only — do NOT affect the mask
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5,
                ),
                A.GaussNoise(var_limit=(5e-4, 1e-3), p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            ])
        else:
            self.transform = None

    def __len__(self) -> int:
        return len(self.tile_names)

    def __getitem__(self, idx: int):
        name = self.tile_names[idx]
        img_path = os.path.join(self.image_dir, name)
        msk_path = os.path.join(self.mask_dir, name)

        ext = os.path.splitext(name)[1].lower()

        if ext in (".tif", ".tiff"):
            import rasterio
            with rasterio.open(img_path) as src:
                image = src.read().astype(np.float32)  # (C, H, W)
            with rasterio.open(msk_path) as src:
                mask = src.read(1).astype(np.float32)  # (H, W)
        else:
            from PIL import Image
            img = Image.open(img_path).convert("RGB")
            image = np.array(img, dtype=np.float32).transpose(2, 0, 1)  # (3, H, W)
            msk = Image.open(msk_path).convert("L")
            mask = np.array(msk, dtype=np.float32)  # (H, W)

        # Normalize image to [0, 1]
        if image.max() > 1.0:
            if image.max() > 255:
                image = np.clip(image / 10000.0, 0.0, 1.0)  # Sentinel-2
            else:
                image = image / 255.0  # Standard 8-bit

        # Ensure mask is binary [0, 1]
        mask = (mask > 0).astype(np.float32)

        # Augmentation
        if self.augment:
            image, mask = self._augment(image, mask)

        image_tensor = torch.from_numpy(image.copy())
        mask_tensor = torch.from_numpy(mask.copy()).unsqueeze(0)  # (1, H, W)

        return image_tensor, mask_tensor

    def _augment(self, image, mask):
        """Apply augmentations. Uses albumentations if available, else basic flips."""
        if self.transform is not None:
            # albumentations expects (H, W, C) uint8-like float image
            img_hwc = image.transpose(1, 2, 0)  # (C,H,W) -> (H,W,C)
            transformed = self.transform(image=img_hwc, mask=mask)
            image = transformed["image"].transpose(2, 0, 1)  # back to (C,H,W)
            mask = transformed["mask"]
        else:
            # Fallback: basic geometric augmentations
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=2).copy()
                mask = np.flip(mask, axis=1).copy()
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=0).copy()
            k = np.random.randint(0, 4)
            if k > 0:
                image = np.rot90(image, k, axes=(1, 2)).copy()
                mask = np.rot90(mask, k, axes=(0, 1)).copy()
        return image, mask

 
# 3. Synthetic Dataset for Dry Run 

class SyntheticRoadDataset(Dataset):
    """Fake satellite tiles with road-like features for testing."""

    def __init__(self, num_samples=64, size=256):
        self.num_samples = num_samples
        self.size = size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        s = self.size
        image = torch.rand(3, s, s) * 0.4 + 0.1
        mask = torch.zeros(1, s, s)

        # Horizontal road
        y = np.random.randint(20, s - 20)
        w = np.random.randint(3, 8)
        mask[0, y - w:y + w, :] = 1

        # Vertical road
        x = np.random.randint(20, s - 20)
        w = np.random.randint(3, 8)
        mask[0, :, x - w:x + w] = 1

        # Make roads brighter
        for c in range(3):
            image[c][mask[0] == 1] = 0.5 + torch.rand(1).item() * 0.3

        return image, mask

 
# 4. Training & Evaluation 

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_iou = 0
    total_dice = 0
    total_precision = 0
    total_recall = 0
    n = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)
        loss = criterion(preds, masks)
        total_loss += loss.item()

        # Binary predictions
        pred_bin = (torch.sigmoid(preds) > 0.5).float()
        gt = masks.float()

        # Metrics per batch
        tp = (pred_bin * gt).sum()
        fp = (pred_bin * (1 - gt)).sum()
        fn = ((1 - pred_bin) * gt).sum()

        iou = tp / (tp + fp + fn + 1e-7)
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)

        total_iou += iou.item()
        total_dice += dice.item()
        total_precision += precision.item()
        total_recall += recall.item()
        n += 1

    return {
        "loss": total_loss / n,
        "iou": total_iou / n,
        "dice": total_dice / n,
        "precision": total_precision / n,
        "recall": total_recall / n,
    }

 
# 5. Visualization 

def visualize_predictions(model, dataset, device, num_samples=4, save_path=None):
    """Plot: Input RGB | Ground Truth | Prediction | Overlay."""
    model.eval()

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    for row, idx in enumerate(indices):
        image, mask = dataset[idx]

        with torch.no_grad():
            x = image.unsqueeze(0).to(device)
            pred = torch.sigmoid(model(x))[0, 0].cpu().numpy()

        pred_bin = (pred > 0.5).astype(float)
        rgb = image.permute(1, 2, 0).numpy()
        rgb = np.clip(rgb, 0, 1)
        gt = mask[0].numpy()

        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title("Input RGB")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("Ground Truth")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pred_bin, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title("Prediction")
        axes[row, 2].axis("off")

        overlay = rgb.copy()
        tp = (pred_bin == 1) & (gt == 1)
        fn = (pred_bin == 0) & (gt == 1)
        fp = (pred_bin == 1) & (gt == 0)
        overlay[tp] = [0, 1, 0]
        overlay[fn] = [1, 0, 0]
        overlay[fp] = [0, 0, 1]
        axes[row, 3].imshow(overlay)
        axes[row, 3].set_title("Overlay (G=TP R=FN B=FP)")
        axes[row, 3].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved predictions to {save_path}")
    plt.close()

 
# 6. Main 

def main():
    parser = argparse.ArgumentParser(
        description="Train road segmentation — smp U-Net + configurable encoder"
    )
    parser.add_argument("--data_dir", type=str, default="data/mass_roads",
                        help="Dataset root (must have train/ and val/ with images/ & masks/)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="L2 regularization strength (default: 1e-4)")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="auto",
                        help="'cuda', 'cpu', or 'auto'")
    parser.add_argument("--dry_run", action="store_true",
                        help="Test with synthetic data")
    parser.add_argument("--finetune", type=str, default=None,
                        help="Path to .pth file to fine-tune")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze encoder during fine-tuning")
    parser.add_argument("--encoder", type=str, default="resnet34",
                        choices=["resnet18", "resnet34", "resnet50",
                                 "efficientnet-b0", "efficientnet-b1"],
                        help="Encoder backbone (default: resnet34)")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience in epochs (default: 15)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of data loader workers (0 = main process, safest for Windows)")
    args = parser.parse_args()

    # --- Device ---
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # --- Datasets ---
    if args.dry_run:
        print("=" * 60)
        print("DRY RUN — Using synthetic data")
        print("=" * 60)
        train_ds = SyntheticRoadDataset(num_samples=64, size=256)
        val_ds = SyntheticRoadDataset(num_samples=16, size=256)
        args.epochs = 5
    else:
        train_dir = os.path.join(args.data_dir, "train")
        val_dir = os.path.join(args.data_dir, "val")

        if not os.path.isdir(train_dir):
            print(f"ERROR: {train_dir} not found")
            print("Run with --dry_run to test, or download Massachusetts Roads dataset")
            sys.exit(1)

        train_ds = SimpleRoadDataset(train_dir, augment=True)
        val_ds = SimpleRoadDataset(val_dir, augment=False)

    print(f"Train: {len(train_ds)} tiles")
    print(f"Val:   {len(val_ds)} tiles")

    # Use num_workers=0 by default on Windows to avoid multiprocessing crashes
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True,
    )

    # --- Model ---
    encoder_name = args.encoder
    print(f"Encoder: {encoder_name}")
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        decoder_attention_type="scse",  # Squeeze & Excitation — implicit regularization
    )

    # --- Fine-tune from existing .pth ---
    if args.finetune:
        print(f"\nLoading weights from: {args.finetune}")
        state = torch.load(args.finetune, map_location="cpu", weights_only=False)
        # Handle both raw state_dict and checkpoint dict
        if "model_state_dict" in state:
            saved_encoder = state.get("encoder_name", "resnet34")
            if saved_encoder != encoder_name:
                print(f"  WARNING: checkpoint encoder ({saved_encoder}) != requested ({encoder_name})")
                print(f"  Using checkpoint encoder: {saved_encoder}")
                encoder_name = saved_encoder
                model = smp.Unet(
                    encoder_name=encoder_name,
                    encoder_weights=None,
                    in_channels=3,
                    classes=1,
                    decoder_attention_type="scse",
                )
            state = state["model_state_dict"]
        try:
            model.load_state_dict(state, strict=True)
            print("  Loaded successfully (strict match)")
        except RuntimeError:
            model.load_state_dict(state, strict=False)
            print("  Loaded with strict=False (partial match)")

        if args.freeze_encoder:
            for param in model.encoder.parameters():
                param.requires_grad = False
            print("  Encoder frozen — training decoder only")

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable:,} trainable")

    # --- Loss & Optimizer ---
    criterion = smp.losses.DiceLoss(mode="binary")
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,  # L2 regularization
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2,  # restart every 10, 20, 40 epochs
    )

    # --- Training ---
    os.makedirs(args.save_dir, exist_ok=True)
    best_iou = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_dice": []}

    print(f"\n{'=' * 60}")
    print(f"Training: U-Net + {encoder_name} | {args.epochs} epochs | lr={args.lr}")
    print(f"  Weight decay: {args.weight_decay} | Patience: {args.patience}")
    print(f"  Augmentations: {'albumentations' if HAS_ALBUMENTATIONS else 'basic (flip+rotate)'}")
    print(f"  Decoder attention: scSE")
    print(f"{'=' * 60}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val = validate(model, val_loader, criterion, device)

        scheduler.step(epoch)  # CosineAnnealingWarmRestarts uses epoch

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val['loss']:.4f} | "
            f"IoU: {val['iou']:.4f} | "
            f"Dice: {val['dice']:.4f} | "
            f"Prec: {val['precision']:.4f} | "
            f"Rec: {val['recall']:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"{elapsed:.1f}s"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_iou"].append(val["iou"])
        history["val_dice"].append(val["dice"])

        if val["iou"] > best_iou:
            best_iou = val["iou"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_iou": best_iou,
                "val_metrics": val,
                "encoder_name": encoder_name,  # save for predict.py
            }, os.path.join(args.save_dir, "best_model.pth"))
            print(f"  >>> New best model saved (IoU={best_iou:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping: no improvement for {args.patience} epochs")
                break

    # --- Results ---
    print(f"\n{'=' * 60}")
    print(f"Training complete. Best IoU: {best_iou:.4f}")
    print(f"{'=' * 60}")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.set_title("Loss Curves")

    ax2.plot(history["val_iou"], label="Val IoU")
    ax2.plot(history["val_dice"], label="Val Dice")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Score")
    ax2.legend(); ax2.set_title("Validation Metrics")

    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "training_curves.png"), dpi=150)
    print(f"Saved training curves to {args.save_dir}/training_curves.png")

    # --- Visualize ---
    visualize_predictions(
        model, val_ds, device, num_samples=4,
        save_path=os.path.join(args.save_dir, "sample_predictions.png"),
    )


if __name__ == "__main__":
    main()
