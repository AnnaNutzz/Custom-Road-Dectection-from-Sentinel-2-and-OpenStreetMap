"""
predict.py — Run road detection on satellite imagery
=====================================================
Loads a trained model and predicts road masks for input images.

Usage:
    # Predict on a single GeoTIFF:
    python predict.py --model checkpoints/best_model.pth --input data/delhi/sentinel2_rgb.tif

    # Predict on a directory of tiles:
    python predict.py --model checkpoints/best_model.pth --input data/delhi/tiles/images/

    # Custom tile size and threshold:
    python predict.py --model checkpoints/best_model.pth --input image.tif --tile_size 256 --threshold 0.4
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp


def load_model(model_path, device):
    """Load trained U-Net model from checkpoint. Auto-detects encoder."""
    
    # 1. Load checkpoint first to find encoder name
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    
    if "model_state_dict" in state:
        encoder_name = state.get("encoder_name", "resnet34")  # Fallback for old models
        state_dict = state["model_state_dict"]
        epoch = state.get("epoch", "?")
        iou = state.get("best_iou", 0)
        print(f"Loading {encoder_name} model from epoch {epoch} (IoU={iou:.4f})...")
    else:
        encoder_name = "resnet34"
        state_dict = state
        print("Loading legacy model weights...")

    # 2. Build model with correct encoder
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,  # We load our own weights
        in_channels=3,
        classes=1,
        decoder_attention_type="scse",  # Matches training script
    )

    # 3. Load weights
    try:
        model.load_state_dict(state_dict, strict=True)
        print("  Loaded successfully (strict match)")
    except RuntimeError:
        model.load_state_dict(state_dict, strict=False)
        print("  Loaded with strict=False (partial match)")

    model = model.to(device)
    model.eval()
    return model


def predict_tile(model, image_np, device, threshold=0.5):
    """
    Predict road mask for a single image tile.

    Args:
        model: trained model
        image_np: numpy array (C, H, W) or (H, W, C), float32 [0, 1]
        device: torch device
        threshold: probability threshold for binary mask

    Returns:
        pred_prob: (H, W) float probability map
        pred_mask: (H, W) binary mask
    """
    if image_np.ndim == 3 and image_np.shape[2] == 3:
        image_np = image_np.transpose(2, 0, 1)  # HWC → CHW

    tensor = torch.from_numpy(image_np).unsqueeze(0).float().to(device)

    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output)[0, 0].cpu().numpy()

    mask = (prob > threshold).astype(np.uint8)
    return prob, mask


def predict_large_image(model, image, device, tile_size=256, overlap=64, threshold=0.5):
    """
    Predict road mask for a large image by sliding window.

    Args:
        model: trained model
        image: (C, H, W) numpy array
        device: torch device
        tile_size: prediction tile size
        overlap: overlap between tiles
        threshold: probability threshold

    Returns:
        prob_map: (H, W) float probability map
        mask: (H, W) binary mask
    """
    C, H, W = image.shape
    stride = tile_size - overlap

    prob_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y_end = min(y + tile_size, H)
            x_end = min(x + tile_size, W)
            y_start = y_end - tile_size
            x_start = x_end - tile_size

            if y_start < 0: y_start = 0
            if x_start < 0: x_start = 0

            tile = image[:, y_start:y_end, x_start:x_end]

            # Pad if needed
            _, th, tw = tile.shape
            if th < tile_size or tw < tile_size:
                padded = np.zeros((C, tile_size, tile_size), dtype=np.float32)
                padded[:, :th, :tw] = tile
                tile = padded

            prob, _ = predict_tile(model, tile, device, threshold)

            prob_map[y_start:y_start + th, x_start:x_start + tw] += prob[:th, :tw]
            count_map[y_start:y_start + th, x_start:x_start + tw] += 1

    # Average overlapping predictions
    count_map[count_map == 0] = 1
    prob_map /= count_map
    mask = (prob_map > threshold).astype(np.uint8)

    return prob_map, mask


def load_image(path):
    """Load image from GeoTIFF or PNG."""
    ext = os.path.splitext(path)[1].lower()

    if ext in (".tif", ".tiff"):
        import rasterio
        with rasterio.open(path) as src:
            image = src.read().astype(np.float32)  # (C, H, W)
            profile = src.profile.copy()
            crs = src.crs
            transform = src.transform
    else:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        image = np.array(img, dtype=np.float32).transpose(2, 0, 1)
        profile = None
        crs = None
        transform = None

    # Normalize
    if image.max() > 255:
        image = np.clip(image / 10000.0, 0.0, 1.0)
    elif image.max() > 1.0:
        image = image / 255.0

    return image, profile, crs, transform


def save_mask_geotiff(mask, profile, output_path):
    """Save predicted mask as GeoTIFF (0=background, 255=road)."""
    import rasterio
    profile.update(count=1, dtype="uint8", compress="lzw")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write((mask * 255).astype(np.uint8), 1)


def visualize_result(image, prob_map, mask, save_path, title="Road Detection"):
    """Create a 4-panel visualization."""
    rgb = image.transpose(1, 2, 0) if image.shape[0] == 3 else image
    rgb = np.clip(rgb, 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 1. Input
    axes[0].imshow(rgb)
    axes[0].set_title("Satellite Image")
    axes[0].axis("off")

    # 2. Probability map
    axes[1].imshow(prob_map, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Road Probability")
    axes[1].axis("off")

    # 3. Binary mask
    axes[2].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Predicted Road Mask")
    axes[2].axis("off")

    # 4. Overlay — roads in red on satellite image
    overlay = rgb.copy()
    road_pixels = mask == 1
    overlay[road_pixels] = [1.0, 0.2, 0.2]  # Red roads
    axes[3].imshow(overlay)
    axes[3].set_title("Roads Overlay")
    axes[3].axis("off")

    road_pct = mask.sum() / mask.size * 100
    fig.suptitle(f"{title}  |  Road coverage: {road_pct:.1f}%", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization: {save_path}")
    plt.close()


def process_single_image(model, image_path, output_dir, device, tile_size, threshold):
    """Process one satellite image and save results."""
    basename = os.path.splitext(os.path.basename(image_path))[0]

    print(f"\nProcessing: {image_path}")
    image, profile, crs, transform = load_image(image_path)
    C, H, W = image.shape
    print(f"  Size: {W}x{H} px, {C} bands")

    # Predict
    if H <= tile_size and W <= tile_size:
        prob_map, mask = predict_tile(model, image, device, threshold)
    else:
        print(f"  Large image — using sliding window ({tile_size}x{tile_size})...")
        prob_map, mask = predict_large_image(model, image, device, tile_size, threshold=threshold)

    road_pct = mask.sum() / mask.size * 100
    print(f"  Road pixels: {mask.sum():,} ({road_pct:.1f}%)")

    # Save mask as GeoTIFF (if input was GeoTIFF)
    if profile is not None:
        mask_path = os.path.join(output_dir, f"{basename}_road_mask.tif")
        save_mask_geotiff(mask, profile, mask_path)
        print(f"  Saved mask: {mask_path}")

    # Save visualization
    viz_path = os.path.join(output_dir, f"{basename}_prediction.png")
    visualize_result(image, prob_map, mask, viz_path, title=basename)

    return mask


def main():
    parser = argparse.ArgumentParser(
        description="Predict road masks from satellite imagery"
    )
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pth",
                        help="Path to trained model .pth")
    parser.add_argument("--input", type=str, required=True,
                        help="Input image or directory of tiles")
    parser.add_argument("--output_dir", type=str, default="predictions",
                        help="Output directory")
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Road probability threshold (lower = more roads detected)")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load model
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        sys.exit(1)
    model = load_model(args.model, device)

    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Process input
    if os.path.isdir(args.input):
        # Directory of tiles
        exts = {".tif", ".tiff", ".png", ".jpg"}
        files = sorted([
            os.path.join(args.input, f) for f in os.listdir(args.input)
            if os.path.splitext(f)[1].lower() in exts
        ])
        print(f"Found {len(files)} images in {args.input}")

        for f in files:
            try:
                process_single_image(model, f, args.output_dir, device,
                                     args.tile_size, args.threshold)
            except Exception as e:
                print(f"  ERROR: {e}")
    else:
        # Single image
        process_single_image(model, args.input, args.output_dir, device,
                             args.tile_size, args.threshold)

    print(f"\nDone! Results in: {args.output_dir}/")


if __name__ == "__main__":
    main()
