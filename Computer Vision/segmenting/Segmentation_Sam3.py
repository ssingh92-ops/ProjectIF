#!/usr/bin/env python
"""
Segment plant images with SAM3 and count plant pixels.

Requirements (in a fresh env with GPU ideally):

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    pip install transformers pillow matplotlib opencv-python

    # and you need access to facebook/sam3 on Hugging Face:
    huggingface-cli login

Usage:

    python segment_sam3.py --image images/plant.jpg \
                           --prompt "green plant leaves" \
                           --overlay_out plant_segmented.png
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from transformers import Sam3Processor, Sam3Model


# ---------------------------------------------------------------------
# HSV-based center green-point detection (your original logic)
# ---------------------------------------------------------------------
def auto_leaf_point(image_rgb: np.ndarray, crop_frac: float = 1.0):
    """
    image_rgb: H x W x 3, uint8, RGB
    crop_frac: fraction of central region to search (1.0 == full image)

    Returns: [[x, y]] in absolute image coords, or None if no green region.
    """
    h, w, _ = image_rgb.shape

    crop_size = crop_frac
    h0 = int(h * (0.5 - crop_size / 2))
    h1 = int(h * (0.5 + crop_size / 2))
    w0 = int(w * (0.5 - crop_size / 2))
    w1 = int(w * (0.5 + crop_size / 2))

    center_crop = image_rgb[h0:h1, w0:w1, :]

    center_crop_bgr = cv2.cvtColor(center_crop, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(center_crop_bgr, cv2.COLOR_BGR2HSV)

    # Typical "green" range in HSV – tweak if needed
    mask = (
        (hsv[..., 0] > 35) & (hsv[..., 0] < 85) &
        (hsv[..., 1] > 50) & (hsv[..., 2] > 50)
    )

    candidates = np.argwhere(mask)
    if candidates.shape[0] == 0:
        return None

    center_y, center_x = np.array(mask.shape) // 2
    dists = np.sum((candidates - [center_y, center_x]) ** 2, axis=1)
    y_rel, x_rel = candidates[np.argmin(dists)]

    y_abs = y_rel + h0
    x_abs = x_rel + w0
    return [[int(x_abs), int(y_abs)]]


# ---------------------------------------------------------------------
# SAM3 segmentation
# ---------------------------------------------------------------------
def segment_with_sam3(
    image_pil: Image.Image,
    text_prompt: str = "green plant leaves",
    score_thresh: float = 0.5,
    mask_thresh: float = 0.5,
):
    """
    Run SAM3 Promptable Concept Segmentation with a text prompt.

    Returns:
        union_mask: H x W boolean numpy array
        results: original HF post-processed dict for debugging
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SAM3] Using device: {device}")

    print("[SAM3] Loading model and processor (facebook/sam3)...")
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)

    # Prepare inputs
    inputs = processor(images=image_pil, text=text_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process to instance masks sized to the original image
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=score_thresh,
        mask_threshold=mask_thresh,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    masks = results["masks"]  # [N, H, W] tensor
    scores = results["scores"]  # [N]

    if masks is None or masks.shape[0] == 0:
        print("[SAM3] No masks returned.")
        return None, results

    print(f"[SAM3] Found {masks.shape[0]} instance masks for prompt: '{text_prompt}'")

    # Union all masks into a single plant mask
    if isinstance(masks, torch.Tensor):
        union_mask = masks.bool().any(dim=0)  # [H, W] bool tensor
        union_mask = union_mask.cpu().numpy()
    else:
        masks_np = np.asarray(masks)
        union_mask = masks_np.astype(bool).any(axis=0)

    return union_mask, results


# ---------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------
def overlay_union_mask(image_pil: Image.Image, union_mask: np.ndarray):
    """
    Return a new PIL image with the union_mask overlaid in red.
    """
    image = image_pil.convert("RGBA")
    h, w = image.size[1], image.size[0]

    # Ensure mask is H x W
    if union_mask.shape != (h, w):
        # union_mask is [H, W] in image coordinates; just be safe:
        union_mask = cv2.resize(
            union_mask.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    mask_255 = (union_mask.astype(np.uint8) * 255)
    mask_img = Image.fromarray(mask_255, mode="L")

    overlay = Image.new("RGBA", image.size, (255, 0, 0, 0))
    alpha = mask_img.point(lambda v: int(v * 0.5))  # 0.5 opacity
    overlay.putalpha(alpha)

    return Image.alpha_composite(image, overlay)


# ---------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Plant segmentation with SAM3")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to plant image (RGB).")
    parser.add_argument("--prompt", type=str, default="green plant leaves",
                        help="Text prompt for SAM3 (default: 'green plant leaves').")
    parser.add_argument("--overlay_out", type=str, default=None,
                        help="Optional path to save overlay PNG.")
    parser.add_argument("--crop_frac", type=float, default=1.0,
                        help="Fraction of central crop for HSV point search (1.0 = full image).")

    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    # Load RGB image
    image_pil = Image.open(img_path).convert("RGB")
    image_np = np.array(image_pil)

    # HSV auto leaf point (not currently used by SAM3, but useful for debugging)
    plantloc = auto_leaf_point(image_np, crop_frac=args.crop_frac)
    if plantloc is not None:
        print(f"[HSV] Auto-selected leaf point: {plantloc[0]}")
    else:
        print("[HSV] No green region detected in center; continuing anyway.")

    # Run SAM3 segmentation
    union_mask, results = segment_with_sam3(
        image_pil,
        text_prompt=args.prompt,
        score_thresh=0.5,
        mask_thresh=0.5,
    )

    if union_mask is None:
        print("No segmentation produced by SAM3.")
        return

    # Count pixels
    num_pixels = int(union_mask.sum())
    print(f"[RESULT] Number of plant pixels (union of instances): {num_pixels}")

    # Save overlay if requested
    if args.overlay_out is not None:
        overlay_img = overlay_union_mask(image_pil, union_mask)
        overlay_path = Path(args.overlay_out)
        overlay_img.save(overlay_path)
        print(f"[RESULT] Overlay saved to: {overlay_path}")

    # Optional: quick side-by-side visualization (opens a window if backend allows)
    # Comment out if you only want files.
    # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # axs[0].imshow(image_np)
    # axs[0].set_title("Original")
    # axs[0].axis("off")
    # axs[1].imshow(overlay_img)
    # axs[1].set_title(f"SAM3 Segmentation ({num_pixels} px)")
    # axs[1].axis("off")
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()
