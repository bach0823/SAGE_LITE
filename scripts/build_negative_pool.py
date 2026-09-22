"""
build_negative_pool.py
======================
Offline script: scan Crack500 train set, collect negative-crop candidates
(fg_pixels < 20) and save deterministic metadata (with exact crop pixel
coordinates) to an .npz artifact.

Must be run ONCE before training, from repo root on Colab:
    python scripts/build_negative_pool.py [--args]

Output:
    /content/dataset/Crack500/negative_pool.npz

At training time the dataloader loads this artifact and slices the padded
image directly at [crop_y : crop_y+448, crop_x : crop_x+448], guaranteeing
fg_pixels < 20 without any RandomCrop call.

Does NOT modify sage/utils/dataloader.py.
Does NOT train B0.
Does NOT add CLAHE / ElasticTransform / GridDistortion.
Does NOT change fg_pixels threshold (20) or positive pipeline.
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import cv2
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG_PATH = "configs/b0_crack500.yaml"
DEFAULT_OUTPUT_PATH = "/content/dataset/Crack500/negative_pool.npz"
DEFAULT_IMAGE_SIZE  = 448
DEFAULT_FG_THR      = 20
DEFAULT_N_CROPS     = 200
DEFAULT_SEED        = 42


def parse_args():
    p = argparse.ArgumentParser(description="Build Crack500 negative-crop pool")
    p.add_argument("--config",     default=DEFAULT_CONFIG_PATH)
    p.add_argument("--output",     default=DEFAULT_OUTPUT_PATH)
    p.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    p.add_argument("--fg-thr",     type=int, default=DEFAULT_FG_THR)
    p.add_argument("--n-crops",    type=int, default=DEFAULT_N_CROPS)
    p.add_argument("--seed",       type=int, default=DEFAULT_SEED)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Pure-numpy crop implementation (no RandomCrop randomness)
# ─────────────────────────────────────────────────────────────────────────────

def pad_reflect(image: np.ndarray, mask: np.ndarray, target: int):
    """
    Replicates A.PadIfNeeded with BORDER_REFLECT_101.
    Returns (padded_image, padded_mask, pad_top, pad_left).
    """
    h, w = image.shape[:2]
    pad_h = max(0, target - h)
    pad_w = max(0, target - w)
    top  = pad_h // 2
    bot  = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    img_padded  = cv2.copyMakeBorder(image, top, bot, left, right,
                                      cv2.BORDER_REFLECT_101)
    mask_padded = cv2.copyMakeBorder(mask,  top, bot, left, right,
                                      cv2.BORDER_REFLECT_101)
    return img_padded, mask_padded, top, left


def random_crop_coords(padded_h: int, padded_w: int,
                        crop_size: int, rng: np.random.Generator):
    """Return (y, x) top-left of a uniform random crop."""
    max_y = padded_h - crop_size
    max_x = padded_w - crop_size
    y = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
    x = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
    return y, x


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 70)
    print("=== BUILD CRACK500 NEGATIVE CANDIDATE POOL ===")
    print("=" * 70)
    print(f"Config          : {args.config}")
    print(f"Output          : {args.output}")
    print(f"Image size      : {args.image_size}")
    print(f"FG threshold    : fg < {args.fg_thr}")
    print(f"Crops per image : {args.n_crops}")
    print(f"Seed            : {args.seed}")

    # Deterministic RNG per image (child seeds from base seed)
    base_rng = np.random.default_rng(args.seed)

    # ── Load config ────────────────────────────────────────────────────────────
    if not os.path.exists(args.config):
        print(f"ERROR: Config not found: {args.config}")
        sys.exit(1)

    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    root_dir  = cfg.get("root_dir", "")
    train_cfg = cfg.get("train", {})
    img_dir   = train_cfg.get("images", "")
    mask_dir  = train_cfg.get("masks",  "")

    if root_dir and not os.path.isabs(img_dir):
        img_dir  = os.path.join(root_dir, img_dir)
        mask_dir = os.path.join(root_dir, mask_dir)

    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        print(f"ERROR: img_dir='{img_dir}' or mask_dir='{mask_dir}' not found.")
        sys.exit(1)

    # ── Collect samples (same logic as dataloader) ─────────────────────────────
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    mask_map   = {}
    for root, _, files in os.walk(mask_dir):
        for f in files:
            stem = os.path.splitext(f)[0]
            mask_map[stem] = os.path.join(root, f)

    samples = []
    for root, _, files in os.walk(img_dir):
        for f in sorted(files):
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_exts:
                stem = os.path.splitext(f)[0]
                if stem in mask_map:
                    samples.append({
                        "image": os.path.join(root, f),
                        "label": mask_map[stem],
                    })

    n_images = len(samples)
    print(f"\nFound {n_images} image-mask pairs.")
    if n_images == 0:
        print("ERROR: No samples found.")
        sys.exit(1)

    # ── Scan & build pool ──────────────────────────────────────────────────────
    cols = {k: [] for k in ("src_idx", "src_path", "crop_x", "crop_y",
                             "pad_top", "pad_left", "fg_pixels")}
    per_source_counts = np.zeros(n_images, dtype=np.int32)

    for src_idx in tqdm(range(n_images), desc="Scanning"):
        sample = samples[src_idx]
        image  = cv2.imread(sample["image"])
        mask   = cv2.imread(sample["label"], cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"  WARN: Cannot read {sample['image']}. Skipping.")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask  = (mask > 0).astype(np.uint8)

        # Pad once (deterministic, no randomness)
        img_p, mask_p, pad_top, pad_left = pad_reflect(image, mask, args.image_size)
        ph, pw = img_p.shape[:2]

        # Per-image child RNG (reproducible across runs with same seed)
        img_rng = np.random.default_rng(base_rng.integers(0, 2**31))

        for _ in range(args.n_crops):
            cy, cx = random_crop_coords(ph, pw, args.image_size, img_rng)
            crop_mask = mask_p[cy:cy + args.image_size, cx:cx + args.image_size]
            fg = int((crop_mask > 0).sum())

            if fg < args.fg_thr:
                cols["src_idx"].append(src_idx)
                cols["src_path"].append(sample["image"])
                cols["crop_x"].append(cx)
                cols["crop_y"].append(cy)
                cols["pad_top"].append(pad_top)
                cols["pad_left"].append(pad_left)
                cols["fg_pixels"].append(fg)
                per_source_counts[src_idx] += 1

    # ── Summary stats ──────────────────────────────────────────────────────────
    total   = len(cols["src_idx"])
    contrib = int((per_source_counts > 0).sum())

    print(f"\n{'='*50}")
    print(f"Total negative candidates  : {total}")
    print(f"Contributing sources       : {contrib} / {n_images}  "
          f"({contrib / n_images * 100:.1f}%)")
    print(f"Non-contributing sources   : {n_images - contrib}  "
          f"({(n_images - contrib) / n_images * 100:.1f}%)")
    if contrib > 0:
        nz = per_source_counts[per_source_counts > 0]
        print(f"Candidates/contributing-src: min={nz.min()} mean={nz.mean():.1f} "
              f"median={np.median(nz):.1f} max={nz.max()}")

    if total == 0:
        print("ERROR: No negative candidates found. Check threshold and images.")
        sys.exit(1)

    # ── Save artifact ──────────────────────────────────────────────────────────
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    provenance = {
        "seed":                     args.seed,
        "image_size":               args.image_size,
        "fg_threshold":             args.fg_thr,
        "n_crops_per_image":        args.n_crops,
        "n_sources":                n_images,
        "n_contributing_sources":   contrib,
        "total_candidates":         total,
        "border_mode":              "BORDER_REFLECT_101",
        "sampling_strategy":        "uniform_pool",       # Phương án A
        "crop_size":                args.image_size,
        "note": ("crop_y, crop_x are pixel top-left coords in the PADDED image. "
                 "Reproduce: load img → pad_reflect → slice [cy:cy+448, cx:cx+448]")
    }

    np.savez_compressed(
        args.output,
        src_idx   = np.array(cols["src_idx"],   dtype=np.int32),
        crop_x    = np.array(cols["crop_x"],    dtype=np.int16),
        crop_y    = np.array(cols["crop_y"],    dtype=np.int16),
        pad_top   = np.array(cols["pad_top"],   dtype=np.int16),
        pad_left  = np.array(cols["pad_left"],  dtype=np.int16),
        fg_pixels = np.array(cols["fg_pixels"], dtype=np.int32),
        src_paths = np.array(cols["src_path"]),               # object/str array
        provenance= np.array([json.dumps(provenance)]),       # 1-element str array
    )

    print(f"\nPool saved → {args.output}")
    print(f"Provenance:\n{json.dumps(provenance, indent=2)}")
    print("\nDo NOT commit this .npz file to Git.")


if __name__ == "__main__":
    main()
