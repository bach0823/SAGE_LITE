"""
analyze_negative_pool_design.py
================================
Scans Crack500 train set, builds a negative-crop candidate pool,
and analyses source-distribution trade-offs BEFORE any dataloader change.

Run from repo root:
    python scripts/analyze_negative_pool_design.py

Does NOT modify sage/utils/dataloader.py.
Does NOT train B0.
Does NOT add CLAHE / ElasticTransform / GridDistortion.
Does NOT change fg_pixels threshold (20) or positive pipeline.
"""

import os
import sys
import json
import random
import numpy as np
import cv2
from collections import defaultdict
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sage.utils.dataloader import get_dataset_from_config


# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
CONFIG_PATH       = "configs/b0_crack500.yaml"
IMAGE_SIZE        = 448
N_CROPS_PER_IMAGE = 200   # candidates to generate per source image
FG_NEGATIVE_THR   = 20    # identical to dataloader threshold
OUTPUT_JSON       = "results/negative_pool_metadata.json"


def build_crop_transform(image_size: int) -> A.Compose:
    """Same PadIfNeeded + RandomCrop as dataloader, no augmentation."""
    return A.Compose([
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_REFLECT_101,
        ),
        A.RandomCrop(height=image_size, width=image_size),
    ])


def main():
    print("=" * 70)
    print("=== CRACK500 NEGATIVE-POOL DESIGN ANALYSIS ===")
    print("=" * 70)

    if not os.path.exists(CONFIG_PATH):
        print(f"ERROR: {CONFIG_PATH} not found. Run from repo root.")
        sys.exit(1)

    ds = get_dataset_from_config(CONFIG_PATH, split='train', image_size=IMAGE_SIZE)
    samples = ds.samples
    n_images = len(samples)
    print(f"Total training images : {n_images}")
    print(f"Crops per image       : {N_CROPS_PER_IMAGE}")
    print(f"Negative threshold    : fg_pixels < {FG_NEGATIVE_THR}")
    print(f"Scanning ... (this may take a few minutes on Colab T4)\n")

    crop_tf = build_crop_transform(IMAGE_SIZE)

    # Pool metadata
    pool_metadata = []           # list of dicts: {src_idx, src_path, x, y, fg}
    per_source_counts = np.zeros(n_images, dtype=np.int32)

    for src_idx in tqdm(range(n_images), desc="Building Negative Pool"):
        sample = samples[src_idx]
        image  = cv2.imread(sample['image'])
        mask   = cv2.imread(sample['label'], cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            print(f"WARN: Could not read {sample['image']} or its mask. Skipping.")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask  = (mask > 0).astype(np.uint8)

        for _ in range(N_CROPS_PER_IMAGE):
            cropped = crop_tf(image=image, mask=mask)
            fg = int((cropped['mask'] > 0).sum())
            if fg < FG_NEGATIVE_THR:
                # store minimal metadata (no pixel data)
                pool_metadata.append({
                    "src_idx":  src_idx,
                    "src_path": sample['image'],
                    "fg":       fg,
                })
                per_source_counts[src_idx] += 1

    # ──────────────────────────────────────────────────────────
    # Statistics
    # ──────────────────────────────────────────────────────────
    total_candidates = len(pool_metadata)
    contributing     = int((per_source_counts > 0).sum())
    non_contributing = n_images - contributing

    print("\n" + "=" * 70)
    print("=== CANDIDATE POOL STATISTICS ===")
    print("=" * 70)
    print(f"Total negative candidates           : {total_candidates}")
    print(f"Source images contributing  (>= 1)  : {contributing}  "
          f"({contributing / n_images * 100:.1f}%)")
    print(f"Source images NOT contributing (= 0): {non_contributing}  "
          f"({non_contributing / n_images * 100:.1f}%)")

    if contributing > 0:
        nonzero_counts = per_source_counts[per_source_counts > 0]
        print(f"\nCandidates/contributing-source distribution:")
        print(f"  Min    : {nonzero_counts.min()}")
        print(f"  Max    : {nonzero_counts.max()}")
        print(f"  Mean   : {nonzero_counts.mean():.1f}")
        print(f"  Median : {np.median(nonzero_counts):.1f}")
        print(f"  P10    : {np.percentile(nonzero_counts, 10):.1f}")
        print(f"  P90    : {np.percentile(nonzero_counts, 90):.1f}")

    # ──────────────────────────────────────────────────────────
    # Source-distribution skew analysis
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("=== SOURCE DISTRIBUTION SKEW ANALYSIS ===")
    print("=" * 70)
    print("If NEGATIVE request draws uniformly from the pool,")
    print("each candidate is equally likely regardless of source.")
    print("This means high-contributor sources are over-represented.\n")

    # Sampling weight of each CONTRIBUTING source
    if total_candidates > 0:
        weights = per_source_counts / per_source_counts.sum()  # over ALL images

        top_k = 10
        top_indices = np.argsort(per_source_counts)[::-1][:top_k]
        cumulative_top = per_source_counts[top_indices].sum() / total_candidates * 100
        print(f"Top-{top_k} contributor sources cover : {cumulative_top:.1f}% of pool")
        print(f"(Ideally closer to {top_k / n_images * 100:.1f}% if uniform)")

        # Distribution of sampling probabilities
        nonzero_weights = weights[weights > 0]
        print(f"\nSampling probability (per candidate-pool draw):")
        print(f"  Ideal per source  : {1.0 / n_images * 100:.4f}%")
        print(f"  Actual min (contr): {nonzero_weights.min() * 100:.4f}%")
        print(f"  Actual max (contr): {nonzero_weights.max() * 100:.4f}%")
        print(f"  Gini coefficient  : {_gini(nonzero_weights):.4f}  (0=equal, 1=maximal skew)")

    # ──────────────────────────────────────────────────────────
    # Guaranteed-coverage estimate
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("=== GUARANTEED-COVERAGE ESTIMATE ===")
    print("=" * 70)
    print("With pool-based draw: P(getting a valid negative) = 1.0 (by construction)")
    print("There is NO fail-fast risk for NEGATIVE requests if we draw from pool.")
    print("The only residual risk is pool exhaustion if pool is too small.")
    print(f"\nEstimated epoch size: 1896 samples → ~285 NEGATIVE requests (15%).")
    print(f"Pool size          : {total_candidates}")
    if total_candidates > 0:
        ratio = total_candidates / max(int(n_images * 0.15), 1)
        print(f"Pool/epoch-negative : {ratio:.0f}×  ", end="")
        if ratio >= 10:
            print("✓ Sufficient – pool is much larger than per-epoch negative demand.")
        else:
            print("⚠ May be tight – consider increasing N_CROPS_PER_IMAGE at build time.")

    # ──────────────────────────────────────────────────────────
    # Save pool metadata JSON for implementation reference
    # ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "total_candidates":   total_candidates,
            "contributing":       contributing,
            "non_contributing":   non_contributing,
            "n_images":           n_images,
            "fg_threshold":       FG_NEGATIVE_THR,
            "crops_per_image":    N_CROPS_PER_IMAGE,
            "pool":               pool_metadata[:5000],   # keep first 5000 to avoid huge JSON
        }, f, indent=2)
    print(f"\nPool metadata saved (first 5000 entries) → {OUTPUT_JSON}")

    print("\n" + "=" * 70)
    print("=== DESIGN RECOMMENDATION ===")
    print("=" * 70)
    print("""
NEXT STEP (requires user approval before any dataloader change):
  A) Pre-build negative pool at dataset __init__ time (same logic as this script).
  B) POSITIVE request  → existing crop-retry / source-resample loop (unchanged).
  C) NEGATIVE request  → draw directly from pre-built pool (no retry needed).
  D) Pool is built once per training run; shuffle pool before each epoch.
  E) This guarantees final accepted 85% POS / 15% NEG with zero FAIL-FAST.
  
  Source-skew trade-off:
  - Pool over-represents images with few cracks (high P(fg<20)).
  - Consequence: negative patches come from a biased subset of images.
  - For a BASELINE B0, this is acceptable.
  - Optional mitigation (post-B0): cap candidates-per-source to equalise weights.
""")


def _gini(weights: np.ndarray) -> float:
    """Compute Gini coefficient of a normalised distribution."""
    w = np.sort(weights)
    n = len(w)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * w).sum() / (n * w.sum())) - (n + 1) / n)


if __name__ == '__main__':
    main()
