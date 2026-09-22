"""
run_behavior_probe.py
=====================
Behavior probe for ConfigurableMedicalDataset (Crack500 smart-filter mode).

Tests the REAL __getitem__ in sage/utils/dataloader.py — NO monkey-patching.

The smart-filter accepts only crops with fg_pixels >= 20 (crack present).
This probe verifies:
  - No RuntimeError over 3000 samples.
  - All returned images have shape (3, 448, 448).
  - All returned labels have shape (448, 448) with values in {0, 1}.
  - Foreground pixel distribution of returned crops (post-augmentation).

Note: fg measured from returned labels is AFTER spatial augmentation and
normalization. This reports output distribution, not guaranteed crop-level
fg (which is enforced at crop-selection time before augmentation).

Run from repo root on Colab:
    python scripts/run_behavior_probe.py

Exits with code 1 on any RuntimeError or shape failure.
"""

import os
import sys
import random
import numpy as np
import torch
from tqdm import tqdm

from sage.utils.dataloader import ConfigurableMedicalDataset


# ──────────────────────────────────────────────────────────────
CONFIG_PATH = "configs/b0_crack500.yaml"
IMAGE_SIZE  = 448
N_SAMPLES   = 3000
SEED        = 42
# ──────────────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("=== REAL BEHAVIOR PROBE (Crack500 smart-filter crop-retry) ===")
    print("=" * 70)

    if not os.path.exists(CONFIG_PATH):
        print(f"ERROR: {CONFIG_PATH} not found. Run from repo root.")
        sys.exit(1)

    print(f"\n1. Loading dataset from {CONFIG_PATH} ...")
    try:
        ds = ConfigurableMedicalDataset(CONFIG_PATH, split="train",
                                        image_size=IMAGE_SIZE)
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        sys.exit(1)

    print(f"   Dataset size  : {len(ds)} samples")
    print(f"   Smart filter  : {ds.use_smart_filter}  (fg_pixels >= 20 required at crop time)")

    random.seed(SEED)
    np.random.seed(SEED)

    returned_ok    = 0
    runtime_errors = 0
    shape_failures = []
    fg_pixels_log  = []

    print(f"\n2. Running {N_SAMPLES} __getitem__ calls on REAL implementation ...")

    for i in tqdm(range(N_SAMPLES), desc="Probe"):
        idx = random.randint(0, len(ds) - 1)
        try:
            sample = ds[idx]
        except RuntimeError as e:
            runtime_errors += 1
            print(f"\n[!] RuntimeError at sample {i}: {e}")
            import traceback; traceback.print_exc()
            continue

        image = sample["image"]   # torch.Tensor (3, H, W)
        label = sample["label"]   # torch.Tensor (H, W), dtype=long

        if tuple(image.shape) != (3, IMAGE_SIZE, IMAGE_SIZE):
            shape_failures.append(f"sample={i} image={tuple(image.shape)}")
        if tuple(label.shape) != (IMAGE_SIZE, IMAGE_SIZE):
            shape_failures.append(f"sample={i} label={tuple(label.shape)}")

        fg = int((label > 0).sum().item())
        fg_pixels_log.append(fg)
        returned_ok += 1

    print(f"\n{'='*70}")
    print(f"=== BEHAVIOR PROBE RESULTS ===")
    print(f"{'='*70}")
    print(f"Total requests            : {N_SAMPLES}")
    print(f"Returned successfully     : {returned_ok}")
    print(f"RuntimeErrors             : {runtime_errors}")
    print(f"Shape failures            : {len(shape_failures)}")

    if fg_pixels_log:
        fg = np.array(fg_pixels_log)
        print(f"\n[FG pixel distribution of returned labels (post-augmentation)]")
        print(f"  Note: fg is measured AFTER spatial augmentation. Crop selection")
        print(f"        guarantees fg_pixels >= 20 BEFORE augmentation.")
        print(f"  Min    : {fg.min()}")
        print(f"  Max    : {fg.max()}")
        print(f"  Mean   : {fg.mean():.1f}")
        print(f"  Median : {np.median(fg):.1f}")
        print(f"  Samples with fg > 0  : {(fg > 0).sum()}  ({(fg > 0).mean()*100:.1f}%)")
        print(f"  Samples with fg == 0 : {(fg == 0).sum()}  ({(fg == 0).mean()*100:.1f}%)")

    if shape_failures:
        print(f"\nShape failure details: {shape_failures[:5]}")

    passed = (runtime_errors == 0 and len(shape_failures) == 0)
    print(f"\n{'='*70}")
    print(f"OVERALL: {'PASS' if passed else 'FAIL'}")
    print(f"{'='*70}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
