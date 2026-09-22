"""
run_behavior_probe.py
=====================
Behavior probe for ConfigurableMedicalDataset with negative-pool implementation.

Tests the REAL __getitem__ in sage/utils/dataloader.py — NO monkey-patching.

What this probe measures
------------------------
- Output-class distribution: fraction of returned samples where fg < 20 vs >= 20
  AFTER the full pipeline (crop → aug_transform → normalize → tensor).

  Because spatial augmentation (flips, ShiftScaleRotate) can change fg counts,
  this distribution is NOT equivalent to the target sampling ratio (85/15).
  Do NOT use fg from returned labels to assert exact 85/15.

- The true guarantee for negative correctness comes from the assertion inside
  ConfigurableMedicalDataset.__getitem__:
      assert fg_actual < 20, "Pool entry violated fg threshold ..."
  which fires BEFORE augmentation. If that assertion never raises here,
  the pool is intact and no silent-accept occurred.

- To measure exact target ratio (positive/negative targets chosen at
  __getitem__ call time), production telemetry in dataloader itself is required.
  This probe does not attempt to infer it from post-augmentation fg.

Run from repo root on Colab (after building the pool):
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
    print("=== REAL BEHAVIOR PROBE (Crack500 negative-pool implementation) ===")
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

    print(f"   Dataset size     : {len(ds)} samples")
    print(f"   Negative pool    : {ds._neg_pool_size} candidates")

    random.seed(SEED)
    np.random.seed(SEED)

    # ── Metrics ───────────────────────────────────────────────────────────────
    returned_fg_lt20  = 0   # output fg < 20 after full pipeline
    returned_fg_ge20  = 0   # output fg >= 20 after full pipeline
    fg_pixels_log     = []
    runtime_errors    = 0
    shape_failures    = []

    print(f"\n2. Running {N_SAMPLES} __getitem__ calls on REAL implementation ...")

    for i in tqdm(range(N_SAMPLES), desc="Probe"):
        idx = random.randint(0, len(ds) - 1)
        try:
            sample = ds[idx]
        except AssertionError as e:
            # Pool integrity assertion fired inside dataloader — this is a hard failure
            print(f"\n[!] Pool integrity AssertionError at sample {i}: {e}")
            runtime_errors += 1
            import traceback; traceback.print_exc()
            continue
        except RuntimeError as e:
            runtime_errors += 1
            print(f"\n[!] RuntimeError at sample {i}: {e}")
            import traceback; traceback.print_exc()
            continue

        label = sample["label"]   # torch.Tensor (H, W), dtype=long
        image = sample["image"]   # torch.Tensor (3, H, W)

        # Shape checks
        if tuple(image.shape) != (3, IMAGE_SIZE, IMAGE_SIZE):
            shape_failures.append(f"sample={i} image={tuple(image.shape)}")
        if tuple(label.shape) != (IMAGE_SIZE, IMAGE_SIZE):
            shape_failures.append(f"sample={i} label={tuple(label.shape)}")

        fg = int((label > 0).sum().item())
        fg_pixels_log.append(fg)

        if fg < 20:
            returned_fg_lt20 += 1
        else:
            returned_fg_ge20 += 1

    # ── Report ────────────────────────────────────────────────────────────────
    total_returned = returned_fg_lt20 + returned_fg_ge20
    pct_lt20 = returned_fg_lt20 / total_returned * 100 if total_returned else 0
    pct_ge20 = returned_fg_ge20 / total_returned * 100 if total_returned else 0

    print(f"\n{'='*70}")
    print(f"=== BEHAVIOR PROBE RESULTS ===")
    print(f"{'='*70}")
    print(f"Total samples requested   : {N_SAMPLES}")
    print(f"Samples returned OK       : {total_returned}")
    print(f"RuntimeErrors             : {runtime_errors}")
    print(f"Shape failures            : {len(shape_failures)}")

    print(f"\n[Output-class distribution — measured on labels AFTER full pipeline]")
    print(f"  fg >= 20 (crack present) : {returned_fg_ge20:>5}  ({pct_ge20:.1f}%)")
    print(f"  fg <  20 (no crack)      : {returned_fg_lt20:>5}  ({pct_lt20:.1f}%)")
    print()
    print(f"  NOTE: This is the OUTPUT distribution after crop + spatial aug.")
    print(f"        Spatial augmentation can shift fg counts, so this is NOT")
    print(f"        the same as the target sampling ratio (85% pos / 15% neg).")
    print(f"        Pool guarantee: P(valid negative | negative request) = 1,")
    print(f"        enforced by assertion in ConfigurableMedicalDataset.__getitem__")
    print(f"        BEFORE augmentation. Zero AssertionErrors above = pool intact.")

    if fg_pixels_log:
        fg = np.array(fg_pixels_log)
        print(f"\n[FG pixel distribution of ALL returned labels (post-augmentation)]")
        print(f"  Min    : {fg.min()}")
        print(f"  Max    : {fg.max()}")
        print(f"  Mean   : {fg.mean():.1f}")
        print(f"  Median : {np.median(fg):.1f}")

    if shape_failures:
        print(f"\n  Shape failure details: {shape_failures[:5]}")

    passed = (runtime_errors == 0 and len(shape_failures) == 0)
    print(f"\n{'='*70}")
    print(f"OVERALL: {'PASS' if passed else 'FAIL'}")
    print(f"{'='*70}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
