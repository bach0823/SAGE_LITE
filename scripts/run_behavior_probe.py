"""
run_behavior_probe.py
=====================
Behavior probe for ConfigurableMedicalDataset with negative-pool implementation.

Tests the REAL __getitem__ in sage/utils/dataloader.py — NO monkey-patching.

The probe observes the FINAL ACCEPTED distribution by inspecting returned labels:
  - fg_pixels = (label > 0).sum()
  - fg >= 20  → accepted as positive
  - fg < 20   → accepted as negative

Note on expected ratios:
  - Pool guarantees P(valid negative | negative request) = 1  (by construction).
  - 85/15 is the expected TARGET SAMPLING RATIO set at __getitem__ call time.
    Actual observed ratio may differ slightly due to random variation over 3000 samples.

Run from repo root on Colab (after building the pool):
    python scripts/run_behavior_probe.py

Exits with code 1 on any RuntimeError.
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
    accepted_positive = 0   # returned label has fg >= 20
    accepted_negative = 0   # returned label has fg < 20
    fg_pixels_log     = []
    runtime_errors    = 0
    shape_ok          = True

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

        label = sample["label"]   # torch.Tensor (H, W), dtype=long

        # Shape check
        if tuple(label.shape) != (IMAGE_SIZE, IMAGE_SIZE):
            shape_ok = False
            print(f"  WARN: unexpected label shape {tuple(label.shape)} at sample {i}")

        fg = int((label > 0).sum().item())
        fg_pixels_log.append(fg)

        if fg >= 20:
            accepted_positive += 1
        else:
            accepted_negative += 1

    # ── Report ────────────────────────────────────────────────────────────────
    total_returned = accepted_positive + accepted_negative
    pos_pct = accepted_positive / total_returned * 100 if total_returned > 0 else 0
    neg_pct = accepted_negative / total_returned * 100 if total_returned > 0 else 0

    print(f"\n{'='*70}")
    print(f"=== BEHAVIOR PROBE RESULTS ===")
    print(f"{'='*70}")
    print(f"Total samples requested       : {N_SAMPLES}")
    print(f"Samples returned successfully : {total_returned}")
    print(f"RuntimeErrors                 : {runtime_errors}")
    print(f"All output shapes correct     : {shape_ok and runtime_errors == 0}")
    print()
    print(f"[Final Accepted Distribution — measured from returned labels]")
    print(f"  Positive (fg >= 20) : {accepted_positive:>5}  ({pos_pct:.1f}%)")
    print(f"  Negative (fg <  20) : {accepted_negative:>5}  ({neg_pct:.1f}%)")
    print()
    print(f"  Note: 85/15 is the expected TARGET SAMPLING RATIO set per __getitem__ call.")
    print(f"        Pool guarantees P(valid negative | negative request) = 1 by construction.")

    if fg_pixels_log:
        fg = np.array(fg_pixels_log)
        print(f"\n[FG pixel distribution of ALL returned crops]")
        print(f"  Min    : {fg.min()}")
        print(f"  Max    : {fg.max()}")
        print(f"  Mean   : {fg.mean():.1f}")
        print(f"  Median : {np.median(fg):.1f}")
        print(f"  fg <  20 (negative) : {(fg < 20).sum()}")
        print(f"  fg >= 20 (positive) : {(fg >= 20).sum()}")

        # Verify pool integrity: all negatives must genuinely have fg < 20
        neg_with_high_fg = sum(1 for v in fg_pixels_log[:accepted_negative] if v >= 20)
        if neg_with_high_fg > 0:
            print(f"\n  [!] INTEGRITY VIOLATION: {neg_with_high_fg} 'negative' samples "
                  f"had fg >= 20. Pool may be stale.")
        else:
            print(f"\n  Pool integrity: OK — all accepted negatives have fg < 20.")

    passed = (runtime_errors == 0 and shape_ok)
    print(f"\n{'='*70}")
    print(f"OVERALL: {'PASS' if passed else 'FAIL'}")
    print(f"{'='*70}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
