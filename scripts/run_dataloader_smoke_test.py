"""
run_dataloader_smoke_test.py
============================
Smoke test for ConfigurableMedicalDataset (Crack500 smart-filter mode).

Tests the REAL __getitem__ in sage/utils/dataloader.py — NO monkey-patching.

The smart-filter rejects crops with fg_pixels < 20 and resamples source
up to 10 times (20 crop attempts each). This test verifies the pipeline
runs cleanly without RuntimeError across 200 batches.

Run from repo root on Colab:
    python scripts/run_dataloader_smoke_test.py

Exits with code 1 on any failure.
"""

import os
import sys
import math
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sage.utils.dataloader import ConfigurableMedicalDataset


# ──────────────────────────────────────────────────────────────
CONFIG_PATH  = "configs/b0_crack500.yaml"
IMAGE_SIZE   = 448
BATCH_SIZE   = 16
N_BATCHES    = 200   # Max batches to test
SEED         = 42
# ──────────────────────────────────────────────────────────────


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def run_smoke(num_workers: int) -> bool:
    """Returns True if all batches pass; False on any error."""
    print(f"\n{'='*70}")
    print(f"=== SMOKE TEST  num_workers={num_workers} ===")
    print(f"{'='*70}")

    if not os.path.exists(CONFIG_PATH):
        print(f"ERROR: {CONFIG_PATH} not found. Run from repo root.")
        return False

    try:
        ds = ConfigurableMedicalDataset(CONFIG_PATH, split="train",
                                        image_size=IMAGE_SIZE)
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        return False

    expected_batches = math.ceil(len(ds) / BATCH_SIZE)
    target_batches = min(N_BATCHES, expected_batches)

    print(f"Dataset size    : {len(ds)} samples")
    print(f"Expected batches: {expected_batches} (target test: {target_batches})")
    print(f"Smart filter    : {ds.use_smart_filter}  (fg_pixels >= 20 required)")

    g = torch.Generator()
    g.manual_seed(SEED)

    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=False,
    )

    passed_batches = 0
    runtime_errors = 0
    shape_failures  = []
    label_failures  = []

    try:
        for batch_idx, batch in enumerate(
            tqdm(dl, total=target_batches, desc=f"workers={num_workers}"), start=1
        ):
            images = batch["image"]   # (B, 3, H, W)
            labels = batch["label"]   # (B, H, W)

            for b in range(images.shape[0]):
                if tuple(images[b].shape) != (3, IMAGE_SIZE, IMAGE_SIZE):
                    shape_failures.append(
                        f"batch={batch_idx} sample={b} image={tuple(images[b].shape)}"
                    )
                if tuple(labels[b].shape) != (IMAGE_SIZE, IMAGE_SIZE):
                    shape_failures.append(
                        f"batch={batch_idx} sample={b} label={tuple(labels[b].shape)}"
                    )
                unique_vals = labels[b].unique().tolist()
                for v in unique_vals:
                    if v not in (0, 1):
                        label_failures.append(
                            f"batch={batch_idx} sample={b} unexpected label value={v}"
                        )

            passed_batches += 1
            if batch_idx >= target_batches:
                break

    except RuntimeError as e:
        runtime_errors += 1
        print(f"\n[!] RuntimeError at batch {passed_batches + 1}: {e}")
        import traceback; traceback.print_exc()

    print(f"\n--- RESULTS (num_workers={num_workers}) ---")
    print(f"Batches passed    : {passed_batches} / {target_batches}")
    print(f"RuntimeErrors     : {runtime_errors}")
    print(f"Shape failures    : {len(shape_failures)}")
    print(f"Label failures    : {len(label_failures)}")

    if shape_failures:
        print("  Shape details:", shape_failures[:5])
    if label_failures:
        print("  Label details:", label_failures[:5])

    ok = (runtime_errors == 0 and
          len(shape_failures) == 0 and
          len(label_failures) == 0 and
          passed_batches >= target_batches)

    print(f"\n{'PASS' if ok else 'FAIL'} — num_workers={num_workers}")
    return ok


def main():
    overall_ok = True
    overall_ok &= run_smoke(num_workers=4)
    overall_ok &= run_smoke(num_workers=0)

    print(f"\n{'='*70}")
    print(f"OVERALL: {'PASS' if overall_ok else 'FAIL'}")
    print(f"{'='*70}")

    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
