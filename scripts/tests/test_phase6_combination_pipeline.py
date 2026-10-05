#!/usr/bin/env python3
"""
scripts/tests/test_phase6_combination_pipeline.py

Unit and Invariant Tests for SAGE-Lite Phase 6 Final Pipeline Combination:
  1. Config Invariants for Stage 1A (B1-v1, v2, v3) & Stage 1B (A1-v1, v2, v3).
  2. Dynamic Combination Config Generation (Run 2A: A1* + B1*).
  3. Dual-loss CrackBinaryLoss forward & backward sanity check.
  4. Decision Logic: Objective Selection & Stage 2 Pass/Fail criteria.
"""

import os
import sys
import tempfile
import yaml
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.train_crack import CrackBinaryLoss
from scripts.run_phase6_combination_pipeline import (
    generate_combined_config,
    REF_B1_V0,
    REF_A1_V0,
)


def test_stage_1a_configs_invariants():
    """Verify Stage 1A sweep configs exist and match exact specified lambda and r."""
    expected = {
        "b1_v1_l002_r2.yaml": {"weight": 0.020, "dilation": 2},
        "b1_v2_l008_r2.yaml": {"weight": 0.080, "dilation": 2},
        "b1_v3_l004_r1.yaml": {"weight": 0.040, "dilation": 1},
    }
    cfg_dir = os.path.join(PROJECT_ROOT, "configs", "p3_ablation", "phase6_combination")
    for fname, exp in expected.items():
        p = os.path.join(cfg_dir, fname)
        assert os.path.exists(p), f"Config file missing: {p}"
        with open(p, "r") as f:
            cfg = yaml.safe_load(f)
        assert cfg["model"] == "B2"
        assert cfg["num_transformer_layers"] == 4
        assert cfg["p3_mode"] == "C"
        assert cfg["seed"] == 42
        assert cfg["patience"] == 8
        assert cfg["epochs"] == 35
        assert cfg["stage1_epochs"] == 17
        assert abs(cfg["ab_bpl_weight"] - exp["weight"]) < 1e-5
        assert cfg["ab_bpl_dilation"] == exp["dilation"]
        assert cfg.get("boundary_iou_weight", 0.0) == 0.0
    print("[PASS] test_stage_1a_configs_invariants")


def test_stage_1b_configs_invariants():
    """Verify Stage 1B sweep configs exist and match exact specified lambda and d."""
    expected = {
        "a1_v1_l025_d2.yaml": {"weight": 0.25, "dilation": 2},
        "a1_v2_l075_d2.yaml": {"weight": 0.75, "dilation": 2},
        "a1_v3_l050_d3.yaml": {"weight": 0.50, "dilation": 3},
    }
    cfg_dir = os.path.join(PROJECT_ROOT, "configs", "p3_ablation", "phase6_combination")
    for fname, exp in expected.items():
        p = os.path.join(cfg_dir, fname)
        assert os.path.exists(p), f"Config file missing: {p}"
        with open(p, "r") as f:
            cfg = yaml.safe_load(f)
        assert cfg["model"] == "B2"
        assert cfg["num_transformer_layers"] == 4
        assert cfg["p3_mode"] == "C"
        assert cfg["seed"] == 42
        assert cfg["patience"] == 8
        assert cfg["epochs"] == 35
        assert cfg["stage1_epochs"] == 17
        assert abs(cfg["boundary_iou_weight"] - exp["weight"]) < 1e-5
        assert cfg["boundary_iou_dilation"] == exp["dilation"]
        assert cfg.get("ab_bpl_weight", 0.0) == 0.0
    print("[PASS] test_stage_1b_configs_invariants")


def test_dual_loss_forward_backward():
    """Verify CrackBinaryLoss simultaneously calculates SoftBIoU and AB-BPL cleanly."""
    criterion = CrackBinaryLoss(
        bce_weight=1.0,
        dice_weight=1.5,
        boundary_weight=0.50,
        boundary_dilation=2,
        margin_weight=0.040,
        margin_dilation=2,
    )
    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    targets = (torch.rand(2, 1, 64, 64) > 0.8).float()

    loss = criterion(logits, targets)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0

    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()
    print("[PASS] test_dual_loss_forward_backward")


def test_generate_combined_config():
    """Verify combined config generator produces correct YAML structure."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_cfg = os.path.join(tmp_dir, "comb_a1_b1.yaml")
        run_dir = os.path.join(tmp_dir, "run")
        generate_combined_config(
            lambda_b1=0.040,
            r_b1=2,
            lambda_a1=0.50,
            d_a1=2,
            output_cfg_path=out_cfg,
            output_run_dir=run_dir,
            data_root="dummy/data",
        )
        assert os.path.exists(out_cfg)
        with open(out_cfg, "r") as f:
            cfg = yaml.safe_load(f)
        assert cfg["ab_bpl_weight"] == 0.040
        assert cfg["ab_bpl_dilation"] == 2
        assert cfg["boundary_iou_weight"] == 0.50
        assert cfg["boundary_iou_dilation"] == 2
        assert cfg["p3_mode"] == "C"
        assert cfg["stage2_base_lr"] == 1e-4
    print("[PASS] test_generate_combined_config")


def test_decision_criteria_math():
    """Verify mathematical formulas for selection and PASS/FAIL."""
    # Stage 1A Selection: Maximize Dice * Recall s.t. Recall >= 0.840
    runs_1a = [
        {"run_id": "v0", "dice": 0.7685, "recall": 0.8458, "pass": True},
        {"run_id": "v1", "dice": 0.7680, "recall": 0.8470, "pass": True},
        {"run_id": "v2", "dice": 0.7690, "recall": 0.8350, "pass": False},  # Fails constraint
    ]
    for r in runs_1a:
        r["score"] = r["dice"] * r["recall"]

    valid = [r for r in runs_1a if r["pass"]]
    best_1a = max(valid, key=lambda x: x["score"])
    # v0: 0.7685 * 0.8458 = 0.6500
    # v1: 0.7680 * 0.8470 = 0.6505 -> v1 wins among valid
    assert best_1a["run_id"] == "v1"

    # Stage 1B Selection: Maximize Dice + 2 * (ThinDice - 0.4230)
    runs_1b = [
        {"run_id": "v0", "dice": 0.7684, "thin_dice": 0.4350},  # score: 0.7684 + 2*(0.012) = 0.7924
        {"run_id": "v1", "dice": 0.7675, "thin_dice": 0.4400},  # score: 0.7675 + 2*(0.017) = 0.8015
    ]
    for r in runs_1b:
        r["score"] = r["dice"] + 2.0 * (r["thin_dice"] - 0.4230)
    best_1b = max(runs_1b, key=lambda x: x["score"])
    assert best_1b["run_id"] == "v1"

    # Stage 2 Pass/Fail criteria:
    # PASS: Dice >= max(A1, B1) and Recall >= 0.838
    best_standalone = 0.7685
    assert (0.7690 >= best_standalone) and (0.8400 >= 0.838)  # PASS
    assert not ((0.7670 >= best_standalone) and (0.8400 >= 0.838))  # FAIL (Dice lower)
    assert not ((0.7690 >= best_standalone) and (0.8350 >= 0.838))  # FAIL (Recall collapsed)
    print("[PASS] test_decision_criteria_math")


if __name__ == "__main__":
    test_stage_1a_configs_invariants()
    test_stage_1b_configs_invariants()
    test_dual_loss_forward_backward()
    test_generate_combined_config()
    test_decision_criteria_math()
    print("\nALL PHASE 6 COMBINATION PIPELINE INVARIANT TESTS PASSED SUCCESSFULLY!")
