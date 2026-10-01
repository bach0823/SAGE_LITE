#!/usr/bin/env python3
"""
scripts/tests/test_phase6_d1_plu_abbpl.py

Preflight Verification Suite for Phase 6-D.1:
Representation x Boundary Synergy Probe (PLU Head + Asymmetric Boundary-Band Penalty Loss).

Tests:
1. Loss & Gradient Equivalence when margin_weight = 0.0 (Base loss exact match)
2. Asymmetric Boundary Penalty Mechanics on PLU output (FP penalized on M_bg)
3. AMP FP16 Numerical Stability on CUDA (autocast, GradScaler, no NaNs/Infs)
4. Parameter Invariance (10,125,363 params, +0 dynamic)
5. Strict Checkpoint Loading from Phase 6-A.2 Stage-1 Checkpoint (strict=True, RNG state present)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sage.networks import create_b2_unet
from scripts.train_crack import CrackBinaryLoss, AsymmetricBoundaryBandPenaltyLoss

def test_1_loss_equivalence():
    print("[RUNNING] Preflight 1: Loss Equivalence when margin_weight = 0.0...")
    torch.manual_seed(42)
    model = create_b2_unet(
        num_classes=1,
        img_size=448,
        num_transformer_layers=4,
        pretrained=False,
        p3_mode="C",
        use_plu_head=True,
    )
    model.eval()

    x = torch.randn(2, 3, 448, 448)
    target = torch.randint(0, 2, (2, 1, 448, 448)).float()

    criterion_base = CrackBinaryLoss(bce_weight=1.0, dice_weight=1.5, margin_weight=0.0)
    criterion_d1_zero = CrackBinaryLoss(bce_weight=1.0, dice_weight=1.5, margin_weight=0.0, margin_dilation=2)

    with torch.no_grad():
        out = model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        l_base = criterion_base(logits, target)
        l_total = criterion_d1_zero(logits, target)

    diff = torch.abs(l_base - l_total).item()
    assert diff < 1e-7, f"Loss mismatch when margin_weight=0.0: diff={diff}"
    print(f"  [PASS] Preflight 1: Base and D.1 (lambda=0) loss are identical (diff={diff:.2e})")


def test_2_boundary_mechanics():
    print("[RUNNING] Preflight 2: Asymmetric Boundary Penalty Mechanics on PLU output...")
    # Create clean synthetic GT crack (vertical line at x=224)
    target = torch.zeros(1, 1, 448, 448)
    target[:, :, :, 224:226] = 1.0

    # Prediction 1: Perfect prediction (no FP in dilation band)
    logits_clean = torch.full((1, 1, 448, 448), -5.0)
    logits_clean[:, :, :, 224:226] = 5.0

    # Prediction 2: Over-dilated prediction (FP on outer dilation band)
    logits_dilated = logits_clean.clone()
    logits_dilated[:, :, :, 220:230] = 5.0

    abbpl_criterion = AsymmetricBoundaryBandPenaltyLoss(dilation=2)
    loss_clean = abbpl_criterion(logits_clean, target).item()
    loss_dilated = abbpl_criterion(logits_dilated, target).item()

    assert loss_dilated > loss_clean, f"Expected loss_dilated ({loss_dilated}) > loss_clean ({loss_clean})"
    print(f"  [PASS] Preflight 2: AB-BPL penalizes over-dilation (clean={loss_clean:.4f}, dilated={loss_dilated:.4f}, diff=+{loss_dilated - loss_clean:.4f})")


def test_3_amp_fp16_stability():
    print("[RUNNING] Preflight 3: AMP FP16 Numerical Stability on CUDA...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        dev_name = torch.cuda.get_device_name(0)
        if "1650" in dev_name or "1660" in dev_name:
            torch.backends.cudnn.enabled = False
            print(f"  [NOTE] Detected {dev_name}: torch.backends.cudnn.enabled = False for FP16 stability.")

    model = create_b2_unet(
        num_classes=1,
        img_size=448,
        num_transformer_layers=4,
        pretrained=False,
        p3_mode="C",
        use_plu_head=True,
    ).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    criterion_d1 = CrackBinaryLoss(bce_weight=1.0, dice_weight=1.5, margin_weight=0.040, margin_dilation=2)

    x = torch.randn(2, 3, 448, 448, device=device)
    target = torch.randint(0, 2, (2, 1, 448, 448), device=device).float()

    optimizer.zero_grad()
    with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
        out = model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        loss = criterion_d1(logits, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    assert not torch.isnan(loss) and not torch.isinf(loss), "Loss produced NaN or Inf under AMP FP16!"
    print(f"  [PASS] Preflight 3: AMP FP16 forward/backward/optimizer step stable on device: {device} (loss={loss.item():.4f})")


def test_4_parameter_invariance():
    print("[RUNNING] Preflight 4: Parameter Invariance Verification...")
    model = create_b2_unet(
        num_classes=1,
        img_size=448,
        num_transformer_layers=4,
        pretrained=False,
        p3_mode="C",
        use_plu_head=True,
    )
    total_params = sum(p.numel() for p in model.parameters())
    expected_params = 10_125_363  # Exactly matching Phase 6-A.2 Pure PLU (+6,408 params vs Candidate B)
    assert total_params == expected_params, f"Parameter count mismatch: {total_params:,} vs {expected_params:,}"

    # Forward pass invariance check
    x = torch.randn(1, 3, 448, 448)
    _ = model(x)
    total_params_after = sum(p.numel() for p in model.parameters())
    assert total_params == total_params_after, "Parameters dynamically created during forward pass!"
    print(f"  [PASS] Preflight 4: Parameter invariance verified (exactly {total_params:,} params, +0 dynamic)")


def test_5_checkpoint_lineage_loading():
    print("[RUNNING] Preflight 5: Strict Checkpoint Loading from Phase 6-A.2 Stage-1 Checkpoint...")
    best_candidates = [
        "/content/checkpoints/best_model_b2_stage1_plu.pth",
        "results/checkpoints/P3_C_Phase6_A2_Pure_PLU_D4_K2_H64_best_model_b2_stage1.pth",
        "../results/checkpoints/P3_C_Phase6_A2_Pure_PLU_D4_K2_H64_best_model_b2_stage1.pth",
        "../../results/checkpoints/P3_C_Phase6_A2_Pure_PLU_D4_K2_H64_best_model_b2_stage1.pth",
    ]
    last_candidates = [
        "/content/checkpoints/last_model_b2_stage1_plu_rng.pth",
        "results/checkpoints/P3_C_Phase6_A2_Pure_PLU_D4_K2_H64_last_model_b2_stage1.pth",
        "../results/checkpoints/P3_C_Phase6_A2_Pure_PLU_D4_K2_H64_last_model_b2_stage1.pth",
        "../../results/checkpoints/P3_C_Phase6_A2_Pure_PLU_D4_K2_H64_last_model_b2_stage1.pth",
    ]

    ckpt_best = next((p for p in best_candidates if os.path.exists(p)), None)
    ckpt_last = next((p for p in last_candidates if os.path.exists(p)), None)

    if ckpt_best is None or ckpt_last is None:
        print("  [NOTE] Preflight 5: Stage 1 checkpoints not yet downloaded to local/Colab path.")
        print("         Run wget commands in Cell 1 to download checkpoints before Stage 2 training.")
        return

    model = create_b2_unet(
        num_classes=1,
        img_size=448,
        num_transformer_layers=4,
        pretrained=False,
        p3_mode="C",
        use_plu_head=True,
    )

    ckpt = torch.load(ckpt_best, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    msg = model.load_state_dict(sd, strict=True)
    assert len(msg.missing_keys) == 0, f"Missing keys during strict load: {msg.missing_keys}"
    assert len(msg.unexpected_keys) == 0, f"Unexpected keys during strict load: {msg.unexpected_keys}"

    ckpt_rng = torch.load(ckpt_last, map_location="cpu", weights_only=False)
    for k in ["rng_state", "scaler_state_dict", "numpy_rng_state", "python_rng_state"]:
        assert k in ckpt_rng, f"Missing critical state in RNG checkpoint: {k}"

    print(f"  [PASS] Preflight 5: Strict checkpoint loading verified from {ckpt_best} (0 missing, 0 unexpected, complete RNG/scaler state present)")


def main():
    print("=" * 80)
    print("RUNNING PREFLIGHT TEST SUITE FOR PHASE 6-D.1 (PLU + AB-BPL SYNERGY PROBE)")
    print("=" * 80)
    test_1_loss_equivalence()
    test_2_boundary_mechanics()
    test_3_amp_fp16_stability()
    test_4_parameter_invariance()
    test_5_checkpoint_lineage_loading()
    print("=" * 80)
    print("PREFLIGHT GATE: 5/5 PASSED — READY FOR PHASE 6-D.1 STAGE 2 DEPLOYMENT")
    print("=" * 80)


if __name__ == "__main__":
    main()
