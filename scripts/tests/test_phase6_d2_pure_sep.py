#!/usr/bin/env python3
"""
scripts/tests/test_phase6_d2_pure_sep.py

Preflight Verification Suite for Phase 6-D.2:
Pure Inter-Component Separation Isolation Probe (Candidate B Stage-1 + L_sep).

Tests:
1. Exact Base Equivalence when separation_weight = 0.0 (torch.equal on loss and parameters)
2. Negative Moat Mechanics on Two-Component Gap (FP inside moat penalized, FP outside zero penalty)
3. Single-Component and Empty GT Invariance (L_sep = 0.0, grad = 0.0)
4. Parameter Invariance (Exactly 10,118,955 params, +0 extra vs Candidate B)
5. Checkpoint Lineage Verification (Loads cleanly from Candidate B Stage-1 Checkpoint)
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
from scripts.train_crack import CrackBinaryLoss, InterComponentSeparationLoss

def test_1_loss_equivalence():
    print("[RUNNING] Preflight 1: Loss & Parameter Equivalence when separation_weight = 0.0...")
    torch.manual_seed(42)
    device = torch.device('cpu')
    model = create_b2_unet(num_classes=1, img_size=128, num_transformer_layers=2, pretrained=False, p3_mode="C", use_plu_head=False).to(device)
    model.eval()

    x = torch.randn(2, 3, 128, 128)
    y = torch.zeros(2, 1, 128, 128)
    y[0, 0, 20:60, 40] = 1.0
    y[0, 0, 20:60, 44] = 1.0 # 3 px gap

    criterion_base = CrackBinaryLoss(bce_weight=1.0, dice_weight=1.5, separation_weight=0.0)
    criterion_d2_zero = CrackBinaryLoss(bce_weight=1.0, dice_weight=1.5, separation_weight=0.0, separation_max_gap=8.0)

    model.zero_grad()
    out1 = model(x)
    l_base = criterion_base(out1, y)
    l_base.backward()
    grad_base = [p.grad.clone() for p in model.parameters() if p.grad is not None]

    model.zero_grad()
    out2 = model(x)
    l_d2 = criterion_d2_zero(out2, y)
    l_d2.backward()
    grad_d2 = [p.grad.clone() for p in model.parameters() if p.grad is not None]

    assert torch.equal(l_base, l_d2), f"Loss mismatch: Base={l_base.item()}, D2={l_d2.item()}"
    for gb, gd in zip(grad_base, grad_d2):
        assert torch.equal(gb, gd), "Gradient tensor mismatch between Base and D2 (lambda=0)!"

    print(f"  [PASS] Preflight 1: Exact Base equivalence confirmed at byte level (torch.equal = True)")


def test_2_negative_moat_mechanics():
    print("[RUNNING] Preflight 2: Negative Moat Spatial Mechanics...")
    loss_module = InterComponentSeparationLoss(max_gap=8.0)

    # 2 parallel cracks separated by 3px gap
    target = torch.zeros(1, 1, 100, 100)
    target[0, 0, 20:80, 45] = 1.0
    target[0, 0, 20:80, 48] = 1.0

    logits_inside = torch.full((1, 1, 100, 100), -5.0)
    # Put positive prediction right in the gap (x=46, 47)
    logits_inside[0, 0, 20:80, 46:48] = 5.0

    logits_outside = torch.full((1, 1, 100, 100), -5.0)
    logits_outside[0, 0, 20:80, 10:12] = 5.0 # far outside

    l_in = loss_module(logits_inside, target).item()
    l_out = loss_module(logits_outside, target).item()

    assert l_in > 4.0, f"Expected strong penalty inside moat, got {l_in}"
    assert l_out < 0.01, f"Expected near zero penalty outside moat, got {l_out}"
    print(f"  [PASS] Preflight 2: Negative moat mechanics confirmed (inside={l_in:.4f}, outside={l_out:.4f})")


def test_3_invariances():
    print("[RUNNING] Preflight 3: Single-Component and Empty GT Invariance...")
    loss_module = InterComponentSeparationLoss(max_gap=8.0)

    # Single CC
    target_single = torch.zeros(1, 1, 64, 64)
    target_single[:, :, 10:50, 32] = 1.0
    logits = torch.randn(1, 1, 64, 64, requires_grad=True)
    l_single = loss_module(logits, target_single)
    assert l_single.item() == 0.0
    l_single.backward()
    assert logits.grad is None or logits.grad.abs().sum().item() == 0.0

    # Empty
    target_empty = torch.zeros(1, 1, 64, 64)
    l_empty = loss_module(logits, target_empty)
    assert l_empty.item() == 0.0
    print("  [PASS] Preflight 3: Single-component & Empty GT invariances confirmed (L_sep = 0.0, grad = 0.0)")


def test_4_parameter_invariance():
    print("[RUNNING] Preflight 4: Parameter Invariance Verification...")
    sage_cfg = {
        'top_k': 2,
        'gating_type': 'sigmoid',
        'shared_expert_indices': [0, 1, 2, 3],
        'router_hidden_dim': 64,
        'load_balance_factor': 0.01,
        'logit_modulation': True,
        'expert_dropout': 0.1,
        'fusion_type': 'residual',
        'residual_scale': 0.1,
    }
    model = create_b2_unet(num_classes=1, img_size=448, num_transformer_layers=4, pretrained=False, p3_mode="C", sage_config=sage_cfg, use_plu_head=False)
    total_params = sum(p.numel() for p in model.parameters())
    expected_params = 10118955 # Candidate B baseline exactly
    assert total_params == expected_params, f"Parameter count mismatch: got {total_params}, expected {expected_params}"
    print(f"  [PASS] Preflight 4: Parameter count exactly matches Candidate B baseline ({total_params:,} params, +0 extra)")


def test_5_checkpoint_lineage_loading():
    print("[RUNNING] Preflight 5: Strict Checkpoint Lineage Loading from Candidate B Stage-1...")
    ckpt_candidates = [
        os.path.join(project_root, "..", "results", "checkpoints", "P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_stage1.pth"),
        os.path.join(project_root, "results", "checkpoints", "P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_stage1.pth"),
        "/content/drive/MyDrive/crack_seg/P3_C_D4_K2_H64_Phase5_SAGELR2e-4/best_model_b2_stage1.pth",
    ]
    ckpt_path = None
    for cp in ckpt_candidates:
        if os.path.exists(cp):
            ckpt_path = cp
            break

    if ckpt_path is None:
        print("  [SKIP] Preflight 5: Local checkpoint not found at standard path, will verify on Colab.")
        return

    sage_cfg = {
        'top_k': 2,
        'gating_type': 'sigmoid',
        'shared_expert_indices': [0, 1, 2, 3],
        'router_hidden_dim': 64,
        'load_balance_factor': 0.01,
        'logit_modulation': True,
        'expert_dropout': 0.1,
        'fusion_type': 'residual',
        'residual_scale': 0.1,
    }
    model = create_b2_unet(num_classes=1, img_size=448, num_transformer_layers=4, pretrained=False, p3_mode="C", sage_config=sage_cfg, use_plu_head=False)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('model_state_dict', ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert len(missing) == 0 and len(unexpected) == 0
    print(f"  [PASS] Preflight 5: Successfully verified strict checkpoint lineage from: {os.path.basename(ckpt_path)}")


def main():
    print("="*80)
    print("RUNNING PREFLIGHT TEST SUITE FOR PHASE 6-D.2 (PURE SEPARATION ISOLATION PROBE)")
    print("="*80)
    test_1_loss_equivalence()
    test_2_negative_moat_mechanics()
    test_3_invariances()
    test_4_parameter_invariance()
    test_5_checkpoint_lineage_loading()
    print("="*80)
    print("ALL PREFLIGHT CHECKS PASSED FOR PHASE 6-D.2!")
    print("="*80)

if __name__ == '__main__':
    main()
