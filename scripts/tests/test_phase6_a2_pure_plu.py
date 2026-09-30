"""
Preflight Verification Suite for Phase 6-A.2 Pure PLU Representation Probe.

Comprehensive verification of:
1. Shape & Architecture of ProgressiveLearnedUpsamplingHead (+6,408 params, 112->224->448).
2. Stage 1 and Stage 2 Optimizer Parameter Groups (Stage 1: PLU 1e-4, SAGE 2e-4, BB 1e-5; Stage 2: all 1e-4).
3. Loss Probe Isolation (boundary_iou_weight = 0.0 -> boundary_loss is None, pure L_Base).
4. Stage Transition (Stage 1 PLU checkpoint -> Stage 2 strict loading with 0 missing/unexpected).
5. Safeguard: Baseline checkpoint injection raises ValueError.
6. Protocol Identity Test: Config purity, no Candidate B checkpoint reuse, exact hyperparameter locking.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn

# Ensure project root is in sys.path
test_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.abspath(os.path.join(test_dir, '..'))
project_root = os.path.abspath(os.path.join(test_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from sage.networks import create_b2_unet
from sage.networks.decoder_block import ProgressiveLearnedUpsamplingHead
from train_crack import CrackBinaryLoss, get_optimizer_groups, create_stage2_optimizer


def test_plu_architecture_and_shape():
    print("\n" + "=" * 75)
    print("TEST 1: PLU Architecture & Shape Verification (48x112^2 -> 1x448^2)")
    print("=" * 75)

    head = ProgressiveLearnedUpsamplingHead(
        in_channels=48,
        mid_channels=24,
        up_channels=16,
        num_classes=1,
    )

    total_params = sum(p.numel() for p in head.parameters())
    trainable_params = sum(p.numel() for p in head.parameters() if p.requires_grad)

    # Baseline head: Conv2d(48, 24, 3, 1, 1) + BN(24) + Conv2d(24, 1, 1) = 10,392 + 48 + 25 = 10,465
    # PLU head: Conv2d(48, 24, 3, 1, 1) + BN(24) + ConvT(24, 16, 4, 2, 1) + BN(16) + ConvT(16, 1, 4, 2, 1)
    #           = 10,392 + 48 + 6,144 + 32 + 257 = 16,873
    # Delta = 16,873 - 10,465 = exactly +6,408 params
    expected_params = 16873
    assert total_params == expected_params, f"Param mismatch: got {total_params}, expected {expected_params}"
    assert trainable_params == expected_params, "All PLU params must be trainable"
    print(f"  [PASS] PLU Head Parameters: {total_params:,} (Delta vs baseline: +6,408 params)")

    x = torch.randn(2, 48, 112, 112)
    out = head(x)
    assert out.shape == (2, 1, 448, 448), f"Output shape mismatch: got {out.shape}, expected (2, 1, 448, 448)"
    print(f"  [PASS] Forward Pass Output Shape: {tuple(out.shape)} exactly matches (2, 1, 448, 448)")

    loss = out.sum()
    loss.backward()
    for name, param in head.named_parameters():
        assert param.grad is not None, f"Gradient missing for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
    print("  [PASS] Gradient Backpropagation: Clean gradients across all PLU layers (no NaNs)")


def test_optimizer_parameter_groups():
    print("\n" + "=" * 75)
    print("TEST 2: Optimizer Parameter Groups (Stage 1 & Stage 2)")
    print("=" * 75)

    sage_cfg = {"top_k": 2, "router_hidden_dim": 64, "load_balance_factor": 0.01}
    model = create_b2_unet(
        num_classes=1,
        img_size=448,
        num_transformer_layers=4,
        pretrained=False,
        sage_config=sage_cfg,
        p3_mode="C",
        use_plu_head=True,
    )

    # 1. Stage 1 Optimizer Groups
    lr_base = 1e-4
    lr_backbone = lr_base * 0.1  # 1e-5
    lr_decoder = lr_base         # 1e-4
    lr_sage = 2e-4
    lr_p3 = 1e-4

    s1_groups = get_optimizer_groups(
        model,
        lr_backbone=lr_backbone,
        lr_decoder=lr_decoder,
        lr_sage=lr_sage,
        lr_p3=lr_p3,
        weight_decay=0.05,
    )

    # Locate PLU parameters in Stage 1
    plu_names = [
        'decoder.segmentation_head.conv112.weight',
        'decoder.segmentation_head.conv112.bias',
        'decoder.segmentation_head.norm112.weight',
        'decoder.segmentation_head.norm112.bias',
        'decoder.segmentation_head.up224.weight',
        'decoder.segmentation_head.norm224.weight',
        'decoder.segmentation_head.norm224.bias',
        'decoder.segmentation_head.up448.weight',
        'decoder.segmentation_head.up448.bias',
    ]

    for p_name in plu_names:
        param = dict(model.named_parameters())[p_name]
        found = False
        for g in s1_groups:
            for p in g['params']:
                if id(p) == id(param):
                    found = True
                    assert g['lr'] == 1e-4, f"PLU param {p_name} has unexpected Stage 1 LR: {g['lr']}"
                    if 'norm' in p_name or p_name.endswith('.bias'):
                        assert g['weight_decay'] == 0.0, f"PLU norm/bias {p_name} must have WD=0.0, got {g['weight_decay']}"
                    else:
                        assert g['weight_decay'] == 0.05, f"PLU weight {p_name} must have WD=0.05, got {g['weight_decay']}"
        assert found, f"PLU parameter {p_name} not assigned to any Stage 1 optimizer group!"

    print("  [PASS] Stage 1 Optimizer: All PLU parameters assigned to Decoder group (LR=1e-4, WD=0.05/0.0)")

    # 2. Stage 2 Optimizer Groups
    s2_opt = create_stage2_optimizer(
        model,
        stage2_base_lr=1e-4,
        stage2_shared_lr=1e-4,
        stage2_sage_lr=1e-4,
        stage2_p3_lr=1e-4,
        weight_decay=0.05,
    )

    for p_name in plu_names:
        param = dict(model.named_parameters())[p_name]
        found = False
        for g in s2_opt.param_groups:
            for p in g['params']:
                if id(p) == id(param):
                    found = True
                    assert g['lr'] == 1e-4, f"PLU param {p_name} has unexpected Stage 2 LR: {g['lr']}"
                    if 'norm' in p_name or p_name.endswith('.bias'):
                        assert g['weight_decay'] == 0.0, f"PLU norm/bias {p_name} must have WD=0.0, got {g['weight_decay']}"
                    else:
                        assert g['weight_decay'] == 0.05, f"PLU weight {p_name} must have WD=0.05, got {g['weight_decay']}"
        assert found, f"PLU parameter {p_name} not assigned to any Stage 2 optimizer group!"

    print("  [PASS] Stage 2 Optimizer: All PLU parameters assigned to Base/Others group (LR=1e-4, WD=0.05/0.0)")


def test_loss_probe_isolation():
    print("\n" + "=" * 75)
    print("TEST 3: Loss Probe Isolation (Pure L_Base, boundary_weight = 0.0)")
    print("=" * 75)

    criterion = CrackBinaryLoss(
        bce_weight=1.0,
        dice_weight=1.5,
        boundary_weight=0.0,
        boundary_dilation=2,
    )

    assert criterion.boundary_loss is None, "boundary_loss must be None when boundary_weight=0.0"
    print("  [PASS] criterion.boundary_loss is None (BoundaryIoU completely disabled)")

    logits = torch.randn(2, 1, 448, 448, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 448, 448)).float()

    loss = criterion(logits, targets)
    assert not torch.isnan(loss), "Loss cannot be NaN"

    # Compute manual L_Base
    bce = nn.BCEWithLogitsLoss()(logits, targets)
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = 1.0 - ((2.0 * intersection + 1e-5) / (union + 1e-5)).mean()
    expected_loss = 1.0 * bce + 1.5 * dice

    assert torch.allclose(loss, expected_loss, atol=1e-6), "Loss computation deviates from pure L_Base formula"
    print(f"  [PASS] Loss strictly equals L_Base (1.0*BCE + 1.5*Dice): {loss.item():.4f}")


def test_stage_transition_and_strict_loading():
    print("\n" + "=" * 75)
    print("TEST 4: Stage 1 Save -> Stage 2 Strict Loading")
    print("=" * 75)

    sage_cfg = {"top_k": 2, "router_hidden_dim": 64, "load_balance_factor": 0.01}
    model_s1 = create_b2_unet(
        num_classes=1,
        img_size=448,
        num_transformer_layers=4,
        pretrained=False,
        sage_config=sage_cfg,
        p3_mode="C",
        use_plu_head=True,
    )

    # Save mock Stage 1 state_dict
    s1_sd = model_s1.state_dict()

    # Stage 2 model loads Stage 1 state_dict
    model_s2 = create_b2_unet(
        num_classes=1,
        img_size=448,
        num_transformer_layers=4,
        pretrained=False,
        sage_config=sage_cfg,
        p3_mode="C",
        use_plu_head=True,
    )

    missing, unexpected = model_s2.load_stage1_state_dict(s1_sd)
    assert len(missing) == 0, f"Unexpected missing keys: {missing}"
    assert len(unexpected) == 0, f"Unexpected extra keys: {unexpected}"
    print(f"  [PASS] Strict Stage 1 -> Stage 2 checkpoint load: 0 missing, 0 unexpected across {len(s1_sd)} tensors")


def test_baseline_safeguard_rejection():
    print("\n" + "=" * 75)
    print("TEST 5: Safeguard Against Baseline Checkpoint Injection")
    print("=" * 75)

    sage_cfg = {"top_k": 2, "router_hidden_dim": 64, "load_balance_factor": 0.01}
    model = create_b2_unet(
        num_classes=1,
        img_size=448,
        num_transformer_layers=4,
        pretrained=False,
        sage_config=sage_cfg,
        p3_mode="C",
        use_plu_head=True,
    )

    # Create mock legacy baseline state dict containing segmentation_head.0 and segmentation_head.3
    mock_baseline_sd = {
        'decoder.segmentation_head.0.weight': torch.randn(24, 48, 3, 3),
        'decoder.segmentation_head.0.bias': torch.randn(24),
        'decoder.segmentation_head.1.weight': torch.randn(24),
        'decoder.segmentation_head.1.bias': torch.randn(24),
        'decoder.segmentation_head.3.weight': torch.randn(1, 24, 1, 1),
        'decoder.segmentation_head.3.bias': torch.randn(1),
    }

    caught = False
    try:
        model.load_stage1_state_dict(mock_baseline_sd)
    except ValueError as e:
        caught = True
        print(f"  [PASS] Successfully caught and rejected baseline checkpoint: {e}")

    assert caught, "Safeguard failed: B2 PLU model must reject legacy baseline checkpoints!"


def test_protocol_identity():
    print("\n" + "=" * 75)
    print("TEST 6: Protocol Identity Test (Pure PLU Configuration)")
    print("=" * 75)

    config_path = os.path.join(project_root, "configs", "p3_ablation", "b2_p3_run_c_d4_k2_h64_phase6_a2_pure_plu.yaml")
    if not os.path.exists(config_path):
        # Fallback to repo root if path differ
        config_path = os.path.join(test_dir, "..", "..", "..", "results", "configs", "b2_p3_run_c_d4_k2_h64_phase6_a2_pure_plu.yaml")

    assert os.path.exists(config_path), f"Config file not found at {config_path}"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 1. Checkpoint source purity
    assert cfg.get("checkpoint") is None, "A2 Pure PLU must start from scratch: checkpoint must be None"
    print("  [PASS] Startup Checkpoint: None (Trains from scratch, no Candidate B reuse)")

    # 2. PLU flag & Objective isolation
    assert cfg.get("use_plu_head") is True, "use_plu_head must be True"
    assert float(cfg.get("boundary_iou_weight", -1.0)) == 0.0, "boundary_iou_weight must be strictly 0.0"
    print("  [PASS] Architectural & Objective Flags: use_plu_head=True, boundary_iou_weight=0.0")

    # 3. Two-Stage schedule & epochs
    assert cfg.get("two_stage") is True, "two_stage must be True"
    assert cfg.get("epochs") == 35, "Total epochs must be 35"
    assert cfg.get("stage1_epochs") == 17, "Stage 1 epochs must be 17"
    assert cfg.get("warmup_epochs") == 3, "Warmup epochs must be 3"
    assert cfg.get("patience") == 6, "Patience must be 6"
    print("  [PASS] Budget & Schedule: 35 epochs (Stage 1 = 17, Stage 2 = 18, Warmup = 3, Patience = 6)")

    # 4. Learning Rates
    assert float(cfg.get("lr")) == 1e-4, "Base LR must be 1e-4"
    assert float(cfg.get("sage_lr")) == 2e-4, "Stage 1 SAGE LR must be 2e-4"
    assert float(cfg.get("p3_lr")) == 1e-4, "P3 LR must be 1e-4"
    assert float(cfg.get("stage2_base_lr")) == 1e-4, "Stage 2 Base LR must be 1e-4"
    assert float(cfg.get("stage2_shared_lr")) == 1e-4, "Stage 2 Shared LR must be 1e-4"
    print("  [PASS] Learning Rates: Stage 1 (Dec=1e-4, SAGE=2e-4), Stage 2 (Base=1e-4, Shared=1e-4)")

    # 5. Core Architectural Invariants (D4, K2, H64, LB=0.010, P3-C)
    assert cfg.get("model") == "B2", "Model must be B2"
    assert cfg.get("num_transformer_layers") == 4, "ViT depth must be 4 (D4)"
    assert cfg.get("p3_mode") == "C", "P3 mode must be C (ASDW)"
    assert cfg["sage_config"]["top_k"] == 2, "Top-K must be 2 (K2)"
    assert cfg["sage_config"]["router_hidden_dim"] == 64, "Router hidden dim must be 64 (H64)"
    assert float(cfg["sage_config"]["load_balance_factor"]) == 0.010, "LB factor must be 0.010"
    print("  [PASS] Invariant Architecture: D4, K2, H64, LB=0.010, P3-C ASDW fully verified!")


if __name__ == "__main__":
    print("\n" + "=" * 75)
    print("RUNNING ALL PREFLIGHT TESTS FOR PHASE 6-A.2 PURE PLU PROBE")
    print("=" * 75)
    test_plu_architecture_and_shape()
    test_optimizer_parameter_groups()
    test_loss_probe_isolation()
    test_stage_transition_and_strict_loading()
    test_baseline_safeguard_rejection()
    test_protocol_identity()
    print("\n" + "=" * 75)
    print("ALL 6 PREFLIGHT TESTS PASSED CLEANLY! PROTOCOL IS CERTIFIED.")
    print("=" * 75 + "\n")
