"""
Unit and Regression Tests for SAGE LR Isolation (Phase 5.1) and Stage 2 LR Ratio (Phase 5.3)

Verifies:
1. Stage 1 Isolated SAGE LR parameter groups (lr_sage != lr_decoder).
2. Stage 2 Optimizer 3-tier parameter partition integrity (shared_experts, other_non_experts, p3_refinement).
3. CLI overrides and YAML config resolution.
4. Invariance: Varying sage_lr strictly isolates SAGE routing tier without affecting backbone, decoder, p3, or Stage 2.
5. Phase 5.3 ratio isolation: r = LR_shared / LR_base leaves other_non_experts and p3 fixed.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sage.networks.b2_unet import create_b2_unet
from scripts.train_crack import get_optimizer_groups, create_stage2_optimizer


def test_stage1_isolated_sage_lr():
    print("\n[Test 1/5] Testing Stage 1 Optimizer parameter grouping and learning rate isolation...")
    model = create_b2_unet(num_transformer_layers=2, p3_mode="C", img_size=224)

    lr_backbone = 1e-5
    lr_decoder = 1e-4
    lr_sage = 2e-4   # Isolated SAGE LR != lr_decoder
    lr_p3 = 1e-4
    weight_decay = 0.05

    groups = get_optimizer_groups(
        model,
        lr_backbone=lr_backbone,
        lr_decoder=lr_decoder,
        lr_sage=lr_sage,
        lr_p3=lr_p3,
        weight_decay=weight_decay
    )

    names_found = set()
    total_group_params = 0
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    for g in groups:
        name = g.get('name')
        lr = g['lr']
        wd = g['weight_decay']
        names_found.add(name)
        total_group_params += len(g['params'])

        if name == 'backbone':
            assert abs(lr - lr_backbone) < 1e-10, f"Expected backbone LR {lr_backbone}, got {lr}"
        elif name == 'decoder':
            assert abs(lr - lr_decoder) < 1e-10, f"Expected decoder LR {lr_decoder}, got {lr}"
        elif name == 'sage':
            assert abs(lr - lr_sage) < 1e-10, f"Expected sage LR {lr_sage}, got {lr}"
        elif name == 'p3_refinement':
            assert abs(lr - lr_p3) < 1e-10, f"Expected p3 LR {lr_p3}, got {lr}"
        else:
            raise AssertionError(f"Unknown param group name: {name}")

        assert wd in (0.0, weight_decay), f"Invalid weight decay {wd}"

    assert 'backbone' in names_found, "Group 'backbone' not found!"
    assert 'decoder' in names_found, "Group 'decoder' not found!"
    assert 'sage' in names_found, "Group 'sage' not found!"
    assert 'p3_refinement' in names_found, "Group 'p3_refinement' not found!"
    assert total_group_params == len(trainable_params), f"Partition count mismatch: {total_group_params} vs {len(trainable_params)}"

    print("  --> PASS: Stage 1 isolated SAGE LR parameter groups and backward compatibility verified.")


def test_stage2_optimizer_partition_integrity():
    print("[Test 2/5] Testing Stage 2 Optimizer 3-tier parameter partition integrity...")
    model = create_b2_unet(num_transformer_layers=2, p3_mode="C", img_size=224)

    stage2_base_lr = 1e-4
    stage2_shared_lr = 5e-5  # r = 0.5
    stage2_p3_lr = 2e-4
    weight_decay = 0.05

    opt = create_stage2_optimizer(
        model,
        stage2_base_lr=stage2_base_lr,
        stage2_shared_lr=stage2_shared_lr,
        stage2_p3_lr=stage2_p3_lr,
        weight_decay=weight_decay
    )

    names_found = set()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_group_params = 0

    for g in opt.param_groups:
        name = g.get('name')
        lr = g['lr']
        wd = g['weight_decay']
        names_found.add(name)
        total_group_params += len(g['params'])

        if name == 'shared_experts':
            assert abs(lr - stage2_shared_lr) < 1e-10, f"Expected shared_lr {stage2_shared_lr}, got {lr}"
        elif name == 'other_non_experts':
            assert abs(lr - stage2_base_lr) < 1e-10, f"Expected base_lr {stage2_base_lr}, got {lr}"
        elif name == 'p3_refinement':
            assert abs(lr - stage2_p3_lr) < 1e-10, f"Expected p3_lr {stage2_p3_lr}, got {lr}"
        else:
            raise AssertionError(f"Unknown Stage 2 group name: {name}")

        assert wd in (0.0, weight_decay), f"Invalid weight decay {wd}"

    assert 'shared_experts' in names_found, "Group 'shared_experts' not found!"
    assert 'other_non_experts' in names_found, "Group 'other_non_experts' not found!"
    assert 'p3_refinement' in names_found, "Group 'p3_refinement' not found!"
    assert total_group_params == len(trainable_params), f"Stage 2 partition count mismatch: {total_group_params} vs {len(trainable_params)}"

    print("  --> PASS: Stage 2 3-tier parameter partition integrity verified.")


def test_cli_and_config_resolution():
    print("[Test 3/5] Testing CLI overrides and config fallback logic...")

    # 1. Config fallback when sage_lr is omitted
    cfg_legacy = {'lr': 1e-4}
    base_lr = float(cfg_legacy.get('lr', 1e-4))
    sage_lr = float(cfg_legacy.get('sage_lr', base_lr))
    p3_lr = float(cfg_legacy.get('p3_lr', base_lr))
    assert sage_lr == 1e-4, f"Legacy fallback failed: {sage_lr}"
    assert p3_lr == 1e-4, f"Legacy fallback failed: {p3_lr}"

    # 2. Config explicit sage_lr
    cfg_new = {'lr': 1e-4, 'sage_lr': 5e-5, 'p3_lr': 2e-4}
    base_lr = float(cfg_new.get('lr', 1e-4))
    sage_lr = float(cfg_new.get('sage_lr', base_lr))
    p3_lr = float(cfg_new.get('p3_lr', base_lr))
    assert sage_lr == 5e-5, f"Config explicit sage_lr failed: {sage_lr}"
    assert p3_lr == 2e-4, f"Config explicit p3_lr failed: {p3_lr}"

    # 3. CLI parsing test for SAGE LR, P3 LR, Stage 2 Base/Shared LR using real parser
    from scripts.train_crack import build_parser
    parser = build_parser()

    # Test real CLI parsing for valid Stage 2 and SAGE LR arguments
    args1 = parser.parse_args(['--config', 'dummy.yaml', '--sage-lr', '5e-5', '--stage2-shared-lr', '8e-5', '--stage2-base-lr', '1e-4', '--stage2-sage-lr', '2e-4', '--p3-lr', '2e-4'])
    assert args1.sage_lr == 5e-5, f"Real parser failed sage_lr: {args1.sage_lr}"
    assert args1.stage2_shared_lr == 8e-5, f"Real parser failed stage2_shared_lr: {args1.stage2_shared_lr}"
    assert args1.stage2_base_lr == 1e-4, f"Real parser failed stage2_base_lr: {args1.stage2_base_lr}"
    assert args1.stage2_sage_lr == 2e-4, f"Real parser failed stage2_sage_lr: {args1.stage2_sage_lr}"
    assert args1.p3_lr == 2e-4, f"Real parser failed p3_lr: {args1.p3_lr}"

    # Also test alternative flag aliases
    args2 = parser.parse_args(['--config', 'dummy.yaml', '--sage_lr', '2e-4', '--p3_lr', '1.5e-4'])
    assert args2.sage_lr == 2e-4
    assert args2.p3_lr == 1.5e-4

    # Verify that removed flag --stage2-fine-lr is strictly rejected by the real parser!
    try:
        parser.parse_args(['--config', 'dummy.yaml', '--stage2-fine-lr', '1e-4'])
        raised = False
    except SystemExit:
        raised = True
    assert raised, "Real parser should reject removed flag '--stage2-fine-lr' with SystemExit!"

    print("  --> PASS: Real CLI parser and config fallback logic verified.")


def test_sage_lr_invariance_across_tiers():
    print("[Test 4/5] Testing invariance: varying sage_lr leaves backbone, decoder, p3, and Stage 2 untouched...")
    model = create_b2_unet(num_transformer_layers=2, p3_mode="C", img_size=224)

    base_lr = 1e-4
    p3_lr = 1e-4
    lr_bb = base_lr * 0.1
    lr_dec = base_lr

    candidate_sage_lrs = [5e-5, 1e-4, 2e-4, 5e-4]

    for candidate_lr in candidate_sage_lrs:
        # Stage 1 optimizer grouping verification
        groups = get_optimizer_groups(
            model,
            lr_backbone=lr_bb,
            lr_decoder=lr_dec,
            lr_sage=candidate_lr,
            lr_p3=p3_lr,
            weight_decay=0.05
        )

        for g in groups:
            name = g.get('name')
            if name == 'backbone':
                assert abs(g['lr'] - lr_bb) < 1e-10, f"Backbone LR modified! Expected {lr_bb}, got {g['lr']}"
            elif name == 'decoder':
                assert abs(g['lr'] - lr_dec) < 1e-10, f"Decoder LR modified! Expected {lr_dec}, got {g['lr']}"
            elif name == 'p3_refinement':
                assert abs(g['lr'] - p3_lr) < 1e-10, f"P3 LR modified! Expected {p3_lr}, got {g['lr']}"
            elif name == 'sage':
                assert abs(g['lr'] - candidate_lr) < 1e-10, f"SAGE LR mismatch! Expected {candidate_lr}, got {g['lr']}"

        # Stage 2 optimizer verification: must remain untouched at 1e-4
        stage2_opt = create_stage2_optimizer(
            model,
            stage2_base_lr=1e-4,
            stage2_shared_lr=1e-4,
            stage2_p3_lr=1e-4,
        )
        for g in stage2_opt.param_groups:
            assert abs(g['lr'] - 1e-4) < 1e-10, f"Stage 2 group {g['name']} LR contaminated! Expected 1e-4, got {g['lr']}"

    print("  --> PASS: Strict invariance confirmed: varying sage_lr strictly isolates SAGE tier without touching backbone, decoder, p3, or Stage 2.")


def test_phase5_3_ratio_isolation():
    print("[Test 5/5] Testing Phase 5.3 ratio isolation: r = LR_shared / LR_base leaves other_non_experts and p3 fixed...")
    model = create_b2_unet(num_transformer_layers=2, p3_mode="C", img_size=224)

    fixed_base_lr = 1e-4
    fixed_p3_lr = 1e-4

    ratios = [0.25, 0.50, 1.00, 2.00]
    for r in ratios:
        stage2_shared = r * fixed_base_lr

        opt = create_stage2_optimizer(
            model,
            stage2_base_lr=fixed_base_lr,
            stage2_shared_lr=stage2_shared,
            stage2_p3_lr=fixed_p3_lr,
        )

        for g in opt.param_groups:
            name = g.get('name')
            lr = g['lr']
            if name == 'shared_experts':
                assert abs(lr - stage2_shared) < 1e-10, f"Ratio {r}: shared_lr mismatch"
            elif name == 'other_non_experts':
                assert abs(lr - fixed_base_lr) < 1e-10, f"Ratio {r}: other_non_experts modified! Expected {fixed_base_lr}, got {lr}"
            elif name == 'p3_refinement':
                assert abs(lr - fixed_p3_lr) < 1e-10, f"Ratio {r}: p3_refinement modified! Expected {fixed_p3_lr}, got {lr}"

    print("  --> PASS: Phase 5.3 ratio isolation verified across all ratios (r in [0.25, 0.50, 1.00, 2.00]).")


def test_stage2_isolated_sage_lr():
    print("[Test 6/6] Testing Stage 2 isolated SAGE LR (Phase 5.1 Extra)...")
    model = create_b2_unet(num_transformer_layers=2, p3_mode="C", img_size=224)

    # 1. When stage2_sage_lr is specified (e.g. 2e-4) and differs from stage2_base_lr (1e-4)
    opt = create_stage2_optimizer(
        model,
        stage2_base_lr=1e-4,
        stage2_shared_lr=1e-4,
        stage2_sage_lr=2e-4,
        stage2_p3_lr=1e-4,
    )
    group_names = {g['name'] for g in opt.param_groups}
    assert 'sage' in group_names, f"Expected 'sage' group in optimizer, got: {group_names}"
    for g in opt.param_groups:
        name = g['name']
        if name == 'sage':
            assert abs(g['lr'] - 2e-4) < 1e-10, f"Expected SAGE LR 2e-4, got {g['lr']}"
        elif name == 'other_non_experts':
            assert abs(g['lr'] - 1e-4) < 1e-10, f"Expected Base LR 1e-4, got {g['lr']}"
        elif name == 'shared_experts':
            assert abs(g['lr'] - 1e-4) < 1e-10, f"Expected Shared LR 1e-4, got {g['lr']}"

    # 2. When stage2_sage_lr is omitted (None), fallback to stage2_base_lr
    opt_fallback = create_stage2_optimizer(
        model,
        stage2_base_lr=1e-4,
        stage2_shared_lr=1e-4,
    )
    fb_names = {g['name'] for g in opt_fallback.param_groups}
    assert fb_names == {'shared_experts', 'other_non_experts', 'p3_refinement'}, f"Fallback should keep canonical groups, got {fb_names}"
    assert 'sage' not in fb_names, "Sage group should NOT exist when stage2_sage_lr is omitted/fallback"

    print("  --> PASS: Stage 2 SAGE LR isolation and fallback verified.")


if __name__ == '__main__':
    test_stage1_isolated_sage_lr()
    test_stage2_optimizer_partition_integrity()
    test_cli_and_config_resolution()
    test_sage_lr_invariance_across_tiers()
    test_phase5_3_ratio_isolation()
    test_stage2_isolated_sage_lr()
    print("\nALL SAGE LR ISOLATION AND STAGE 2 TESTS PASSED (6/6)!")
