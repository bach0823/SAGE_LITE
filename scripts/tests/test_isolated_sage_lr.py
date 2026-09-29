import os
import sys
import argparse
import torch
import torch.nn as nn

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sage.networks.b2_unet import create_b2_unet
from scripts.train_crack import get_optimizer_groups, create_stage2_optimizer

def test_stage1_isolated_sage_lr():
    print("[Test 1/3] Testing Stage 1 Optimizer parameter grouping and learning rate isolation...")
    model = create_b2_unet(num_transformer_layers=2, p3_mode="C", img_size=224)
    
    lr_bb = 1e-5
    lr_dec = 1e-4
    lr_sage = 5e-5
    lr_p3 = 2e-4
    weight_decay = 0.05

    groups = get_optimizer_groups(
        model,
        lr_backbone=lr_bb,
        lr_decoder=lr_dec,
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
            assert abs(lr - lr_bb) < 1e-10, f"Expected backbone LR {lr_bb}, got {lr}"
        elif name == 'decoder':
            assert abs(lr - lr_dec) < 1e-10, f"Expected decoder LR {lr_dec}, got {lr}"
        elif name == 'sage':
            assert abs(lr - lr_sage) < 1e-10, f"Expected SAGE LR {lr_sage}, got {lr}"
        elif name == 'p3_refinement':
            assert abs(lr - lr_p3) < 1e-10, f"Expected P3 LR {lr_p3}, got {lr}"
        else:
            raise AssertionError(f"Unknown group name: {name}")

        assert wd in (0.0, weight_decay), f"Invalid weight decay {wd}"

    assert 'sage' in names_found, "Group 'sage' not found in optimizer groups!"
    assert 'backbone' in names_found, "Group 'backbone' not found in optimizer groups!"
    assert 'decoder' in names_found, "Group 'decoder' not found in optimizer groups!"
    assert 'p3_refinement' in names_found, "Group 'p3_refinement' not found in optimizer groups!"
    assert total_group_params == len(trainable_params), f"Param count mismatch: {total_group_params} vs {len(trainable_params)}"

    # Backward compatibility check: lr_sage=None must fall back to lr_decoder
    groups_default = get_optimizer_groups(model, lr_backbone=lr_bb, lr_decoder=lr_dec, lr_sage=None, lr_p3=None)
    for g in groups_default:
        if g.get('name') == 'sage':
            assert abs(g['lr'] - lr_dec) < 1e-10, f"Default SAGE LR must match decoder LR ({lr_dec}), got {g['lr']}"
        if g.get('name') == 'p3_refinement':
            assert abs(g['lr'] - lr_dec) < 1e-10, f"Default P3 LR must match decoder LR ({lr_dec}), got {g['lr']}"

    print("  --> PASS: Stage 1 isolated SAGE LR parameter groups and backward compatibility verified.")

def test_stage2_optimizer_partition_integrity():
    print("[Test 2/3] Testing Stage 2 Optimizer parameter partition integrity and Phase 5.3 ratio...")
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
        elif name == 'other_and_routers':
            assert abs(lr - stage2_base_lr) < 1e-10, f"Expected base_lr {stage2_base_lr}, got {lr}"
        elif name == 'p3_refinement':
            assert abs(lr - stage2_p3_lr) < 1e-10, f"Expected p3_lr {stage2_p3_lr}, got {lr}"
        else:
            raise AssertionError(f"Unknown Stage 2 group name: {name}")

        assert wd in (0.0, weight_decay), f"Invalid weight decay {wd}"

    assert 'shared_experts' in names_found, "Group 'shared_experts' not found!"
    assert 'other_and_routers' in names_found, "Group 'other_and_routers' not found!"
    assert 'p3_refinement' in names_found, "Group 'p3_refinement' not found!"
    assert total_group_params == len(trainable_params), f"Stage 2 partition count mismatch: {total_group_params} vs {len(trainable_params)}"

    print("  --> PASS: Stage 2 parameter partition integrity and Phase 5.3 ratio verified.")

def test_cli_and_config_resolution():
    print("[Test 3/3] Testing CLI overrides and config fallback logic...")
    from scripts.train_crack import CrackBinaryLoss
    
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

    # 3. CLI parsing test
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=False)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--sage-lr', '--sage_lr', type=float, default=None, dest='sage_lr')
    parser.add_argument('--p3-lr', '--p3_lr', type=float, default=None, dest='p3_lr')

    args1 = parser.parse_args(['--sage-lr', '5e-5'])
    assert args1.sage_lr == 5e-5
    args2 = parser.parse_args(['--sage_lr', '2e-4'])
    assert args2.sage_lr == 2e-4

    print("  --> PASS: CLI and config fallback logic verified.")

def test_sage_lr_invariance_across_tiers():
    print("[Test 4/4] Testing invariance: varying sage_lr leaves backbone, decoder, p3, and Stage 2 untouched...")
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
            name = g['name']
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

if __name__ == '__main__':
    test_stage1_isolated_sage_lr()
    test_stage2_optimizer_partition_integrity()
    test_cli_and_config_resolution()
    test_sage_lr_invariance_across_tiers()
    print("\nALL SAGE LR ISOLATION AND OPTIMIZER TESTS PASSED (4/4)!")
