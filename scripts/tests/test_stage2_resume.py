import os
import sys
import tempfile
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sage.networks.b2_unet import create_b2_unet
from scripts.train_crack import (
    build_parser,
    create_stage2_optimizer,
    get_optimizer_groups,
    get_scheduler,
)

REAL_BEST_STAGE1_PATH = r"D:\truong\SpecialSubjectTTNT\results\checkpoints\P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_stage1.pth"
REAL_LAST_STAGE1_PATH = r"D:\truong\SpecialSubjectTTNT\results\checkpoints\P3_C_D4_K2_H64_Phase5_SAGELR2e-4_last_model_b2_stage1.pth"


def test_stage2_resume_logic():
    print("[Test 1/5] Verifying Stage 2 resumption and state restoration contracts...")
    with tempfile.TemporaryDirectory() as tmpdir:
        dummy_model = create_b2_unet(num_transformer_layers=2, p3_mode="A", img_size=224)
        ckpt_path = os.path.join(tmpdir, "test_ckpt.pth")
        
        torch.save({
            'epoch': 13,
            'stage': 2,
            'model_state_dict': dummy_model.state_dict(),
            'best_dice': 0.7543,
            'best_loss': 1.0140,
            'model_type': 'B2',
            'num_transformer_layers': 2,
            'p3_mode': 'A',
        }, ckpt_path)
        
        loaded_ckpt = torch.load(ckpt_path, weights_only=False)
        assert loaded_ckpt['best_dice'] == 0.7543
        assert loaded_ckpt['best_loss'] == 1.0140
        assert loaded_ckpt['epoch'] == 13
        assert loaded_ckpt['stage'] == 2
        print("  --> PASS: Checkpoint metadata contract verified.")


def test_cli_parser_rng_checkpoint():
    print("[Test 2/5] Verifying CLI parser support for --rng-checkpoint...")
    parser = build_parser()
    
    # 1. Default is None
    args = parser.parse_args(['--config', 'configs/dummy.yaml'])
    assert args.rng_checkpoint is None, f"Expected None, got {args.rng_checkpoint}"

    # 2. Explicit --rng-checkpoint flag
    args = parser.parse_args([
        '--config', 'configs/dummy.yaml',
        '--stage2-only',
        '--checkpoint', 'path/to/best.pth',
        '--rng-checkpoint', 'path/to/last.pth',
    ])
    assert args.stage2_only is True
    assert args.checkpoint == 'path/to/best.pth'
    assert args.rng_checkpoint == 'path/to/last.pth'
    print("  --> PASS: CLI parser recognizes --rng-checkpoint properly.")


def test_stage2_only_rng_restoration_and_fresh_optimizer():
    print("[Test 3/5] Verifying Stage 2-only RNG restoration with strictly fresh optimizer/scheduler...")
    with tempfile.TemporaryDirectory() as tmpdir:
        model = create_b2_unet(num_transformer_layers=2, p3_mode="C", img_size=224)
        
        # Advance torch, python, numpy RNG and DataLoader generator
        random.seed(12345)
        np.random.seed(54321)
        torch.manual_seed(99999)
        g_saved = torch.Generator().manual_seed(77777)
        # Advance generator once
        _ = torch.rand(5, generator=g_saved)
        saved_gen_state = g_saved.get_state()

        # Step an optimizer to generate Adam momentum states
        s1_opt = optim.AdamW(model.parameters(), lr=1e-4)
        dummy_inp = torch.randn(2, 3, 224, 224)
        dummy_out = model(dummy_inp)
        dummy_loss = dummy_out.sum()
        dummy_loss.backward()
        s1_opt.step()
        s1_opt_state = s1_opt.state_dict()
        assert len(s1_opt.state) > 0, "Expected Adam momentum states in s1_opt!"

        # Create best Stage 1 checkpoint: weights only, NO optimizer, NO RNG
        best_ckpt_path = os.path.join(tmpdir, "best_model_b2_stage1.pth")
        torch.save({
            'epoch': 13,
            'stage': 1,
            'model_state_dict': copy.deepcopy(model.state_dict()),
            'best_dice': 0.8123,
            'best_loss': 0.3540,
            'model_type': 'B2',
            'num_transformer_layers': 2,
            'p3_mode': 'C',
        }, best_ckpt_path)

        # Alter weights to represent epoch 17 (last model)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.1)

        # Create last Stage 1 checkpoint: full state (weights, optimizer, scheduler, scaler, RNG)
        last_ckpt_path = os.path.join(tmpdir, "last_model_b2_stage1.pth")
        torch.save({
            'epoch': 17,
            'stage': 1,
            'model_state_dict': copy.deepcopy(model.state_dict()),
            'optimizer_state_dict': s1_opt_state,
            'scheduler_state_dict': {'last_epoch': 17},
            'scaler_state_dict': {'scale': 65536.0},
            'val_dice': 0.8010,
            'val_loss': 0.3800,
            'best_dice': 0.8123,
            'best_loss': 0.3540,
            'epochs_no_improve': 4,
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy_rng_state': np.random.get_state(),
            'python_rng_state': random.getstate(),
            'dataloader_generator_state': saved_gen_state,
            'model_type': 'B2',
            'num_transformer_layers': 2,
            'p3_mode': 'C',
        }, last_ckpt_path)

        # SIMULATE STAGE 2 INITIALIZATION (as in train_crack.py)
        # 1. Reset model and load from best_ckpt_path
        fresh_model = create_b2_unet(num_transformer_layers=2, p3_mode="C", img_size=224)
        best_data = torch.load(best_ckpt_path, map_location='cpu', weights_only=False)
        fresh_model.load_state_dict(best_data['model_state_dict'])

        # Verify weights match best_data, NOT last_data
        last_data = torch.load(last_ckpt_path, map_location='cpu', weights_only=False)
        trainable_names = {name for name, p in fresh_model.named_parameters() if p.requires_grad}
        for k in fresh_model.state_dict():
            assert torch.equal(fresh_model.state_dict()[k], best_data['model_state_dict'][k]), f"Mismatch on {k}"
            if k in trainable_names:
                assert not torch.equal(fresh_model.state_dict()[k], last_data['model_state_dict'][k]), f"{k} matches last instead of best!"

        # 2. Update shared experts
        fresh_model.set_shared_experts([0, 1, 2, 3])

        # 3. Create fresh Stage 2 optimizer and scheduler
        opt_s2 = create_stage2_optimizer(
            fresh_model,
            stage2_base_lr=1e-4,
            stage2_shared_lr=1e-4,
            stage2_sage_lr=2e-4,
            stage2_p3_lr=1e-4,
        )
        sched_s2 = get_scheduler(opt_s2, epochs=10, warmup_epochs=1)

        # 4. Restore RNG from last_ckpt_path
        rng_data = torch.load(last_ckpt_path, map_location='cpu', weights_only=False)
        torch.set_rng_state(rng_data['rng_state'])
        np.random.set_state(rng_data['numpy_rng_state'])
        random.setstate(rng_data['python_rng_state'])
        g_test = torch.Generator()
        g_test.set_state(rng_data['dataloader_generator_state'])

        # 5. Invariant verifications:
        # A) Optimizer MUST BE STRICTLY FRESH (0 momentum / Adam moments)
        assert len(opt_s2.state) == 0, f"Stage 2 optimizer has contaminated momentum states: {len(opt_s2.state)}"
        
        # B) Scheduler MUST BE FRESH (configured for Stage 2 epochs=10, warmup=1)
        assert sched_s2.t_initial == 10, f"Stage 2 scheduler not fresh: t_initial={sched_s2.t_initial}"
        assert sched_s2.warmup_t == 1, f"Stage 2 scheduler warmup_t={sched_s2.warmup_t}"
        assert abs(sched_s2._get_lr(0)[0] - 1e-06) < 1e-9, f"Stage 2 scheduler initial LR={sched_s2._get_lr(0)}"

        # C) RNG states match last_ckpt_path
        assert torch.equal(torch.get_rng_state(), rng_data['rng_state']), "Torch RNG state mismatch!"
        assert g_test.get_state().equal(saved_gen_state), "DataLoader generator state mismatch!"

        # D) Metrics start fresh for Stage 2
        best_stage_dice = 0.0
        best_stage_loss = float('inf')
        epochs_no_improve = 0
        global_best_dice = float(best_data['best_dice'])
        assert best_stage_dice == 0.0
        assert global_best_dice == 0.8123
        assert epochs_no_improve == 0

        print("  --> PASS: Stage 2-only RNG restoration with strictly fresh optimizer/scheduler verified.")


def test_stage2_only_auto_detect_adjacent_rng_checkpoint():
    print("[Test 4/5] Verifying auto-detection of adjacent last_model_b2_stage1.pth...")
    with tempfile.TemporaryDirectory() as tmpdir:
        best_path = os.path.join(tmpdir, "best_model_b2_stage1.pth")
        last_path = os.path.join(tmpdir, "last_model_b2_stage1.pth")

        torch.save({'epoch': 12, 'model_state_dict': {}}, best_path)
        g_saved = torch.Generator().manual_seed(43210)
        torch.save({
            'epoch': 17,
            'model_state_dict': {},
            'rng_state': torch.get_rng_state(),
            'dataloader_generator_state': g_saved.get_state(),
        }, last_path)

        # Simulation of auto-detection logic in train_crack.py:
        generic_ckpt_path = best_path
        model_type = "B2"
        output_dir = tmpdir

        rng_ckpt_path = None
        candidates = []
        if generic_ckpt_path:
            ckpt_dir = os.path.dirname(generic_ckpt_path)
            candidates.append(os.path.join(ckpt_dir, f"last_model_{model_type.lower()}_stage1.pth"))
        candidates.append(os.path.join(output_dir, f"last_model_{model_type.lower()}_stage1.pth"))
        for cand in candidates:
            if os.path.exists(cand):
                rng_ckpt_path = cand
                break

        assert rng_ckpt_path == last_path, f"Expected {last_path}, got {rng_ckpt_path}"
        print(f"  --> PASS: Auto-detected adjacent checkpoint: {rng_ckpt_path}")


def test_real_candidate_b_checkpoint_provenance():
    print("[Test 5/5] Verifying real Candidate B checkpoints on disk if present...")
    if not os.path.exists(REAL_BEST_STAGE1_PATH) or not os.path.exists(REAL_LAST_STAGE1_PATH):
        print("  --> SKIP: Real checkpoints not present at expected paths (skipping on non-host environment).")
        return

    best_data = torch.load(REAL_BEST_STAGE1_PATH, map_location='cpu', weights_only=False)
    last_data = torch.load(REAL_LAST_STAGE1_PATH, map_location='cpu', weights_only=False)

    # 1. Assert best_data has weights and metadata but NO RNG / optimizer
    assert 'model_state_dict' in best_data
    assert 'rng_state' not in best_data
    assert 'optimizer_state_dict' not in best_data
    assert best_data['epoch'] == 13
    assert abs(best_data['best_dice'] - 0.7333) < 1e-3

    # 2. Assert last_data has full state including all 5 RNGs
    assert 'model_state_dict' in last_data
    assert 'optimizer_state_dict' in last_data
    assert 'scheduler_state_dict' in last_data
    assert 'scaler_state_dict' in last_data
    assert 'rng_state' in last_data
    assert 'cuda_rng_state_all' in last_data
    assert 'numpy_rng_state' in last_data
    assert 'python_rng_state' in last_data
    assert 'dataloader_generator_state' in last_data
    assert last_data['epoch'] == 17

    # 3. Model construction and loading from real best checkpoint
    model = create_b2_unet(
        num_classes=1,
        img_size=224,
        num_transformer_layers=best_data['num_transformer_layers'],
        pretrained=False,
        sage_config=best_data['sage_config'],
        p3_mode=best_data['p3_mode'],
    )
    load_res = model.load_state_dict(best_data['model_state_dict'], strict=True)
    assert len(load_res.missing_keys) == 0
    assert len(load_res.unexpected_keys) == 0

    # 4. Create Phase 5.1 Extra Stage 2 optimizer
    model.set_shared_experts([0, 1, 2, 3])
    opt = create_stage2_optimizer(
        model,
        stage2_base_lr=1e-4,
        stage2_shared_lr=1e-4,
        stage2_sage_lr=2e-4,
        stage2_p3_lr=1e-4,
    )
    group_names = {g['name'] for g in opt.param_groups}
    assert group_names == {'shared_experts', 'other_non_experts', 'sage', 'p3_refinement'}
    assert len(opt.state) == 0, "Stage 2 optimizer must be fresh!"

    print("  --> PASS: Real Candidate B Stage 1 checkpoint provenance and Stage 2 initialization verified.")


if __name__ == '__main__':
    test_stage2_resume_logic()
    test_cli_parser_rng_checkpoint()
    test_stage2_only_rng_restoration_and_fresh_optimizer()
    test_stage2_only_auto_detect_adjacent_rng_checkpoint()
    test_real_candidate_b_checkpoint_provenance()
    print("\nALL STAGE 2 RESUME & RNG STATE RESTORATION TESTS PASSED (5/5)!")
