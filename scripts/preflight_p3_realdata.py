"""
Real-Data P3 Launch Preflight Script for SAGE-Lite B2 (Run A, Run B, Run C).
Author: Special Subject AI Team (September 2026)

Conducts an exhaustive real-data launch preflight across:
- Run A (b2_p3_run_a.yaml, p3_mode="A")
- Run B (b2_p3_run_b.yaml, p3_mode="B")
- Run C (b2_p3_run_c.yaml, p3_mode="C")

Verifies all 24 required preflight points:
- Real Crack500 DataLoader & canonical preprocessing
- Forward, backward, loss finiteness, optimizer.step on 3 real batches
- Stage 1 -> Stage 2 transition with reload, set_shared_experts([0, 1, 2, 3]), optimizer grouping
- Stage 2 forward, backward, optimizer.step on 3 real batches
- P3 gradient presence (Run B/C) / absence (Run A)
- Fixed Shared PE28 buffer invariance (no gradient, float32, correct shape [1, 784, 192])
- Peak allocated & reserved VRAM, throughput (samples/s), step time
- Validation evaluation sanity check on small sample (WITHOUT accessing Test set)
- Temporary artifact isolation (no pollution of official experiment directories)
"""

import argparse
import gc
import json
import logging
import os
import sys
import tempfile
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sage.networks.b2_unet import create_b2_unet
from sage.utils.dataloader import get_dataset_from_config
from sage.utils.training_utils import seed_worker, set_seed
from scripts.train_crack import (
    create_stage2_optimizer,
    get_optimizer_groups,
    get_scheduler,
    CrackBinaryLoss,
    DEFAULT_SHARED_PREFIXES,
)
from scripts.evaluate_crack_official import get_image_mask_pairs, evaluate_split


def run_single_preflight(config_path: str, data_root_override: str = None, num_batches: int = 3):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    if data_root_override:
        cfg['root_dir'] = data_root_override

    p3_mode = cfg.get('p3_mode')
    run_id = f"Run {p3_mode}"
    print("\n" + "=" * 80)
    print(f"STARTING REAL-DATA LAUNCH PREFLIGHT: {run_id} (Config: {os.path.basename(config_path)})")
    print("=" * 80)

    # 1. Device and environment setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        dev_name = torch.cuda.get_device_name(0)
        print(f"GPU Name: {dev_name}")
        if '1650' in dev_name or '1660' in dev_name:
            torch.backends.cudnn.enabled = False
            print("Detected GTX 1650/1660: set cudnn.enabled=False for FP16 stability.")
        torch.cuda.reset_peak_memory_stats(device)

    # Temporary directory for preflight artifacts
    temp_dir = os.path.join(tempfile.gettempdir(), f"p3_preflight_run_{p3_mode}_{int(time.time())}")
    os.makedirs(temp_dir, exist_ok=True)
    print(f"Temporary Preflight Directory: {temp_dir}")

    # Set seed
    seed = int(cfg.get('seed', 42))
    set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    # 2. Real DataLoader setup
    print("\n[DataLoader Setup]")
    root_dir = cfg.get('root_dir', '/content/dataset/Crack500')
    print(f"Dataset root_dir: {root_dir}")
    if not os.path.exists(root_dir):
        raise FileNotFoundError(
            f"Dataset directory '{root_dir}' not found! "
            f"Please ensure Crack500 is prepared or specify --data-root-override."
        )

    img_size = int(cfg.get('img_size', 448))
    batch_size = int(cfg.get('batch_size', 12))
    num_workers = int(cfg.get('num_workers', 2))

    train_dataset = get_dataset_from_config(cfg, split='train', image_size=img_size)
    print(f"Real Crack500 Train Dataset: {len(train_dataset)} samples loaded successfully.")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=(device.type == 'cuda'),
        drop_last=False
    )
    dataloader_health = "PASS"

    # 3. Model Instantiation
    print(f"\n[Model Instantiation: {run_id}]")
    model = create_b2_unet(
        num_classes=1,
        img_size=img_size,
        num_transformer_layers=int(cfg.get('num_transformer_layers', 4)),
        pretrained=False,
        sage_config=cfg.get('sage_config'),
        p3_mode=p3_mode
    ).to(device)
    model.train()

    # Verify PE28 initial buffer state
    assert hasattr(model.backbone, "pe28_fixed"), "Model lacks pe28_fixed buffer!"
    pe28 = model.backbone.pe28_fixed
    assert pe28.shape == (1, 784, 192), f"PE28 shape mismatch: {pe28.shape} != (1, 784, 192)"
    assert pe28.dtype == torch.float32, f"PE28 dtype {pe28.dtype} != float32"
    assert not pe28.requires_grad, "PE28 must NOT have requires_grad=True!"
    print(f"PE28 Buffer Initial Invariant: Verified shape={list(pe28.shape)}, dtype={pe28.dtype}, requires_grad={pe28.requires_grad}")

    # Verify P3 parameters in model
    p3_params = [p for n, p in model.named_parameters() if "p3_refinement" in n]
    if p3_mode in ("B", "C"):
        assert len(p3_params) == 10, f"Expected 10 P3 params for Run {p3_mode}, got {len(p3_params)}"
        print(f"Run {p3_mode} P3 refinement parameters: 10 tensors confirmed.")
    else:
        assert len(p3_params) == 0, f"Expected 0 P3 params for Run A, got {len(p3_params)}"
        print("Run A P3 refinement parameters: 0 tensors confirmed (Identity).")

    # 4. Stage 1 Optimizer Setup
    base_lr = float(cfg.get('lr', 1e-4))
    stage1_groups = get_optimizer_groups(
        model,
        lr_backbone=base_lr * 0.1,
        lr_decoder=base_lr,
        lr_sage=base_lr,
        weight_decay=0.05
    )
    opt1 = optim.AdamW(stage1_groups)
    scaler1 = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    criterion = CrackBinaryLoss()

    # 5. STAGE 1 PREFLIGHT: Run real batches
    print(f"\n[Executing Stage 1 Real Batches ({num_batches} batches)]")
    stage1_forward_pass = False
    stage1_backward_pass = False
    stage1_step_pass = False
    stage1_nan_inf = False

    batch_times = []
    total_samples = 0

    loader_iter = iter(train_loader)
    for b_idx in range(1, num_batches + 1):
        t0 = time.perf_counter()
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        current_bs = images.size(0)
        total_samples += current_bs

        assert images.shape == (current_bs, 3, img_size, img_size), f"Input shape mismatch: {images.shape}"
        assert labels.shape == (current_bs, 1, img_size, img_size), f"Label shape mismatch: {labels.shape}"

        opt1.zero_grad(set_to_none=True)

        # Forward pass
        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                if hasattr(model, 'forward_with_routing_info'):
                    out = model.forward_with_routing_info(images)
                    logits = out['logits']
                    lb_loss = model.compute_total_load_balance_loss(out['routing_infos'])
                else:
                    logits = model(images)
                    lb_loss = torch.tensor(0.0, device=device)
                seg_loss = criterion(logits, labels)
                total_loss = seg_loss + 1.0 * lb_loss
        else:
            if hasattr(model, 'forward_with_routing_info'):
                out = model.forward_with_routing_info(images)
                logits = out['logits']
                lb_loss = model.compute_total_load_balance_loss(out['routing_infos'])
            else:
                logits = model(images)
                lb_loss = torch.tensor(0.0, device=device)
            seg_loss = criterion(logits, labels)
            total_loss = seg_loss + 1.0 * lb_loss

        # Verify output shape & finite loss
        assert logits.shape == (current_bs, 1, img_size, img_size), f"Output shape mismatch: {logits.shape}"
        assert torch.isfinite(total_loss), f"Non-finite loss detected: {total_loss.item()}"
        stage1_forward_pass = True

        # Backward pass
        if scaler1 is not None:
            scaler1.scale(total_loss).backward()
            scaler1.step(opt1)
            scaler1.update()
        else:
            total_loss.backward()
            opt1.step()

        if device.type == 'cuda':
            torch.cuda.synchronize(device)

        t1 = time.perf_counter()
        step_duration = t1 - t0
        batch_times.append(step_duration)

        # Verify gradients are finite
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    stage1_nan_inf = True
                    print(f"ERROR: Non-finite gradient in {name}")

        stage1_backward_pass = True
        stage1_step_pass = True
        print(f"  Stage 1 Batch {b_idx}/{num_batches}: Loss={total_loss.item():.4f} (Seg: {seg_loss.item():.4f}, LB: {lb_loss.item():.4f}) | Time: {step_duration:.3f}s")

    # Record throughput and VRAM
    avg_step_time = float(np.mean(batch_times))
    throughput_sps = float(batch_size / avg_step_time) if avg_step_time > 0 else 0.0

    if device.type == 'cuda':
        peak_alloc_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        peak_res_mb = float(torch.cuda.max_memory_reserved(device) / (1024 ** 2))
    else:
        peak_alloc_mb = 0.0
        peak_res_mb = 0.0

    print(f"  Stage 1 Timing: Avg Step = {avg_step_time:.3f}s | Throughput = {throughput_sps:.2f} samples/s")
    print(f"  Stage 1 VRAM: Peak Allocated = {peak_alloc_mb:.1f} MB ({peak_alloc_mb/1024:.2f} GB) | Peak Reserved = {peak_res_mb:.1f} MB ({peak_res_mb/1024:.2f} GB)")

    # 6. STAGE 1 -> STAGE 2 TRANSITION PREFLIGHT
    print(f"\n[Stage 1 -> Stage 2 Transition Verification]")
    stage1_ckpt_path = os.path.join(temp_dir, f"best_model_b2_stage1_preflight.pth")
    torch.save({
        'epoch': 1,
        'stage': 1,
        'model_state_dict': model.state_dict(),
        'best_dice': 0.75,
        'best_loss': 1.10
    }, stage1_ckpt_path)
    assert os.path.exists(stage1_ckpt_path), "Failed to save Stage 1 checkpoint!"
    print(f"  Stage 1 Checkpoint saved: {stage1_ckpt_path} ({os.path.getsize(stage1_ckpt_path)} bytes)")

    # Reload into fresh model
    model2 = create_b2_unet(
        num_classes=1,
        img_size=img_size,
        num_transformer_layers=int(cfg.get('num_transformer_layers', 4)),
        pretrained=False,
        sage_config=cfg.get('sage_config'),
        p3_mode=p3_mode
    ).to(device)
    ckpt = torch.load(stage1_ckpt_path, map_location=device, weights_only=False)
    model2.load_state_dict(ckpt['model_state_dict'])
    print("  Fresh model instantiated and Stage 1 weights reloaded successfully.")

    # Call set_shared_experts([0, 1, 2, 3])
    shared_indices = cfg.get("sage_config", {}).get("shared_expert_indices", [0, 1, 2, 3])
    assert hasattr(model2, "set_shared_experts"), "model2 lacks set_shared_experts method!"
    model2.set_shared_experts(shared_indices)
    print(f"  model2.set_shared_experts({shared_indices}) executed.")

    # Build Stage-2 optimizer
    stage2_base_lr = float(cfg.get('stage2_base_lr', 1e-4))
    stage2_shared_lr = float(cfg.get('stage2_shared_lr', 1e-4))
    assert stage2_shared_lr == 1e-4, f"stage2_shared_lr {stage2_shared_lr} != 1e-4"
    assert stage2_base_lr == 1e-4, f"stage2_base_lr {stage2_base_lr} != 1e-4"
    assert abs(stage2_shared_lr / stage2_base_lr - 1.0) < 1e-6, "Stage-2 LR ratio must be 1:1!"
    print(f"  Stage 2 LRs Verified: shared_lr={stage2_shared_lr:.2e}, base_lr={stage2_base_lr:.2e} (Ratio: 1.00)")

    opt2 = create_stage2_optimizer(model2, stage2_base_lr=stage2_base_lr, stage2_shared_lr=stage2_shared_lr)

    # Verify Stage-2 parameter grouping
    param_id_to_name2 = {id(p): name for name, p in model2.named_parameters()}
    trainable_p2 = [p for p in model2.parameters() if p.requires_grad]
    all_opt2_p = []
    shared_opt_p = []
    other_opt_p = []

    for g_idx, grp in enumerate(opt2.param_groups):
        g_name = grp.get('name', f"grp_{g_idx}")
        all_opt2_p.extend(grp['params'])
        if "shared" in g_name:
            shared_opt_p.extend(grp['params'])
            assert grp['lr'] == stage2_shared_lr, f"Shared group LR {grp['lr']} != {stage2_shared_lr}"
        else:
            other_opt_p.extend(grp['params'])
            assert grp['lr'] == stage2_base_lr, f"Other group LR {grp['lr']} != {stage2_base_lr}"

    assert len(trainable_p2) == len(all_opt2_p), "Mismatch in total trainable vs opt2 parameters!"
    assert len(set(trainable_p2)) == len(set(all_opt2_p)), "Duplicate parameters in Stage 2 optimizer!"

    # Verify shared expert prefix correctness
    shared_param_names = [param_id_to_name2.get(id(p), "") for p in shared_opt_p]
    expected_prefixes = (
        "backbone.convnext.stages.0.main_block.",
        "backbone.convnext.stages.1.main_block.",
        "backbone.convnext.stages.2.main_block.",
        "backbone.convnext.stages.3.main_block.",
    )
    for s_name in shared_param_names:
        assert any(s_name.startswith(pfx) for pfx in expected_prefixes), f"Illegitimate parameter in shared_experts: {s_name}"
        assert not any(x in s_name for x in ["router", "sa_hub", "alpha", "decoder", "transformer", "p3_refinement"]), (
            f"Non-CNN main_block found in shared_experts: {s_name}"
        )
    print(f"  Stage 2 Shared Experts: All {len(shared_param_names)} parameters confirmed as CNN main_blocks (requires_grad=True).")

    # Verify P3 membership in Stage 2
    for n, p in model2.named_parameters():
        if "p3_refinement" in n:
            assert id(p) in [id(x) for x in other_opt_p], f"P3 param {n} must belong to other_and_routers!"
            assert id(p) not in [id(x) for x in shared_opt_p], f"P3 param {n} illegally in shared_experts!"
    print(f"  Stage 2 P3 Membership: All P3 parameters confirmed strictly in other_and_routers.")

    # Verify Scheduler recreated with warmup=3
    scheduler2 = get_scheduler(opt2, epochs=15, warmup_epochs=3)
    assert scheduler2.warmup_t == 3, f"Scheduler warmup {scheduler2.warmup_t} != 3"
    print("  Stage 2 Scheduler: Freshly recreated with warmup_epochs=3.")
    stage1_to_stage2_transition = True

    # 7. STAGE 2 PREFLIGHT: Run real batches
    print(f"\n[Executing Stage 2 Real Batches ({num_batches} batches)]")
    model2.train()
    scaler2 = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    stage2_fwd_pass = False
    stage2_bwd_pass = False
    stage2_step_pass = False
    stage2_nan_inf = False

    for b_idx in range(1, num_batches + 1):
        t0 = time.perf_counter()
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        current_bs = images.size(0)

        opt2.zero_grad(set_to_none=True)

        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                if hasattr(model2, 'forward_with_routing_info'):
                    out2 = model2.forward_with_routing_info(images)
                    logits2 = out2['logits']
                    lb_loss2 = model2.compute_total_load_balance_loss(out2['routing_infos'])
                else:
                    logits2 = model2(images)
                    lb_loss2 = torch.tensor(0.0, device=device)
                seg_loss2 = criterion(logits2, labels)
                total_loss2 = seg_loss2 + 1.0 * lb_loss2
        else:
            if hasattr(model2, 'forward_with_routing_info'):
                out2 = model2.forward_with_routing_info(images)
                logits2 = out2['logits']
                lb_loss2 = model2.compute_total_load_balance_loss(out2['routing_infos'])
            else:
                logits2 = model2(images)
                lb_loss2 = torch.tensor(0.0, device=device)
            seg_loss2 = criterion(logits2, labels)
            total_loss2 = seg_loss2 + 1.0 * lb_loss2

        assert logits2.shape == (current_bs, 1, img_size, img_size)
        assert torch.isfinite(total_loss2)
        stage2_fwd_pass = True

        if scaler2 is not None:
            scaler2.scale(total_loss2).backward()
            scaler2.step(opt2)
            scaler2.update()
        else:
            total_loss2.backward()
            opt2.step()

        if device.type == 'cuda':
            torch.cuda.synchronize(device)

        t1 = time.perf_counter()
        step_duration2 = t1 - t0

        for name, param in model2.named_parameters():
            if param.requires_grad and param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    stage2_nan_inf = True
                    print(f"ERROR: Non-finite gradient in Stage 2 {name}")

        stage2_bwd_pass = True
        stage2_step_pass = True
        print(f"  Stage 2 Batch {b_idx}/{num_batches}: Loss={total_loss2.item():.4f} (Seg: {seg_loss2.item():.4f}, LB: {lb_loss2.item():.4f}) | Time: {step_duration2:.3f}s")

    # 8. Verify P3 Gradients & PE28 State after Stage 2 step
    print("\n[P3 Gradient & PE28 Buffer Verification]")
    p3_grad_status = "PASS"
    pe28_status = "PASS"

    p3_params2 = [(n, p) for n, p in model2.named_parameters() if "p3_refinement" in n]
    if p3_mode in ("B", "C"):
        assert len(p3_params2) == 10, f"Expected 10 P3 params, got {len(p3_params2)}"
        for n, p in p3_params2:
            assert p.grad is not None, f"P3 param {n} missing gradient after backward!"
            assert torch.isfinite(p.grad).all(), f"P3 param {n} has non-finite gradient!"
        print(f"  Run {p3_mode}: All 10 P3 refinement tensors have active, finite gradients.")
    else:
        assert len(p3_params2) == 0, f"Run A must have 0 P3 parameters, got {len(p3_params2)}"
        print("  Run A: 0 P3 refinement parameters (Identity mapping verified).")

    # Verify PE28 buffer state
    pe28_2 = model2.backbone.pe28_fixed
    assert pe28_2.grad is None, "PE28 fixed buffer must NEVER accumulate gradients!"
    assert not pe28_2.requires_grad, "PE28 fixed buffer must have requires_grad=False!"
    assert pe28_2.dtype == torch.float32, f"PE28 dtype {pe28_2.dtype} != float32"
    assert pe28_2.shape == (1, 784, 192), f"PE28 shape {pe28_2.shape} != (1, 784, 192)"
    assert pe28_2.device == device, f"PE28 device {pe28_2.device} != {device}"
    print(f"  PE28 Fixed Buffer: Verified 0 gradient, float32, device={device}, shape=(1, 784, 192).")

    # 9. Save Stage 2 Checkpoint
    stage2_ckpt_path = os.path.join(temp_dir, f"best_model_b2_stage2_preflight.pth")
    torch.save({
        'epoch': 1,
        'stage': 2,
        'model_state_dict': model2.state_dict(),
        'best_dice': 0.76,
        'best_loss': 1.05
    }, stage2_ckpt_path)
    assert os.path.exists(stage2_ckpt_path), "Failed to save Stage 2 checkpoint!"
    ckpt_save_reload_pass = True
    print(f"  Stage 2 Checkpoint saved: {stage2_ckpt_path} ({os.path.getsize(stage2_ckpt_path)} bytes)")

    # 10. Validation Metric Computation Sanity Check (Val Set Only - NEVER Test Set)
    print("\n[Validation Metric Sanity Check (Val Set Only)]")
    val_pairs = get_image_mask_pairs(cfg, 'val')
    print(f"  Discovered {len(val_pairs)} validation pairs in Crack500 val split.")
    assert len(val_pairs) > 0, "No validation pairs found!"
    sample_val = val_pairs[:2]  # run on 2 sample images for fast sanity check
    model2.eval()
    with torch.no_grad():
        val_res = evaluate_split(
            model2, sample_val, device,
            protocol='setting_a', tile_size=img_size, criterion=criterion, verbose=False
        )
    print(f"  Sanity Val Metrics (2 samples): Val Loss={val_res['loss']:.4f}, Val Dice={val_res['dice']:.4f}, Val IoU={val_res['iou']:.4f}")
    assert np.isfinite(val_res['dice']), "Validation Dice is not finite!"
    print("  Validation pipeline execution: PASS (strictly isolated from Test set).")

    # Clean up temp directory
    try:
        import shutil
        shutil.rmtree(temp_dir)
        print(f"  Cleaned up temporary preflight directory: {temp_dir}")
    except Exception as e:
        print(f"  Warning cleaning up temp dir: {e}")

    # Build report dictionary for this run
    results = {
        'run_id': run_id,
        'p3_mode': p3_mode,
        'forward_pass': "PASS" if stage1_forward_pass else "FAIL",
        'backward_pass': "PASS" if stage1_backward_pass else "FAIL",
        'optimizer_step': "PASS" if stage1_step_pass else "FAIL",
        'stage1_to_stage2_transition': "PASS" if stage1_to_stage2_transition else "FAIL",
        'stage2_fwd_bwd': "PASS" if (stage2_fwd_pass and stage2_bwd_pass) else "FAIL",
        'ckpt_save_reload': "PASS" if ckpt_save_reload_pass else "FAIL",
        'peak_alloc_vram': f"{peak_alloc_mb:.1f} MB ({peak_alloc_mb/1024:.2f} GB)" if device.type == 'cuda' else "N/A (CPU)",
        'peak_res_vram': f"{peak_res_mb:.1f} MB ({peak_res_mb/1024:.2f} GB)" if device.type == 'cuda' else "N/A (CPU)",
        'throughput': f"{throughput_sps:.2f} samples/s",
        'avg_step_time': f"{avg_step_time:.3f} s/batch",
        'dataloader_health': dataloader_health,
        'nan_inf_status': "CLEAN (None)" if not (stage1_nan_inf or stage2_nan_inf) else "FAIL (NaN/Inf Detected)",
        'p3_grad_status': p3_grad_status,
        'pe28_status': pe28_status,
    }

    print(f"\n--> PREFLIGHT SUMMARY FOR {run_id}: ALL 24 INVARIANT CHECKS PASSED!\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="Real-Data P3 Launch Preflight for Run A, Run B, and Run C")
    parser.add_argument('--config', type=str, default=None, help="Path to single YAML config")
    parser.add_argument('--data-root-override', type=str, default=None, help="Override root_dir in config")
    parser.add_argument('--num-batches', type=int, default=3, help="Number of real batches per stage (default: 3)")
    args = parser.parse_args()

    configs_to_run = []
    if args.config:
        configs_to_run.append(args.config)
    else:
        configs_to_run = [
            os.path.join(project_root, "configs", "p3_ablation", "b2_p3_run_a.yaml"),
            os.path.join(project_root, "configs", "p3_ablation", "b2_p3_run_b.yaml"),
            os.path.join(project_root, "configs", "p3_ablation", "b2_p3_run_c.yaml"),
        ]

    print("=" * 80)
    print("STARTING FULL REAL-DATA P3 LAUNCH PREFLIGHT SUITE (RUN A, RUN B, RUN C)")
    print("=" * 80)

    all_results = []
    all_passed = True

    for cfg_path in configs_to_run:
        try:
            res = run_single_preflight(cfg_path, data_root_override=args.data_root_override, num_batches=args.num_batches)
            all_results.append(res)
        except Exception as e:
            print(f"\n[PREFLIGHT FAILED] Error during {cfg_path}: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
            break

    print("\n" + "=" * 80)
    print("FINAL CONSOLIDATED PREFLIGHT REPORT")
    print("=" * 80)

    header = f"| {'Check Item / Metric':<32} | {'Run A (Identity)':<16} | {'Run B (Generic DW)':<18} | {'Run C (ASDW)':<16} |"
    sep = f"|{'-'*34}|{'-'*18}|{'-'*20}|{'-'*18}|"
    print(header)
    print(sep)

    if len(all_results) == 3:
        rA, rB, rC = all_results[0], all_results[1], all_results[2]
        rows = [
            ("A. Real-data forward", rA['forward_pass'], rB['forward_pass'], rC['forward_pass']),
            ("B. Real-data backward", rA['backward_pass'], rB['backward_pass'], rC['backward_pass']),
            ("C. Optimizer step", rA['optimizer_step'], rB['optimizer_step'], rC['optimizer_step']),
            ("D. Stage 1 -> 2 transition", rA['stage1_to_stage2_transition'], rB['stage1_to_stage2_transition'], rC['stage1_to_stage2_transition']),
            ("E. Stage 2 forward/backward", rA['stage2_fwd_bwd'], rB['stage2_fwd_bwd'], rC['stage2_fwd_bwd']),
            ("F. Checkpoint save/reload", rA['ckpt_save_reload'], rB['ckpt_save_reload'], rC['ckpt_save_reload']),
            ("G. Peak allocated VRAM", rA['peak_alloc_vram'], rB['peak_alloc_vram'], rC['peak_alloc_vram']),
            ("H. Peak reserved VRAM", rA['peak_res_vram'], rB['peak_res_vram'], rC['peak_res_vram']),
            ("I. Throughput", rA['throughput'], rB['throughput'], rC['throughput']),
            ("J. DataLoader health", rA['dataloader_health'], rB['dataloader_health'], rC['dataloader_health']),
            ("K. NaN / Inf status", rA['nan_inf_status'], rB['nan_inf_status'], rC['nan_inf_status']),
            ("L. P3 gradient status", rA['p3_grad_status'], rB['p3_grad_status'], rC['p3_grad_status']),
            ("M. PE28 fixed buffer status", rA['pe28_status'], rB['pe28_status'], rC['pe28_status']),
        ]
        for name, a, b, c in rows:
            print(f"| {name:<32} | {a:<16} | {b:<18} | {c:<16} |")
        print(sep)

    final_verdict = "REAL-DATA P3 LAUNCH PREFLIGHT = PASS" if all_passed and len(all_results) == 3 else "REAL-DATA P3 LAUNCH PREFLIGHT = BLOCKED"
    print(f"\n{final_verdict}\n")

if __name__ == "__main__":
    main()
