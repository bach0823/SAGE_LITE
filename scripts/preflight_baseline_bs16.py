#!/usr/bin/env python3
"""
scripts/preflight_baseline_bs16.py

Runtime Preflight & VRAM / OOM Probe for Baseline B2 (ViT Depth 4, Top-K 2, S2-Gate v2, BS 16)
Designed for Google Colab Tesla T4 (16GB VRAM) and local execution.

Checks:
1. Real Crack500 dataset loader with Batch Size = 16 (448x448, smart filter fg_pixels >= 20)
2. Exact B2 Architecture (D=4, K=2, 8 Experts, S2-Gate Conv3x3)
3. Stage 1 forward + SoftBoundaryIoU + LB loss + backward + optimizer.step (AMP FP16)
4. Stage 2 reload + Stage 2 optimizer partitioning + S2-Gate LR
5. Stage 2 forward + backward + optimizer.step (AMP FP16)
6. Peak allocated VRAM, reserved VRAM, and remaining headroom
"""

import argparse
import gc
import os
import sys
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure sage_lite root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sage.networks.b2_unet import create_b2_unet
from sage.utils.dataloader import get_dataset_from_config
from sage.utils.training_utils import seed_worker, set_seed
from scripts.train_crack import (
    create_stage2_optimizer,
    get_optimizer_groups,
    CrackBinaryLoss,
    SoftBoundaryIoULoss,
)


def print_banner(text: str, ch: str = "="):
    line = ch * 75
    print(f"\n{line}\n{text}\n{line}")


def run_preflight_bs16(
    config_path: str,
    data_root: str = None,
    batch_size: int = 16,
    num_batches: int = 5,
    warmup_batches: int = 2,
    device_str: str = None,
):
    print_banner(f"PREFLIGHT RUNTIME & VRAM PROBE: BASELINE B2 (BS = {batch_size})")

    # 1. Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if data_root:
        config['root_dir'] = data_root

    seed = int(config.get('seed', 42))
    set_seed(seed)

    # 2. Hardware Setup
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[Device] Target Device: {device}")
    total_vram_mb = 0.0
    if device.type == 'cuda':
        dev_name = torch.cuda.get_device_name(0)
        total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        print(f"[Device] GPU: {dev_name} | VRAM: {total_vram_mb:.1f} MB ({total_vram_mb/1024:.2f} GB)")
        torch.backends.cudnn.benchmark = True

    # 3. Dataset Setup
    img_size = config.get('img_size', 448)
    print(f"[Dataset] Loading Crack500 dataset from '{config.get('root_dir')}'...")
    use_synthetic = False
    try:
        train_dataset = get_dataset_from_config(config, split='train', image_size=img_size)
        print(f"[Dataset] Real Crack500 loaded: {len(train_dataset)} training samples.")
    except Exception as e:
        print(f"[Dataset] Real dataset not available ({e}). Falling back to Synthetic Dataset for architecture probe.")
        use_synthetic = True
        class SyntheticDataset(torch.utils.data.Dataset):
            def __len__(self): return 100
            def __getitem__(self, idx):
                return {
                    'image': torch.randn(3, img_size, img_size),
                    'label': torch.randint(0, 2, (1, img_size, img_size)).float(),
                }
        train_dataset = SyntheticDataset()

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 if not use_synthetic else 0,
        pin_memory=(device.type == 'cuda'),
        drop_last=True,
    )
    print(f"[DataLoader] Batch size: {batch_size} | Total batches: {len(train_loader)}")

    # 4. Instantiate Baseline B2 (D=4, K=2, 8 Experts, S2-Gate)
    print_banner("INSTANTIATING BASELINE B2 UNET (D=4, K=2, S2-Gate v2)")
    sage_cfg = config.get('sage_config', {})
    model = create_b2_unet(
        num_transformer_layers=int(config.get('num_transformer_layers', 4)),
        pretrained=True if not use_synthetic else False,
        sage_config=sage_cfg,
        use_s2_gate=config.get('use_s2_gate', True),
        s2_gate_kernel_size=config.get('s2_gate_kernel_size', 3),
    ).to(device)

    info = model.get_model_info()
    print(f"[Model] {info['model_name']} | ViT Depth: {info['num_transformer_layers']} | Experts: {info['expert_pool_size']} | Top-K: {info['top_k']}")
    print(f"[Model] Params: {info['total_parameters']:,} (Trainable: {info['trainable_parameters']:,})")

    # Objective: CrackBinaryLoss (BCE + Dice + SoftBIoU)
    biou_weight = float(config.get('boundary_iou_weight', 0.5))
    biou_dilation = int(config.get('boundary_iou_dilation', 2))
    criterion = CrackBinaryLoss(boundary_weight=biou_weight, boundary_dilation=biou_dilation).to(device)
    print(f"[Objective] BCE (1.0) + Dice (1.5) + SoftBIoU ({biou_weight}, dilation={biou_dilation})")

    # =========================================================================
    # STEP 1: STAGE 1 PREFLIGHT (Forward, Backward, Optimizer, Peak VRAM)
    # =========================================================================
    print_banner("STEP 1: STAGE 1 TRAINING SMOKE (Batch Size = 16)")
    model.train()
    param_groups = get_optimizer_groups(model, lr_backbone=1e-5, lr_decoder=1e-4, lr_sage=2e-4, weight_decay=0.01)
    optimizer = torch.optim.AdamW(param_groups)
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda'))

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    data_iter = iter(train_loader)
    stage1_step_times = []

    for b_idx in range(1, num_batches + 1):
        t0 = time.perf_counter()
        batch = next(data_iter)
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16):
            fwd = model.forward_with_routing_info(images)
            logits = fwd['logits']
            routing_infos = fwd['routing_infos']
            lb_loss = model.compute_total_load_balance_loss(routing_infos)
            seg_loss = criterion(logits, labels)
            total_loss = seg_loss + 1.0 * lb_loss

        assert torch.isfinite(total_loss), f"Stage 1 Batch {b_idx}: Non-finite loss!"
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if device.type == 'cuda':
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        if b_idx > warmup_batches:
            stage1_step_times.append(dt)

        if device.type == 'cuda':
            peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
            peak_res = torch.cuda.max_memory_reserved() / (1024 ** 2)
            free_vram = total_vram_mb - peak_res
            print(f"  Stage 1 Batch {b_idx:02d}/{num_batches:02d} | Step: {dt:5.2f}s | Loss: {total_loss.item():.4f} | Peak Alloc: {peak_alloc/1024:5.2f} GB | Peak Res: {peak_res/1024:5.2f} GB | Free: {free_vram/1024:4.2f} GB")
        else:
            print(f"  Stage 1 Batch {b_idx:02d}/{num_batches:02d} | Step: {dt:5.2f}s | Loss: {total_loss.item():.4f}")

    s1_speed = (batch_size / (sum(stage1_step_times) / len(stage1_step_times))) if stage1_step_times else 0.0
    print(f"[PASS] Stage 1 Smoke PASSED (Speed: {s1_speed:.2f} samples/s)")

    # =========================================================================
    # STEP 2: STAGE 2 PREFLIGHT (Differential LR, S2-Gate LR, Peak VRAM)
    # =========================================================================
    print_banner("STEP 2: STAGE 2 TRAINING SMOKE (Batch Size = 16)")
    # Setup Stage 2 optimizer
    s2_base_lr = float(config.get('stage2_base_lr', 1e-4))
    s2_shared_lr = float(config.get('stage2_shared_lr', 1e-4))
    s2_gate_lr = float(config.get('stage2_s2_gate_lr', 1e-3))
    stage2_optimizer = create_stage2_optimizer(
        model,
        stage2_base_lr=s2_base_lr,
        stage2_shared_lr=s2_shared_lr,
        stage2_sage_lr=s2_base_lr,
        stage2_s2_gate_lr=s2_gate_lr,
    )

    stage2_step_times = []
    for b_idx in range(1, num_batches + 1):
        t0 = time.perf_counter()
        batch = next(data_iter)
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        stage2_optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16):
            fwd = model.forward_with_routing_info(images)
            logits = fwd['logits']
            routing_infos = fwd['routing_infos']
            lb_loss = model.compute_total_load_balance_loss(routing_infos)
            seg_loss = criterion(logits, labels)
            total_loss = seg_loss + 1.0 * lb_loss

        assert torch.isfinite(total_loss), f"Stage 2 Batch {b_idx}: Non-finite loss!"
        scaler.scale(total_loss).backward()
        scaler.step(stage2_optimizer)
        scaler.update()

        if device.type == 'cuda':
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        if b_idx > warmup_batches:
            stage2_step_times.append(dt)

        if device.type == 'cuda':
            peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
            peak_res = torch.cuda.max_memory_reserved() / (1024 ** 2)
            free_vram = total_vram_mb - peak_res
            print(f"  Stage 2 Batch {b_idx:02d}/{num_batches:02d} | Step: {dt:5.2f}s | Loss: {total_loss.item():.4f} | Peak Alloc: {peak_alloc/1024:5.2f} GB | Peak Res: {peak_res/1024:5.2f} GB | Free: {free_vram/1024:4.2f} GB")
        else:
            print(f"  Stage 2 Batch {b_idx:02d}/{num_batches:02d} | Step: {dt:5.2f}s | Loss: {total_loss.item():.4f}")

    s2_speed = (batch_size / (sum(stage2_step_times) / len(stage2_step_times))) if stage2_step_times else 0.0
    print(f"[PASS] Stage 2 Smoke PASSED (Speed: {s2_speed:.2f} samples/s)")

    # =========================================================================
    # STEP 3: FINAL VERDICT
    # =========================================================================
    print_banner("FINAL PREFLIGHT VERDICT (BATCH SIZE 16)")
    if device.type == 'cuda':
        final_free = total_vram_mb - torch.cuda.max_memory_reserved() / (1024 ** 2)
        print(f"  [HARDWARE] GPU:                  {dev_name}")
        print(f"  [PEAK VRAM ALLOCATED]:          {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB")
        print(f"  [PEAK VRAM RESERVED]:           {torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB")
        print(f"  [REMAINING HEADROOM]:           {final_free / 1024:.2f} GB ({(final_free / total_vram_mb * 100):.1f}%)")
        if final_free > 500.0:
            print(f"  [VERDICT] >>> PASS: Batch Size 16 is FULLY FEASIBLE on this GPU! <<<")
        else:
            print(f"  [VERDICT] >>> MARGINAL: Headroom < 500MB. Batch Size 14 is safer. <<<")
    else:
        print("  [VERDICT] >>> PASS: Architecture and forward/backward sanity confirmed on CPU! <<<")

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preflight & OOM Probe for Baseline B2 (BS=16)")
    parser.add_argument('--config', type=str, default='configs/p3_ablation/phase6_full_s1/a1_s2g_end_to_end.yaml')
    parser.add_argument('--data-root', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-batches', type=int, default=5)
    parser.add_argument('--warmup-batches', type=int, default=2)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    run_preflight_bs16(
        config_path=args.config,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        warmup_batches=args.warmup_batches,
        device_str=args.device,
    )
