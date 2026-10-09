#!/usr/bin/env python3
"""
scripts/benchmark_bs_sweep.py

Automated Batch Size Capacity & Throughput Sweep for Baseline B2 UNet
Sweeps Batch Sizes: [16, 20, 24, 28, 32]
Profiles on Real Crack500 dataset (or synthetic fallback) under AMP FP16.
Measures:
- Peak Allocated VRAM & Peak Reserved VRAM
- Remaining Headroom & OOM Boundary
- Throughput (samples/sec) for Stage 1 (All modules) & Stage 2 (Fine-tuning)
- Generates a neat comparison summary table.
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
)


def print_banner(text: str, ch: str = "="):
    line = ch * 80
    print(f"\n{line}\n{text}\n{line}")


def test_single_batch_size(
    config: dict,
    batch_size: int,
    train_dataset,
    device,
    num_batches: int = 5,
    warmup_batches: int = 2,
):
    print_banner(f"TESTING BATCH SIZE: {batch_size}")
    total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2) if device.type == 'cuda' else 0.0

    # Clean memory before test
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == 'cuda'),
        drop_last=True,
    )

    sage_cfg = config.get('sage_config', {})
    try:
        model = create_b2_unet(
            num_transformer_layers=int(config.get('num_transformer_layers', 4)),
            pretrained=False,  # Skip download during fast sweep
            sage_config=sage_cfg,
            use_s2_gate=config.get('use_s2_gate', True),
            s2_gate_kernel_size=config.get('s2_gate_kernel_size', 3),
        ).to(device)

        biou_weight = float(config.get('boundary_iou_weight', 0.5))
        biou_dilation = int(config.get('boundary_iou_dilation', 2))
        criterion = CrackBinaryLoss(boundary_weight=biou_weight, boundary_dilation=biou_dilation).to(device)

        # -------------------------------------------------------------
        # Stage 1 Smoke & Throughput
        # -------------------------------------------------------------
        model.train()
        param_groups = get_optimizer_groups(model, lr_backbone=1e-5, lr_decoder=1e-4, lr_sage=2e-4, weight_decay=0.01)
        optimizer = torch.optim.AdamW(param_groups)
        scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda'))

        data_iter = iter(train_loader)
        s1_times = []

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

            if not torch.isfinite(total_loss):
                raise RuntimeError(f"Non-finite loss at Stage 1 batch {b_idx}")

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if device.type == 'cuda':
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if b_idx > warmup_batches:
                s1_times.append(dt)

        s1_throughput = (batch_size / (sum(s1_times) / len(s1_times))) if s1_times else 0.0

        # -------------------------------------------------------------
        # Stage 2 Smoke & Throughput
        # -------------------------------------------------------------
        s2_optimizer = create_stage2_optimizer(
            model,
            stage2_base_lr=1e-4,
            stage2_shared_lr=1e-4,
            stage2_sage_lr=1e-4,
            stage2_s2_gate_lr=1e-3,
        )

        s2_times = []
        for b_idx in range(1, num_batches + 1):
            t0 = time.perf_counter()
            batch = next(data_iter)
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)

            s2_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16):
                fwd = model.forward_with_routing_info(images)
                logits = fwd['logits']
                routing_infos = fwd['routing_infos']
                lb_loss = model.compute_total_load_balance_loss(routing_infos)
                seg_loss = criterion(logits, labels)
                total_loss = seg_loss + 1.0 * lb_loss

            if not torch.isfinite(total_loss):
                raise RuntimeError(f"Non-finite loss at Stage 2 batch {b_idx}")

            scaler.scale(total_loss).backward()
            scaler.step(s2_optimizer)
            scaler.update()

            if device.type == 'cuda':
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if b_idx > warmup_batches:
                s2_times.append(dt)

        s2_throughput = (batch_size / (sum(s2_times) / len(s2_times))) if s2_times else 0.0

        peak_alloc_gb = (torch.cuda.max_memory_allocated() / (1024 ** 3)) if device.type == 'cuda' else 0.0
        peak_res_gb = (torch.cuda.max_memory_reserved() / (1024 ** 3)) if device.type == 'cuda' else 0.0
        headroom_gb = (total_vram_mb / 1024.0) - peak_res_gb if device.type == 'cuda' else 0.0

        print(f"  [RESULT] BS={batch_size} PASS! Peak Alloc: {peak_alloc_gb:.2f} GB | Peak Res: {peak_res_gb:.2f} GB | Headroom: {headroom_gb:.2f} GB")
        print(f"  [SPEED]  Stage 1 Throughput: {s1_throughput:.2f} samples/s | Stage 2 Throughput: {s2_throughput:.2f} samples/s")

        return {
            'batch_size': batch_size,
            'status': 'PASS',
            'peak_alloc_gb': peak_alloc_gb,
            'peak_res_gb': peak_res_gb,
            'headroom_gb': headroom_gb,
            'headroom_pct': (headroom_gb / (total_vram_mb / 1024.0) * 100) if total_vram_mb > 0 else 0.0,
            's1_throughput': s1_throughput,
            's2_throughput': s2_throughput,
        }

    except torch.cuda.OutOfMemoryError as e:
        print(f"  [RESULT] BS={batch_size} FAILED with CUDA OutOfMemoryError (OOM)!")
        return {
            'batch_size': batch_size,
            'status': 'OOM',
            'peak_alloc_gb': total_vram_mb / 1024.0,
            'peak_res_gb': total_vram_mb / 1024.0,
            'headroom_gb': 0.0,
            'headroom_pct': 0.0,
            's1_throughput': 0.0,
            's2_throughput': 0.0,
        }
    except Exception as e:
        print(f"  [RESULT] BS={batch_size} FAILED with Error: {e}")
        return {
            'batch_size': batch_size,
            'status': f'ERROR: {str(e)[:30]}',
            'peak_alloc_gb': 0.0,
            'peak_res_gb': 0.0,
            'headroom_gb': 0.0,
            'headroom_pct': 0.0,
            's1_throughput': 0.0,
            's2_throughput': 0.0,
        }
    finally:
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Sweep Batch Sizes and measure throughput / VRAM on Baseline B2")
    parser.add_argument('--config', type=str, default='configs/p3_ablation/phase6_full_s1/a1_s2g_end_to_end.yaml')
    parser.add_argument('--data-root', type=str, default=None)
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[16, 20, 24, 28, 32])
    parser.add_argument('--num-batches', type=int, default=5)
    parser.add_argument('--warmup-batches', type=int, default=2)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.data_root:
        config['root_dir'] = args.data_root

    set_seed(int(config.get('seed', 42)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_banner(f"BATCH SIZE CAPACITY & THROUGHPUT SWEEP (GPU: {torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})")

    img_size = config.get('img_size', 448)
    try:
        train_dataset = get_dataset_from_config(config, split='train', image_size=img_size)
        print(f"[Dataset] Real Crack500 loaded with {len(train_dataset)} samples.")
    except Exception as e:
        print(f"[Dataset] Loading fallback synthetic dataset due to: {e}")
        class SyntheticDataset(torch.utils.data.Dataset):
            def __len__(self): return 500
            def __getitem__(self, idx):
                return {
                    'image': torch.randn(3, img_size, img_size),
                    'label': torch.randint(0, 2, (1, img_size, img_size)).float(),
                }
        train_dataset = SyntheticDataset()

    results = []
    for bs in args.batch_sizes:
        res = test_single_batch_size(
            config=config,
            batch_size=bs,
            train_dataset=train_dataset,
            device=device,
            num_batches=args.num_batches,
            warmup_batches=args.warmup_batches,
        )
        results.append(res)
        # If OOM, break early to save time
        if res['status'] == 'OOM':
            print(f"\n[INFO] Hit hardware VRAM ceiling at BS={bs}. Stopping further sweep.")
            break

    # Summary Table
    print_banner("BATCH SIZE SWEEP SUMMARY & THROUGHPUT COMPARISON")
    print(f"{'BS':<6} | {'Status':<6} | {'Peak Alloc':<11} | {'Peak Res':<10} | {'Headroom':<15} | {'S1 Throughput':<15} | {'S2 Throughput':<15}")
    print("-" * 95)
    for r in results:
        status = r['status']
        alloc = f"{r['peak_alloc_gb']:.2f} GB" if status == 'PASS' else "—"
        res_mem = f"{r['peak_res_gb']:.2f} GB" if status == 'PASS' else "—"
        headroom = f"{r['headroom_gb']:.2f} GB ({r['headroom_pct']:.1f}%)" if status == 'PASS' else "0 GB (OOM)"
        s1_speed = f"{r['s1_throughput']:.2f} s/s" if status == 'PASS' else "—"
        s2_speed = f"{r['s2_throughput']:.2f} s/s" if status == 'PASS' else "—"
        print(f"{r['batch_size']:<6} | {status:<6} | {alloc:<11} | {res_mem:<10} | {headroom:<15} | {s1_speed:<15} | {s2_speed:<15}")


if __name__ == '__main__':
    main()
