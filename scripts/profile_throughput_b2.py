"""
Coarse Throughput Profiler for SAGE-Lite B2 on Crack500

Measures:
1. Data Wait (DataLoader stall): time.perf_counter() around next(loader_iter)
2. Forward Pass: CUDA Event (forward_with_routing_info + loss)
3. Backward Pass: CUDA Event (scaler.scale(loss).backward())
4. Optimizer Step: CUDA Event (scaler.step + scaler.update)
5. Peak VRAM & CPU/GPU utilization
6. Parameterized comparison across num_workers = [0, 2, 4]

Usage:
    python scripts/profile_throughput_b2.py --config configs/b2_crack500_depth12.yaml
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional
import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sage.networks.b2_unet import create_b2_unet
from sage.utils.dataloader import get_dataset_from_config
from sage.utils.training_utils import seed_worker, set_seed
from scripts.train_crack import get_optimizer_groups, create_stage2_optimizer, CrackBinaryLoss

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def print_banner(text: str, ch: str = "="):
    line = ch * 70
    print(f"\n{line}\n{text}\n{line}")


def resolve_data_root(config_root: str) -> str:
    if os.path.exists(config_root):
        return config_root
    # Fallback to local workspace dataset if available
    local_candidates = [
        os.path.join(project_root, "..", "datasets", "canonical_benchmarks", "crack500_and_deepcrack", "CRACK500"),
        os.path.join(project_root, "datasets", "canonical_benchmarks", "crack500_and_deepcrack", "CRACK500"),
    ]
    for cand in local_candidates:
        cand_abs = os.path.abspath(cand)
        if os.path.exists(cand_abs):
            return cand_abs
    return config_root


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, length: int = 158 * 12, image_size: int = 448):
        self.length = length
        self.image_size = image_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            'image': torch.randn(3, self.image_size, self.image_size),
            'label': torch.randint(0, 2, (1, self.image_size, self.image_size)).float()
        }


def profile_worker_setting(
    config: dict,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    num_workers: int,
    batch_size: int,
    num_batches: int = 12,
    warmup_batches: int = 2,
    use_synthetic: bool = False,
) -> Dict[str, float]:
    img_size = int(config.get('img_size', 448))
    seed = int(config.get('seed', 42))

    # Build dataset and dataloader
    if use_synthetic:
        print("[DataLoader] Using SyntheticDataset for benchmarking")
        train_dataset = SyntheticDataset(length=max(100, num_batches * batch_size * 2), image_size=img_size)
    else:
        try:
            train_dataset = get_dataset_from_config(config, split='train', image_size=img_size)
        except FileNotFoundError as e:
            print(f"[DataLoader Warning] Real dataset not found ({e}). Falling back to SyntheticDataset.")
            train_dataset = SyntheticDataset(length=max(100, num_batches * batch_size * 2), image_size=img_size)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker if not use_synthetic else None,
        generator=g,
        pin_memory=(device.type == 'cuda'),
        drop_last=False,
    )


    loader_iter = iter(train_loader)
    measured_count = num_batches - warmup_batches

    data_wait_times = []
    forward_times = []
    backward_times = []
    optimizer_times = []
    step_times = []
    cpu_utils = []

    start_fwd = torch.cuda.Event(enable_timing=True)
    end_fwd = torch.cuda.Event(enable_timing=True)
    start_bwd = torch.cuda.Event(enable_timing=True)
    end_bwd = torch.cuda.Event(enable_timing=True)
    start_opt = torch.cuda.Event(enable_timing=True)
    end_opt = torch.cuda.Event(enable_timing=True)

    print(f"\n--- Profiling with num_workers = {num_workers} ({num_batches} batches: {warmup_batches} warmup, {measured_count} measured) ---")

    for i in range(1, num_batches + 1):
        if HAS_PSUTIL:
            cpu_utils.append(psutil.cpu_percent(interval=None))

        # 1. Measure Data Wait time on CPU (NO cuda synchronize here)
        t_data_start = time.perf_counter()
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)
        t_data_end = time.perf_counter()
        data_wait_ms = (t_data_end - t_data_start) * 1000.0

        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # 2. Measure Forward Pass
        start_fwd.record()
        with torch.amp.autocast('cuda'):
            forward_out = model.forward_with_routing_info(images)
            logits = forward_out['logits']
            routing_infos = forward_out['routing_infos']
            lb_loss = model.compute_total_load_balance_loss(routing_infos)
            seg_loss = criterion(logits, labels)
            loss = seg_loss + 1.0 * lb_loss
        end_fwd.record()

        # 3. Measure Backward Pass
        start_bwd.record()
        scaler.scale(loss).backward()
        end_bwd.record()

        # 4. Measure Optimizer Step
        start_opt.record()
        scaler.step(optimizer)
        scaler.update()
        end_opt.record()

        # Final synchronize to collect CUDA event durations
        torch.cuda.synchronize()

        fwd_ms = start_fwd.elapsed_time(end_fwd)
        bwd_ms = start_bwd.elapsed_time(end_bwd)
        opt_ms = start_opt.elapsed_time(end_opt)
        total_step_ms = data_wait_ms + fwd_ms + bwd_ms + opt_ms

        is_warmup = (i <= warmup_batches)
        tag = "[WARMUP]" if is_warmup else "[MEASURED]"

        if not is_warmup:
            data_wait_times.append(data_wait_ms)
            forward_times.append(fwd_ms)
            backward_times.append(bwd_ms)
            optimizer_times.append(opt_ms)
            step_times.append(total_step_ms)

        print(
            f"  Batch {i:02d}/{num_batches:02d} {tag:10s} | "
            f"DataWait: {data_wait_ms:6.1f}ms | "
            f"Fwd: {fwd_ms:6.1f}ms | "
            f"Bwd: {bwd_ms:6.1f}ms | "
            f"Opt: {opt_ms:5.1f}ms | "
            f"Step: {total_step_ms:6.1f}ms | "
            f"Loss: {loss.item():.4f}"
        )

    # Compute statistics
    mean_wait = float(np.mean(data_wait_times))
    mean_fwd = float(np.mean(forward_times))
    mean_bwd = float(np.mean(backward_times))
    mean_opt = float(np.mean(optimizer_times))
    mean_step = float(np.mean(step_times))
    std_step = float(np.std(step_times))

    throughput_sps = (batch_size / (mean_step / 1000.0)) if mean_step > 0 else 0.0
    est_epoch_sec = (158 * mean_step) / 1000.0
    est_epoch_min = est_epoch_sec / 60.0

    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == 'cuda' else 0.0
    avg_cpu = float(np.mean(cpu_utils)) if cpu_utils else 0.0

    return {
        'num_workers': num_workers,
        'data_wait_ms': mean_wait,
        'forward_ms': mean_fwd,
        'backward_ms': mean_bwd,
        'optimizer_ms': mean_opt,
        'total_step_ms': mean_step,
        'total_step_std': std_step,
        'throughput_sps': throughput_sps,
        'est_epoch_min': est_epoch_min,
        'peak_vram_mb': peak_vram_mb,
        'avg_cpu_percent': avg_cpu,
    }


def main():
    parser = argparse.ArgumentParser(description="Coarse Throughput Profiler for SAGE-Lite B2")
    parser.add_argument('--config', type=str, required=True, help="Path to config YAML")
    parser.add_argument('--workers', type=str, default="0,2,4", help="Comma-separated list of num_workers to test")
    parser.add_argument('--batch-size', type=int, default=None, help="Batch size override (default: from config)")
    parser.add_argument('--batches', type=int, default=12, help="Total batches per setting (default: 12)")
    parser.add_argument('--warmup', type=int, default=2, help="Warmup batches (default: 2)")

    parser.add_argument('--data-root', type=str, default=None, help="Override root_dir in config")
    parser.add_argument('--synthetic', action='store_true', help="Force use of synthetic data for testing")

    args = parser.parse_args()

    print_banner("COARSE THROUGHPUT PROFILER (SAGE-Lite B2)")

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Resolve data root
    if args.data_root:
        resolved_root = args.data_root
    else:
        resolved_root = resolve_data_root(config.get('root_dir', ''))
    config['root_dir'] = resolved_root
    print(f"[Config] File: {args.config}")
    print(f"[Config] Resolved Data Root: {resolved_root}")
    if args.synthetic:
        print("[Config] Synthetic Mode: ENABLED")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] Target: {device}")
    if device.type == 'cuda':
        dev_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[Device] GPU: {dev_name} ({vram_total:.2f} GB)")
        if '1650' in dev_name or '1660' in dev_name:
            torch.backends.cudnn.enabled = False
            print("[Device] Set torch.backends.cudnn.enabled = False for GTX 1650/1660 FP16 stability.")


    # Model parameters
    vit_depth = int(config.get('num_transformer_layers', 12))
    sage_cfg = config.get('sage_config', {})
    batch_size = args.batch_size if args.batch_size is not None else int(config.get('batch_size', 12))

    base_lr = float(config.get('lr', 1e-4))
    stage2_base_lr = float(config.get('stage2_base_lr', base_lr))
    stage2_shared_lr = float(config.get('stage2_shared_lr', base_lr))

    print(f"[Model Config] Model: B2 | ViT Depth: {vit_depth} | Batch Size: {batch_size}")
    print(f"[Model Config] SAGE: Top-K={sage_cfg.get('top_k', 4)}, Gating={sage_cfg.get('gating_type', 'sigmoid')}")

    # Instantiate model
    model = create_b2_unet(
        num_transformer_layers=vit_depth,
        pretrained=True,
        sage_config=sage_cfg,
    ).to(device)
    model.train()

    # Configure Stage 2 optimizer
    model.set_shared_experts([0, 1, 2, 3])
    optimizer = create_stage2_optimizer(model, stage2_base_lr=stage2_base_lr, stage2_shared_lr=stage2_shared_lr)
    criterion = CrackBinaryLoss()
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda'))

    workers_to_test = [int(w.strip()) for w in args.workers.split(',')]

    results = []
    for nw in workers_to_test:
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        res = profile_worker_setting(
            config=config,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            num_workers=nw,
            batch_size=batch_size,
            num_batches=args.batches,
            warmup_batches=args.warmup,
            use_synthetic=args.synthetic,
        )

        results.append(res)

    # -------------------------------------------------------------------------
    # Summary Report
    # -------------------------------------------------------------------------
    print_banner("COARSE THROUGHPUT PROFILING SUMMARY REPORT")
    print(f"{'Workers':^7} | {'DataWait':^10} | {'Forward':^10} | {'Backward':^10} | {'Optimizer':^10} | {'TotalStep':^12} | {'Thpt (img/s)':^12} | {'Est 1 Epoch':^12} | {'Peak VRAM':^10}")
    print("-" * 105)

    for r in results:
        dw_pct = (r['data_wait_ms'] / r['total_step_ms']) * 100.0 if r['total_step_ms'] > 0 else 0
        fwd_pct = (r['forward_ms'] / r['total_step_ms']) * 100.0 if r['total_step_ms'] > 0 else 0
        bwd_pct = (r['backward_ms'] / r['total_step_ms']) * 100.0 if r['total_step_ms'] > 0 else 0

        print(
            f"{r['num_workers']:^7d} | "
            f"{r['data_wait_ms']:7.1f}ms  | "
            f"{r['forward_ms']:7.1f}ms  | "
            f"{r['backward_ms']:7.1f}ms  | "
            f"{r['optimizer_ms']:7.1f}ms  | "
            f"{r['total_step_ms']:7.1f} +/- {r['total_step_std']:4.1f}ms | "

            f"{r['throughput_sps']:10.2f}   | "
            f"{r['est_epoch_min']:9.2f} min | "
            f"{r['peak_vram_mb']:7.1f} MB"
        )

    print("\n[Breakdown Percentages (% of Total Step)]")
    for r in results:
        tot = r['total_step_ms']
        print(
            f"  Workers={r['num_workers']}: DataWait={r['data_wait_ms']/tot*100:4.1f}% | "
            f"Forward={r['forward_ms']/tot*100:4.1f}% | "
            f"Backward={r['backward_ms']/tot*100:4.1f}% | "
            f"Optimizer={r['optimizer_ms']/tot*100:4.1f}% | "
            f"CPU Usage={r['avg_cpu_percent']:.1f}%"
        )

    print_banner("PROFILING COMPLETE")


if __name__ == '__main__':
    main()
