"""
B2 (Full SAGE-Lite) Real-Data Runtime Preflight & Benchmark Script for Crack500

Purpose:
Verify that batch size 12 runs safely, stably, and without OOM under the ACTUAL
Crack500 dataset pipeline (ConfigurableMedicalDataset, Albumentations transforms,
smart filter fg_pixels >= 20, random 448x448 crops) rather than synthetic tensors.

Verifications:
1. Real Data Pipeline Integration:
   - Uses exact ConfigurableMedicalDataset and DataLoader settings from Crack500 train.
   - Preserves all canonical preprocessing and data augmentations.
2. Exact B2 Architecture & Optimizer:
   - B2 ConvNeXtV2-ViT Hybrid UNet (Depth 12, 16 injected routers, Top-K=4).
   - Training mode enabled (exploration noise ON, expert dropout ON).
   - AMP FP16 forward_with_routing_info -> CrackBinaryLoss + LB loss -> backward -> optimizer.step.
3. Rigorous Profiling & Throughput Benchmarking:
   - CUDA synchronization before and after every batch.
   - Skips first 2 batches (warmup / autotuning) for steady-state throughput calculation.
   - Reports per-batch time, steady-state mean/median, samples/sec.
   - Checks finite values for logits, losses, and gradients on every batch.
   - Gathers routing statistics across all 16 experts on real image patches.

Usage:
    python scripts/preflight_b2_realdata.py --config configs/b2_crack500_depth12.yaml --batch-size 12 --num-batches 12
"""

import argparse
import gc
import os
import statistics
import sys
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure sage_lite root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sage.networks.b2_unet import create_b2_unet
from sage.utils.dataloader import get_dataset_from_config
from sage.utils.training_utils import seed_worker, set_seed
from scripts.train_crack import get_optimizer_groups, CrackBinaryLoss


def print_banner(text: str, ch: str = "="):
    line = ch * 70
    print(f"\n{line}\n{text}\n{line}")


def run_realdata_preflight(
    config_path: str,
    data_root_override: str = None,
    batch_size: int = 12,
    num_batches: int = 12,
    warmup_batches: int = 2,
    num_workers: int = 2,
):
    print_banner(f"B2 REAL-DATA RUNTIME PREFLIGHT (CRACK500)")
    print(f"[Config Path]  : {config_path}")
    print(f"[Target Batch] : {batch_size}")
    print(f"[Total Batches]: {num_batches} (Warmup: {warmup_batches}, Measured: {num_batches - warmup_batches})")

    # -------------------------------------------------------------------------
    # 1. Load Configuration
    # -------------------------------------------------------------------------
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if data_root_override:
        config['root_dir'] = data_root_override
        print(f"[Data Root Override]: {data_root_override}")

    seed = int(config.get('seed', 42))
    set_seed(seed)

    # -------------------------------------------------------------------------
    # 2. Hardware & Device Setup
    # -------------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Device] Target Device: {device}")

    if device.type == 'cuda':
        dev_name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        total_vram_gb = total_vram_mb / 1024.0

        if '1650' in dev_name or '1660' in dev_name:
            torch.backends.cudnn.enabled = False
            print(f"[Device] Detected {dev_name}. Set cudnn.enabled=False for FP16 stability.")
        else:
            torch.backends.cudnn.benchmark = True

        print(f"[Device] GPU: {dev_name} | Compute Cap: {cap[0]}.{cap[1]} | Total VRAM: {total_vram_mb:.1f} MB ({total_vram_gb:.2f} GB)")
        print(f"[Device] cuDNN: {torch.backends.cudnn.version()} (enabled={torch.backends.cudnn.enabled}, benchmark={torch.backends.cudnn.benchmark})")
    else:
        total_vram_mb = 0.0
        total_vram_gb = 0.0
        print("[Device] WARNING: Running on CPU! VRAM metrics will not be measured.")

    # -------------------------------------------------------------------------
    # 3. Real Crack500 Dataset & DataLoader Setup
    # -------------------------------------------------------------------------
    img_size = config.get('img_size', 448)
    print_banner(f"LOADING REAL CRACK500 DATASET (Image Size: {img_size}x{img_size})")

    # Use exact dataset pipeline from config (ConfigurableMedicalDataset)
    train_dataset = get_dataset_from_config(config, split='train', image_size=img_size)
    print(f"[Dataset] Train samples count: {len(train_dataset)}")
    print(f"[Dataset] Preprocessing mode: crop_mode='{getattr(train_dataset, 'crop_mode', 'random')}', smart_filter={getattr(train_dataset, 'use_smart_filter', True)}")

    g = torch.Generator()
    g.manual_seed(seed)

    nw = num_workers if num_workers is not None else int(config.get('num_workers', 2))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=(device.type == 'cuda'),
        drop_last=False,
    )
    print(f"[DataLoader] Batch size: {batch_size} | Num workers: {nw} | Total available batches: {len(train_loader)}")

    # -------------------------------------------------------------------------
    # 4. Instantiate Model B2 & Training Components
    # -------------------------------------------------------------------------
    vit_depth = int(config.get('num_transformer_layers', 12))
    sage_cfg = config.get('sage_config', {})
    base_lr = float(config.get('lr', 1e-4))
    lr_backbone = base_lr * 0.1
    lr_decoder = base_lr
    lr_sage = base_lr

    print_banner("INSTANTIATING B2 MODEL (Full SAGE-Lite)")
    model = create_b2_unet(
        num_transformer_layers=vit_depth,
        pretrained=True,
        sage_config=sage_cfg,
    ).to(device)
    model.train()  # Exploration noise ON, dropout ON

    info = model.get_model_info()
    print(f"[Model] Name:                 {info['model_name']}")
    print(f"[Model] ViT Depth:            {info['num_transformer_layers']}")
    print(f"[Model] Injected Routers:     {info['num_injected_routers']}")
    print(f"[Model] Expert Pool Size:     {info['expert_pool_size']}")
    print(f"[Model] Top-K:                {info['top_k']}")
    print(f"[Model] Fusion:               {info['fusion_type']} (scale={info['residual_scale']})")

    param_groups = get_optimizer_groups(model, lr_backbone=lr_backbone, lr_decoder=lr_decoder, lr_sage=lr_sage, weight_decay=0.05)
    optimizer = torch.optim.AdamW(param_groups)
    criterion = CrackBinaryLoss()
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda'))

    # -------------------------------------------------------------------------
    # 5. Execute Real-Data Benchmark Loop
    # -------------------------------------------------------------------------
    print_banner(f"RUNNING BENCHMARK ({num_batches} Batches on Real Crack500)")

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    batch_metrics = []
    oom_occurred = False
    data_iter = iter(train_loader)

    try:
        for b_idx in range(1, num_batches + 1):
            # Fetch real batch
            t_data_0 = time.perf_counter()
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            t_data_1 = time.perf_counter()
            data_fetch_time = t_data_1 - t_data_0

            # Step synchronization before timing computation
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_step_0 = time.perf_counter()

            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16):
                forward_out = model.forward_with_routing_info(images)
                logits = forward_out['logits']
                routing_infos = forward_out['routing_infos']
                lb_loss = model.compute_total_load_balance_loss(routing_infos)
                seg_loss = criterion(logits, labels)
                total_loss = seg_loss + 1.0 * lb_loss

            # Strict finite checks
            if not torch.isfinite(logits).all():
                raise RuntimeError(f"Batch {b_idx}: Non-finite logits detected!")
            if not torch.isfinite(total_loss):
                raise RuntimeError(f"Batch {b_idx}: Non-finite total loss detected: {total_loss.item()}")

            scaler.scale(total_loss).backward()

            # Gradient check
            for name, param in model.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    raise RuntimeError(f"Batch {b_idx}: Non-finite gradient in {name}!")

            scaler.step(optimizer)
            scaler.update()

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_step_1 = time.perf_counter()
            step_time = t_step_1 - t_step_0

            # VRAM stats
            if device.type == 'cuda':
                cur_alloc_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                cur_peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                cur_peak_res_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
                free_mb = total_vram_mb - cur_peak_res_mb
            else:
                cur_alloc_mb = cur_peak_alloc_mb = cur_peak_res_mb = free_mb = 0.0

            is_warmup = (b_idx <= warmup_batches)
            tag = "[WARMUP]" if is_warmup else "[MEASURED]"

            print(
                f"  Batch {b_idx:02d}/{num_batches:02d} {tag} | "
                f"Step: {step_time:6.2f}s (Data: {data_fetch_time:4.2f}s) | "
                f"Loss: {total_loss.item():.4f} (Seg: {seg_loss.item():.4f}, LB: {lb_loss.item():.4f}) | "
                f"Peak Alloc: {cur_peak_alloc_mb/1024:5.2f} GB | "
                f"Peak Res: {cur_peak_res_mb/1024:5.2f} GB | "
                f"Free: {free_mb/1024:4.2f} GB"
            )

            batch_metrics.append({
                'batch_idx': b_idx,
                'is_warmup': is_warmup,
                'step_time': step_time,
                'data_fetch_time': data_fetch_time,
                'loss': total_loss.item(),
                'seg_loss': seg_loss.item(),
                'lb_loss': lb_loss.item(),
                'peak_alloc_mb': cur_peak_alloc_mb,
                'peak_res_mb': cur_peak_res_mb,
                'free_mb': free_mb,
            })

    except torch.cuda.OutOfMemoryError as e:
        oom_occurred = True
        print_banner("OOM OCCURRED DURING REAL-DATA PREFLIGHT", ch="!")
        print(f"CUDA OutOfMemoryError caught at batch {b_idx}:\n{e}")
        print(f"Conclusion: Batch size {batch_size} cannot safely run on {device} with real Crack500 data.")
        return False, None

    # -------------------------------------------------------------------------
    # 6. Steady-State Throughput Analysis
    # -------------------------------------------------------------------------
    print_banner("THROUGHPUT & VRAM BENCHMARK RESULTS")

    measured_batches = [m for m in batch_metrics if not m['is_warmup']]
    step_times = [m['step_time'] for m in measured_batches]

    mean_step_time = statistics.mean(step_times) if step_times else 0.0
    median_step_time = statistics.median(step_times) if step_times else 0.0
    min_step_time = min(step_times) if step_times else 0.0
    max_step_time = max(step_times) if step_times else 0.0
    samples_per_sec = (batch_size / mean_step_time) if mean_step_time > 0 else 0.0

    final_peak_alloc = batch_metrics[-1]['peak_alloc_mb']
    final_peak_res = batch_metrics[-1]['peak_res_mb']
    final_free = batch_metrics[-1]['free_mb']

    print(f"  Execution Status:          PASS (0 OOM errors across {num_batches} batches)")
    print(f"  Finite Gradients & Losses: PASS (All strictly finite)")
    print(f"\n  --- VRAM Profiling (Batch Size = {batch_size}) ---")
    print(f"  Peak VRAM Allocated:       {final_peak_alloc:8.1f} MB ({final_peak_alloc / 1024:.2f} GB)")
    print(f"  Peak VRAM Reserved:        {final_peak_res:8.1f} MB ({final_peak_res / 1024:.2f} GB) / {total_vram_gb:.2f} GB")
    print(f"  VRAM Free Remaining:       {final_free:8.1f} MB ({final_free / 1024:.2f} GB)")
    print(f"  VRAM Utilization Ratio:    {(final_peak_res / total_vram_mb * 100):.1f}%")

    print(f"\n  --- Throughput Timing (Excluding First {warmup_batches} Warmup Batches) ---")
    print(f"  Measured Batches Count:    {len(measured_batches)}")
    print(f"  Mean Step Time:            {mean_step_time:.3f} s / batch")
    print(f"  Median Step Time:          {median_step_time:.3f} s / batch")
    print(f"  Min / Max Step Time:       {min_step_time:.3f} s / {max_step_time:.3f} s")
    print(f"  Processing Throughput:     {samples_per_sec:.2f} samples / second")

    # -------------------------------------------------------------------------
    # 7. Routing Usage Inspection on Real Data
    # -------------------------------------------------------------------------
    print_banner("ROUTING USAGE DIAGNOSTICS (Real Crack500 Batches)")
    usage_stats = model.get_expert_usage_statistics()
    all_usage = [0.0] * info['expert_pool_size']

    for group_key in ['cnn_stages', 'transformer_blocks']:
        for r_stat in usage_stats.get(group_key, []):
            counts = r_stat.get('router_stats', {}).get('expert_usage_count', [])
            for e_idx, c in enumerate(counts):
                if e_idx < len(all_usage):
                    all_usage[e_idx] += c

    total_selections = sum(all_usage)
    zero_selection_experts = []

    print("  --- Per-Expert Selection Breakdown ---")
    for e_idx, count in enumerate(all_usage):
        pct = (count / total_selections * 100.0) if total_selections > 0 else 0.0
        expert_tag = "[SHARED]" if e_idx in sage_cfg.get('shared_expert_indices', [0, 1, 2, 3]) else "        "
        name = f"CNN Stage {e_idx}" if e_idx < 4 else f"ViT Block {e_idx - 4:02d}"
        print(f"  Expert {e_idx:02d} ({name}) {expert_tag}: {int(count):6d} selections ({pct:5.1f}%)")
        if count == 0:
            zero_selection_experts.append(e_idx)

    print(f"\n  Total Routing Selection Events: {int(total_selections)}")
    if zero_selection_experts:
        print(f"  [WARN] Inactive Experts Detected: {zero_selection_experts}")
    else:
        print("  [PASS] All 16 experts actively received routing assignments on real data.")

    # -------------------------------------------------------------------------
    # 8. Final Verdict
    # -------------------------------------------------------------------------
    print_banner("FINAL REAL-DATA PREFLIGHT VERDICT")
    if not oom_occurred and final_free > 200.0:
        print(f"  [VERDICT] PASS: Batch Size {batch_size} is FEASIBLE on real Crack500 data!")
        print(f"  Remaining VRAM Headroom: {final_free/1024:.2f} GB ({(final_free / total_vram_mb * 100):.1f}%)")
        print(f"  Steady-State Speed:      {samples_per_sec:.2f} samples/s ({mean_step_time:.2f}s/batch)")
    else:
        print(f"  [VERDICT] MARGINAL / RISKY: Batch Size {batch_size} has very little headroom ({final_free/1024:.2f} GB).")
        print(f"  Recommendation: Consider batch_size=10 for safer continuous training without risk of OOM during validation.")

    return True, batch_metrics


def main():
    parser = argparse.ArgumentParser(description="Real-data Preflight & Benchmark for B2 on Crack500")
    parser.add_argument('--config', type=str, default='configs/b2_crack500_depth12.yaml', help='Path to config YAML')
    parser.add_argument('--data-root', type=str, default=None, help='Override dataset root_dir (e.g. for local or custom path)')
    parser.add_argument('--batch-size', type=int, default=12, help='Batch size to benchmark (default: 12)')
    parser.add_argument('--num-batches', type=int, default=12, help='Total training batches to run (default: 12)')
    parser.add_argument('--warmup-batches', type=int, default=2, help='Warmup batches to exclude from throughput calculation (default: 2)')
    parser.add_argument('--num-workers', type=int, default=2, help='DataLoader workers (default: 2)')
    args = parser.parse_args()

    success, _ = run_realdata_preflight(
        config_path=args.config,
        data_root_override=args.data_root,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        warmup_batches=args.warmup_batches,
        num_workers=args.num_workers,
    )
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
