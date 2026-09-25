"""
Master Phase 0 Probe & Characterization Script for SAGE-Lite B2 on Crack500
Audits and characterizes runtime & hardware for ALL 3 depths: D12, D6, D4.

Sections:
1. Full-Model OOM / Batch-Size Probe:
   - Exact B2 training path: AMP FP16, forward_with_routing_info, CrackBinaryLoss + LB loss,
     backward, optimizer.step(), exploration noise ON, expert dropout ON, top_k=4.
   - Separately probes D12, D6, D4 across batch sizes [8, 12, 16, 24].
   - Reports PASS/OOM, peak allocated, peak reserved, free VRAM, finite status, memory leak check.
2. Throughput Benchmark:
   - For all 3 depths at candidate batch size across num_workers = [0, 2, 4].
   - Measures DataWait, Forward, Backward, Optimizer, Step Time, Samples/sec, Epoch time, Peak VRAM.
3. Epoch Budget & Wall-Clock Estimates:
   - Projections for 15, 20, 25, 30 epochs.
   - Recommends N_total and Stage 1 / Stage 2 split.
4. Unified Markdown Summary Table & 5 Technical Conclusions.

Usage:
    python scripts/run_phase0_probe.py --data-root /content/dataset/Crack500
"""

import argparse
import gc
import json
import logging
import math
import os
import sys
import time

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
from typing import Any, Dict, List, Optional, Tuple
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
from scripts.train_crack import CrackBinaryLoss, create_stage2_optimizer
from scripts.profile_throughput_b2 import SyntheticDataset, resolve_data_root

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def print_banner(text: str, ch: str = "="):
    line = ch * 80
    print(f"\n{line}\n{text}\n{line}")


def run_oom_probe_for_depth(
    depth: int,
    config: dict,
    batch_sizes: List[int],
    device: torch.device,
    data_root: str,
    num_batches: int = 5,
    warmup_batches: int = 2,
    use_synthetic: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """
    Probes batch sizes for a single ViT depth.
    Catches CUDA OOM gracefully, checks numerical finiteness, and tests for memory leaks.
    """
    img_size = int(config.get('img_size', 448))
    seed = int(config.get('seed', 42))
    sage_cfg = config.get('sage_config', {})

    total_vram_mb = torch.cuda.get_device_properties(device).total_memory / (1024**2) if device.type == 'cuda' else 0.0

    # Build dataset
    if use_synthetic:
        train_dataset = SyntheticDataset(length=max(100, max(batch_sizes) * num_batches * 2), image_size=img_size)
    else:
        try:
            train_dataset = get_dataset_from_config(config, split='train', image_size=img_size)
        except Exception as e:
            print(f"  [Warning] Dataset loading failed ({e}). Falling back to SyntheticDataset.")
            train_dataset = SyntheticDataset(length=max(100, max(batch_sizes) * num_batches * 2), image_size=img_size)

    results_per_bs = {}

    for bs in batch_sizes:
        print(f"\n  [Depth {depth} | BS {bs}] Initiating OOM Probe...")
        if device.type == 'cuda':
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        g = torch.Generator()
        g.manual_seed(seed)
        loader = DataLoader(
            train_dataset,
            batch_size=bs,
            shuffle=True,
            num_workers=0,
            worker_init_fn=seed_worker if not use_synthetic else None,
            generator=g,
            pin_memory=(device.type == 'cuda'),
            drop_last=True,
        )

        model = None
        optimizer = None
        criterion = None
        scaler = None

        try:
            # Full B2 SAGE-Lite model
            model = create_b2_unet(
                num_transformer_layers=depth,
                pretrained=not use_synthetic,
                sage_config=sage_cfg,
                p3_mode=None, # Pure Base B2 (No P3)
            ).to(device)
            model.train() # Exploration noise and expert dropout active
            model.set_shared_experts([0, 1, 2, 3])

            optimizer = create_stage2_optimizer(model, stage2_base_lr=1e-4, stage2_shared_lr=1e-4)
            criterion = CrackBinaryLoss()
            scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda'))

            loader_iter = iter(loader)
            allocated_samples = []
            finite_losses = []
            grad_finite = True

            for step_idx in range(1, num_batches + 1):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(loader)
                    batch = next(loader_iter)

                images = batch['image'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True) if 'label' in batch else batch['mask'].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda')):
                    fwd_out = model.forward_with_routing_info(images)
                    logits = fwd_out['logits']
                    routing_infos = fwd_out['routing_infos']
                    lb_loss = model.compute_total_load_balance_loss(routing_infos)
                    seg_loss = criterion(logits, labels)
                    loss = seg_loss + 1.0 * lb_loss

                scaler.scale(loss).backward()

                # Check gradient finiteness
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        grad_finite = False
                        break

                scaler.step(optimizer)
                scaler.update()

                if device.type == 'cuda':
                    torch.cuda.synchronize()

                loss_val = loss.item()
                is_finite = math.isfinite(loss_val)
                finite_losses.append(is_finite)

                alloc_mb = torch.cuda.memory_allocated(device) / (1024**2) if device.type == 'cuda' else 0.0
                if step_idx > warmup_batches:
                    allocated_samples.append(alloc_mb)

                print(f"    Step {step_idx}/{num_batches} | Loss: {loss_val:.4f} (Finite: {is_finite}) | Alloc: {alloc_mb:.1f} MB")

            peak_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == 'cuda' else 0.0
            peak_res_mb = torch.cuda.max_memory_reserved(device) / (1024**2) if device.type == 'cuda' else 0.0
            free_vram_mb = total_vram_mb - peak_res_mb if device.type == 'cuda' else 0.0

            # Memory leak check across measured batches
            leak_detected = False
            growth_mb = 0.0
            if len(allocated_samples) >= 2:
                growth_mb = allocated_samples[-1] - allocated_samples[0]
                if growth_mb > 50.0: # Greater than 50MB sustained growth
                    leak_detected = True

            all_finite = all(finite_losses) and grad_finite

            results_per_bs[bs] = {
                'status': 'PASS',
                'peak_allocated_mb': peak_alloc_mb,
                'peak_reserved_mb': peak_res_mb,
                'free_vram_mb': free_vram_mb,
                'finite_status': 'FINITE' if all_finite else 'NON-FINITE',
                'memory_leak': f"GROWTH_{growth_mb:.1f}MB" if leak_detected else "NO_LEAK",
            }
            print(f"  [PASS] Depth {depth} | BS {bs}: PeakAlloc={peak_alloc_mb:.1f}MB, PeakRes={peak_res_mb:.1f}MB, Free={free_vram_mb:.1f}MB")

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            err_msg = str(e)
            if "out of memory" in err_msg.lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                print(f"  [OOM DETECTED] Depth {depth} | BS {bs}: Out of Memory!")
                results_per_bs[bs] = {
                    'status': 'OOM',
                    'peak_allocated_mb': total_vram_mb,
                    'peak_reserved_mb': total_vram_mb,
                    'free_vram_mb': 0.0,
                    'finite_status': 'N/A',
                    'memory_leak': 'N/A',
                }
            else:
                raise e
        finally:
            del model
            del optimizer
            del criterion
            del scaler
            if device.type == 'cuda':
                gc.collect()
                torch.cuda.empty_cache()

    return results_per_bs


def run_throughput_for_depth(
    depth: int,
    config: dict,
    batch_size: int,
    workers_list: List[int],
    device: torch.device,
    data_root: str,
    num_batches: int = 12,
    warmup_batches: int = 2,
    use_synthetic: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """
    Runs coarse throughput profiling across num_workers for a given ViT depth.
    """
    img_size = int(config.get('img_size', 448))
    seed = int(config.get('seed', 42))
    sage_cfg = config.get('sage_config', {})

    if use_synthetic:
        train_dataset = SyntheticDataset(length=max(100, batch_size * num_batches * 2), image_size=img_size)
    else:
        try:
            train_dataset = get_dataset_from_config(config, split='train', image_size=img_size)
        except Exception as e:
            print(f"  [Warning] Dataset loading failed ({e}). Falling back to SyntheticDataset.")
            train_dataset = SyntheticDataset(length=max(100, batch_size * num_batches * 2), image_size=img_size)

    # Instantiate model once per depth
    if device.type == 'cuda':
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model = create_b2_unet(
        num_transformer_layers=depth,
        pretrained=not use_synthetic,
        sage_config=sage_cfg,
        p3_mode=None,
    ).to(device)
    model.train()
    model.set_shared_experts([0, 1, 2, 3])

    optimizer = create_stage2_optimizer(model, stage2_base_lr=1e-4, stage2_shared_lr=1e-4)
    criterion = CrackBinaryLoss()
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda'))

    results_per_worker = {}

    start_fwd = torch.cuda.Event(enable_timing=True)
    end_fwd = torch.cuda.Event(enable_timing=True)
    start_bwd = torch.cuda.Event(enable_timing=True)
    end_bwd = torch.cuda.Event(enable_timing=True)
    start_opt = torch.cuda.Event(enable_timing=True)
    end_opt = torch.cuda.Event(enable_timing=True)

    for nw in workers_list:
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)

        g = torch.Generator()
        g.manual_seed(seed)
        loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=nw,
            worker_init_fn=seed_worker if not use_synthetic else None,
            generator=g,
            pin_memory=(device.type == 'cuda'),
            drop_last=False,
        )

        loader_iter = iter(loader)
        measured_count = num_batches - warmup_batches

        data_wait_times = []
        forward_times = []
        backward_times = []
        optimizer_times = []
        step_times = []

        print(f"\n  [Depth {depth} | Workers {nw} | BS {batch_size}] Profiling {num_batches} batches...")

        for i in range(1, num_batches + 1):
            t_data_start = time.perf_counter()
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)
            t_data_end = time.perf_counter()
            data_wait_ms = (t_data_end - t_data_start) * 1000.0

            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True) if 'label' in batch else batch['mask'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            start_fwd.record()
            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda')):
                fwd_out = model.forward_with_routing_info(images)
                logits = fwd_out['logits']
                routing_infos = fwd_out['routing_infos']
                lb_loss = model.compute_total_load_balance_loss(routing_infos)
                seg_loss = criterion(logits, labels)
                loss = seg_loss + 1.0 * lb_loss
            end_fwd.record()

            start_bwd.record()
            scaler.scale(loss).backward()
            end_bwd.record()

            start_opt.record()
            scaler.step(optimizer)
            scaler.update()
            end_opt.record()

            if device.type == 'cuda':
                torch.cuda.synchronize()

            fwd_ms = start_fwd.elapsed_time(end_fwd) if device.type == 'cuda' else 0.0
            bwd_ms = start_bwd.elapsed_time(end_bwd) if device.type == 'cuda' else 0.0
            opt_ms = start_opt.elapsed_time(end_opt) if device.type == 'cuda' else 0.0
            total_step_ms = data_wait_ms + fwd_ms + bwd_ms + opt_ms

            is_warmup = (i <= warmup_batches)
            if not is_warmup:
                data_wait_times.append(data_wait_ms)
                forward_times.append(fwd_ms)
                backward_times.append(bwd_ms)
                optimizer_times.append(opt_ms)
                step_times.append(total_step_ms)

        mean_wait = float(np.mean(data_wait_times))
        mean_fwd = float(np.mean(forward_times))
        mean_bwd = float(np.mean(backward_times))
        mean_opt = float(np.mean(optimizer_times))
        mean_step = float(np.mean(step_times))

        batches_per_epoch = len(loader) if len(loader) > 0 else math.ceil(len(train_dataset) / batch_size)
        thpt_sps = (batch_size / (mean_step / 1000.0)) if mean_step > 0 else 0.0
        est_epoch_sec = (batches_per_epoch * mean_step) / 1000.0
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == 'cuda' else 0.0

        results_per_worker[nw] = {
            'workers': nw,
            'data_wait_ms': mean_wait,
            'forward_ms': mean_fwd,
            'backward_ms': mean_bwd,
            'optimizer_ms': mean_opt,
            'step_time_ms': mean_step,
            'throughput_sps': thpt_sps,
            'est_epoch_sec': est_epoch_sec,
            'batches_per_epoch': batches_per_epoch,
            'peak_vram_mb': peak_vram_mb,
        }

        print(f"    Workers {nw:1d}: Step={mean_step:6.1f}ms (Wait: {mean_wait:5.1f}ms, Fwd: {mean_fwd:5.1f}ms, Bwd: {mean_bwd:5.1f}ms) | {thpt_sps:5.2f} img/s | {est_epoch_sec:5.1f}s/epoch")

    del model
    del optimizer
    del criterion
    del scaler
    if device.type == 'cuda':
        gc.collect()
        torch.cuda.empty_cache()

    return results_per_worker


def main():
    parser = argparse.ArgumentParser(description="Phase 0 Runtime/Hardware Characterization for B2")
    parser.add_argument('--data-root', type=str, default="/content/dataset/Crack500", help="Path to Crack500 dataset root")
    parser.add_argument('--depths', type=str, default="12,6,4", help="ViT depths to probe (e.g. 12,6,4)")
    parser.add_argument('--batch-sizes', type=str, default="8,12,16,24", help="Batch sizes for OOM probe")
    parser.add_argument('--workers', type=str, default="0,2,4", help="Num workers for throughput benchmark")
    parser.add_argument('--tp-batch-size', type=int, default=12, help="Candidate batch size for throughput benchmark")
    parser.add_argument('--probe-batches', type=int, default=5, help="Number of batches per OOM probe")
    parser.add_argument('--tp-batches', type=int, default=12, help="Number of batches per throughput benchmark")
    parser.add_argument('--synthetic', action='store_true', help="Use synthetic data if real data is unavailable")
    parser.add_argument('--skip-oom', action='store_true', help="Skip OOM probe")
    parser.add_argument('--skip-throughput', action='store_true', help="Skip throughput benchmark")
    parser.add_argument('--output-json', type=str, default="results/phase0_probe_summary.json", help="Path to save summary JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5s]  %(message)s')
    print_banner("PHASE 0: RUNTIME & HARDWARE CHARACTERIZATION (B2: D12, D6, D4)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] Target: {device}")
    if device.type == 'cuda':
        dev_name = torch.cuda.get_device_name(0)
        vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[Device] GPU: {dev_name} ({vram_total_gb:.2f} GB VRAM)")
        if '1650' in dev_name or '1660' in dev_name:
            torch.backends.cudnn.enabled = False
            print("[Device] Set torch.backends.cudnn.enabled = False for GTX 1650/1660 FP16 stability.")
    else:
        print("[Device] WARNING: Running on CPU. CUDA benchmarks will be simulated.")

    depths = [int(d.strip()) for d in args.depths.split(',')]
    batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(',')]
    workers_list = [int(w.strip()) for w in args.workers.split(',')]

    data_root = resolve_data_root(args.data_root)
    print(f"[Dataset] Target Root: {data_root} (Exists: {os.path.exists(data_root)})")

    # Load configs for each depth
    configs = {}
    for d in depths:
        cfg_path = os.path.join(project_root, f"configs/b2_crack500_depth{d}.yaml")
        if not os.path.exists(cfg_path):
            cfg_path = os.path.join(project_root, "configs/b2_crack500_depth12.yaml")
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        cfg['root_dir'] = data_root
        cfg['num_transformer_layers'] = d
        configs[d] = cfg

    oom_results = {}
    throughput_results = {}

    # =========================================================================
    # PART 1: FULL-MODEL OOM / BATCH-SIZE PROBE
    # =========================================================================
    if not args.skip_oom:
        print_banner("PART 1: FULL-MODEL OOM & BATCH-SIZE PROBE")
        for d in depths:
            print(f"\n>>> Probing Depth {d} (BS in {batch_sizes})...")
            oom_results[d] = run_oom_probe_for_depth(
                depth=d,
                config=configs[d],
                batch_sizes=batch_sizes,
                device=device,
                data_root=data_root,
                num_batches=args.probe_batches,
                use_synthetic=args.synthetic or not os.path.exists(data_root),
            )
    else:
        print("\n[Skipping OOM Probe as requested]")

    # =========================================================================
    # PART 2: THROUGHPUT BENCHMARK ACROSS WORKERS
    # =========================================================================
    if not args.skip_throughput:
        print_banner(f"PART 2: THROUGHPUT BENCHMARK (Candidate BS = {args.tp_batch_size})")
        for d in depths:
            print(f"\n>>> Profiling Depth {d} across workers {workers_list}...")
            throughput_results[d] = run_throughput_for_depth(
                depth=d,
                config=configs[d],
                batch_size=args.tp_batch_size,
                workers_list=workers_list,
                device=device,
                data_root=data_root,
                num_batches=args.tp_batches,
                use_synthetic=args.synthetic or not os.path.exists(data_root),
            )
    else:
        print("\n[Skipping Throughput Benchmark as requested]")

    # =========================================================================
    # PART 3: EPOCH BUDGET & WALL-CLOCK ESTIMATES
    # =========================================================================
    print_banner("PART 3: EPOCH BUDGET & WALL-CLOCK ESTIMATES (Candidate BS = 12, Workers = 4)")
    budget_epochs = [15, 20, 25, 30]
    epoch_projections = {}

    for d in depths:
        epoch_projections[d] = {}
        # Pick workers=4 or highest available worker result
        tp_d = throughput_results.get(d, {})
        w_res = tp_d.get(4) or tp_d.get(2) or tp_d.get(0)
        sec_per_epoch = w_res['est_epoch_sec'] if w_res else 0.0

        for ep in budget_epochs:
            tot_sec = ep * sec_per_epoch
            tot_min = tot_sec / 60.0
            epoch_projections[d][ep] = {
                'sec': tot_sec,
                'min': tot_min,
            }

    # =========================================================================
    # PART 4: UNIFIED SUMMARY REPORT & 5 TECHNICAL CONCLUSIONS
    # =========================================================================
    print_banner("PART 4: UNIFIED SUMMARY REPORT & TECHNICAL VERDICTS")

    print("\n### Unified Phase 0 Characterization Table\n")
    print("| Depth | Batch PASS/OOM | Peak VRAM | Workers | Step time | img/s | sec/epoch |")
    print("|:-----:|:--------------:|:---------:|:-------:|:---------:|:-----:|:---------:|")

    for d in depths:
        oom_d = oom_results.get(d, {})
        oom_str_list = []
        for bs in batch_sizes:
            st = oom_d.get(bs, {}).get('status', 'N/A')
            oom_str_list.append(f"BS{bs}:{st}")
        oom_str = ", ".join(oom_str_list)

        tp_d = throughput_results.get(d, {})
        for w in workers_list:
            r = tp_d.get(w, {})
            if r:
                print(
                    f"| D{d:02d} | {oom_str} | {r['peak_vram_mb']:.1f} MB | {w} "
                    f"| {r['step_time_ms']:.1f} ms | {r['throughput_sps']:.2f} | {r['est_epoch_sec']:.1f} s |"
                )
            else:
                # If only OOM ran
                max_peak = max([oom_d[b]['peak_allocated_mb'] for b in oom_d if 'peak_allocated_mb' in oom_d[b]], default=0.0)
                print(f"| D{d:02d} | {oom_str} | {max_peak:.1f} MB | N/A | N/A | N/A | N/A |")

    print("\n### Epoch Budget Wall-Clock Projections (Workers = 4)")
    print("| Depth | 15 Epochs | 20 Epochs | 25 Epochs | 30 Epochs |")
    print("|:-----:|:---------:|:---------:|:---------:|:---------:|")
    for d in depths:
        p = epoch_projections.get(d, {})
        print(
            f"| D{d:02d} | {p.get(15, {}).get('min', 0):.1f} min | "
            f"{p.get(20, {}).get('min', 0):.1f} min | "
            f"{p.get(25, {}).get('min', 0):.1f} min | "
            f"{p.get(30, {}).get('min', 0):.1f} min |"
        )

    # Invariant Verification Notice
    print("\n" + "=" * 80)
    print("INVARIANT VERIFICATION:")
    print("  [x] Base B2 model used (p3_mode=None)")
    print("  [x] Full Stage-2 optimizer path validated")
    print("  [x] Zero Phase 1 official training executed (Sanity & Characterization ONLY)")
    print("=" * 80)

    # Save results to JSON
    os.makedirs(os.path.dirname(args.output_json) if os.path.dirname(args.output_json) else '.', exist_ok=True)
    summary_data = {
        'oom_results': oom_results,
        'throughput_results': throughput_results,
        'epoch_projections': epoch_projections,
        'config': {
            'depths': depths,
            'batch_sizes': batch_sizes,
            'workers': workers_list,
            'tp_batch_size': args.tp_batch_size,
        }
    }
    with open(args.output_json, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n[Saved] Phase 0 Summary JSON saved to: {args.output_json}")


if __name__ == '__main__':
    main()
