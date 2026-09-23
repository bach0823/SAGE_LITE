"""
Deep Runtime Profiler for SAGE-Lite B2 vs B1 Baseline on Crack500 (Tesla T4)

Instruments:
1. B1-D12 Baseline Timing (batch=12, workers=2, AMP, Crack500)
2. B2-D12 Overall Step Breakdown (DataWait, Forward, Backward, Optimizer, Step Time, VRAM)
3. Per-SageLayer Timing (Layers 0..15: 4 CNN stages + 12 ViT blocks)
4. Selected Expert Counts & Sample Counts (Experts 0..15)
5. Routing Transition Categories:
   - CNN -> CNN
   - CNN -> ViT
   - ViT -> CNN
   - ViT -> ViT
   - Riêng Stage0 -> ViT (112x112 -> 14x14 token adaptation)
6. SAHub Timing & Adaptation Overhead:
   - Input Adapt vs Expert Compute vs Output Adapt vs Gating & index_add_
   - _infer_expert_type() CPU overhead
   - Self-Selection Bypass frequency and time saved

Usage:
    python scripts/profile_deep_b2.py --config configs/b2_crack500_depth12.yaml --measured 5 --warmup 2
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sage.components.sa_hub import SAHub
from sage.components.sage_layer import SageLayer
from sage.networks.b1_unet import B1ConvNeXtViTUNet
from sage.networks.b2_unet import create_b2_unet
from sage.utils.dataloader import get_dataset_from_config
from sage.utils.training_utils import seed_worker, set_seed
from scripts.train_crack import CrackBinaryLoss, get_optimizer_groups


def print_header(title: str, ch: str = "=", width: int = 75):
    line = ch * width
    print(f"\n{line}\n{title}\n{line}")


def resolve_data_root(config_root: str) -> str:
    if os.path.exists(config_root):
        return config_root
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
            "image": torch.randn(3, self.image_size, self.image_size),
            "label": torch.randint(0, 2, (1, self.image_size, self.image_size)).float(),
        }


# ==============================================================================
# 1. BENCHMARK B1-D12 BASELINE
# ==============================================================================
def benchmark_b1_baseline(
    config: dict,
    device: torch.device,
    num_workers: int = 2,
    batch_size: int = 12,
    warmup_steps: int = 2,
    measured_steps: int = 5,
    use_synthetic: bool = False,
) -> Dict[str, float]:
    print_header("BENCHMARKING B1-D12 BASELINE (NO SAGE)")
    img_size = int(config.get("img_size", 448))
    seed = int(config.get("seed", 42))
    set_seed(seed)

    # Build model
    print("[Model] Initializing B1ConvNeXtViTUNet(num_transformer_layers=12)...")
    model = B1ConvNeXtViTUNet(
        num_classes=1,
        img_size=img_size,
        num_transformer_layers=12,
        pretrained=True,
    ).to(device)
    model.train()

    criterion = CrackBinaryLoss(bce_weight=0.5, dice_weight=0.5).to(device)
    optimizer = torch.optim.AdamW(
        get_optimizer_groups(model, lr_backbone=1e-5, lr_decoder=1e-4, lr_sage=1e-4, weight_decay=1e-4)
    )
    scaler = torch.amp.GradScaler("cuda")

    # DataLoader
    if use_synthetic:
        dataset = SyntheticDataset(length=max(100, (warmup_steps + measured_steps) * batch_size * 2), image_size=img_size)
    else:
        dataset = get_dataset_from_config(config, split="train", image_size=img_size)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        drop_last=True,
    )

    data_wait_times = []
    fwd_times = []
    bwd_times = []
    opt_times = []
    step_times = []

    loader_iter = iter(loader)
    total_steps = warmup_steps + measured_steps

    torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, total_steps + 1):
        is_measured = step > warmup_steps

        t_wait_start = time.perf_counter()
        batch = next(loader_iter)
        t_wait = (time.perf_counter() - t_wait_start) * 1000.0

        images = batch["image"].to(device, non_blocking=True)
        masks = batch["label"].to(device, non_blocking=True)

        ev_fwd_start = torch.cuda.Event(enable_timing=True)
        ev_fwd_end = torch.cuda.Event(enable_timing=True)
        ev_bwd_start = torch.cuda.Event(enable_timing=True)
        ev_bwd_end = torch.cuda.Event(enable_timing=True)
        ev_opt_start = torch.cuda.Event(enable_timing=True)
        ev_opt_end = torch.cuda.Event(enable_timing=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward
        ev_fwd_start.record()
        with torch.amp.autocast("cuda", enabled=True):
            logits = model(images)
            loss = criterion(logits, masks)
        ev_fwd_end.record()

        # Backward
        ev_bwd_start.record()
        scaler.scale(loss).backward()
        ev_bwd_end.record()

        # Optimizer
        ev_opt_start.record()
        scaler.step(optimizer)
        scaler.update()
        ev_opt_end.record()

        torch.cuda.synchronize(device)

        fwd_ms = ev_fwd_start.elapsed_time(ev_fwd_end)
        bwd_ms = ev_bwd_start.elapsed_time(ev_bwd_end)
        opt_ms = ev_opt_start.elapsed_time(ev_opt_end)
        total_ms = t_wait + fwd_ms + bwd_ms + opt_ms

        tag = "MEASURED" if is_measured else "WARMUP"
        print(f"  B1 Step {step:02d}/{total_steps:02d} [{tag}] | Data: {t_wait:6.1f}ms | Fwd: {fwd_ms:6.1f}ms | Bwd: {bwd_ms:6.1f}ms | Opt: {opt_ms:5.1f}ms | Step: {total_ms:6.1f}ms")

        if is_measured:
            data_wait_times.append(t_wait)
            fwd_times.append(fwd_ms)
            bwd_times.append(bwd_ms)
            opt_times.append(opt_ms)
            step_times.append(total_ms)

    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    results = {
        "data_wait_ms": float(np.mean(data_wait_times)),
        "fwd_ms": float(np.mean(fwd_times)),
        "bwd_ms": float(np.mean(bwd_times)),
        "opt_ms": float(np.mean(opt_times)),
        "step_ms": float(np.mean(step_times)),
        "thpt": (batch_size / (np.mean(step_times) / 1000.0)),
        "peak_vram_mb": peak_vram,
    }

    # Clean up model to free VRAM for B2
    del model, optimizer, criterion
    torch.cuda.empty_cache()
    return results


# ==============================================================================
# 2. DEEP INSTRUMENTATION ENGINE FOR B2-D12
# ==============================================================================
class DeepProfilingCollector:
    """Collects per-layer, per-expert, transition, and SA-Hub micro-timings."""
    def __init__(self, num_experts: int = 16):
        self.num_experts = num_experts
        self.active = False

        # Per-layer timings (layer_index -> list of ms)
        self.layer_fwd_times = defaultdict(list)
        self.layer_main_times = defaultdict(list)
        self.layer_expert_times = defaultdict(list)

        # Expert selection counts & sample counts (expert_index -> int)
        self.expert_selection_counts = defaultdict(int)
        self.expert_sample_counts = defaultdict(int)

        # Transitions: (source_layer, target_expert) -> count of samples
        self.transitions = defaultdict(int)

        # Micro-components of expert path (list of ms per step)
        self.step_sahub_input_adapt = []
        self.step_expert_compute = []
        self.step_sahub_output_adapt = []
        self.step_index_add = []
        self.step_self_bypass = []

        # _infer_expert_type CPU timing
        self.infer_expert_type_calls = 0
        self.infer_expert_type_total_sec = 0.0

        # Current step accumulators
        self._cur_step_input_adapt_events = []
        self._cur_step_expert_compute_events = []
        self._cur_step_output_adapt_events = []
        self._cur_step_index_add_events = []
        self._cur_step_self_bypass_events = []

    def start_step(self):
        self._cur_step_input_adapt_events.clear()
        self._cur_step_expert_compute_events.clear()
        self._cur_step_output_adapt_events.clear()
        self._cur_step_index_add_events.clear()
        self._cur_step_self_bypass_events.clear()

    def end_step(self):
        if not self.active:
            return
        # Resolve all recorded event pairs for micro-timings
        def sum_events(event_pairs):
            total_ms = 0.0
            for start_ev, end_ev in event_pairs:
                total_ms += start_ev.elapsed_time(end_ev)
            return total_ms

        self.step_sahub_input_adapt.append(sum_events(self._cur_step_input_adapt_events))
        self.step_expert_compute.append(sum_events(self._cur_step_expert_compute_events))
        self.step_sahub_output_adapt.append(sum_events(self._cur_step_output_adapt_events))
        self.step_index_add.append(sum_events(self._cur_step_index_add_events))
        self.step_self_bypass.append(sum_events(self._cur_step_self_bypass_events))


collector = DeepProfilingCollector(num_experts=16)


def instrument_sage_layer(orig_forward, orig_execute_expert_path):
    def instrumented_forward(self, x: torch.Tensor, expert_pool: Optional[nn.ModuleList] = None) -> torch.Tensor:
        if not collector.active:
            return orig_forward(self, x, expert_pool)

        pool = expert_pool if expert_pool is not None else self.expert_pool
        ev_layer_start = torch.cuda.Event(enable_timing=True)
        ev_main_start = torch.cuda.Event(enable_timing=True)
        ev_main_end = torch.cuda.Event(enable_timing=True)
        ev_expert_start = torch.cuda.Event(enable_timing=True)
        ev_expert_end = torch.cuda.Event(enable_timing=True)
        ev_layer_end = torch.cuda.Event(enable_timing=True)

        ev_layer_start.record()

        # Step 1: Main Path
        ev_main_start.record()
        main_output = self._execute_main_path(x)
        ev_main_end.record()

        # Step 2: Expert Path
        ev_expert_start.record()
        expert_output, routing_info = self._execute_expert_path(x, main_output, pool)
        ev_expert_end.record()

        # Step 3: Fusion
        if self.fusion_type == "adaptive":
            alpha = torch.clamp(self.alpha, 0.1, 1.0)
            final_output = alpha * main_output + (1.0 - alpha) * self.expert_dropout(expert_output)
        else:
            final_output = main_output + self.expert_dropout(self.residual_scale * expert_output)

        routing_info.update({
            "forward_call_count": self.forward_calls.item(),
            "expert_success_rate": (self.expert_successes / max(self.forward_calls, 1)).item(),
            "my_index": self.my_index,
        })
        self._last_routing_info = routing_info
        ev_layer_end.record()

        torch.cuda.synchronize(x.device)
        layer_idx = self.my_index if self.my_index is not None else -1
        collector.layer_fwd_times[layer_idx].append(ev_layer_start.elapsed_time(ev_layer_end))
        collector.layer_main_times[layer_idx].append(ev_main_start.elapsed_time(ev_main_end))
        collector.layer_expert_times[layer_idx].append(ev_expert_start.elapsed_time(ev_expert_end))

        return final_output

    def instrumented_execute_expert_path(self, x: torch.Tensor, main_output: torch.Tensor, expert_pool: nn.ModuleList):
        if not collector.active:
            return orig_execute_expert_path(self, x, main_output, expert_pool)

        B, *dims = main_output.shape
        device = x.device
        source_layer = self.my_index if self.my_index is not None else -1

        top_k_indices, gating_weights, routing_info = self.router(x)
        self._last_lb_loss = routing_info.get("load_balance_loss", None)

        flat_indices = top_k_indices.flatten()
        flat_weights = gating_weights.flatten()
        batch_map = torch.arange(B, device=device).repeat_interleave(self.router.top_k)
        final_expert_output = torch.zeros_like(main_output)

        # Track transitions and selections
        for idx in flat_indices.tolist():
            collector.expert_selection_counts[idx] += 1

        for expert_idx, expert in enumerate(expert_pool):
            expert_mask = (flat_indices == expert_idx)
            if not expert_mask.any():
                continue

            original_batch_indices = batch_map[expert_mask]
            sample_cnt = len(original_batch_indices)
            collector.expert_sample_counts[expert_idx] += sample_cnt
            collector.transitions[(source_layer, expert_idx)] += sample_cnt

            if self.my_index is not None and expert_idx == self.my_index:
                ev_byp_start = torch.cuda.Event(enable_timing=True)
                ev_byp_end = torch.cuda.Event(enable_timing=True)
                ev_byp_start.record()
                adapted_expert_output = main_output[original_batch_indices]
                ev_byp_end.record()
                collector._cur_step_self_bypass_events.append((ev_byp_start, ev_byp_end))
            else:
                # 4a: Input Adapt
                ev_in_start = torch.cuda.Event(enable_timing=True)
                ev_in_end = torch.cuda.Event(enable_timing=True)
                ev_in_start.record()
                adapted_input, _ = self.sa_hub.adapt(x[original_batch_indices], expert)
                ev_in_end.record()
                collector._cur_step_input_adapt_events.append((ev_in_start, ev_in_end))

                # 4b: Expert Compute
                ev_exp_start = torch.cuda.Event(enable_timing=True)
                ev_exp_end = torch.cuda.Event(enable_timing=True)
                ev_exp_start.record()
                expert_raw_output = expert(adapted_input)
                if isinstance(expert_raw_output, tuple):
                    expert_raw_output = expert_raw_output[0]
                ev_exp_end.record()
                collector._cur_step_expert_compute_events.append((ev_exp_start, ev_exp_end))

                # 4c: Output Adapt
                main_path_shape_subset = main_output[original_batch_indices].shape
                ev_out_start = torch.cuda.Event(enable_timing=True)
                ev_out_end = torch.cuda.Event(enable_timing=True)
                ev_out_start.record()
                adapted_expert_output, _ = self.sa_hub.adapt(
                    expert_raw_output,
                    self.main_block,
                    main_path_shape=main_path_shape_subset,
                )
                ev_out_end.record()
                collector._cur_step_output_adapt_events.append((ev_out_start, ev_out_end))

            # 4d & 5: Index Add
            weights_for_expert = flat_weights[expert_mask]
            reshape_dims = [-1] + [1] * (adapted_expert_output.dim() - 1)
            weighted_output = adapted_expert_output * weights_for_expert.view(reshape_dims)

            ev_add_start = torch.cuda.Event(enable_timing=True)
            ev_add_end = torch.cuda.Event(enable_timing=True)
            ev_add_start.record()
            final_expert_output.index_add_(0, original_batch_indices, weighted_output.to(final_expert_output.dtype))
            ev_add_end.record()
            collector._cur_step_index_add_events.append((ev_add_start, ev_add_end))

        self.expert_successes += 1
        return final_expert_output, routing_info

    return instrumented_forward, instrumented_execute_expert_path


def instrument_sa_hub(orig_infer_expert_type):
    def instrumented_infer_expert_type(self, expert_module: nn.Module) -> str:
        if not collector.active:
            return orig_infer_expert_type(self, expert_module)
        t0 = time.perf_counter()
        ret = orig_infer_expert_type(self, expert_module)
        t_elapsed = time.perf_counter() - t0
        collector.infer_expert_type_calls += 1
        collector.infer_expert_type_total_sec += t_elapsed
        return ret
    return instrumented_infer_expert_type


# ==============================================================================
# 3. RUN DEEP PROFILING ON B2-D12
# ==============================================================================
def profile_b2_deep(
    config: dict,
    device: torch.device,
    num_workers: int = 2,
    batch_size: int = 12,
    warmup_steps: int = 2,
    measured_steps: int = 5,
    use_synthetic: bool = False,
) -> Dict[str, Any]:
    print_header("DEEP RUNTIME PROFILING B2-D12 (SAGE-Lite)")
    img_size = int(config.get("img_size", 448))
    seed = int(config.get("seed", 42))
    set_seed(seed)

    # Monkey-patch instrumentation
    orig_layer_fwd = SageLayer.forward
    orig_layer_exp = SageLayer._execute_expert_path
    orig_sahub_infer = SAHub._infer_expert_type

    instr_fwd, instr_exp = instrument_sage_layer(orig_layer_fwd, orig_layer_exp)
    SageLayer.forward = instr_fwd
    SageLayer._execute_expert_path = instr_exp
    SAHub._infer_expert_type = instrument_sa_hub(orig_sahub_infer)

    print("[Model] Initializing B2 model from config...")
    vit_depth = int(config.get("num_transformer_layers", 12))
    sage_cfg = config.get("sage_config", {})
    model = create_b2_unet(
        num_transformer_layers=vit_depth,
        img_size=img_size,
        pretrained=True,
        sage_config=sage_cfg,
    ).to(device)
    model.train()

    criterion = CrackBinaryLoss(bce_weight=0.5, dice_weight=0.5).to(device)
    optimizer = torch.optim.AdamW(
        get_optimizer_groups(model, lr_backbone=1e-5, lr_decoder=1e-4, lr_sage=1e-4, weight_decay=1e-4)
    )
    scaler = torch.amp.GradScaler("cuda")

    if use_synthetic:
        dataset = SyntheticDataset(length=max(100, (warmup_steps + measured_steps) * batch_size * 2), image_size=img_size)
    else:
        dataset = get_dataset_from_config(config, split="train", image_size=img_size)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        drop_last=True,
    )

    data_wait_times = []
    fwd_times = []
    bwd_times = []
    opt_times = []
    step_times = []

    loader_iter = iter(loader)
    total_steps = warmup_steps + measured_steps

    torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, total_steps + 1):
        is_measured = step > warmup_steps
        collector.active = is_measured
        collector.start_step()

        t_wait_start = time.perf_counter()
        batch = next(loader_iter)
        t_wait = (time.perf_counter() - t_wait_start) * 1000.0

        images = batch["image"].to(device, non_blocking=True)
        masks = batch["label"].to(device, non_blocking=True)

        ev_fwd_start = torch.cuda.Event(enable_timing=True)
        ev_fwd_end = torch.cuda.Event(enable_timing=True)
        ev_bwd_start = torch.cuda.Event(enable_timing=True)
        ev_bwd_end = torch.cuda.Event(enable_timing=True)
        ev_opt_start = torch.cuda.Event(enable_timing=True)
        ev_opt_end = torch.cuda.Event(enable_timing=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward
        ev_fwd_start.record()
        with torch.amp.autocast("cuda", enabled=True):
            fwd_out = model.forward_with_routing_info(images)
            logits = fwd_out["logits"]
            routing_infos = fwd_out["routing_infos"]
            seg_loss = criterion(logits, masks)
            lb_loss = model.compute_total_load_balance_loss(routing_infos)
            total_loss = seg_loss + lb_loss
        ev_fwd_end.record()

        # Backward
        ev_bwd_start.record()
        scaler.scale(total_loss).backward()
        ev_bwd_end.record()

        # Optimizer
        ev_opt_start.record()
        scaler.step(optimizer)
        scaler.update()
        ev_opt_end.record()

        torch.cuda.synchronize(device)
        collector.end_step()

        fwd_ms = ev_fwd_start.elapsed_time(ev_fwd_end)
        bwd_ms = ev_bwd_start.elapsed_time(ev_bwd_end)
        opt_ms = ev_opt_start.elapsed_time(ev_opt_end)
        total_ms = t_wait + fwd_ms + bwd_ms + opt_ms

        tag = "MEASURED" if is_measured else "WARMUP"
        print(f"  B2 Step {step:02d}/{total_steps:02d} [{tag}] | Data: {t_wait:6.1f}ms | Fwd: {fwd_ms:6.1f}ms | Bwd: {bwd_ms:6.1f}ms | Opt: {opt_ms:5.1f}ms | Step: {total_ms:6.1f}ms | Loss: {total_loss.item():.4f}")

        if is_measured:
            data_wait_times.append(t_wait)
            fwd_times.append(fwd_ms)
            bwd_times.append(bwd_ms)
            opt_times.append(opt_ms)
            step_times.append(total_ms)

    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    # Restore uninstrumented methods
    SageLayer.forward = orig_layer_fwd
    SageLayer._execute_expert_path = orig_layer_exp
    SAHub._infer_expert_type = orig_sahub_infer

    return {
        "data_wait_ms": float(np.mean(data_wait_times)),
        "fwd_ms": float(np.mean(fwd_times)),
        "bwd_ms": float(np.mean(bwd_times)),
        "opt_ms": float(np.mean(opt_times)),
        "step_ms": float(np.mean(step_times)),
        "thpt": (batch_size / (np.mean(step_times) / 1000.0)),
        "peak_vram_mb": peak_vram,
    }


# ==============================================================================
# 4. GENERATE SUMMARY REPORTS & METRIC BREAKDOWNS
# ==============================================================================
def print_deep_profiling_report(b1_res: Optional[Dict[str, float]], b2_res: Dict[str, Any]):
    print_header("DEEP RUNTIME PROFILING RESULTS & BREAKDOWN", ch="=")

    # 1. Macro Comparison Table
    print("\n1. STEP TIMING & RUNTIME COMPARISON (B1 vs B2-D12)")
    print("-" * 88)
    print(f"{'Metric':<25} | {'B1-D12 (No SAGE)':<18} | {'B2-D12 (SAGE-Lite)':<20} | {'Delta / Overhead':<16}")
    print("-" * 88)
    if b1_res:
        fwd_delta = b2_res['fwd_ms'] - b1_res['fwd_ms']
        bwd_delta = b2_res['bwd_ms'] - b1_res['bwd_ms']
        step_delta = b2_res['step_ms'] - b1_res['step_ms']
        vram_delta = b2_res['peak_vram_mb'] - b1_res['peak_vram_mb']
        print(f"{'Data Wait (ms)':<25} | {b1_res['data_wait_ms']:>14.1f} ms | {b2_res['data_wait_ms']:>16.1f} ms | {b2_res['data_wait_ms'] - b1_res['data_wait_ms']:>+14.1f} ms")
        print(f"{'Forward Pass (ms)':<25} | {b1_res['fwd_ms']:>14.1f} ms | {b2_res['fwd_ms']:>16.1f} ms | {fwd_delta:>+14.1f} ms ({b2_res['fwd_ms']/b1_res['fwd_ms']:.2f}x)")
        print(f"{'Backward Pass (ms)':<25} | {b1_res['bwd_ms']:>14.1f} ms | {b2_res['bwd_ms']:>16.1f} ms | {bwd_delta:>+14.1f} ms ({b2_res['bwd_ms']/b1_res['bwd_ms']:.2f}x)")
        print(f"{'Optimizer Step (ms)':<25} | {b1_res['opt_ms']:>14.1f} ms | {b2_res['opt_ms']:>16.1f} ms | {b2_res['opt_ms'] - b1_res['opt_ms']:>+14.1f} ms")
        print(f"{'Total Step Time (ms)':<25} | {b1_res['step_ms']:>14.1f} ms | {b2_res['step_ms']:>16.1f} ms | {step_delta:>+14.1f} ms ({b2_res['step_ms']/b1_res['step_ms']:.2f}x)")
        print(f"{'Throughput (img/s)':<25} | {b1_res['thpt']:>14.2f} /s | {b2_res['thpt']:>16.2f} /s | {b2_res['thpt'] - b1_res['thpt']:>+14.2f} /s")
        print(f"{'Est 1 Epoch (158 btch)':<25} | {158*b1_res['step_ms']/60000:>14.2f} m  | {158*b2_res['step_ms']/60000:>16.2f} m  | {158*step_delta/60000:>+14.2f} min")
        print(f"{'Peak VRAM (MB)':<25} | {b1_res['peak_vram_mb']:>14.1f} MB | {b2_res['peak_vram_mb']:>16.1f} MB | {vram_delta:>+14.1f} MB")
    else:
        print(f"{'Data Wait (ms)':<25} | {'N/A':>18} | {b2_res['data_wait_ms']:>16.1f} ms | {'-':>16}")
        print(f"{'Forward Pass (ms)':<25} | {'N/A':>18} | {b2_res['fwd_ms']:>16.1f} ms | {'-':>16}")
        print(f"{'Backward Pass (ms)':<25} | {'N/A':>18} | {b2_res['bwd_ms']:>16.1f} ms | {'-':>16}")
        print(f"{'Total Step Time (ms)':<25} | {'N/A':>18} | {b2_res['step_ms']:>16.1f} ms | {'-':>16}")
        print(f"{'Throughput (img/s)':<25} | {'N/A':>18} | {b2_res['thpt']:>16.2f} /s | {'-':>16}")
        print(f"{'Peak VRAM (MB)':<25} | {'N/A':>18} | {b2_res['peak_vram_mb']:>16.1f} MB | {'-':>16}")
    print("-" * 88)

    # 2. Per-SageLayer Timing Table
    print("\n2. PER-SAGELAYER FORWARD TIMING BREAKDOWN (16 Layers)")
    print("-" * 80)
    print(f"{'Layer / Block Name':<28} | {'Type':<8} | {'Total (ms)':<11} | {'Main (ms)':<11} | {'Expert (ms)':<11}")
    print("-" * 80)
    for idx in range(16):
        if idx < 4:
            name = f"SageLayer {idx:02d} (CNN Stage {idx})"
            ltype = "CNN"
        else:
            name = f"SageLayer {idx:02d} (ViT Block {idx-4:02d})"
            ltype = "ViT"
        tot = np.mean(collector.layer_fwd_times[idx]) if collector.layer_fwd_times[idx] else 0.0
        main = np.mean(collector.layer_main_times[idx]) if collector.layer_main_times[idx] else 0.0
        exp = np.mean(collector.layer_expert_times[idx]) if collector.layer_expert_times[idx] else 0.0
        print(f"{name:<28} | {ltype:<8} | {tot:>9.2f}ms | {main:>9.2f}ms | {exp:>9.2f}ms")
    print("-" * 80)

    # 3. Transitions Matrix & Categories
    print("\n3. SAGE ROUTING TRANSITION PATTERNS & MODALITIES")
    print("-" * 75)
    total_transitions = sum(collector.transitions.values())
    cnn_to_cnn = sum(collector.transitions[(s, t)] for s in range(4) for t in range(4))
    cnn_to_vit = sum(collector.transitions[(s, t)] for s in range(4) for t in range(4, 16))
    vit_to_cnn = sum(collector.transitions[(s, t)] for s in range(4, 16) for t in range(4))
    vit_to_vit = sum(collector.transitions[(s, t)] for s in range(4, 16) for t in range(4, 16))
    stage0_to_vit = sum(collector.transitions[(0, t)] for t in range(4, 16))

    def fmt_pct(cnt, tot):
        return f"{cnt:>6d} ({cnt / max(1, tot) * 100:>5.1f}%)"

    print(f"{'Transition Category':<35} | {'Samples Routed':<20} | {'Status':<15}")
    print("-" * 75)
    print(f"{'CNN -> CNN (Intra-CNN)':<35} | {fmt_pct(cnn_to_cnn, total_transitions):<20} | {'Active':<15}")
    print(f"{'CNN -> ViT (Cross-Modal CNN->ViT)':<35} | {fmt_pct(cnn_to_vit, total_transitions):<20} | {'Active':<15}")
    print(f"{'  * Stage0 -> ViT (112->14 tok)':<35} | {fmt_pct(stage0_to_vit, total_transitions):<20} | {'Special Focus':<15}")
    print(f"{'ViT -> CNN (Cross-Modal ViT->CNN)':<35} | {fmt_pct(vit_to_cnn, total_transitions):<20} | {'Active':<15}")
    print(f"{'ViT -> ViT (Intra-ViT)':<35} | {fmt_pct(vit_to_vit, total_transitions):<20} | {'Active':<15}")
    print(f"{'TOTAL Routing Transitions':<35} | {fmt_pct(total_transitions, total_transitions):<20} | {'100%':<15}")
    print("-" * 75)

    # 4. Expert Usage Breakdown
    print("\n4. SELECTED EXPERT USAGE & SAMPLE COUNTS (16 Experts)")
    print("-" * 75)
    total_selections = sum(collector.expert_selection_counts.values())
    total_samples = sum(collector.expert_sample_counts.values())
    print(f"{'Expert Index & Name':<28} | {'Type':<8} | {'Selections':<16} | {'Samples':<16}")
    print("-" * 75)
    for idx in range(16):
        if idx < 4:
            ename = f"Expert {idx:02d} (CNN Stage {idx}) [SHR]"
            etype = "CNN"
        else:
            ename = f"Expert {idx:02d} (ViT Block {idx-4:02d})"
            etype = "ViT"
        sel = collector.expert_selection_counts[idx]
        smp = collector.expert_sample_counts[idx]
        sel_pct = (sel / max(1, total_selections)) * 100.0
        smp_pct = (smp / max(1, total_samples)) * 100.0
        print(f"{ename:<28} | {etype:<8} | {sel:>6d} ({sel_pct:>4.1f}%) | {smp:>6d} ({smp_pct:>4.1f}%)")
    print("-" * 75)

    # 5. SAHub & Micro-component Breakdown
    print("\n5. SAHUB ADAPTATION & EXPERT PATH MICRO-TIMINGS")
    print("-" * 75)
    in_adapt_ms = np.mean(collector.step_sahub_input_adapt) if collector.step_sahub_input_adapt else 0.0
    exp_comp_ms = np.mean(collector.step_expert_compute) if collector.step_expert_compute else 0.0
    out_adapt_ms = np.mean(collector.step_sahub_output_adapt) if collector.step_sahub_output_adapt else 0.0
    idx_add_ms = np.mean(collector.step_index_add) if collector.step_index_add else 0.0
    bypass_ms = np.mean(collector.step_self_bypass) if collector.step_self_bypass else 0.0
    total_micro_ms = in_adapt_ms + exp_comp_ms + out_adapt_ms + idx_add_ms + bypass_ms

    print(f"{'Micro-Component':<35} | {'Mean Time/Step':<16} | {'% of Expert Loop':<16}")
    print("-" * 75)
    print(f"{'SAHub Input Adapt (Shape/Chan)':<35} | {in_adapt_ms:>11.2f} ms | {in_adapt_ms/max(1e-3, total_micro_ms)*100:>13.1f} %")
    print(f"{'Expert Forward Computation':<35} | {exp_comp_ms:>11.2f} ms | {exp_comp_ms/max(1e-3, total_micro_ms)*100:>13.1f} %")
    print(f"{'SAHub Output Adapt (Match Main)':<35} | {out_adapt_ms:>11.2f} ms | {out_adapt_ms/max(1e-3, total_micro_ms)*100:>13.1f} %")
    print(f"{'Gating & index_add_ Accumulation':<35} | {idx_add_ms:>11.2f} ms | {idx_add_ms/max(1e-3, total_micro_ms)*100:>13.1f} %")
    print(f"{'Self-Selection Bypass (Zero-Cost)':<35} | {bypass_ms:>11.2f} ms | {bypass_ms/max(1e-3, total_micro_ms)*100:>13.1f} %")
    print("-" * 75)
    print(f"{'TOTAL Expert Path Sub-Operations':<35} | {total_micro_ms:>11.2f} ms | {100.0:>13.1f} %")
    print("-" * 75)

    # 6. _infer_expert_type CPU Overhead
    print("\n6. _INFER_EXPERT_TYPE() CPU OVERHEAD ANALYSIS")
    print("-" * 75)
    calls = collector.infer_expert_type_calls
    tot_cpu_ms = collector.infer_expert_type_total_sec * 1000.0
    avg_cpu_us = (tot_cpu_ms / max(1, calls)) * 1000.0
    print(f"  Total Calls Across Measured Steps: {calls} calls")
    print(f"  Total CPU Time:                    {tot_cpu_ms:.2f} ms")
    print(f"  Average Time Per Call:             {avg_cpu_us:.2f} microseconds")
    print(f"  CPU Time Per Forward Step:         {tot_cpu_ms / max(1, len(collector.step_sahub_input_adapt)):.2f} ms/step")
    print("-" * 75)
    print_header("PROFILING RUN COMPLETED", ch="=")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Deep Runtime Profiler for SAGE-Lite B2 vs B1")
    parser.add_argument("--config", type=str, default="configs/b2_crack500_depth12.yaml")
    parser.add_argument("--b1-config", type=str, default="configs/b1_crack500_depth12.yaml")
    parser.add_argument("--skip-b1", action="store_true", help="Skip B1 baseline benchmark")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--measured", type=int, default=5)
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset for test/smoke")
    args = parser.parse_args()

    print_header("SAGE-LITE DEEP RUNTIME PROFILER INITIALIZATION")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Target Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        print(f"[Device] GPU: {gpu_name} | Total VRAM: {total_vram:.1f} MB ({total_vram/1024:.2f} GB)")

    # Load B2 config
    with open(args.config, "r") as f:
        config_b2 = yaml.safe_load(f)
    config_b2["batch_size"] = args.batch_size
    config_b2["num_workers"] = args.workers
    config_b2["root_dir"] = resolve_data_root(config_b2.get("root_dir", "/content/dataset/Crack500"))

    # Part 1: B1 Benchmark
    b1_results = None
    if not args.skip_b1:
        if os.path.exists(args.b1_config):
            with open(args.b1_config, "r") as f:
                config_b1 = yaml.safe_load(f)
        else:
            config_b1 = dict(config_b2)
        config_b1["batch_size"] = args.batch_size
        config_b1["num_workers"] = args.workers
        config_b1["root_dir"] = config_b2["root_dir"]

        b1_results = benchmark_b1_baseline(
            config=config_b1,
            device=device,
            num_workers=args.workers,
            batch_size=args.batch_size,
            warmup_steps=args.warmup,
            measured_steps=args.measured,
            use_synthetic=args.synthetic,
        )

    # Part 2: B2 Deep Profiling
    b2_results = profile_b2_deep(
        config=config_b2,
        device=device,
        num_workers=args.workers,
        batch_size=args.batch_size,
        warmup_steps=args.warmup,
        measured_steps=args.measured,
        use_synthetic=args.synthetic,
    )

    # Part 3: Print Complete Diagnostics Report
    print_deep_profiling_report(b1_results, b2_results)


if __name__ == "__main__":
    main()
