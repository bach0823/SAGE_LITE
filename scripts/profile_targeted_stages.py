"""
Targeted Runtime Profiler for SAGE-Lite B2-D12 on Crack500 (Tesla T4)

Deep Investigation of SageLayer 0 (CNN Stage 0) and SageLayer 1 (CNN Stage 1):
- Input shape & actual spatial token count passed to each expert (CNN vs ViT)
- Per-expert breakdown: selection count, call count, bypass count, compute time, ms/call
- SAHub input adaptation vs output adaptation vs expert forward compute vs index_add_
- Accounting reconciliation: Expert Path Time vs Sum of Sub-Components (Delta analysis)
- Aggregate comparison: Source Layer -> Target Family (CNN vs ViT)

Usage:
    python scripts/profile_targeted_stages.py --config configs/b2_crack500_depth12.yaml --warmup 2 --measured 10
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
from sage.networks.b2_unet import create_b2_unet
from sage.utils.dataloader import get_dataset_from_config
from sage.utils.training_utils import seed_worker, set_seed
from scripts.train_crack import CrackBinaryLoss, get_optimizer_groups


def print_banner(text: str, ch: str = "=", width: int = 85):
    line = ch * width
    print(f"\n{line}\n{text}\n{line}")


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
# TARGETED PROFILING COLLECTOR
# ==============================================================================
class TargetedCollector:
    def __init__(self):
        self.active = False

        # Macro step timings
        self.step_data_wait_ms = []
        self.step_fwd_ms = []
        self.step_bwd_ms = []
        self.step_opt_ms = []
        self.step_total_ms = []

        # Layer-level total time (layer_idx -> list of ms)
        self.layer_total_ms = defaultdict(list)
        self.layer_main_ms = defaultdict(list)
        self.layer_expert_path_ms = defaultdict(list)

        # Micro-timings per layer (layer_idx -> list of ms)
        self.layer_router_ms = defaultdict(list)
        self.layer_sahub_in_ms = defaultdict(list)
        self.layer_expert_compute_ms = defaultdict(list)
        self.layer_sahub_out_ms = defaultdict(list)
        self.layer_index_add_ms = defaultdict(list)
        self.layer_bypass_ms = defaultdict(list)

        # Detailed stats for (source_layer, target_expert):
        # Key: (source_layer, target_expert)
        self.trans_selections = defaultdict(int)         # Selections count (sample-level assignments)
        self.trans_forward_calls = defaultdict(int)      # Actual forward execution count
        self.trans_bypass_calls = defaultdict(int)       # Self-bypass count
        self.trans_input_shapes = defaultdict(set)       # Set of shapes passed to expert
        self.trans_tokens_per_sample = defaultdict(int)  # Spatial tokens per sample
        self.trans_total_tokens = defaultdict(int)       # Total tokens processed
        self.trans_expert_time_ms = defaultdict(float)   # Total expert forward time
        self.trans_sahub_in_time_ms = defaultdict(float) # Total SAHub input adapt time
        self.trans_sahub_out_time_ms = defaultdict(float)# Total SAHub output adapt time

        # Temporary event storage for current step
        self._current_step_events = []

    def start_step(self):
        self._current_step_events.clear()

    def record_event_pair(self, category: str, key: Any, start_ev: torch.cuda.Event, end_ev: torch.cuda.Event):
        if self.active:
            self._current_step_events.append((category, key, start_ev, end_ev))

    def resolve_step_events(self):
        if not self.active:
            return

        step_layer_router = defaultdict(float)
        step_layer_sahub_in = defaultdict(float)
        step_layer_compute = defaultdict(float)
        step_layer_sahub_out = defaultdict(float)
        step_layer_add = defaultdict(float)
        step_layer_bypass = defaultdict(float)

        for cat, key, start_ev, end_ev in self._current_step_events:
            dur = start_ev.elapsed_time(end_ev)
            if cat == "router":
                layer_idx = key
                step_layer_router[layer_idx] += dur
            elif cat == "sahub_in":
                src_layer, tgt_expert = key
                step_layer_sahub_in[src_layer] += dur
                self.trans_sahub_in_time_ms[(src_layer, tgt_expert)] += dur
            elif cat == "expert_compute":
                src_layer, tgt_expert = key
                step_layer_compute[src_layer] += dur
                self.trans_expert_time_ms[(src_layer, tgt_expert)] += dur
            elif cat == "sahub_out":
                src_layer, tgt_expert = key
                step_layer_sahub_out[src_layer] += dur
                self.trans_sahub_out_time_ms[(src_layer, tgt_expert)] += dur
            elif cat == "index_add":
                src_layer = key
                step_layer_add[src_layer] += dur
            elif cat == "bypass":
                src_layer = key
                step_layer_bypass[src_layer] += dur

        for layer_idx in range(16):
            self.layer_router_ms[layer_idx].append(step_layer_router[layer_idx])
            self.layer_sahub_in_ms[layer_idx].append(step_layer_sahub_in[layer_idx])
            self.layer_expert_compute_ms[layer_idx].append(step_layer_compute[layer_idx])
            self.layer_sahub_out_ms[layer_idx].append(step_layer_sahub_out[layer_idx])
            self.layer_index_add_ms[layer_idx].append(step_layer_add[layer_idx])
            self.layer_bypass_ms[layer_idx].append(step_layer_bypass[layer_idx])


collector = TargetedCollector()


# ==============================================================================
# INSTRUMENTATION HOOKS (ZERO ARCHITECTURE CHANGES)
# ==============================================================================
def instrument_sage_layer(orig_forward, orig_execute_expert_path):
    def instrumented_forward(self, x: torch.Tensor, expert_pool: Optional[nn.ModuleList] = None) -> torch.Tensor:
        if not collector.active:
            return orig_forward(self, x, expert_pool)

        pool = expert_pool if expert_pool is not None else self.expert_pool
        ev_layer_start = torch.cuda.Event(enable_timing=True)
        ev_layer_end = torch.cuda.Event(enable_timing=True)
        ev_main_start = torch.cuda.Event(enable_timing=True)
        ev_main_end = torch.cuda.Event(enable_timing=True)
        ev_exp_start = torch.cuda.Event(enable_timing=True)
        ev_exp_end = torch.cuda.Event(enable_timing=True)

        ev_layer_start.record()

        # Step 1: Main Path
        ev_main_start.record()
        main_output = self._execute_main_path(x)
        ev_main_end.record()

        # Step 2: Expert Path
        ev_exp_start.record()
        expert_output, routing_info = self._execute_expert_path(x, main_output, pool)
        ev_exp_end.record()

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
        l_idx = self.my_index if self.my_index is not None else -1
        collector.layer_total_ms[l_idx].append(ev_layer_start.elapsed_time(ev_layer_end))
        collector.layer_main_ms[l_idx].append(ev_main_start.elapsed_time(ev_main_end))
        collector.layer_expert_path_ms[l_idx].append(ev_exp_start.elapsed_time(ev_exp_end))

        return final_output

    def instrumented_execute_expert_path(self, x: torch.Tensor, main_output: torch.Tensor, expert_pool: nn.ModuleList):
        if not collector.active:
            return orig_execute_expert_path(self, x, main_output, expert_pool)

        B, *dims = main_output.shape
        device = x.device
        source_layer = self.my_index if self.my_index is not None else -1

        # Router timing
        ev_r_start = torch.cuda.Event(enable_timing=True)
        ev_r_end = torch.cuda.Event(enable_timing=True)
        ev_r_start.record()
        top_k_indices, gating_weights, routing_info = self.router(x)
        ev_r_end.record()
        collector.record_event_pair("router", source_layer, ev_r_start, ev_r_end)

        self._last_lb_loss = routing_info.get("load_balance_loss", None)

        flat_indices = top_k_indices.flatten()
        flat_weights = gating_weights.flatten()
        batch_map = torch.arange(B, device=device).repeat_interleave(self.router.top_k)
        final_expert_output = torch.zeros_like(main_output)

        for expert_idx, expert in enumerate(expert_pool):
            expert_mask = (flat_indices == expert_idx)
            if not expert_mask.any():
                continue

            original_batch_indices = batch_map[expert_mask]
            sub_b = len(original_batch_indices)
            collector.trans_selections[(source_layer, expert_idx)] += sub_b

            if self.my_index is not None and expert_idx == self.my_index:
                collector.trans_bypass_calls[(source_layer, expert_idx)] += 1
                ev_byp_s = torch.cuda.Event(enable_timing=True)
                ev_byp_e = torch.cuda.Event(enable_timing=True)
                ev_byp_s.record()
                adapted_expert_output = main_output[original_batch_indices]
                ev_byp_e.record()
                collector.record_event_pair("bypass", source_layer, ev_byp_s, ev_byp_e)
            else:
                collector.trans_forward_calls[(source_layer, expert_idx)] += 1

                # 4a: SAHub Input Adapt
                ev_in_s = torch.cuda.Event(enable_timing=True)
                ev_in_e = torch.cuda.Event(enable_timing=True)
                ev_in_s.record()
                adapted_input, _ = self.sa_hub.adapt(x[original_batch_indices], expert)
                ev_in_e.record()
                collector.record_event_pair("sahub_in", (source_layer, expert_idx), ev_in_s, ev_in_e)

                # Record input shape & spatial positions/tokens
                in_shape = tuple(adapted_input.shape)
                collector.trans_input_shapes[(source_layer, expert_idx)].add(str(in_shape))
                if adapted_input.dim() == 4:
                    # CNN format: (B_sub, C, H, W)
                    tokens_per_sample = adapted_input.shape[2] * adapted_input.shape[3]
                elif adapted_input.dim() == 3:
                    # Transformer format: (B_sub, N, D)
                    tokens_per_sample = adapted_input.shape[1]
                else:
                    tokens_per_sample = 1

                collector.trans_tokens_per_sample[(source_layer, expert_idx)] = tokens_per_sample
                collector.trans_total_tokens[(source_layer, expert_idx)] += sub_b * tokens_per_sample

                # 4b: Expert Compute
                ev_comp_s = torch.cuda.Event(enable_timing=True)
                ev_comp_e = torch.cuda.Event(enable_timing=True)
                ev_comp_s.record()
                expert_raw_output = expert(adapted_input)
                if isinstance(expert_raw_output, tuple):
                    expert_raw_output = expert_raw_output[0]
                ev_comp_e.record()
                collector.record_event_pair("expert_compute", (source_layer, expert_idx), ev_comp_s, ev_comp_e)

                # 4c: SAHub Output Adapt
                main_path_shape_subset = main_output[original_batch_indices].shape
                ev_out_s = torch.cuda.Event(enable_timing=True)
                ev_out_e = torch.cuda.Event(enable_timing=True)
                ev_out_s.record()
                adapted_expert_output, _ = self.sa_hub.adapt(
                    expert_raw_output,
                    self.main_block,
                    main_path_shape=main_path_shape_subset,
                )
                ev_out_e.record()
                collector.record_event_pair("sahub_out", (source_layer, expert_idx), ev_out_s, ev_out_e)

            # 4d & 5: Index Add
            weights_for_expert = flat_weights[expert_mask]
            reshape_dims = [-1] + [1] * (adapted_expert_output.dim() - 1)
            weighted_output = adapted_expert_output * weights_for_expert.view(reshape_dims)

            ev_add_s = torch.cuda.Event(enable_timing=True)
            ev_add_e = torch.cuda.Event(enable_timing=True)
            ev_add_s.record()
            final_expert_output.index_add_(0, original_batch_indices, weighted_output.to(final_expert_output.dtype))
            ev_add_e.record()
            collector.record_event_pair("index_add", source_layer, ev_add_s, ev_add_e)

        self.expert_successes += 1
        return final_expert_output, routing_info

    return instrumented_forward, instrumented_execute_expert_path


# ==============================================================================
# MAIN PROFILING RUN
# ==============================================================================
def run_targeted_profiling(
    config: dict,
    device: torch.device,
    batch_size: int = 12,
    num_workers: int = 2,
    warmup_steps: int = 2,
    measured_steps: int = 10,
    use_synthetic: bool = False,
):
    print_banner("TARGETED RUNTIME PROFILING B2-D12 (SAGE-Lite)")
    img_size = int(config.get("img_size", 448))
    seed = int(config.get("seed", 42))
    set_seed(seed)

    orig_layer_fwd = SageLayer.forward
    orig_layer_exp = SageLayer._execute_expert_path

    instr_fwd, instr_exp = instrument_sage_layer(orig_layer_fwd, orig_layer_exp)
    SageLayer.forward = instr_fwd
    SageLayer._execute_expert_path = instr_exp

    print("[Model] Loading B2-D12 model...")
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

    loader_iter = iter(loader)
    total_steps = warmup_steps + measured_steps

    torch.cuda.reset_peak_memory_stats(device)

    for step in range(1, total_steps + 1):
        is_measured = step > warmup_steps
        collector.active = is_measured
        collector.start_step()

        t_wait_s = time.perf_counter()
        batch = next(loader_iter)
        t_wait_ms = (time.perf_counter() - t_wait_s) * 1000.0

        images = batch["image"].to(device, non_blocking=True)
        masks = batch["label"].to(device, non_blocking=True)

        ev_fwd_s = torch.cuda.Event(enable_timing=True)
        ev_fwd_e = torch.cuda.Event(enable_timing=True)
        ev_bwd_s = torch.cuda.Event(enable_timing=True)
        ev_bwd_e = torch.cuda.Event(enable_timing=True)
        ev_opt_s = torch.cuda.Event(enable_timing=True)
        ev_opt_e = torch.cuda.Event(enable_timing=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward
        ev_fwd_s.record()
        with torch.amp.autocast("cuda", enabled=True):
            fwd_out = model.forward_with_routing_info(images)
            logits = fwd_out["logits"]
            routing_infos = fwd_out["routing_infos"]
            seg_loss = criterion(logits, masks)
            lb_loss = model.compute_total_load_balance_loss(routing_infos)
            total_loss = seg_loss + lb_loss
        ev_fwd_e.record()

        # Backward
        ev_bwd_s.record()
        scaler.scale(total_loss).backward()
        ev_bwd_e.record()

        # Optimizer
        ev_opt_s.record()
        scaler.step(optimizer)
        scaler.update()
        ev_opt_e.record()

        torch.cuda.synchronize(device)
        collector.resolve_step_events()

        fwd_ms = ev_fwd_s.elapsed_time(ev_fwd_e)
        bwd_ms = ev_bwd_s.elapsed_time(ev_bwd_e)
        opt_ms = ev_opt_s.elapsed_time(ev_opt_e)
        tot_ms = t_wait_ms + fwd_ms + bwd_ms + opt_ms

        tag = "MEASURED" if is_measured else "WARMUP"
        print(f"  Step {step:02d}/{total_steps:02d} [{tag}] | Data: {t_wait_ms:5.1f}ms | Fwd: {fwd_ms:6.1f}ms | Bwd: {bwd_ms:6.1f}ms | Opt: {opt_ms:5.1f}ms | Step: {tot_ms:6.1f}ms | Loss: {total_loss.item():.4f}")

        if is_measured:
            collector.step_data_wait_ms.append(t_wait_ms)
            collector.step_fwd_ms.append(fwd_ms)
            collector.step_bwd_ms.append(bwd_ms)
            collector.step_opt_ms.append(opt_ms)
            collector.step_total_ms.append(tot_ms)

    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    # Restore uninstrumented methods
    SageLayer.forward = orig_layer_fwd
    SageLayer._execute_expert_path = orig_layer_exp

    return peak_vram_mb, measured_steps, batch_size


# ==============================================================================
# REPORTING GENERATOR
# ==============================================================================
def print_targeted_report(peak_vram_mb: float, measured_steps: int, batch_size: int):
    print_banner("TARGETED PROFILING REPORT: SAGELAYER 0 & 1 FOCUS", ch="=")

    # 1. Macro Overview
    mean_fwd = np.mean(collector.step_fwd_ms)
    mean_bwd = np.mean(collector.step_bwd_ms)
    mean_step = np.mean(collector.step_total_ms)
    print(f"\n1. MACRO SUMMARY (Across {measured_steps} Measured Steps)")
    print("-" * 75)
    print(f"  Data Wait:    {np.mean(collector.step_data_wait_ms):8.2f} ms")
    print(f"  Forward Pass: {mean_fwd:8.2f} ms")
    print(f"  Backward Pass:{mean_bwd:8.2f} ms")
    print(f"  Optimizer:    {np.mean(collector.step_opt_ms):8.2f} ms")
    print(f"  Total Step:   {mean_step:8.2f} ms ({mean_step/1000.0:.2f} s/batch)")
    print(f"  Throughput:   {batch_size / (mean_step / 1000.0):8.2f} images/sec")
    print(f"  Peak VRAM:    {peak_vram_mb:8.1f} MB ({peak_vram_mb/1024:.2f} GB)")
    print("-" * 75)

    # 2. Per-SageLayer Timing Breakdown
    print("\n2. PER-SAGELAYER FORWARD TIMINGS (16 Layers)")
    print("-" * 80)
    print(f"{'Layer':<25} | {'Total (ms)':<11} | {'Main Path':<11} | {'Expert Path':<12} | {'% Forward':<10}")
    print("-" * 80)
    for idx in range(16):
        lname = f"SageLayer {idx:02d} (CNN Stg {idx})" if idx < 4 else f"SageLayer {idx:02d} (ViT Blk {idx-4:02d})"
        tot = np.mean(collector.layer_total_ms[idx])
        main = np.mean(collector.layer_main_ms[idx])
        exp = np.mean(collector.layer_expert_path_ms[idx])
        pct = (tot / mean_fwd) * 100.0
        print(f"{lname:<25} | {tot:>9.2f}ms | {main:>9.2f}ms | {exp:>10.2f}ms | {pct:>8.1f}%")
    print("-" * 80)

    # Helper function for stage breakdown
    def print_stage_detailed_breakdown(stage_idx: int):
        stage_name = f"SageLayer {stage_idx:02d} (CNN Stage {stage_idx})"
        print(f"\n--- DETAILED EXPERT-BY-EXPERT BREAKDOWN FOR {stage_name} ---")
        print("-" * 115)
        print(f"{'Target Expert':<28} | {'Type':<5} | {'Select':<6} | {'Calls':<5} | {'Bypass':<6} | {'Input Shape':<20} | {'Tokens/Smp':<10} | {'Total ms':<10} | {'ms/call':<9}")
        print("-" * 115)

        cnn_total_ms = 0.0
        cnn_calls = 0
        cnn_tokens = 0
        cnn_selects = 0

        vit_total_ms = 0.0
        vit_calls = 0
        vit_tokens = 0
        vit_selects = 0

        for tgt in range(16):
            ttype = "CNN" if tgt < 4 else "ViT"
            tname = f"Expert {tgt:02d} (CNN Stg {tgt})" if tgt < 4 else f"Expert {tgt:02d} (ViT Blk {tgt-4:02d})"
            sel = collector.trans_selections[(stage_idx, tgt)]
            calls = collector.trans_forward_calls[(stage_idx, tgt)]
            byp = collector.trans_bypass_calls[(stage_idx, tgt)]
            shapes = list(collector.trans_input_shapes[(stage_idx, tgt)])
            shape_str = shapes[0] if len(shapes) > 0 else "N/A"
            tok_per_smp = collector.trans_tokens_per_sample[(stage_idx, tgt)]
            tot_tok = collector.trans_total_tokens[(stage_idx, tgt)]
            t_ms = collector.trans_expert_time_ms[(stage_idx, tgt)]
            ms_per_call = (t_ms / calls) if calls > 0 else 0.0

            if tgt < 4:
                cnn_total_ms += t_ms
                cnn_calls += calls
                cnn_tokens += tot_tok
                cnn_selects += sel
            else:
                vit_total_ms += t_ms
                vit_calls += calls
                vit_tokens += tot_tok
                vit_selects += sel

            print(f"{tname:<28} | {ttype:<5} | {sel:>6d} | {calls:>5d} | {byp:>6d} | {shape_str:<20} | {tok_per_smp:>10d} | {t_ms:>8.2f}ms | {ms_per_call:>7.2f}ms")

        print("-" * 115)
        print(f"  [SUBTOTAL CNN Experts 00-03]: Selections={cnn_selects}, Calls={cnn_calls}, Total Tokens={cnn_tokens:,}, Time={cnn_total_ms:.2f}ms, Mean={cnn_total_ms/max(1,cnn_calls):.2f}ms/call")
        print(f"  [SUBTOTAL ViT Experts 04-15]: Selections={vit_selects}, Calls={vit_calls}, Total Tokens={vit_tokens:,}, Time={vit_total_ms:.2f}ms, Mean={vit_total_ms/max(1,vit_calls):.2f}ms/call")
        print("-" * 115)

    # 3. Print Stage 0 and Stage 1 Detailed Breakdown
    print("\n3. STAGE 0 & STAGE 1 TARGETED PROFILING (Requirement B)")
    print_stage_detailed_breakdown(0)
    print_stage_detailed_breakdown(1)

    # 4. Aggregate Table (Requirement C)
    print("\n4. AGGREGATE SUMMARY TABLE (Requirement C)")
    print("-" * 105)
    print(f"{'Source Layer':<24} | {'Target Family':<14} | {'Selections':<10} | {'Actual Tokens':<15} | {'% Select':<9} | {'Expert Time':<12} | {'ms/call':<9}")
    print("-" * 105)

    tot_selections_all = sum(collector.trans_selections.values())

    combos = [
        ("SageLayer 0 (CNN Stage 0)", 0, "CNN", range(4)),
        ("SageLayer 0 (CNN Stage 0)", 0, "ViT", range(4, 16)),
        ("SageLayer 1 (CNN Stage 1)", 1, "CNN", range(4)),
        ("SageLayer 1 (CNN Stage 1)", 1, "ViT", range(4, 16)),
        ("All ViT Layers (04-15)", range(4, 16), "CNN", range(4)),
        ("All ViT Layers (04-15)", range(4, 16), "ViT", range(4, 16)),
    ]

    for label, s_range, t_family, t_range in combos:
        if isinstance(s_range, int):
            sources = [s_range]
        else:
            sources = list(s_range)

        sel = sum(collector.trans_selections[(s, t)] for s in sources for t in t_range)
        calls = sum(collector.trans_forward_calls[(s, t)] for s in sources for t in t_range)
        tokens = sum(collector.trans_total_tokens[(s, t)] for s in sources for t in t_range)
        t_ms = sum(collector.trans_expert_time_ms[(s, t)] for s in sources for t in t_range)
        pct = (sel / max(1, tot_selections_all)) * 100.0
        ms_per_call = (t_ms / calls) if calls > 0 else 0.0

        print(f"{label:<24} | {t_family:<14} | {sel:>10d} | {tokens:>15,d} | {pct:>8.1f}% | {t_ms:>10.2f}ms | {ms_per_call:>7.2f}ms")
    print("-" * 105)

    # 5. Clarification on "Samples" vs "Tokens" (Requirement D)
    print("\n5. AUDIT OF 'SAMPLES' VS 'ACTUAL TOKENS' (Requirement D)")
    print("-" * 85)
    print("  * 'Selections / Sample Assignments': Number of image instances in batch (B=12) routed.")
    print("  * 'Actual Tokens / Spatial Positions': B_sub * (H * W) for CNN, or B_sub * N for ViT.")
    print("  * Stage 0 Spatial Dimension: 112x112 = 12,544 spatial tokens per image.")
    print("  * Stage 1 Spatial Dimension: 56x56   =  3,136 spatial tokens per image.")
    print("  * ViT Spatial Dimension:     14x14   =    196 spatial tokens per image.")
    print("  * CONCLUSION: Routing Stage 0/1 to ViT triggers attention across 12,544 / 3,136 tokens!")
    print("-" * 85)

    # 6. Micro-Timing Reconciliation & Accounting (Requirement E)
    print("\n6. MICRO-TIMING RECONCILIATION & ACCOUNTING (Requirement E)")
    print("-" * 110)
    print(f"{'Layer':<22} | {'ExpPath (ms)':<12} | {'Router':<9} | {'SAHubIn':<9} | {'Compute':<9} | {'SAHubOut':<9} | {'IdxAdd':<8} | {'Bypass':<7} | {'Residual (ms)':<13}")
    print("-" * 110)
    for idx in range(16):
        lname = f"Layer {idx:02d} (CNN {idx})" if idx < 4 else f"Layer {idx:02d} (ViT {idx-4:02d})"
        t_exp = np.mean(collector.layer_expert_path_ms[idx])
        r = np.mean(collector.layer_router_ms[idx])
        sin = np.mean(collector.layer_sahub_in_ms[idx])
        comp = np.mean(collector.layer_expert_compute_ms[idx])
        sout = np.mean(collector.layer_sahub_out_ms[idx])
        add = np.mean(collector.layer_index_add_ms[idx])
        byp = np.mean(collector.layer_bypass_ms[idx])

        sum_components = r + sin + comp + sout + add + byp
        residual = t_exp - sum_components
        residual_pct = (residual / max(1e-3, t_exp)) * 100.0

        print(f"{lname:<22} | {t_exp:>10.2f}ms | {r:>7.2f}ms | {sin:>7.2f}ms | {comp:>7.2f}ms | {sout:>7.2f}ms | {add:>6.2f}ms | {byp:>5.2f}ms | {residual:>7.2f}ms ({residual_pct:>4.1f}%)")
    print("-" * 110)
    print("  * Accounting Explanation: Residual = Python loop dispatch overhead + tensor slicing/indexing x[indices] + cuda event boundary latencies.")
    print("-" * 110)
    print_banner("TARGETED PROFILING COMPLETED", ch="=")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Targeted Runtime Profiler for SageLayer 0 & 1")
    parser.add_argument("--config", type=str, default="configs/b2_crack500_depth12.yaml")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--measured", type=int, default=10)
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset for test")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_banner("INITIALIZING TARGETED PROFILER")
    print(f"[Device] Target Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        print(f"[Device] GPU: {gpu_name} | Total VRAM: {total_vram:.1f} MB ({total_vram/1024:.2f} GB)")

    with open(args.config, "r") as f:
        config_b2 = yaml.safe_load(f)
    config_b2["batch_size"] = args.batch_size
    config_b2["num_workers"] = args.workers
    config_b2["root_dir"] = resolve_data_root(config_b2.get("root_dir", "/content/dataset/Crack500"))

    peak_vram, measured_steps, batch_size = run_targeted_profiling(
        config=config_b2,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.workers,
        warmup_steps=args.warmup,
        measured_steps=args.measured,
        use_synthetic=args.synthetic,
    )

    print_targeted_report(peak_vram, measured_steps, batch_size)


if __name__ == "__main__":
    main()
