"""
Phase-1 Hyperparameter Screening Suite for P3-C (ASDW) Base Model
==================================================================

Research Paradigm:
- P3-C (Adaptive Spatial-Detail Wavelet Refinement) is the NEW BASE MODEL of this research branch.
- B2 Base (p3_mode=None) serves exclusively as the parent/initialization checkpoint.
- Fixed Invariants:
    * Backbone: ConvNeXt-V2-Femto + ViT Depth = 12 (D12)
    * Batch Size: 14 (BS14) - empirically validated on Tesla T4 for P3-C
    * Workers: 2
    * Architecture: P3-C (ASDW refinement on ConvNeXt stages 0 & 1, PE28 fixed buffer)
    * Gating: Sigmoid, Logit Modulation = True, Shared Experts = [0, 1, 2, 3]
    * Exploration Noise: Active in model.train()
    * Input resolution: 448x448
    * Dataset: Real Crack500 (1896 train samples, 348 val pairs)
    * Isolation: Validation split ONLY. Zero access to Test split.
    * Seed: 42 (deterministic)

Screening Grid (9 Candidates):
    * P3 LR: [5e-5, 1e-4, 2e-4]
    * gamma_init: [0.001, 0.01, 0.05]

Metrics Logged:
    * Val Dice, Val IoU, Val Loss (Final & at Best Dice)
    * gamma_S0, gamma_S1 (initial -> final values)
    * P3 gradient norm (mean across training batches)
    * P3 parameter delta (L2 distance from initial P3 weights)
    * sample_routing_entropy (mean scalar from full batch gating weights)
    * expert_utilization_entropy (Shannon entropy of expert usage)
    * active_experts / pool_size
    * VRAM Allocated / Reserved
    * Empirical timing (s/epoch)
    * NaN/Inf numerical stability check

Usage Examples:
    # 1. Screen all 9 P3-C candidates from parent checkpoint:
    python scripts/screen_p3_c_hyperparams.py --parent-checkpoint checkpoints/b2_base_d12.pt

    # 2. Test a single candidate:
    python scripts/screen_p3_c_hyperparams.py --p3-lr 1e-4 --gamma-init 0.01

    # 3. Dry-run verification on local CPU / small mock:
    python scripts/screen_p3_c_hyperparams.py --dry-run
"""

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from timm.scheduler.cosine_lr import CosineLRScheduler
import yaml

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sage.networks.b2_unet import create_b2_unet, B2ConvNeXtViTUNet
from sage.components.router import SageRouter
from sage.utils.dataloader import get_dataset_from_config
from sage.utils.training_utils import seed_worker, set_seed
from sage.utils.model_utils import load_locked_base_into_p3
from scripts.train_crack import (
    CrackBinaryLoss,
    calculate_binary_metrics,
    DEFAULT_SHARED_PREFIXES,
)
from scripts.evaluate_crack_official import evaluate_split, get_image_mask_pairs, resolve_protocol


class MockDataset(torch.utils.data.Dataset):
    def __init__(self, length: int = 56, image_size: int = 448):
        self.length = length
        self.image_size = image_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            'image': torch.randn(3, self.image_size, self.image_size),
            'label': torch.randint(0, 2, (1, self.image_size, self.image_size)).float()
        }


def print_banner(text: str, ch: str = "="):
    line = ch * 80
    print(f"\n{line}\n{text}\n{line}")


def get_git_commit_hash() -> str:
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        return subprocess.check_output(cmd, cwd=project_root).decode("ascii").strip()
    except Exception:
        return "UNKNOWN_COMMIT"


def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_sample_gating_entropy(routing_infos: Any) -> float:
    """
    Aggregates the lightweight scalar 'routing_entropy_mean' computed across the full batch
    by each active SAGE router: H = mean_b[-sum_k p_bk * log2(p_bk)].
    """
    if isinstance(routing_infos, dict):
        infos = routing_infos.get('all', [])
    elif isinstance(routing_infos, list):
        infos = routing_infos
    else:
        infos = []

    entropies = []
    for info in infos:
        if isinstance(info, dict) and 'routing_entropy_mean' in info:
            entropies.append(float(info['routing_entropy_mean']))
    return float(np.mean(entropies)) if entropies else 0.0


def calculate_routing_statistics(model: nn.Module) -> Dict[str, Any]:
    """
    Calculates expert-utilization statistics across all SageRouters.
    H_usage = -sum(u_e * log2(u_e)) measures the entropy of expert selection frequency across the pool.
    """
    routers = [m for m in model.modules() if isinstance(m, SageRouter)]
    if not routers:
        return {
            "expert_utilization_entropy": 0.0,
            "active_experts": 0,
            "pool_size": 16,
            "expert_utilization_pct": 0.0,
            "total_calls": 0,
        }

    usage_entropies = []
    total_usage = None
    total_calls = 0

    for r in routers:
        if hasattr(r, 'expert_usage_count'):
            usage = r.expert_usage_count.detach().cpu().numpy().astype(np.float64)
            if total_usage is None:
                total_usage = np.zeros_like(usage)
            total_usage += usage

            sum_usage = usage.sum()
            if sum_usage > 0:
                p = usage / sum_usage
                p_pos = p[p > 0]
                entropy = -float(np.sum(p_pos * np.log2(p_pos)))
                usage_entropies.append(entropy)

        if hasattr(r, 'total_calls'):
            total_calls += int(r.total_calls.item())

    pool_size = len(total_usage) if total_usage is not None else 16
    active_count = int(np.sum(total_usage > 0)) if total_usage is not None else 0
    utilization_pct = (active_count / pool_size * 100.0) if pool_size > 0 else 0.0
    mean_usage_entropy = float(np.mean(usage_entropies)) if usage_entropies else 0.0

    return {
        "expert_utilization_entropy": round(mean_usage_entropy, 4),
        "active_experts": active_count,
        "pool_size": pool_size,
        "expert_utilization_pct": round(utilization_pct, 2),
        "total_calls": total_calls,
    }


def create_reproducible_train_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    seed: int = 42,
    pin_memory: bool = True,
) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=pin_memory,
        drop_last=False,
    )


def set_p3_gamma_init(model: nn.Module, gamma_val: float):
    """Initializes gamma parameters in all P3 refinement stages to the candidate value."""
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'convnext') and hasattr(model.backbone.convnext, 'stages'):
        for stage in model.backbone.convnext.stages:
            if hasattr(stage, 'p3_refinement') and stage.p3_refinement is not None:
                if hasattr(stage.p3_refinement, 'gamma'):
                    with torch.no_grad():
                        stage.p3_refinement.gamma.fill_(gamma_val)


def get_p3_gamma_values(model: nn.Module) -> Tuple[float, float]:
    """Retrieves current gamma values for Stage 0 and Stage 1."""
    g0, g1 = float('nan'), float('nan')
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'convnext') and hasattr(model.backbone.convnext, 'stages'):
        stages = model.backbone.convnext.stages
        if len(stages) > 0 and hasattr(stages[0], 'p3_refinement') and stages[0].p3_refinement is not None:
            if hasattr(stages[0].p3_refinement, 'gamma'):
                g0 = float(stages[0].p3_refinement.gamma.item())
        if len(stages) > 1 and hasattr(stages[1], 'p3_refinement') and stages[1].p3_refinement is not None:
            if hasattr(stages[1].p3_refinement, 'gamma'):
                g1 = float(stages[1].p3_refinement.gamma.item())
    return g0, g1


def get_p3_c_optimizer_groups(
    model: nn.Module,
    lr_base: float,
    lr_p3: float,
    weight_decay: float = 0.05,
) -> List[Dict[str, Any]]:
    """Constructs Stage 1 optimizer parameter groups with dedicated P3 tier."""
    interface_keys = (
        'convnext_to_transformer',
        'transformer_to_decoder',
        'pre_transformer_norm',
        'post_transformer_norm',
    )
    lr_backbone = lr_base * 0.1
    lr_decoder = lr_base
    lr_sage = lr_base

    groups = {
        'backbone_decay': {'params': [], 'weight_decay': weight_decay, 'lr': lr_backbone, 'name': 'backbone'},
        'backbone_no_decay': {'params': [], 'weight_decay': 0.0, 'lr': lr_backbone, 'name': 'backbone'},
        'decoder_decay': {'params': [], 'weight_decay': weight_decay, 'lr': lr_decoder, 'name': 'decoder'},
        'decoder_no_decay': {'params': [], 'weight_decay': 0.0, 'lr': lr_decoder, 'name': 'decoder'},
        'sage_decay': {'params': [], 'weight_decay': weight_decay, 'lr': lr_sage, 'name': 'sage'},
        'sage_no_decay': {'params': [], 'weight_decay': 0.0, 'lr': lr_sage, 'name': 'sage'},
        'p3_decay': {'params': [], 'weight_decay': weight_decay, 'lr': lr_p3, 'name': 'p3_refinement'},
        'p3_no_decay': {'params': [], 'weight_decay': 0.0, 'lr': lr_p3, 'name': 'p3_refinement'},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_no_decay = 'layernorm' in name.lower() or 'norm' in name.lower() or name.endswith('.bias') or 'gamma' in name.lower()

        if 'p3_refinement' in name:
            tier = 'p3'
        elif 'router' in name or 'sa_hub' in name or 'alpha' in name:
            tier = 'sage'
        elif name.startswith('decoder') or any(k in name for k in interface_keys):
            tier = 'decoder'
        else:
            tier = 'backbone'

        group_key = f"{tier}_no_decay" if is_no_decay else f"{tier}_decay"
        groups[group_key]['params'].append(param)

    return [g for g in groups.values() if len(g['params']) > 0]


def create_p3_c_stage2_optimizer(
    model: nn.Module,
    stage2_base_lr: float,
    stage2_shared_lr: float,
    stage2_p3_lr: float,
    shared_prefixes: Optional[Set[str]] = None,
    weight_decay: float = 0.05,
) -> optim.Optimizer:
    """Constructs Stage 2 optimizer parameter groups separating shared experts, base, and P3."""
    if shared_prefixes is None:
        shared_prefixes = DEFAULT_SHARED_PREFIXES

    groups = {
        'shared_decay': {'params': [], 'weight_decay': weight_decay, 'lr': stage2_shared_lr, 'name': 'shared_experts'},
        'shared_no_decay': {'params': [], 'weight_decay': 0.0, 'lr': stage2_shared_lr, 'name': 'shared_experts'},
        'p3_decay': {'params': [], 'weight_decay': weight_decay, 'lr': stage2_p3_lr, 'name': 'p3_refinement'},
        'p3_no_decay': {'params': [], 'weight_decay': 0.0, 'lr': stage2_p3_lr, 'name': 'p3_refinement'},
        'others_decay': {'params': [], 'weight_decay': weight_decay, 'lr': stage2_base_lr, 'name': 'other_and_routers'},
        'others_no_decay': {'params': [], 'weight_decay': 0.0, 'lr': stage2_base_lr, 'name': 'other_and_routers'},
    }

    shared_param_ids = set()
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'convnext') and hasattr(model.backbone.convnext, 'stages'):
        for stage_idx in range(min(4, len(model.backbone.convnext.stages))):
            stage = model.backbone.convnext.stages[stage_idx]
            block = stage.main_block if hasattr(stage, 'main_block') else stage
            for p in block.parameters():
                shared_param_ids.add(id(p))

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_no_decay = 'layernorm' in name.lower() or 'norm' in name.lower() or name.endswith('.bias') or 'gamma' in name.lower()

        if 'p3_refinement' in name:
            tier = 'p3'
        elif (id(param) in shared_param_ids) or any(name.startswith(p) for p in shared_prefixes):
            tier = 'shared'
        else:
            tier = 'others'

        group_key = f"{tier}_no_decay" if is_no_decay else f"{tier}_decay"
        groups[group_key]['params'].append(param)

    param_groups = [g for g in groups.values() if len(g['params']) > 0]
    return optim.AdamW(param_groups)


def run_p3_c_screening_trial(
    candidate_id: str,
    base_config: dict,
    p3_lr: float,
    gamma_init: float,
    parent_checkpoint: Optional[str],
    train_dataset: torch.utils.data.Dataset,
    loader_workers: int,
    val_pairs: List[Tuple[str, str]],
    device: torch.device,
    stage1_epochs: int = 3,
    stage2_epochs: int = 3,
    warmup_epochs: int = 1,
    dry_run: bool = False,
    max_val_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Executes a two-stage screening run for a single P3-C candidate."""
    cfg = copy.deepcopy(base_config)
    sage_cfg = cfg.get('sage_config', {})

    vit_depth = int(cfg.get('num_transformer_layers', 12))
    batch_size = int(cfg.get('batch_size', 14))
    img_size = int(cfg.get('img_size', 448))
    seed = int(cfg.get('seed', 42))
    base_lr = float(cfg.get('lr', 1e-4))
    stage2_base_lr = float(cfg.get('stage2_base_lr', 1e-4))
    stage2_shared_lr = float(cfg.get('stage2_shared_lr', 1e-4))

    set_seed(seed)

    print_banner(f"STARTING P3-C SCREENING TRIAL: {candidate_id}")
    print(f"Hyperparameters: P3 LR = {p3_lr:.2e} | gamma_init = {gamma_init:.4f}")
    print(f"Model Invariants: ViT Depth = {vit_depth}, BS = {batch_size}, Seed = {seed}, p3_mode = 'C' (ASDW)")
    print(f"Screening Protocol: Stage 1 = {stage1_epochs} ep, Stage 2 = {stage2_epochs} ep (warmup = {warmup_epochs})")

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # 1. Instantiate P3-C UNet Model
    model = create_b2_unet(
        num_transformer_layers=vit_depth,
        pretrained=not dry_run and (parent_checkpoint is None),
        sage_config=sage_cfg,
        p3_mode='C', # P3-C ASDW refinement
    ).to(device)

    assert isinstance(model, B2ConvNeXtViTUNet), "Model instantiation error!"
    assert model.p3_mode == 'C', f"Expected p3_mode='C', got '{model.p3_mode}'!"

    # 2. Ingest Parent Base Checkpoint if provided
    if parent_checkpoint and os.path.exists(parent_checkpoint):
        print(f"  Ingesting parent base checkpoint from {parent_checkpoint}...")
        load_locked_base_into_p3(model, parent_checkpoint, p3_mode='C')
    elif dry_run:
        print("  [Notice] Dry-run mode: initialized P3-C without parent checkpoint.")
    else:
        print("  [Notice] Initializing P3-C directly from pretrained backbone (no parent checkpoint).")

    # 3. Apply candidate gamma_init
    set_p3_gamma_init(model, gamma_init)
    gamma_s0_init, gamma_s1_init = get_p3_gamma_values(model)
    print(f"  P3 gamma initialized: S0 = {gamma_s0_init:.4f}, S1 = {gamma_s1_init:.4f}")

    # 4. Snapshot initial P3 parameters to compute delta
    p3_initial_weights = {
        name: param.detach().clone()
        for name, param in model.named_parameters() if 'p3_refinement' in name
    }

    criterion = CrackBinaryLoss(bce_weight=1.0, dice_weight=1.5)
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda'))

    eval_protocol = resolve_protocol(cfg)
    active_val_pairs = val_pairs[:max_val_samples] if max_val_samples else val_pairs

    start_trial_time = time.perf_counter()
    nan_inf_detected = False
    epoch_durations = []
    p3_batch_grad_norms = []

    # =========================================================================
    # STAGE 1: Free Exploration & P3 Refinement Learning
    # =========================================================================
    print(f"\n--- [Stage 1 Training ({stage1_epochs} Epochs)] ---")
    set_seed(seed)
    stage1_loader = create_reproducible_train_loader(
        train_dataset,
        batch_size=batch_size,
        num_workers=loader_workers,
        seed=seed,
        pin_memory=(device.type == 'cuda'),
    )

    stage1_groups = get_p3_c_optimizer_groups(
        model,
        lr_base=base_lr,
        lr_p3=p3_lr,
        weight_decay=0.05,
    )
    opt1 = optim.AdamW(stage1_groups)
    sched1 = CosineLRScheduler(
        opt1,
        t_initial=stage1_epochs,
        lr_min=1e-6,
        warmup_t=warmup_epochs,
        warmup_lr_init=1e-6,
        t_in_epochs=True,
    )

    stage1_final_val_dice = 0.0
    stage1_final_val_loss = float('inf')
    stage1_best_val_dice = -1.0
    stage1_val_loss_at_best_dice = float('inf')
    best_stage1_weights = None
    s1_sample_entropies = []

    for epoch in range(1, stage1_epochs + 1):
        t_ep_start = time.perf_counter()
        model.train() # Exploration noise is active
        train_loss, train_seg_loss, train_lb_loss = 0.0, 0.0, 0.0
        train_dice, train_iou = 0.0, 0.0
        train_sample_entropy = 0.0
        batches_processed = 0

        for b_idx, batch in enumerate(stage1_loader, 1):
            if dry_run and b_idx > 2:
                break

            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True) if 'label' in batch else batch['mask'].to(device, non_blocking=True)
            if labels.dim() == 3:
                labels = labels.unsqueeze(1)

            opt1.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda')):
                fwd_out = model.forward_with_routing_info(images)
                logits = fwd_out['logits']
                routing_infos = fwd_out['routing_infos']
                lb_loss = model.compute_total_load_balance_loss(routing_infos)
                seg_loss = criterion(logits, labels)
                # Keep exact LB loss: factor already multiplied inside SageRouter
                total_loss = seg_loss + 1.0 * lb_loss

            if not torch.isfinite(total_loss):
                print(f"  [ERROR] Non-finite loss at Stage 1, Epoch {epoch}, Batch {b_idx}: {total_loss.item()}")
                nan_inf_detected = True
                break

            scaler.scale(total_loss).backward()

            # Measure P3 gradient norm before unscaling/optimizer step
            p3_grads = [p.grad for name, p in model.named_parameters() if 'p3_refinement' in name and p.grad is not None]
            if p3_grads:
                gnorm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in p3_grads]), 2).item()
                p3_batch_grad_norms.append(gnorm)

            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    nan_inf_detected = True
                    break

            scaler.step(opt1)
            scaler.update()

            probs = torch.sigmoid(logits)
            _, d_val, iou_val = calculate_binary_metrics(probs, labels)
            sample_h = compute_sample_gating_entropy(routing_infos)

            train_loss += total_loss.item()
            train_seg_loss += seg_loss.item()
            train_lb_loss += lb_loss.item()
            train_dice += d_val
            train_iou += iou_val
            train_sample_entropy += sample_h
            batches_processed += 1

        sched1.step(epoch)

        num_b = max(1, batches_processed)
        avg_train_loss = train_loss / num_b
        avg_train_dice = train_dice / num_b
        avg_train_iou = train_iou / num_b
        avg_train_lb = train_lb_loss / num_b
        s1_sample_entropies.append(train_sample_entropy / num_b)

        model.eval()
        with torch.no_grad():
            if dry_run:
                val_metrics = {'loss': 2.0, 'dice': 0.15, 'iou': 0.08}
            else:
                val_metrics = evaluate_split(
                    model,
                    active_val_pairs,
                    device,
                    protocol=eval_protocol,
                    tile_size=img_size,
                    criterion=criterion,
                    verbose=False,
                )

        ep_duration = time.perf_counter() - t_ep_start
        epoch_durations.append(ep_duration)
        val_dice = float(val_metrics.get('dice', 0.0))
        val_loss = float(val_metrics.get('loss', 0.0))

        stage1_final_val_dice = val_dice
        stage1_final_val_loss = val_loss

        if val_dice > stage1_best_val_dice:
            stage1_best_val_dice = val_dice
            stage1_val_loss_at_best_dice = val_loss
            best_stage1_weights = copy.deepcopy(model.state_dict())

        cur_g0, cur_g1 = get_p3_gamma_values(model)
        print(
            f"  [Stage 1][Epoch {epoch:02d}/{stage1_epochs:02d}] "
            f"Train Loss: {avg_train_loss:.4f} (LB: {avg_train_lb:.4f}, Dice: {avg_train_dice:.4f}) | "
            f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f} | "
            f"gamma: [S0={cur_g0:.4f}, S1={cur_g1:.4f}] | "
            f"Time: {ep_duration:.1f}s"
        )

    # =========================================================================
    # STAGE 2: Shared Experts Locked & Two-Speed Optimization
    # =========================================================================
    print(f"\n--- [Stage 2 Training ({stage2_epochs} Epochs)] ---")
    if best_stage1_weights is not None:
        model.load_state_dict(best_stage1_weights)
        print("  Reloaded best Stage 1 weights successfully.")

    shared_indices = [0, 1, 2, 3]
    model.set_shared_experts(shared_indices)
    print(f"  Stage 2 shared experts initialized: {shared_indices}")

    set_seed(seed + 1000)
    stage2_loader = create_reproducible_train_loader(
        train_dataset,
        batch_size=batch_size,
        num_workers=loader_workers,
        seed=seed + 1000,
        pin_memory=(device.type == 'cuda'),
    )

    opt2 = create_p3_c_stage2_optimizer(
        model,
        stage2_base_lr=stage2_base_lr,
        stage2_shared_lr=stage2_shared_lr,
        stage2_p3_lr=p3_lr,
    )
    sched2 = CosineLRScheduler(
        opt2,
        t_initial=stage2_epochs,
        lr_min=1e-6,
        warmup_t=warmup_epochs,
        warmup_lr_init=1e-6,
        t_in_epochs=True,
    )

    stage2_final_val_dice = 0.0
    stage2_final_val_loss = float('inf')
    stage2_final_val_iou = 0.0
    stage2_best_val_dice = -1.0
    stage2_val_loss_at_best_dice = float('inf')
    stage2_val_iou_at_best_dice = 0.0
    s2_sample_entropies = []

    for epoch in range(1, stage2_epochs + 1):
        t_ep_start = time.perf_counter()
        model.train() # Exploration noise is active
        train_loss, train_seg_loss, train_lb_loss = 0.0, 0.0, 0.0
        train_dice, train_iou = 0.0, 0.0
        train_sample_entropy = 0.0
        batches_processed = 0

        for b_idx, batch in enumerate(stage2_loader, 1):
            if dry_run and b_idx > 2:
                break

            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True) if 'label' in batch else batch['mask'].to(device, non_blocking=True)
            if labels.dim() == 3:
                labels = labels.unsqueeze(1)

            opt2.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda')):
                fwd_out = model.forward_with_routing_info(images)
                logits = fwd_out['logits']
                routing_infos = fwd_out['routing_infos']
                lb_loss = model.compute_total_load_balance_loss(routing_infos)
                seg_loss = criterion(logits, labels)
                total_loss = seg_loss + 1.0 * lb_loss

            if not torch.isfinite(total_loss):
                print(f"  [ERROR] Non-finite loss at Stage 2, Epoch {epoch}, Batch {b_idx}: {total_loss.item()}")
                nan_inf_detected = True
                break

            scaler.scale(total_loss).backward()

            p3_grads = [p.grad for name, p in model.named_parameters() if 'p3_refinement' in name and p.grad is not None]
            if p3_grads:
                gnorm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in p3_grads]), 2).item()
                p3_batch_grad_norms.append(gnorm)

            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    nan_inf_detected = True
                    break

            scaler.step(opt2)
            scaler.update()

            probs = torch.sigmoid(logits)
            _, d_val, iou_val = calculate_binary_metrics(probs, labels)
            sample_h = compute_sample_gating_entropy(routing_infos)

            train_loss += total_loss.item()
            train_seg_loss += seg_loss.item()
            train_lb_loss += lb_loss.item()
            train_dice += d_val
            train_iou += iou_val
            train_sample_entropy += sample_h
            batches_processed += 1

        sched2.step(epoch)

        num_b = max(1, batches_processed)
        avg_train_loss = train_loss / num_b
        avg_train_dice = train_dice / num_b
        avg_train_lb = train_lb_loss / num_b
        s2_sample_entropies.append(train_sample_entropy / num_b)

        model.eval()
        with torch.no_grad():
            if dry_run:
                val_metrics = {'loss': 1.95, 'dice': 0.22, 'iou': 0.12}
            else:
                val_metrics = evaluate_split(
                    model,
                    active_val_pairs,
                    device,
                    protocol=eval_protocol,
                    tile_size=img_size,
                    criterion=criterion,
                    verbose=False,
                )

        ep_duration = time.perf_counter() - t_ep_start
        epoch_durations.append(ep_duration)
        val_dice = float(val_metrics.get('dice', 0.0))
        val_loss = float(val_metrics.get('loss', 0.0))
        val_iou = float(val_metrics.get('iou', 0.0))

        stage2_final_val_dice = val_dice
        stage2_final_val_loss = val_loss
        stage2_final_val_iou = val_iou

        if val_dice > stage2_best_val_dice:
            stage2_best_val_dice = val_dice
            stage2_val_loss_at_best_dice = val_loss
            stage2_val_iou_at_best_dice = val_iou

        cur_g0, cur_g1 = get_p3_gamma_values(model)
        print(
            f"  [Stage 2][Epoch {epoch:02d}/{stage2_epochs:02d}] "
            f"Train Loss: {avg_train_loss:.4f} (LB: {avg_train_lb:.4f}, Dice: {avg_train_dice:.4f}) | "
            f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f} | "
            f"gamma: [S0={cur_g0:.4f}, S1={cur_g1:.4f}] | "
            f"Time: {ep_duration:.1f}s"
        )

    total_trial_sec = time.perf_counter() - start_trial_time
    routing_stats = calculate_routing_statistics(model)
    mean_epoch_time_s = float(np.mean(epoch_durations)) if epoch_durations else 0.0
    avg_sample_entropy = float(np.mean(s2_sample_entropies)) if s2_sample_entropies else 0.0
    mean_p3_grad_norm = float(np.mean(p3_batch_grad_norms)) if p3_batch_grad_norms else 0.0

    # Compute P3 parameter delta: ||theta_P3^(final) - theta_P3^(init)||
    p3_param_deltas = [
        torch.norm(p.detach() - p3_initial_weights[name], 2)
        for name, p in model.named_parameters() if name in p3_initial_weights
    ]
    p3_total_param_delta = float(torch.norm(torch.stack(p3_param_deltas), 2).item()) if p3_param_deltas else 0.0

    final_gamma_s0, final_gamma_s1 = get_p3_gamma_values(model)

    peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == 'cuda' else 0.0
    peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024**2) if device.type == 'cuda' else 0.0

    trial_summary = {
        "candidate_id": candidate_id,
        "p3_lr": p3_lr,
        "gamma_init": gamma_init,
        # Segmentation Metrics
        "s1_final_val_dice": round(stage1_final_val_dice, 4),
        "s1_final_val_loss": round(stage1_final_val_loss, 4),
        "s1_best_val_dice": round(stage1_best_val_dice, 4),
        "s1_val_loss_at_best_dice": round(stage1_val_loss_at_best_dice, 4),
        "s2_final_val_dice": round(stage2_final_val_dice, 4),
        "s2_final_val_loss": round(stage2_final_val_loss, 4),
        "s2_final_val_iou": round(stage2_final_val_iou, 4),
        "s2_best_val_dice": round(stage2_best_val_dice, 4),
        "s2_val_loss_at_best_dice": round(stage2_val_loss_at_best_dice, 4),
        "s2_val_iou_at_best_dice": round(stage2_val_iou_at_best_dice, 4),
        # P3 Learning Dynamics Metrics
        "gamma_s0_init": round(gamma_s0_init, 4),
        "gamma_s0_final": round(final_gamma_s0, 4),
        "gamma_s1_init": round(gamma_s1_init, 4),
        "gamma_s1_final": round(final_gamma_s1, 4),
        "p3_param_delta": round(p3_total_param_delta, 6),
        "p3_grad_norm": round(mean_p3_grad_norm, 6),
        # Routing Metrics
        "sample_routing_entropy": round(avg_sample_entropy, 4),
        "expert_utilization_entropy": routing_stats['expert_utilization_entropy'],
        "active_experts": routing_stats['active_experts'],
        "pool_size": routing_stats['pool_size'],
        "expert_utilization_pct": routing_stats['expert_utilization_pct'],
        # Hardware & Stability
        "mean_epoch_time_s": round(mean_epoch_time_s, 2),
        "peak_allocated_mb": round(peak_allocated_mb, 1),
        "peak_reserved_mb": round(peak_reserved_mb, 1),
        "nan_inf_status": "FAIL" if nan_inf_detected else "CLEAN",
        "runtime_sec": round(total_trial_sec, 1),
        "runtime_min": round(total_trial_sec / 60.0, 2),
    }

    del model, opt1, opt2, sched1, sched2, scaler, best_stage1_weights
    del stage1_loader, stage2_loader, p3_initial_weights
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return trial_summary


def main():
    parser = argparse.ArgumentParser(description="Phase-1 P3-C (ASDW) Base Hyperparameter Screening Suite")
    parser.add_argument("--config", type=str, default="configs/p3_ablation/b2_p3_run_c.yaml",
                        help="Base YAML configuration file for Run C")
    parser.add_argument("--data-root", type=str, default="/content/dataset/Crack500",
                        help="Root directory of Crack500 dataset")
    parser.add_argument("--parent-checkpoint", type=str, default=None,
                        help="Path to parent B2 base checkpoint to initialize from (alias: --locked-base)")
    parser.add_argument("--locked-base", type=str, default=None,
                        help="Alias for --parent-checkpoint")
    parser.add_argument("--depth", type=int, default=12,
                        help="ViT Depth (default: 12)")
    parser.add_argument("--batch-size", type=int, default=14,
                        help="Batch Size (default: 14 - Locked sweet spot)")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader worker processes (default: 2)")
    parser.add_argument("--stage1-epochs", type=int, default=3,
                        help="Number of epochs for Stage 1 short-horizon screening (default: 3)")
    parser.add_argument("--stage2-epochs", type=int, default=3,
                        help="Number of epochs for Stage 2 short-horizon screening (default: 3)")
    parser.add_argument("--warmup-epochs", type=int, default=1,
                        help="Screening-specific warmup epochs (default: 1; canonical full confirmation uses warmup=3)")
    parser.add_argument("--confirmation-epochs", type=int, default=None,
                        help="Optional canonical confirmation epoch count to compute full-training extrapolation (default: None)")

    # P3-C Hyperparameter Grid Parameters
    parser.add_argument("--p3-lr", type=float, default=None,
                        help="Single P3 LR candidate to test (e.g. 1e-4)")
    parser.add_argument("--p3-lr-candidates", type=str, default="5e-5,1e-4,2e-4",
                        help="Comma-separated P3 LR candidates (default: '5e-5,1e-4,2e-4')")
    parser.add_argument("--gamma-init", type=float, default=None,
                        help="Single gamma_init candidate to test (e.g. 0.01)")
    parser.add_argument("--gamma-candidates", type=str, default="0.001,0.01,0.05",
                        help="Comma-separated gamma_init candidates (default: '0.001,0.01,0.05')")

    parser.add_argument("--output-dir", type=str, default="results/screening",
                        help="Output directory to save screening logs and report")
    parser.add_argument("--max-val-samples", type=int, default=None,
                        help="Max validation samples to evaluate per epoch for ultra-fast screening")
    parser.add_argument("--dry-run", action="store_true",
                        help="Execute a rapid 2-batch mock screening on CPU to verify mechanics")

    args = parser.parse_args()

    parent_ckpt = args.parent_checkpoint or args.locked_base
    if parent_ckpt and not os.path.isabs(parent_ckpt):
        parent_ckpt = os.path.join(project_root, parent_ckpt)

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        base_cfg = yaml.safe_load(f)

    if args.data_root:
        base_cfg['root_dir'] = args.data_root
    if args.depth:
        base_cfg['num_transformer_layers'] = args.depth
    if args.batch_size:
        base_cfg['batch_size'] = args.batch_size
    if args.num_workers is not None:
        base_cfg['num_workers'] = args.num_workers

    if args.dry_run:
        base_cfg['num_transformer_layers'] = 2
        base_cfg['batch_size'] = 2
        args.stage1_epochs = 1
        args.stage2_epochs = 1
        args.warmup_epochs = 1

    device = torch.device('cuda' if torch.cuda.is_available() and not args.dry_run else 'cpu')
    dev_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else "CPU"
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if device.type == 'cuda' else 0.0

    print_banner("PHASE-1 P3-C (ASDW) BASE HYPERPARAMETER SCREENING SUITE")
    print(f"Research Model: P3-C (New Base Model)")
    print(f"Device: {device} ({dev_name}, Total VRAM: {total_vram_gb:.2f} GB)")
    print(f"Config: {os.path.basename(config_path)}")
    print(f"Git Commit: {get_git_commit_hash()}")
    print(f"Parent Checkpoint: {parent_ckpt or 'NONE (Pretrained ConvNeXt initialization)'}")
    if parent_ckpt and os.path.exists(parent_ckpt):
        print(f"Parent SHA256: {compute_file_sha256(parent_ckpt)[:16]}...")
    print(f"Dataset Root: {base_cfg['root_dir']}")
    print(f"Batch Size: {base_cfg['batch_size']} | Workers: {base_cfg['num_workers']} | Depth: {base_cfg['num_transformer_layers']}")
    print(f"Protocol: Two-Stage (Stage 1={args.stage1_epochs} ep, Stage 2={args.stage2_epochs} ep, warmup={args.warmup_epochs} ep)")

    os.makedirs(args.output_dir, exist_ok=True)

    img_size = int(base_cfg.get('img_size', 448))
    if args.dry_run:
        print("[Notice] Running in --dry-run mode: generating mock dataset.")
        train_dataset = MockDataset(length=14 * 4, image_size=img_size)
        val_pairs = [("mock_img.png", "mock_mask.png")]
        loader_workers = 0
    else:
        train_dataset = get_dataset_from_config(base_cfg, split='train', image_size=img_size)
        val_pairs = get_image_mask_pairs(base_cfg, 'val')
        assert len(val_pairs) > 0, f"No validation pairs found in {base_cfg['root_dir']}/val!"
        loader_workers = base_cfg['num_workers']

    print(f"Dataset Loaded: Train = {len(train_dataset)} samples | Val = {len(val_pairs)} pairs")

    # Construct Screening Grid (3 P3 LR x 3 gamma_init = 9 Candidates)
    if args.p3_lr is not None and args.gamma_init is not None:
        p3_lr_list = [args.p3_lr]
        gamma_list = [args.gamma_init]
    elif args.p3_lr is not None:
        p3_lr_list = [args.p3_lr]
        gamma_list = [float(x.strip()) for x in args.gamma_candidates.split(',') if x.strip()]
    elif args.gamma_init is not None:
        p3_lr_list = [float(x.strip()) for x in args.p3_lr_candidates.split(',') if x.strip()]
        gamma_list = [args.gamma_init]
    else:
        p3_lr_list = [float(x.strip()) for x in args.p3_lr_candidates.split(',') if x.strip()]
        gamma_list = [float(x.strip()) for x in args.gamma_candidates.split(',') if x.strip()]

    trials_to_run: List[Tuple[str, float, float]] = []
    for cand_lr in p3_lr_list:
        for cand_gamma in gamma_list:
            cid = f"P3C_LR_{cand_lr:.2e}_gamma_{cand_gamma:.3f}"
            trials_to_run.append((cid, cand_lr, cand_gamma))

    print(f"Total P3-C candidates queued: {len(trials_to_run)}")
    for idx, (cid, clr, cgamma) in enumerate(trials_to_run, 1):
        print(f"  [{idx:02d}/{len(trials_to_run):02d}] {cid} -> P3 LR: {clr:.2e}, gamma_init: {cgamma:.4f}")

    results = []
    for cand_id, cand_lr, cand_gamma in trials_to_run:
        res = run_p3_c_screening_trial(
            candidate_id=cand_id,
            base_config=base_cfg,
            p3_lr=cand_lr,
            gamma_init=cand_gamma,
            parent_checkpoint=parent_ckpt,
            train_dataset=train_dataset,
            loader_workers=loader_workers,
            val_pairs=val_pairs,
            device=device,
            stage1_epochs=args.stage1_epochs,
            stage2_epochs=args.stage2_epochs,
            warmup_epochs=args.warmup_epochs,
            dry_run=args.dry_run,
            max_val_samples=args.max_val_samples,
        )
        results.append(res)

    # Sort results by Stage 2 Final Val Dice descending, then S2 Best Val Dice
    results.sort(key=lambda x: (x['s2_final_val_dice'], x['s2_best_val_dice']), reverse=True)

    avg_ep_sec_all = float(np.mean([r['mean_epoch_time_s'] for r in results])) if results else 0.0

    # Build Markdown Summary Report
    parent_provenance_str = os.path.basename(parent_ckpt) if parent_ckpt else "None (Pretrained initialization)"
    md_lines = [
        "# SAGE-Lite B2 Phase 1 P3-C (ASDW) Base Hyperparameter Screening Report",
        "",
        "> [!IMPORTANT]",
        "> **Bản chất Nghiên cứu: P3-C là Mô hình Base Mới**",
        "> - Nhánh nghiên cứu chuyển sang **P3-C (ASDW)** làm mô hình chuẩn (Base Model) thay cho B2 Base thuần.",
        "> - B2 Base đóng vai trò là checkpoint khởi tạo (parent checkpoint); toàn bộ candidate đều bắt đầu từ cùng một checkpoint này.",
        f"> - **Screening-specific protocol**: Stage 1 = {args.stage1_epochs} epochs (warmup = {args.warmup_epochs} epoch), Stage 2 = {args.stage2_epochs} epochs (warmup = {args.warmup_epochs} epoch).",
        "> - **Canonical Confirmation protocol**: Lượt huấn luyện đầy đủ chính thức sẽ áp dụng chuẩn `warmup = 3` epochs cho scheduler.",
        "",
        f"*Execution Date: {time.strftime('%Y-%m-%d %H:%M:%S')}*",
        f"*Hardware: {dev_name} ({total_vram_gb:.2f} GB VRAM)*",
        f"*Git Commit HEAD: `{get_git_commit_hash()}`*",
        f"*ViT Depth: {base_cfg['num_transformer_layers']} | Batch Size: {base_cfg['batch_size']} | Workers: {base_cfg['num_workers']}*",
        f"*Parent Checkpoint (Provenance): `{parent_provenance_str}`*",
        f"*DataLoader Isolation: Fresh Generator(seed=42) per trial per stage (100% batch-order reproducibility)*",
        f"*Objective Scaling: Total Loss = Seg_Loss + 1.0 * LB_Loss*",
        f"*Evaluation: Strictly Crack500 Val Split ({len(val_pairs)} pairs). Test split unaccessed.*",
        "",
        "---",
        "",
        "## 1. Bảng Xếp Hạng Kết Quả Thử Nghiệm P3-C (Leaderboard by S2 Final Val Dice)",
        "",
        "| Rank | Candidate | P3 LR | gamma_init | S1 Final (Best) Dice | S2 Final (Best) Dice | S2 Final (Best) IoU | gamma_S0 (init -> fin) | gamma_S1 (init -> fin) | P3 Param Delta | P3 Grad Norm | Sample Routing Entropy | Utilization Entropy | Active Experts | Measured Time / Ep | Peak VRAM (Alloc / Res) | Stability |",
        "|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for rank, r in enumerate(results, 1):
        s1_dice_str = f"{r['s1_final_val_dice']:.4f} ({r['s1_best_val_dice']:.4f})"
        s2_dice_str = f"**{r['s2_final_val_dice']:.4f}** ({r['s2_best_val_dice']:.4f})"
        s2_iou_str = f"{r['s2_final_val_iou']:.4f} ({r['s2_val_iou_at_best_dice']:.4f})"
        g0_str = f"{r['gamma_s0_init']:.3f} -> {r['gamma_s0_final']:.3f}"
        g1_str = f"{r['gamma_s1_init']:.3f} -> {r['gamma_s1_final']:.3f}"
        vram_str = f"{r['peak_allocated_mb']:.0f} / {r['peak_reserved_mb']:.0f} MB"
        time_str = f"{r['mean_epoch_time_s']:.1f}s"
        experts_str = f"{r['active_experts']}/{r.get('pool_size', 16)} ({r['expert_utilization_pct']}%)"

        md_lines.append(
            f"| **{rank}** | `{r['candidate_id']}` | {r['p3_lr']:.2e} | {r['gamma_init']:.3f} | "
            f"{s1_dice_str} | {s2_dice_str} | {s2_iou_str} | "
            f"{g0_str} | {g1_str} | "
            f"{r['p3_param_delta']:.4f} | {r['p3_grad_norm']:.4e} | "
            f"{r['sample_routing_entropy']:.2f} bits | {r['expert_utilization_entropy']:.2f} bits | "
            f"{experts_str} | {time_str} | {vram_str} | **{r['nan_inf_status']}** |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 2. Nhận Xét Khoa Học & Phân Tích Kỹ Thuật",
        "",
        f"1. **Ứng viên Dẫn đầu (Top Pick):** `{results[0]['candidate_id']}` đạt **S2 Final Val Dice = {results[0]['s2_final_val_dice']:.4f}** (Best: {results[0]['s2_best_val_dice']:.4f}) và **S2 Final IoU = {results[0]['s2_final_val_iou']:.4f}** (tại epoch Best Dice: Val Loss = {results[0]['s2_val_loss_at_best_dice']:.4f}, Val IoU = {results[0]['s2_val_iou_at_best_dice']:.4f}).",
        f"2. **Động lực Học của P3 Refinement (P3 Dynamics):**",
        f"   - gamma_S0 thay đổi: `{results[0]['gamma_s0_init']:.4f}` -> `{results[0]['gamma_s0_final']:.4f}`.",
        f"   - gamma_S1 thay đổi: `{results[0]['gamma_s1_init']:.4f}` -> `{results[0]['gamma_s1_final']:.4f}`.",
        f"   - Độ dịch chuyển tham số P3 (Parameter Delta): `{results[0]['p3_param_delta']:.6f}`.",
        f"   - Gradient Norm trung bình của P3: `{results[0]['p3_grad_norm']:.6e}`.",
        f"   *(Bằng chứng cho thấy các module ASDW thực sự tiếp nhận gradient và tham gia vào quá trình tinh chỉnh chi tiết khe nứt)*.",
        f"3. **Router SAGE Invariants & Entropy:**",
        f"   - Sample Routing Entropy: `{results[0]['sample_routing_entropy']:.2f}` bits / 2.0 bits tối đa.",
        f"   - Expert Utilization Entropy: `{results[0]['expert_utilization_entropy']:.2f}` bits / 4.0 bits tối đa.",
        f"   - Số chuyên gia hoạt động: `{results[0]['active_experts']}/{results[0].get('pool_size', 16)}` ({results[0]['expert_utilization_pct']}%).",
        f"4. **Đo lường Tốc độ Thực tế (Empirical Timing):**",
        f"   - Thời gian thực tế đo được trên môi trường: **{avg_ep_sec_all:.1f} giây / epoch** ({avg_ep_sec_all / 60.0:.2f} phút / epoch).",
    ])

    if args.confirmation_epochs is not None:
        est_conf_hours = (avg_ep_sec_all * args.confirmation_epochs) / 3600.0
        md_lines.append(f"   - Ước tính ngân sách thời gian cho {args.confirmation_epochs} epochs confirmation (dựa trên tốc độ thực tế): **xấp xỉ {est_conf_hours:.2f} giờ**.")

    md_lines.extend([
        "5. **Bước tiếp theo trong Lộ trình:** Khóa ứng viên P3-C tối ưu nhất, tiến hành canonical training (warmup=3, full epochs) để tạo thành **P3-C Base Model chính thức** trước khi chạy ablation A/B/C.",
        "",
    ])

    report_path = os.path.join(args.output_dir, "phase1_p3_c_screening_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    csv_path = os.path.join(args.output_dir, "phase1_p3_c_screening_summary.csv")
    import csv
    if results:
        keys = list(results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)

    print_banner("P3-C SCREENING SUITE COMPLETED SUCCESSFULLY")
    print(f"Summary Report: {report_path}")
    print(f"Summary CSV:    {csv_path}")
    try:
        print("\n" + "\n".join(md_lines[15:28]))
    except UnicodeEncodeError:
        print(f"\nReport generated with {len(results)} candidate results.")


if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    main()
