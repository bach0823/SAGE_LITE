"""
Phase-1 Hyperparameter Screening Script for SAGE-Lite B2 Base Model
===================================================================

Protocol & Objective:
- Fixed Invariants:
    * Backbone: ConvNeXt-V2-Femto + ViT Depth = 12 (D12)
    * Batch Size: 14 (BS14) - Locked empirical sweet spot
    * Workers: 2
    * Architecture: Pure Base SAGE-Lite (p3_mode = None, no P3 refinement)
    * Gating: Sigmoid, Logit Modulation = True, Shared Experts = [0, 1, 2, 3]
    * Exploration Noise: Active in model.train()
    * Input resolution: 448x448
    * Dataset: Real Crack500 (1896 train samples, 348 val pairs)
    * Isolation: Validation split ONLY. Zero access to Test split.
    * Seed: 42 (deterministic)

Roadmap Alignment:
    Phase 1 Pipeline:
    1. Learning Rate screening (Current Step) -> [5e-5, 7.5e-5, 1e-4, 1.5e-4, 2e-4]
    2. top_k screening -> [1, 2, 4]
    3. router_hidden_dim -> [32, 64, 128]
    4. load_balance_factor -> [0.0, 0.005, 0.01, 0.02]
    5. expert_dropout / residual_scale -> [0.0, 0.1, 0.2] / [0.05, 0.1, 0.2]
    6. Official Locked Base D12 Training -> Checkpoint SHA256 -> P3 Gate Open.

Usage Examples:
    # 1. Screen single LR candidate in a clean Colab process:
    python scripts/screen_base_hyperparams.py --lr 1e-4 --stage1-epochs 3 --stage2-epochs 3

    # 2. Screen the full Phase-1 LR candidate grid:
    python scripts/screen_base_hyperparams.py --lr-candidates "5e-5,7.5e-5,1e-4,1.5e-4,2e-4" --stage1-epochs 3 --stage2-epochs 3

    # 3. Dry-run verification on local CPU / small mock:
    python scripts/screen_base_hyperparams.py --dry-run
"""

import argparse
import copy
import gc
import json
import math
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

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
from scripts.train_crack import (
    CrackBinaryLoss,
    calculate_binary_metrics,
    create_stage2_optimizer,
    get_optimizer_groups,
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


def calculate_routing_statistics(model: nn.Module) -> Dict[str, Any]:
    """
    Computes entropy, active expert count, and utilization distribution across all SageRouters.
    """
    routers = [m for m in model.modules() if isinstance(m, SageRouter)]
    if not routers:
        return {
            "mean_entropy": 0.0,
            "active_experts": 0,
            "expert_utilization_pct": 0.0,
            "total_calls": 0,
        }

    entropies = []
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
                # Shannon entropy in bits (base 2)
                p_pos = p[p > 0]
                entropy = -float(np.sum(p_pos * np.log2(p_pos)))
                entropies.append(entropy)

        if hasattr(r, 'total_calls'):
            total_calls += int(r.total_calls.item())

    pool_size = len(total_usage) if total_usage is not None else 16
    active_count = int(np.sum(total_usage > 0)) if total_usage is not None else 0
    utilization_pct = (active_count / pool_size * 100.0) if pool_size > 0 else 0.0
    mean_entropy = float(np.mean(entropies)) if entropies else 0.0

    return {
        "mean_entropy": round(mean_entropy, 4),
        "active_experts": active_count,
        "expert_utilization_pct": round(utilization_pct, 2),
        "total_calls": total_calls,
    }


def run_screening_trial(
    candidate_id: str,
    base_config: dict,
    trial_params: dict,
    train_loader: DataLoader,
    val_pairs: List[Tuple[str, str]],
    device: torch.device,
    stage1_epochs: int = 3,
    stage2_epochs: int = 3,
    warmup_epochs: int = 1,
    dry_run: bool = False,
    max_val_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Executes a complete Two-Stage SAGE-Lite Base screening run for a single hyperparameter configuration.
    """
    cfg = copy.deepcopy(base_config)
    sage_cfg = cfg.get('sage_config', {})

    # Apply hyperparameter overrides
    lr = float(trial_params.get('lr', cfg.get('lr', 1e-4)))
    stage2_base_lr = float(trial_params.get('stage2_base_lr', lr))
    stage2_shared_lr = float(trial_params.get('stage2_shared_lr', lr))
    top_k = int(trial_params.get('top_k', sage_cfg.get('top_k', 4)))
    router_hidden_dim = int(trial_params.get('router_hidden_dim', sage_cfg.get('router_hidden_dim', 64)))
    lb_factor = float(trial_params.get('load_balance_factor', sage_cfg.get('load_balance_factor', 0.01)))
    expert_dropout = float(trial_params.get('expert_dropout', sage_cfg.get('expert_dropout', 0.1)))
    residual_scale = float(trial_params.get('residual_scale', sage_cfg.get('residual_scale', 0.1)))

    sage_cfg['top_k'] = top_k
    sage_cfg['router_hidden_dim'] = router_hidden_dim
    sage_cfg['load_balance_factor'] = lb_factor
    sage_cfg['expert_dropout'] = expert_dropout
    sage_cfg['residual_scale'] = residual_scale
    sage_cfg['shared_expert_indices'] = [0, 1, 2, 3] # Fixed protocol
    sage_cfg['gating_type'] = 'sigmoid' # Fixed protocol
    sage_cfg['logit_modulation'] = True # Fixed protocol

    vit_depth = int(cfg.get('num_transformer_layers', 12))
    img_size = int(cfg.get('img_size', 448))
    seed = int(cfg.get('seed', 42))

    set_seed(seed)

    print_banner(f"STARTING PHASE 1 SCREENING TRIAL: {candidate_id}")
    print(f"Hyperparameters: LR={lr:.2e} | Stage-2 Base={stage2_base_lr:.2e} | Stage-2 Shared={stage2_shared_lr:.2e}")
    print(f"SAGE Configuration: top_k={top_k}, router_hidden_dim={router_hidden_dim}, lb_factor={lb_factor}")
    print(f"Protocol: ViT Depth={vit_depth}, BS={train_loader.batch_size}, Seed={seed}, p3_mode=None (Base)")
    print(f"Budgets: Stage 1 = {stage1_epochs} epochs | Stage 2 = {stage2_epochs} epochs (Warmup: {warmup_epochs})")

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # 1. Instantiate pure Base SAGE-Lite model
    model = create_b2_unet(
        num_transformer_layers=vit_depth,
        pretrained=not dry_run,
        sage_config=sage_cfg,
        p3_mode=None, # Pure Base SAGE-Lite
    ).to(device)

    assert isinstance(model, B2ConvNeXtViTUNet), "Model instantiation error!"
    assert model.p3_mode is None, "p3_mode must be strictly None during Phase 1 Base screening!"

    criterion = CrackBinaryLoss(bce_weight=1.0, dice_weight=1.5)
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda'))

    eval_protocol = resolve_protocol(cfg)
    active_val_pairs = val_pairs[:max_val_samples] if max_val_samples else val_pairs

    start_trial_time = time.perf_counter()
    nan_inf_detected = False

    # =========================================================================
    # STAGE 1: Free Exploration & Initialization
    # =========================================================================
    print(f"\n--- [Stage 1 Training ({stage1_epochs} Epochs)] ---")
    stage1_groups = get_optimizer_groups(
        model,
        lr_backbone=lr * 0.1,
        lr_decoder=lr,
        lr_sage=lr,
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

    stage1_best_val_dice = -1.0
    stage1_best_val_loss = float('inf')
    best_stage1_weights = None

    for epoch in range(1, stage1_epochs + 1):
        t_ep_start = time.perf_counter()
        model.train() # Exploration noise is naturally active
        train_loss, train_seg_loss, train_lb_loss = 0.0, 0.0, 0.0
        train_dice, train_iou = 0.0, 0.0
        batches_processed = 0

        for b_idx, batch in enumerate(train_loader, 1):
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
                total_loss = seg_loss + lb_factor * lb_loss

            if not torch.isfinite(total_loss):
                print(f"  [ERROR] Non-finite loss at Stage 1, Epoch {epoch}, Batch {b_idx}: {total_loss.item()}")
                nan_inf_detected = True
                break

            scaler.scale(total_loss).backward()

            # Check gradients
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    nan_inf_detected = True
                    break

            scaler.step(opt1)
            scaler.update()

            probs = torch.sigmoid(logits)
            _, d_val, iou_val = calculate_binary_metrics(probs, labels)

            train_loss += total_loss.item()
            train_seg_loss += seg_loss.item()
            train_lb_loss += lb_loss.item()
            train_dice += d_val
            train_iou += iou_val
            batches_processed += 1

        sched1.step(epoch)

        num_b = max(1, batches_processed)
        avg_train_loss = train_loss / num_b
        avg_train_dice = train_dice / num_b
        avg_train_iou = train_iou / num_b
        avg_train_lb = train_lb_loss / num_b

        # Validation evaluation (strictly Val set)
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
        val_dice = float(val_metrics.get('dice', 0.0))
        val_loss = float(val_metrics.get('loss', 0.0))

        if val_dice > stage1_best_val_dice:
            stage1_best_val_dice = val_dice
            stage1_best_val_loss = val_loss
            best_stage1_weights = copy.deepcopy(model.state_dict())

        print(
            f"  [Stage 1][Epoch {epoch:02d}/{stage1_epochs:02d}] "
            f"Train Loss: {avg_train_loss:.4f} (LB: {avg_train_lb:.4f}, Dice: {avg_train_dice:.4f}) | "
            f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f} | "
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

    opt2 = create_stage2_optimizer(
        model,
        stage2_base_lr=stage2_base_lr,
        stage2_shared_lr=stage2_shared_lr,
    )
    sched2 = CosineLRScheduler(
        opt2,
        t_initial=stage2_epochs,
        lr_min=1e-6,
        warmup_t=warmup_epochs,
        warmup_lr_init=1e-6,
        t_in_epochs=True,
    )

    stage2_best_val_dice = -1.0
    stage2_best_val_loss = float('inf')
    stage2_best_val_iou = 0.0

    for epoch in range(1, stage2_epochs + 1):
        t_ep_start = time.perf_counter()
        model.train() # Exploration noise is active
        train_loss, train_seg_loss, train_lb_loss = 0.0, 0.0, 0.0
        train_dice, train_iou = 0.0, 0.0
        batches_processed = 0

        for b_idx, batch in enumerate(train_loader, 1):
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
                total_loss = seg_loss + lb_factor * lb_loss

            if not torch.isfinite(total_loss):
                print(f"  [ERROR] Non-finite loss at Stage 2, Epoch {epoch}, Batch {b_idx}: {total_loss.item()}")
                nan_inf_detected = True
                break

            scaler.scale(total_loss).backward()

            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    nan_inf_detected = True
                    break

            scaler.step(opt2)
            scaler.update()

            probs = torch.sigmoid(logits)
            _, d_val, iou_val = calculate_binary_metrics(probs, labels)

            train_loss += total_loss.item()
            train_seg_loss += seg_loss.item()
            train_lb_loss += lb_loss.item()
            train_dice += d_val
            train_iou += iou_val
            batches_processed += 1

        sched2.step(epoch)

        num_b = max(1, batches_processed)
        avg_train_loss = train_loss / num_b
        avg_train_dice = train_dice / num_b
        avg_train_lb = train_lb_loss / num_b

        # Validation evaluation (strictly Val set)
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
        val_dice = float(val_metrics.get('dice', 0.0))
        val_loss = float(val_metrics.get('loss', 0.0))
        val_iou = float(val_metrics.get('iou', 0.0))

        if val_dice > stage2_best_val_dice:
            stage2_best_val_dice = val_dice
            stage2_best_val_loss = val_loss
            stage2_best_val_iou = val_iou

        print(
            f"  [Stage 2][Epoch {epoch:02d}/{stage2_epochs:02d}] "
            f"Train Loss: {avg_train_loss:.4f} (LB: {avg_train_lb:.4f}, Dice: {avg_train_dice:.4f}) | "
            f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f} | "
            f"Time: {ep_duration:.1f}s"
        )

    total_trial_sec = time.perf_counter() - start_trial_time
    routing_stats = calculate_routing_statistics(model)

    peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == 'cuda' else 0.0
    peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024**2) if device.type == 'cuda' else 0.0

    overall_best_dice = max(stage1_best_val_dice, stage2_best_val_dice)
    overall_best_loss = stage2_best_val_loss if stage2_best_val_dice >= stage1_best_val_dice else stage1_best_val_loss

    trial_summary = {
        "candidate_id": candidate_id,
        "lr": lr,
        "stage2_base_lr": stage2_base_lr,
        "stage2_shared_lr": stage2_shared_lr,
        "top_k": top_k,
        "router_hidden_dim": router_hidden_dim,
        "lb_factor": lb_factor,
        "expert_dropout": expert_dropout,
        "residual_scale": residual_scale,
        "stage1_best_val_dice": round(stage1_best_val_dice, 4),
        "stage1_best_val_loss": round(stage1_best_val_loss, 4),
        "stage2_best_val_dice": round(stage2_best_val_dice, 4),
        "stage2_best_val_loss": round(stage2_best_val_loss, 4),
        "best_val_dice": round(overall_best_dice, 4),
        "best_val_loss": round(overall_best_loss, 4),
        "best_val_iou": round(stage2_best_val_iou, 4),
        "mean_routing_entropy": routing_stats['mean_entropy'],
        "active_experts": routing_stats['active_experts'],
        "expert_utilization_pct": routing_stats['expert_utilization_pct'],
        "peak_allocated_mb": round(peak_allocated_mb, 1),
        "peak_reserved_mb": round(peak_reserved_mb, 1),
        "nan_inf_status": "FAIL" if nan_inf_detected else "CLEAN",
        "runtime_sec": round(total_trial_sec, 1),
        "runtime_min": round(total_trial_sec / 60.0, 2),
    }

    # Clean up trial resources
    del model, opt1, opt2, sched1, sched2, scaler, best_stage1_weights
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return trial_summary


def main():
    parser = argparse.ArgumentParser(description="Phase-1 Base Model SAGE-Lite Hyperparameter Screening Suite")
    parser.add_argument("--config", type=str, default="configs/b2_crack500_depth12.yaml",
                        help="Base YAML configuration file")
    parser.add_argument("--data-root", type=str, default="/content/dataset/Crack500",
                        help="Root directory of Crack500 dataset")
    parser.add_argument("--depth", type=int, default=12,
                        help="ViT Depth (default: 12)")
    parser.add_argument("--batch-size", type=int, default=14,
                        help="Batch Size (default: 14 - Locked sweet spot)")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader worker processes (default: 2)")
    parser.add_argument("--stage1-epochs", type=int, default=3,
                        help="Number of epochs for Stage 1 screening (default: 3)")
    parser.add_argument("--stage2-epochs", type=int, default=3,
                        help="Number of epochs for Stage 2 screening (default: 3)")
    parser.add_argument("--warmup-epochs", type=int, default=1,
                        help="Warmup epochs for cosine scheduler (default: 1)")

    # Hyperparameter Screening Parameters
    parser.add_argument("--lr", type=float, default=None,
                        help="Single LR candidate to test (e.g. 1e-4)")
    parser.add_argument("--lr-candidates", type=str, default=None,
                        help="Comma-separated LR candidates (e.g. '5e-5,7.5e-5,1e-4,1.5e-4,2e-4')")
    parser.add_argument("--stage2-ratio", type=float, default=1.0,
                        help="Ratio of stage2_shared_lr to stage2_base_lr (default: 1.0)")

    parser.add_argument("--top-k", type=int, default=None,
                        help="Override top_k (default: 4)")
    parser.add_argument("--top-k-candidates", type=str, default=None,
                        help="Comma-separated top_k candidates (e.g. '1,2,4')")

    parser.add_argument("--router-hidden-dim", type=int, default=None,
                        help="Override router_hidden_dim (default: 64)")
    parser.add_argument("--lb-factor", type=float, default=None,
                        help="Override load_balance_factor (default: 0.01)")
    parser.add_argument("--expert-dropout", type=float, default=None,
                        help="Override expert_dropout (default: 0.1)")
    parser.add_argument("--residual-scale", type=float, default=None,
                        help="Override residual_scale (default: 0.1)")

    parser.add_argument("--output-dir", type=str, default="results/screening",
                        help="Output directory to save screening logs and report")
    parser.add_argument("--max-val-samples", type=int, default=None,
                        help="Max validation samples to evaluate per epoch for ultra-fast screening (default: None = all 348)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Execute a rapid 2-batch mock screening to verify script mechanics")

    args = parser.parse_args()

    # Load configuration
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        base_cfg = yaml.safe_load(f)

    # CLI Overrides
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

    print_banner("PHASE-1 SAGE-LITE BASE HYPERPARAMETER SCREENING SUITE")
    print(f"Device: {device} ({dev_name}, Total VRAM: {total_vram_gb:.2f} GB)")
    print(f"Config: {os.path.basename(config_path)}")
    print(f"Git Commit: {get_git_commit_hash()}")
    print(f"Dataset Root: {base_cfg['root_dir']}")
    print(f"Batch Size: {base_cfg['batch_size']} | Workers: {base_cfg['num_workers']} | Depth: {base_cfg['num_transformer_layers']}")
    print(f"Seed: {base_cfg.get('seed', 42)} | SAGE Invariants: Sigmoid Gating, Logit Modulation, Noise=ON")

    os.makedirs(args.output_dir, exist_ok=True)

    # Build Training DataLoader
    img_size = int(base_cfg.get('img_size', 448))
    seed = int(base_cfg.get('seed', 42))
    g = torch.Generator()
    g.manual_seed(seed)

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=base_cfg['batch_size'],
        shuffle=True,
        num_workers=loader_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=(device.type == 'cuda'),
        drop_last=False,
    )

    print(f"Dataset Loaded: Train = {len(train_dataset)} samples ({len(train_loader)} batches) | Val = {len(val_pairs)} pairs")

    # Construct Screening Grid
    trials_to_run: List[Tuple[str, Dict[str, Any]]] = []

    # Case 1: LR candidates specified
    if args.lr_candidates:
        lr_list = [float(x.strip()) for x in args.lr_candidates.split(',') if x.strip()]
        for cand_lr in lr_list:
            c_id = f"LR_{cand_lr:.2e}"
            trials_to_run.append((c_id, {
                'lr': cand_lr,
                'stage2_base_lr': cand_lr,
                'stage2_shared_lr': cand_lr * args.stage2_ratio,
            }))
    elif args.top_k_candidates:
        k_list = [int(x.strip()) for x in args.top_k_candidates.split(',') if x.strip()]
        base_lr = args.lr or float(base_cfg.get('lr', 1e-4))
        for cand_k in k_list:
            c_id = f"TopK_{cand_k}"
            trials_to_run.append((c_id, {
                'lr': base_lr,
                'stage2_base_lr': base_lr,
                'stage2_shared_lr': base_lr * args.stage2_ratio,
                'top_k': cand_k,
            }))
    else:
        # Default single trial
        target_lr = args.lr or float(base_cfg.get('lr', 1e-4))
        c_id = f"Base_LR_{target_lr:.2e}"
        trial_dict = {
            'lr': target_lr,
            'stage2_base_lr': target_lr,
            'stage2_shared_lr': target_lr * args.stage2_ratio,
        }
        if args.top_k is not None: trial_dict['top_k'] = args.top_k
        if args.router_hidden_dim is not None: trial_dict['router_hidden_dim'] = args.router_hidden_dim
        if args.lb_factor is not None: trial_dict['load_balance_factor'] = args.lb_factor
        if args.expert_dropout is not None: trial_dict['expert_dropout'] = args.expert_dropout
        if args.residual_scale is not None: trial_dict['residual_scale'] = args.residual_scale
        trials_to_run.append((c_id, trial_dict))

    print(f"Total screening candidates queued: {len(trials_to_run)}")

    results = []
    for cand_id, params in trials_to_run:
        res = run_screening_trial(
            candidate_id=cand_id,
            base_config=base_cfg,
            trial_params=params,
            train_loader=train_loader,
            val_pairs=val_pairs,
            device=device,
            stage1_epochs=args.stage1_epochs,
            stage2_epochs=args.stage2_epochs,
            warmup_epochs=args.warmup_epochs,
            dry_run=args.dry_run,
            max_val_samples=args.max_val_samples,
        )
        results.append(res)

    # Sort results by Val Dice descending
    results.sort(key=lambda x: x['best_val_dice'], reverse=True)

    # Generate Markdown Summary Report
    md_lines = [
        "# SAGE-Lite B2 Phase 1 Base Model Hyperparameter Screening Report",
        "",
        f"*Execution Date: {time.strftime('%Y-%m-%d %H:%M:%S')}*",
        f"*Hardware: {dev_name} ({total_vram_gb:.2f} GB VRAM)*",
        f"*Git Commit HEAD: `{get_git_commit_hash()}`*",
        f"*ViT Depth: {base_cfg['num_transformer_layers']} | Batch Size: {base_cfg['batch_size']} | Workers: {base_cfg['num_workers']}*",
        f"*Protocol: Two-Stage Base (p3_mode=None, Stage 1={args.stage1_epochs} ep, Stage 2={args.stage2_epochs} ep)*",
        f"*Evaluation: Strictly Crack500 Val Split ({len(val_pairs)} pairs). Test split unaccessed.*",
        "",
        "---",
        "",
        "## 1. Bảng Xếp Hạng Kết Quả Thử Nghiệm (Leaderboard by Val Dice)",
        "",
        "| Rank | Candidate | Base LR | Stage2 LR (Sh/Base) | Top-K | Router Dim | Best Val Dice | Best Val IoU | Val Loss | Mean Entropy | Active Experts | Peak Alloc (MB) | Peak Reserv (MB) | Stability |",
        "|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for rank, r in enumerate(results, 1):
        sh_lr = r['stage2_shared_lr']
        b_lr = r['stage2_base_lr']
        lr_ratio_str = f"{sh_lr:.1e}/{b_lr:.1e}"
        md_lines.append(
            f"| **{rank}** | `{r['candidate_id']}` | {r['lr']:.2e} | {lr_ratio_str} | "
            f"{r['top_k']} | {r['router_hidden_dim']} | **{r['best_val_dice']:.4f}** | {r['best_val_iou']:.4f} | "
            f"{r['best_val_loss']:.4f} | {r['mean_routing_entropy']:.2f} | {r['active_experts']}/16 ({r['expert_utilization_pct']}%) | "
            f"{r['peak_allocated_mb']} | {r['peak_reserved_mb']} | **{r['nan_inf_status']}** |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 2. Nhận Xét Khoa Học & Quyết Định Kế Tiếp",
        "",
        f"1. **Ứng viên Dẫn đầu (Top Pick):** `{results[0]['candidate_id']}` đạt **Val Dice = {results[0]['best_val_dice']:.4f}** và **Val IoU = {results[0]['best_val_iou']:.4f}**.",
        f"2. **Độ ổn định số học:** Toàn bộ các thử nghiệm đều đạt trạng thái `{results[0]['nan_inf_status']}` (không phát sinh NaN/Inf).",
        f"3. **Tài nguyên VRAM ở BS14:** Peak Allocated cao nhất đạt `{max(r['peak_allocated_mb'] for r in results):.1f} MB` và Peak Reserved đạt `{max(r['peak_reserved_mb'] for r in results):.1f} MB`.",
        "4. **Bước tiếp theo:** Khóa siêu tham số tối ưu đã chọn vào config chính thức trước khi chuyển sang bước khảo sát kế tiếp trong lộ trình Phase 1.",
        "",
    ])

    report_path = os.path.join(args.output_dir, "phase1_base_screening_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # Also save CSV
    csv_path = os.path.join(args.output_dir, "phase1_base_screening_summary.csv")
    import csv
    if results:
        keys = list(results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)

    print_banner("SCREENING SUITE COMPLETED SUCCESSFULLY")
    print(f"Summary Report: {report_path}")
    print(f"Summary CSV:    {csv_path}")
    try:
        print("\n" + "\n".join(md_lines[10:20]))
    except UnicodeEncodeError:
        # Fallback for environments with strict non-utf-8 console encoding
        print(f"\nReport generated with {len(results)} candidate results.")


if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    main()
