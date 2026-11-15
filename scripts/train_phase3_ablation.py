"""
Phase 3 Ablation Training Script — SAGE-Lite Final Protocol.

Variant D: Thay router + expert_pool trong mỗi SageLayer bằng DenseAdapter
           có số tham số tương đương (~6,713,256).
Variant R: Giữ expert_pool, thay router bằng RandomFixedRouter (zero trainable params,
           top-k ngẫu nhiên theo seed global, cố định suốt training).

Usage:
    python scripts/train_phase3_ablation.py --variant D --seed 42 --output-dir /content/runs/phase3_D_seed42
    python scripts/train_phase3_ablation.py --variant R --seed 43 --output-dir /content/runs/phase3_R_seed43

Flags đầy đủ:
    --variant   : 'D' hoặc 'R' (bắt buộc)
    --seed      : int (mặc định 42)
    --config    : path YAML (mặc định configs/sage_b2_crack500.yaml)
    --data-root : override root_dir trong config
    --output-dir: thư mục lưu checkpoint + log (mặc định results/phase3_<variant>_seed<seed>)
"""

import argparse
import json
import logging
import os
import random
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# --- Path setup ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sage.networks import create_b2_unet
from sage.components.sage_layer import SageLayer
from sage.components.router import SageRouter
from sage.utils.dataloader import get_dataset_from_config
from sage.utils.training_utils import setup_logging, set_seed, seed_worker

# ============================================================================
# Variant D — DenseAdapter
# ============================================================================

# Hidden dims khớp với total router+expert_pool params = 6,713,256
# H = floor(target_per_layer / (2*C + 1))  với target_per_layer = 6713256 / 8
# Keyed by OUTPUT channels of main_block (not router.in_channels):
#   stage0=48, stage1=96, stage2=192, stage3=384, transformer0-3=192
_DENSE_H: Dict[int, int] = {48: 8651, 96: 4347, 192: 2179, 384: 1091}
_TOTAL_DENSE_TARGET = 6_713_256
_RESIDUAL_SCALE = 0.1  # giống canonical fusion_type=residual


def _probe_main_block_out_channels(main_block: nn.Module, in_channels: int) -> int:
    """Probe output channels of main_block via dummy forward (CPU, no grad)."""
    orig = main_block.training
    main_block.eval()
    try:
        with torch.no_grad():
            # Try 4D CNN input (B, C, H, W)
            try:
                dummy = torch.zeros(1, in_channels, 16, 16)
                out = main_block(dummy)
                if isinstance(out, tuple):
                    out = out[0]
                if out.dim() == 4:
                    return out.shape[1]
                if out.dim() == 3:
                    return out.shape[-1]
            except Exception:
                pass
            # Try 3D Transformer input (B, L, C)
            dummy = torch.zeros(1, 196, in_channels)
            out = main_block(dummy)
            if isinstance(out, tuple):
                out = out[0]
            return out.shape[-1]
    except Exception:
        return in_channels
    finally:
        if orig:
            main_block.train()


class DenseAdapter(nn.Module):
    """2-layer MLP bottleneck thay router + expert_path.

    out = main_out + residual_scale * up(gelu(down(x)))

    Dùng Linear trên dim cuối (C), broadcast qua spatial.
    """

    def __init__(self, in_channels: int, hidden_dim: int, residual_scale: float = 0.1):
        super().__init__()
        self.residual_scale = residual_scale
        self.down = nn.Linear(in_channels, hidden_dim, bias=True)
        self.act = nn.GELU()
        self.up = nn.Linear(hidden_dim, in_channels, bias=True)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        # (B, C, H, W) → (B, H*W, C)
        if x.dim() == 4:
            B, C, H, W = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        elif x.dim() == 3:
            x_flat = x
        else:
            return x
        delta = self.up(self.act(self.down(x_flat)))
        if x.dim() == 4:
            B, C, H, W = orig_shape
            delta = delta.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x + self.residual_scale * delta

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _inject_variant_D(model: nn.Module) -> nn.ModuleList:
    """Replace router + expert path with DenseAdapter in every SageLayer.

    Output channels of main_block are probed dynamically (not taken from router.in_channels,
    which is the INPUT to the stage — different from output for CNN stages that double channels).

    Returns ModuleList of adapters (registered on model so optimizer sees them).
    """
    adapters = []
    for name, m in model.named_modules():
        if isinstance(m, SageLayer):
            in_c  = m.router.in_channels
            out_c = _probe_main_block_out_channels(m.main_block, in_c)
            if out_c not in _DENSE_H:
                raise ValueError(
                    f"SageLayer '{name}' main_block output channels={out_c} "
                    f"not in DENSE_H={list(_DENSE_H.keys())}"
                )
            adapter = DenseAdapter(out_c, _DENSE_H[out_c], _RESIDUAL_SCALE)
            m._variant_D_adapter = adapter
            adapters.append(adapter)

    if not adapters:
        raise RuntimeError("No SageLayer found in model — cannot inject Variant D.")

    # Register so optimizer can iterate params
    model.phase3_dense_adapters = nn.ModuleList(adapters)

    # Monkey-patch SageLayer.forward for Variant D
    _orig_forward = SageLayer.forward

    def _variant_D_forward(self_layer, x, expert_pool=None):
        self_layer.forward_calls += 1
        main_out = self_layer.main_block(x)
        if isinstance(main_out, tuple):
            main_out = main_out[0]
        # Apply DenseAdapter residual
        out = self_layer._variant_D_adapter(main_out)
        self_layer._last_lb_loss = None
        self_layer._last_routing_info = {
            "variant": "D",
            "load_balance_loss": torch.tensor(0.0, device=x.device),
        }
        return out

    SageLayer.forward = _variant_D_forward  # type: ignore[method-assign]

    total_adapter_params = sum(p.numel() for p in model.phase3_dense_adapters.parameters())
    return model.phase3_dense_adapters, total_adapter_params


# ============================================================================
# Variant R — RandomFixedRouter
# ============================================================================


class RandomFixedRouter(nn.Module):
    """Zero-trainable-param router: selects top_k experts randomly per sample.

    Random state controlled by global seed set before training starts.
    """

    def __init__(self, in_channels: int, expert_pool_size: int, top_k: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.expert_pool_size = expert_pool_size
        self.top_k = top_k
        self.register_buffer("total_calls", torch.tensor(0))
        self.register_buffer("expert_usage_count", torch.zeros(expert_pool_size))

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        device = x.device
        top_k_indices = torch.zeros(B, self.top_k, dtype=torch.long, device=device)
        for b in range(B):
            perm = torch.randperm(self.expert_pool_size, device=device)
            top_k_indices[b] = perm[: self.top_k]
        gating_weights = torch.full((B, self.top_k), 1.0 / self.top_k, device=device)
        self.total_calls += B
        for b in range(B):
            self.expert_usage_count.index_add_(
                0, top_k_indices[b], torch.ones(self.top_k, device=device)
            )
        routing_info = {
            "load_balance_loss": torch.tensor(0.0, device=device),
            "g_s_score_sample_0": 0.5,
            "routing_entropy_mean": float(np.log(max(self.expert_pool_size, 1))),
        }
        return top_k_indices, gating_weights, routing_info

    def get_usage_statistics(self) -> Dict[str, Any]:
        counts = self.expert_usage_count.cpu().numpy()
        total = counts.sum()
        return {
            "total_forward_calls": self.total_calls.item(),
            "expert_usage_count": counts.tolist(),
            "expert_usage_ratio": (counts / max(total, 1)).tolist(),
            "shared_expert_indices": [],
        }


def _inject_variant_R(model: nn.Module) -> None:
    """Replace every SageRouter with RandomFixedRouter (same top_k)."""
    replaced = 0
    for name, m in model.named_modules():
        if isinstance(m, SageLayer):
            orig = m.router
            m.router = RandomFixedRouter(
                in_channels=orig.in_channels,
                expert_pool_size=orig.expert_pool_size,
                top_k=orig.top_k,
            )
            replaced += 1
    if replaced == 0:
        raise RuntimeError("No SageLayer found in model — cannot inject Variant R.")
    return replaced


# ============================================================================
# Soft-BIoU Loss (clone từ canonical training, không import riêng)
# ============================================================================


def soft_biou_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    lam: float = 0.5,
    d: int = 2,
) -> torch.Tensor:
    """SoftBIoU = (1-lam)*DiceLoss + lam*BCE — boundary dilation handled externally."""
    pred = torch.sigmoid(pred_logits)
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice_loss = 1.0 - (2.0 * inter + 1.0) / (union + 1.0)
    bce = F.binary_cross_entropy_with_logits(
        pred_logits, target.float(), reduction="mean"
    )
    return (1.0 - lam) * dice_loss.mean() + lam * bce


# ============================================================================
# Training Loop
# ============================================================================


def _collect_lb_loss(model: nn.Module) -> Optional[torch.Tensor]:
    """Sum load-balance losses from all SageLayers (Variant R only)."""
    total = None
    for m in model.modules():
        if isinstance(m, SageLayer) and m._last_lb_loss is not None:
            total = m._last_lb_loss if total is None else total + m._last_lb_loss
    return total


def train_one_stage(
    model: nn.Module,
    train_loader,
    val_loader,
    n_epochs: int,
    lr: float,
    wd: float,
    output_dir: str,
    stage_name: str,
    seed: int,
    device: torch.device,
    logger: logging.Logger,
    cfg: Dict,
) -> float:
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_dice = 0.0
    best_path = os.path.join(output_dir, f"best_model_b2_{stage_name}.pth")
    last_path = os.path.join(output_dir, f"last_model_b2_{stage_name}.pth")

    lam = float(cfg.get("boundary_iou_weight", 0.5))

    for epoch in range(1, n_epochs + 1):
        model.train()
        set_seed(seed + epoch)
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            imgs  = batch['image'].to(device, non_blocking=True)
            masks = batch['label'].to(device, non_blocking=True)
            # Dataloader returns (B, H, W); loss expects (B, 1, H, W)
            if masks.dim() == 3:
                masks = masks.unsqueeze(1).float()
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                out = model(imgs)
                if isinstance(out, tuple):
                    out = out[0]
                loss = soft_biou_loss(out, masks, lam=lam)
                lb = _collect_lb_loss(model)
                if lb is not None:
                    loss = loss + lb

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # Val
        model.eval()
        all_dice = []
        with torch.no_grad():
            for batch in val_loader:
                imgs  = batch['image'].to(device, non_blocking=True)
                masks = batch['label'].to(device, non_blocking=True)
                if masks.dim() == 3:
                    masks = masks.unsqueeze(1).float()
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    out = model(imgs)
                    if isinstance(out, tuple):
                        out = out[0]
                pred = (torch.sigmoid(out) > 0.5).float()
                for b in range(pred.shape[0]):
                    p = pred[b, 0].cpu().numpy().astype(np.uint8)
                    g = (masks[b, 0].cpu().numpy() > 0.5).astype(np.uint8)
                    tp = int((p & g).sum())
                    fp = int((p & (1 - g)).sum())
                    fn = int(((1 - p) & g).sum())
                    d_val = 2 * tp / (2 * tp + fp + fn + 1e-5)
                    all_dice.append(d_val)

        mean_dice = float(np.mean(all_dice)) if all_dice else 0.0
        elapsed = time.time() - t0
        logger.info(
            f"[{stage_name}] Epoch {epoch}/{n_epochs}  loss={avg_loss:.4f}  "
            f"val_dice={mean_dice:.4f}  ({elapsed:.0f}s)"
        )

        # Save last
        torch.save(
            {
                "epoch": epoch,
                "stage": stage_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "val_dice": mean_dice,
                "best_dice": best_dice,
            },
            last_path,
        )

        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save({"model_state_dict": model.state_dict(), "val_dice": best_dice}, best_path)
            logger.info(f"  → New best {stage_name}: {best_dice:.6f}")

    return best_dice


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Ablation — Variant D or R")
    parser.add_argument("--variant", required=True, choices=["D", "R"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(PROJECT_ROOT, "configs", "sage_b2_crack500.yaml"),
    )
    parser.add_argument("--data-root", "--data_root", dest="data_root", type=str, default=None)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default=None)
    args = parser.parse_args()

    variant = args.variant
    seed = args.seed

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.data_root:
        cfg["root_dir"] = args.data_root

    output_dir = args.output_dir or os.path.join(
        PROJECT_ROOT, f"results/phase3_variant{variant}_seed{seed}"
    )
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logging(output_dir, experiment_name=f"phase3_{variant}_s{seed}")
    logger.info("=" * 70)
    logger.info(f"PHASE 3 ABLATION — Variant {variant}, Seed {seed}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Output dir: {output_dir}")
    logger.info("=" * 70)

    # Save git hash
    try:
        import subprocess

        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT
        ).decode().strip()
        with open(os.path.join(output_dir, "git_hash.txt"), "w") as f:
            f.write(git_hash + "\n")
        logger.info(f"Git hash: {git_hash}")
    except Exception:
        logger.warning("Could not get git hash.")

    # Save config snapshot
    shutil.copy(args.config, os.path.join(output_dir, "config_snapshot.yaml"))

    # Set seed
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Build model (canonical, với pretrained)
    img_size = int(cfg.get("img_size", 448))
    vit_depth = int(cfg.get("num_transformer_layers", 4))
    sage_cfg = cfg.get("sage_config", {})

    model = create_b2_unet(
        num_classes=1,
        img_size=img_size,
        num_transformer_layers=vit_depth,
        pretrained=True,
        sage_config=sage_cfg,
        p3_mode=cfg.get("p3_mode"),
        use_plu_head=cfg.get("use_plu_head", False),
        use_s2_gate=cfg.get("use_s2_gate", False),
        s2_gate_kernel_size=cfg.get("s2_gate_kernel_size", 3),
        use_s2_gate_block2=cfg.get("use_s2_gate_block2", False),
        s2_gate_block2_kernel_size=cfg.get("s2_gate_block2_kernel_size", 3),
        s2_gate_block2_dilation=cfg.get("s2_gate_block2_dilation", 1),
    )
    model = model.to(device)

    base_total = sum(p.numel() for p in model.parameters())
    logger.info(f"Base model params (before variant injection): {base_total:,}")

    # Inject variant
    variant_meta: Dict[str, Any] = {
        "variant": variant,
        "seed": seed,
        "target_replacement_params": _TOTAL_DENSE_TARGET,
        "phase2_mean_val_dice": 0.7786376169868934,
        "phase2_mde": 0.010803939249116433,
    }

    if variant == "D":
        adapters, adapter_params = _inject_variant_D(model)
        logger.info(f"Variant D injected. DenseAdapter total params: {adapter_params:,}")
        logger.info(f"  Target: {_TOTAL_DENSE_TARGET:,}, Actual: {adapter_params:,}")
        variant_meta["dense_adapter_params"] = adapter_params
    else:  # R
        n_replaced = _inject_variant_R(model)
        logger.info(f"Variant R injected. {n_replaced} RandomFixedRouters (zero trainable params).")
        variant_meta["random_routers_replaced"] = n_replaced

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total model params after injection: {total_params:,}")
    variant_meta["total_params_after_injection"] = total_params

    # Re-move to device: Variant R replaces routers AFTER model.to(device),
    # so new RandomFixedRouter buffers (total_calls, expert_usage_count) stay on CPU.
    model = model.to(device)

    # Dataloaders
    from torch.utils.data import DataLoader

    train_ds = get_dataset_from_config(cfg, split="train")
    val_ds = get_dataset_from_config(cfg, split="val")

    g = torch.Generator()
    g.manual_seed(seed)
    bs = int(cfg.get("batch_size", 8))
    nw = int(cfg.get("num_workers", 2))

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=True,
        worker_init_fn=seed_worker, generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=True,
    )
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    stage1_epochs = int(cfg.get("stage1_epochs", 17))
    stage2_epochs = int(cfg.get("epochs", 35)) - stage1_epochs
    wd = 1e-4

    # Stage 1 — backbone frozen
    logger.info("=== STAGE 1 (backbone frozen) ===")
    for name, p in model.named_parameters():
        if "backbone.convnext" in name and "main_block" in name:
            p.requires_grad = False
        elif "backbone.transformer" in name and "main_block" in name:
            p.requires_grad = False
    # Variant D adapters always trainable; Variant R backbone stays frozen too

    s1_lr = float(cfg.get("lr", 1e-4))
    best_s1 = train_one_stage(
        model, train_loader, val_loader,
        stage1_epochs, s1_lr, wd,
        output_dir, "stage1", seed, device, logger, cfg,
    )
    logger.info(f"Stage 1 best val Dice: {best_s1:.6f}")

    # Stage 2 — unfreeze all
    logger.info("=== STAGE 2 (all params unfrozen) ===")
    for p in model.parameters():
        p.requires_grad = True

    s2_lr = float(cfg.get("stage2_base_lr", 1e-4))
    best_s2 = train_one_stage(
        model, train_loader, val_loader,
        stage2_epochs, s2_lr, wd,
        output_dir, "stage2", seed, device, logger, cfg,
    )
    logger.info(f"Stage 2 best val Dice: {best_s2:.6f}")

    # Save best global
    best_global_dice = max(best_s1, best_s2)
    global_path = os.path.join(output_dir, "best_model_b2_global.pth")
    s2_best = os.path.join(output_dir, "best_model_b2_stage2.pth")
    s1_best = os.path.join(output_dir, "best_model_b2_stage1.pth")
    src = s2_best if best_s2 >= best_s1 else s1_best
    if os.path.exists(src):
        shutil.copy(src, global_path)
    logger.info(f"Global best checkpoint: {global_path} (Dice={best_global_dice:.6f})")

    # Save metadata
    variant_meta.update({
        "best_s1_dice": best_s1,
        "best_s2_dice": best_s2,
        "best_global_dice": best_global_dice,
        "delta_vs_phase2_mean": best_global_dice - 0.7786376169868934,
        "mde_exceeded": abs(best_global_dice - 0.7786376169868934) > 0.010803939249116433,
    })
    with open(os.path.join(output_dir, "run_metadata.json"), "w") as f:
        json.dump(variant_meta, f, indent=2)

    logger.info("=" * 70)
    logger.info(f"PHASE 3 VARIANT {variant} SEED {seed} — HOÀN THÀNH")
    logger.info(f"Val Dice: {best_global_dice:.6f}")
    logger.info(f"Phase2 mean: 0.778638 | MDE: 0.010804")
    logger.info(f"Δ: {variant_meta['delta_vs_phase2_mean']:+.6f}")
    logger.info(f"MDE exceeded: {variant_meta['mde_exceeded']}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
