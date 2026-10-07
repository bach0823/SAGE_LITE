#!/usr/bin/env python3
"""
scripts/diagnostics/train_eval_phase7_s2_gate_block2.py

Phase 7: S2-Gate Block 2 Extension (Downstream Spatial Modulation on Skip S0)
=============================================================================

Scientific Scope & Hypothesis (Phase 7):
- In Phase 6, S2-Gate v2 (Conv3x3, 83,073 params) at Decoder Block 1 (modulating Skip S1, 56x56)
  cured 27.91% of wider-gap bridges and reduced HD95 by 11.34 px (53.65 -> 42.31 px).
- Phase 7 extends spatial modulation to Decoder Block 2:
  * Uses S1 context (output of Decoder Block 1, 96 channels, 56x56) upsampled to 112x112
  * Modulates Stage 0 skip connection (48 channels, 112x112) prior to concatenation into Decoder Block 2.
- Protocol & Invariants:
  * Checkpoint: best_model_b2_global.pth with B1 Gate active (from Phase 6 A1+S2G combination).
  * Strict Freeze: model.eval() followed by s2_gate_block2.train().
  * Optimizer: AdamW(s2_gate_block2.parameters(), lr=lr, weight_decay=1e-2).
  * Scheduler: CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5).
  * Trainable parameter count:
    - Kernel 1x1: 4,737 parameters
    - Kernel 3x3: 41,601 parameters
  * Backbone, Routers, Decoder blocks, Seg Head, and S2-Gate Block 1 are 100% strictly FROZEN.
  * Setting A canonical validation: N=348 samples, 448x448 non-overlapping tiling, tau=0.5, FP32 strict.
"""

import argparse
import copy
import glob
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Self-contained path resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
cand_roots = [
    project_root,
    os.path.abspath(os.path.join(project_root, "..")),
    os.getcwd(),
    "d:/truong/SpecialSubjectTTNT",
    "d:/truong/SpecialSubjectTTNT/SAGE_LITE",
    "/content/SAGE_LITE",
    "/content",
]
for p in cand_roots:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from tools.run_phase6_c_topology_diagnostic import (
    load_model_from_checkpoint,
    compute_topology_metrics,
)
from sage.utils.advanced_metrics import calculate_hd95_bf1
from sage.networks import S2GateModule


# -----------------------------------------------------------------------------
# Loss Functions (Strict FP32)
# -----------------------------------------------------------------------------

def dice_loss(pred_logits: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    probs = torch.sigmoid(pred_logits)
    probs_flat = probs.view(-1)
    target_flat = target.view(-1)
    intersection = (probs_flat * target_flat).sum()
    dice = (2.0 * intersection + smooth) / (probs_flat.sum() + target_flat.sum() + smooth)
    return 1.0 - dice


def compute_seg_loss(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss(logits, targets)
    l_total = bce + dice
    metrics = {
        "loss_bce": float(bce.item()),
        "loss_dice": float(dice.item()),
        "loss_total": float(l_total.item()),
    }
    return l_total, metrics


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class Crack500TrainDataset(Dataset):
    def __init__(self, img_dir: str, mask_dir: str, img_size: int = 448, seed: int = 42):
        self.img_size = img_size
        self.samples = []
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        img_files = sorted(os.listdir(img_dir))
        for f in img_files:
            if not f.lower().endswith((".jpg", ".png")):
                continue
            base = os.path.splitext(f)[0]
            mask_path = os.path.join(mask_dir, base + ".png")
            if not os.path.exists(mask_path):
                mask_path = os.path.join(mask_dir, base + ".jpg")
            if os.path.exists(mask_path):
                self.samples.append((os.path.join(img_dir, f), mask_path, base))

        print(f"Loaded {len(self.samples)} training samples from {img_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, mask_path, stem = self.samples[idx]
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            raise RuntimeError(f"Failed to load sample: {img_path}")

        H, W = img.shape[:2]
        crop_sz = self.img_size

        if H < crop_sz or W < crop_sz:
            img = cv2.resize(img, (max(W, crop_sz), max(H, crop_sz)), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (max(W, crop_sz), max(H, crop_sz)), interpolation=cv2.INTER_NEAREST)
            H, W = img.shape[:2]

        top = np.random.randint(0, H - crop_sz + 1)
        left = np.random.randint(0, W - crop_sz + 1)

        img_crop = img[top : top + crop_sz, left : left + crop_sz]
        mask_crop = mask[top : top + crop_sz, left : left + crop_sz]

        if np.random.rand() > 0.5:
            img_crop = np.fliplr(img_crop)
            mask_crop = np.fliplr(mask_crop)
        if np.random.rand() > 0.5:
            img_crop = np.flipud(img_crop)
            mask_crop = np.flipud(mask_crop)

        img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
        norm_img = (img_rgb.astype(np.float32) / 255.0 - self.mean) / self.std
        tensor_img = torch.from_numpy(norm_img.transpose(2, 0, 1)).float()

        mask_bin = (mask_crop > 127).astype(np.float32)
        tensor_mask = torch.from_numpy(mask_bin).unsqueeze(0).float()

        return {"image": tensor_img, "mask": tensor_mask, "stem": stem}


# -----------------------------------------------------------------------------
# Metric Helpers
# -----------------------------------------------------------------------------

def compute_thin_crack_dice(pred_bin: np.ndarray, target_bin: np.ndarray, max_thickness: int = 3) -> float:
    if target_bin.sum() == 0:
        return 1.0 if pred_bin.sum() == 0 else 0.0
    dist = distance_transform_edt(target_bin)
    thin_mask = (dist > 0) & (dist <= max_thickness)
    if thin_mask.sum() == 0:
        return 1.0 if pred_bin.sum() == 0 else 0.0
    intersection = np.logical_and(pred_bin == 1, thin_mask).sum()
    denominator = (pred_bin == 1).sum() + thin_mask.sum()
    if denominator == 0:
        return 1.0
    return float(2.0 * intersection / (denominator + 1e-8))


def compute_boundary_iou(pred_bin: np.ndarray, target_bin: np.ndarray, dilation: int = 2) -> float:
    if pred_bin.sum() == 0 and target_bin.sum() == 0:
        return 1.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation + 1, 2 * dilation + 1))
    pred_dil = cv2.dilate(pred_bin.astype(np.uint8), kernel)
    pred_ero = cv2.erode(pred_bin.astype(np.uint8), kernel)
    pred_b = pred_dil - pred_ero

    gt_dil = cv2.dilate(target_bin.astype(np.uint8), kernel)
    gt_ero = cv2.erode(target_bin.astype(np.uint8), kernel)
    gt_b = gt_dil - gt_ero

    inter = np.logical_and(pred_b > 0, gt_b > 0).sum()
    union = np.logical_or(pred_b > 0, gt_b > 0).sum()
    if union == 0:
        return 1.0
    return float(inter / (union + 1e-8))


def predict_full_image_tiling_setting_a_fp32(
    model: nn.Module,
    img_rgb: np.ndarray,
    device: torch.device,
    tile_size: int = 448,
    batch_size: int = 8,
) -> np.ndarray:
    H, W = img_rgb.shape[:2]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    norm_img = (img_rgb.astype(np.float32) / 255.0 - mean) / std

    y_steps = range(0, H, tile_size)
    x_steps = range(0, W, tile_size)

    patches = []
    coords = []
    pred_logits = np.zeros((H + tile_size, W + tile_size), dtype=np.float32)

    for y in y_steps:
        for x in x_steps:
            patch = norm_img[y : y + tile_size, x : x + tile_size]
            ph, pw = patch.shape[:2]
            if ph < tile_size or pw < tile_size:
                pad_h = tile_size - ph
                pad_w = tile_size - pw
                patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

            t_patch = torch.from_numpy(patch.transpose(2, 0, 1)).float()
            patches.append(t_patch)
            coords.append((y, x))

    for i in range(0, len(patches), batch_size):
        batch = torch.stack(patches[i : i + batch_size]).to(device, non_blocking=True)
        with torch.no_grad():
            logits = model(batch)
            logits_np = logits.squeeze(1).cpu().numpy()

        for j in range(len(batch)):
            y, x = coords[i + j]
            if len(patches) == 1 or len(batch) == 1:
                tile_out = logits_np if logits_np.ndim == 2 else logits_np[0]
            else:
                tile_out = logits_np[j]
            pred_logits[y : y + tile_size, x : x + tile_size] = tile_out

    return pred_logits[:H, :W]


# -----------------------------------------------------------------------------
# Training Routine (S2-Gate Block 2)
# -----------------------------------------------------------------------------

def train_s2_gate_block2(
    model: nn.Module,
    s2_gate_b2: S2GateModule,
    train_loader: DataLoader,
    device: torch.device,
    lr: float = 1e-3,
    epochs: int = 8,
    weight_decay: float = 1e-2,
) -> List[Dict[str, Any]]:
    print("\n" + "=" * 80)
    print(f"TRAINING PHASE 7: S2-Gate Block 2 (LR={lr:.1e}, Kernel={s2_gate_b2.kernel_size})")
    print(f"Epochs: {epochs}, Batch Size: {train_loader.batch_size}, FP32 Strict")
    print(f"AdamW lr={lr:.1e}, wd={weight_decay}, CosineAnnealingLR")
    total_trainable = sum(p.numel() for p in s2_gate_b2.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_trainable}")
    print("=" * 80)

    # 2.1 Parameter group: strictly s2_gate_b2.parameters() only
    optimizer = torch.optim.AdamW(s2_gate_b2.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    logs = []

    for ep in range(1, epochs + 1):
        # 2.2 Execution order: model.eval() then s2_gate_b2.train()
        model.eval()
        s2_gate_b2.train()

        ep_loss_total = 0.0
        ep_loss_bce = 0.0
        ep_loss_dice = 0.0

        pbar = tqdm(train_loader, desc=f"P7-Block2 Ep {ep}/{epochs} (LR={lr:.1e})", ncols=90)
        for batch in pbar:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss, m = compute_seg_loss(logits, masks)

            loss.backward()
            optimizer.step()

            ep_loss_total += m["loss_total"]
            ep_loss_bce += m["loss_bce"]
            ep_loss_dice += m["loss_dice"]

            pbar.set_postfix({
                "tot": f"{m['loss_total']:.3f}",
                "bce": f"{m['loss_bce']:.3f}",
                "dice": f"{m['loss_dice']:.3f}",
            })

        scheduler.step()

        with torch.no_grad():
            w1_norm = float(torch.norm(s2_gate_b2.gate_net[0].weight).item())
            w2_norm = float(torch.norm(s2_gate_b2.gate_net[3].weight).item())
            b2_val = float(s2_gate_b2.gate_net[3].bias.item())
            last_alpha = model.decoder.last_gate_alpha_block2
            alpha_mean = float(last_alpha.mean().item()) if last_alpha is not None else float("nan")
            alpha_min = float(last_alpha.min().item()) if last_alpha is not None else float("nan")
            alpha_max = float(last_alpha.max().item()) if last_alpha is not None else float("nan")

        n_batches = len(train_loader)
        ep_log = {
            "epoch": ep,
            "loss_total": ep_loss_total / n_batches,
            "loss_bce": ep_loss_bce / n_batches,
            "loss_dice": ep_loss_dice / n_batches,
            "lr": optimizer.param_groups[0]["lr"],
            "gate_w1_norm": w1_norm,
            "gate_w2_norm": w2_norm,
            "gate_b2_val": b2_val,
            "alpha_mean": alpha_mean,
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
        }
        logs.append(ep_log)
        print(f"Ep {ep:02d} Summary: Loss={ep_log['loss_total']:.4f} (BCE={ep_log['loss_bce']:.4f}, Dice={ep_log['loss_dice']:.4f}) | Gate W1={w1_norm:.4f}, W2={w2_norm:.4f}, Bias={b2_val:.4f} | Alpha=[{alpha_min:.3f}, {alpha_mean:.3f}, {alpha_max:.3f}]")

    return logs


# -----------------------------------------------------------------------------
# Main CLI & Runner
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 7: S2-Gate Block 2 Extension Runner")
    parser.add_argument("--config", type=str, default="configs/p3_ablation/phase6_full_s1/a1_s2g_end_to_end.yaml")
    parser.add_argument("--checkpoint", type=str, default="results/phase6_combination/phase6_comb_a1_s2g_end_to_end/best_model_b2_global.pth")
    parser.add_argument("--data_root", type=str, default="datasets/Crack500_ready")
    parser.add_argument("--out_dir", type=str, default="results/diagnostics/phase7_s2_gate_block2")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kernel_size", type=int, default=3, choices=[1, 3])
    parser.add_argument("--dilation", type=int, default=1, choices=[1, 2, 4])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=14)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_train", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    # Resolve paths
    for cand in [args.checkpoint, os.path.join(project_root, args.checkpoint), os.path.join("/content/SAGE_LITE", args.checkpoint)]:
        if os.path.exists(cand):
            args.checkpoint = cand
            break

    for cand in [args.config, os.path.join(project_root, args.config), os.path.join("/content/SAGE_LITE", args.config)]:
        if os.path.exists(cand):
            args.config = cand
            break

    cand_data_roots = [
        args.data_root,
        "datasets/Crack500_ready",
        os.path.join(project_root, "datasets", "Crack500_ready"),
        os.path.join(project_root, "..", "datasets", "Crack500_ready"),
        "/content/dataset/Crack500",
        "/content/SAGE_LITE/datasets/Crack500_ready",
    ]
    for cand in cand_data_roots:
        if os.path.isdir(os.path.join(cand, "val", "images")):
            args.data_root = cand
            break

    print(f"Resolved Data Root: {args.data_root}")
    print(f"Resolved Config:    {args.config}")
    print(f"Resolved Checkpoint:{args.checkpoint}")

    # Load Base Model (with B1 Gate)
    print("\nLoading base model from checkpoint...")
    model = load_model_from_checkpoint(args.config, args.checkpoint, device=device)
    model.eval()

    # Instantiate Block 2 Gate
    s2_gate_b2 = S2GateModule(
        s2_channels=96,
        skip_channels=48,
        mid_channels=32,
        kernel_size=args.kernel_size,
        dilation=args.dilation,
    ).to(device)

    model.decoder.s2_gate_block2 = s2_gate_b2
    model.decoder.use_s2_gate_block2 = True

    # 2.2 Strict Freeze
    for p in model.parameters():
        p.requires_grad = False
    for p in s2_gate_b2.parameters():
        p.requires_grad = True

    total_trainable = sum(p.numel() for p in s2_gate_b2.parameters() if p.requires_grad)
    expected_trainable = 41601 if args.kernel_size == 3 else 4737
    print(f"Trainable parameters: {total_trainable} (Expected: {expected_trainable})")
    assert total_trainable == expected_trainable, f"Param mismatch: {total_trainable} vs {expected_trainable}"

    # Setup DataLoaders
    train_img_dir = os.path.join(args.data_root, "train", "images")
    train_mask_dir = os.path.join(args.data_root, "train", "masks")
    val_img_dir = os.path.join(args.data_root, "val", "images")
    val_mask_dir = os.path.join(args.data_root, "val", "masks")

    train_dataset = Crack500TrainDataset(train_img_dir, train_mask_dir, img_size=448, seed=args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    val_files = []
    for f in sorted(os.listdir(val_img_dir)):
        if f.lower().endswith((".jpg", ".png")):
            base = os.path.splitext(f)[0]
            mp = os.path.join(val_mask_dir, base + ".png")
            if not os.path.exists(mp):
                mp = os.path.join(val_mask_dir, base + ".jpg")
            if os.path.exists(mp):
                val_files.append((os.path.join(val_img_dir, f), mp))

    assert len(val_files) == 348, f"Expected 348 validation images, got {len(val_files)}"

    dil_str = f"_d{args.dilation}" if args.dilation > 1 else ""
    ep_str = f"_e{args.epochs}" if args.epochs != 8 else ""
    tag = f"lr_{args.lr:.0e}_k{args.kernel_size}{dil_str}{ep_str}"
    weights_path = os.path.join(args.out_dir, f"s2g_block2_{tag}_weights.pth")
    training_log_path = os.path.join(args.out_dir, f"training_log_{tag}.csv")
    val_csv_path = os.path.join(args.out_dir, f"validation_{tag}_per_sample.csv")
    summary_path = os.path.join(args.out_dir, f"summary_{tag}.json")

    # Training
    if os.path.exists(weights_path) and args.skip_train:
        print(f"Loading weights from {weights_path}")
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        s2_gate_b2.load_state_dict(ckpt["s2_gate_block2_state"])
    else:
        train_logs = train_s2_gate_block2(
            model=model,
            s2_gate_b2=s2_gate_b2,
            train_loader=train_loader,
            device=device,
            lr=args.lr,
            epochs=args.epochs,
        )
        pd.DataFrame(train_logs).to_csv(training_log_path, index=False)
        torch.save({
            "s2_gate_block2_state": s2_gate_b2.state_dict(),
            "trainable_param_count": total_trainable,
            "kernel_size": args.kernel_size,
            "lr": args.lr,
            "epochs": args.epochs,
            "seed": args.seed,
        }, weights_path)
        print(f"Saved weights to {weights_path}")

    # Evaluation
    print("\n" + "=" * 80)
    print(f"CANONICAL SETTING A EVALUATION: Val N=348 (Tag: {tag})")
    print("=" * 80)
    model.eval()

    val_records = []
    runtimes = []

    for img_path, mask_path in tqdm(val_files, desc=f"Eval {tag}", ncols=90):
        case_name = os.path.splitext(os.path.basename(img_path))[0]
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        target_bin = (mask > 127).astype(np.uint8)

        t0 = time.time()
        logits = predict_full_image_tiling_setting_a_fp32(model, img_rgb, device, tile_size=448, batch_size=8)
        dt_ms = (time.time() - t0) * 1000.0
        runtimes.append(dt_ms)

        pred_bin = (logits > 0.0).astype(np.uint8)

        tp = int(np.sum((pred_bin == 1) & (target_bin == 1)))
        fp = int(np.sum((pred_bin == 1) & (target_bin == 0)))
        fn = int(np.sum((pred_bin == 0) & (target_bin == 1)))

        dice = float(2 * tp / (2 * tp + fp + fn + 1e-8))
        iou = float(tp / (tp + fp + fn + 1e-8))
        prec = float(tp / (tp + fp + 1e-8))
        rec = float(tp / (tp + fn + 1e-8))

        topo = compute_topology_metrics(pred_bin, target_bin)
        b_iou = compute_boundary_iou(pred_bin, target_bin, dilation=2)
        hd95, bf1 = calculate_hd95_bf1(pred_bin, target_bin)
        thin_dice = compute_thin_crack_dice(pred_bin, target_bin, max_thickness=3)

        val_records.append({
            "image_id": case_name,
            "dice": dice,
            "iou": iou,
            "precision": prec,
            "recall": rec,
            "cldice": topo["cldice"],
            "boundary_iou": b_iou,
            "hd95": hd95,
            "bf1": bf1,
            "thin_dice": thin_dice,
            "bridge_events": topo["bridge_events"],
            "bridge_status": 1 if topo["bridge_events"] > 0 else 0,
            "break_events": topo["fragmented_gt_components"],
            "break_status": 1 if topo["fragmented_gt_components"] > 0 else 0,
            "spurious_islands": topo.get("spurious_island_count", topo.get("spurious_islands", 0)),
            "runtime_ms": dt_ms,
        })

    df_val = pd.DataFrame(val_records)
    df_val.to_csv(val_csv_path, index=False)

    summary = {
        "tag": tag,
        "lr": args.lr,
        "kernel_size": args.kernel_size,
        "dilation": args.dilation,
        "epochs": args.epochs,
        "seed": args.seed,
        "N": len(df_val),
        "dice": float(df_val["dice"].mean()),
        "iou": float(df_val["iou"].mean()),
        "precision": float(df_val["precision"].mean()),
        "recall": float(df_val["recall"].mean()),
        "cldice": float(df_val["cldice"].mean()),
        "boundary_iou": float(df_val["boundary_iou"].mean()),
        "hd95": float(df_val["hd95"].dropna().mean()),
        "bf1": float(df_val["bf1"].dropna().mean()),
        "thin_dice": float(df_val["thin_dice"].dropna().mean()),
        "total_bridge_events": int(df_val["bridge_events"].sum()),
        "bridge_images": int(df_val["bridge_status"].sum()),
        "total_break_events": int(df_val["break_events"].sum()),
        "break_images": int(df_val["break_status"].sum()),
        "spurious_islands": int(df_val["spurious_islands"].sum()),
        "mean_runtime_ms": float(np.mean(runtimes)),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"PHASE 7 DIAGNOSTIC SUMMARY ({tag}):")
    print(f"  Val Dice:          {summary['dice']:.4f}")
    print(f"  Precision:         {summary['precision']:.4f} | Recall: {summary['recall']:.4f}")
    print(f"  clDice:            {summary['cldice']:.4f} | Thin Dice: {summary['thin_dice']:.4f}")
    print(f"  Boundary IoU:      {summary['boundary_iou']:.4f} | HD95: {summary['hd95']:.2f} px")
    print(f"  Total Bridges:     {summary['total_bridge_events']} ({summary['bridge_images']} images)")
    print(f"  Total Breaks:      {summary['total_break_events']} ({summary['break_images']} images)")
    print(f"  Summary saved to:  {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
