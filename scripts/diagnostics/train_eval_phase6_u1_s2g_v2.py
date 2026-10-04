#!/usr/bin/env python3
"""
scripts/diagnostics/train_eval_phase6_u1_s2g_v2.py

Phase 6: Downstream Spatial Modulation / S2-Gate v2 Experiment (U1-S2G-v2)
==========================================================================

Scientific Hypothesis (U1-S2G-v2):
- From U1-S2G evaluation (results/diagnostics/phase6_u1_s2g/):
  * 12/43 (27.91%) wider-gap events were cured (surpassing zero-training ablation upper bound 11/43 = 25.58%).
  * BUT 11/43 wider-gap events were worsened (delta_z < 0, z_s2g increased from 2.688 -> 3.018).
  * Root cause: Conv1x1 in gate_net operates strictly on 1x1 point-wise features and lacks
    receptive field (RF) to perceive geometric continuity of short corridors between crack tips.
- U1-S2G-v2 Upgrade:
  * Replace Conv1x1(288->32) with Conv2d(288, 32, kernel_size=3, padding=1), expanding RF at 56x56
    resolution from 1px -> 3px (and larger effective RF when combined with S2 28x28 context).
  * Warm-start from trained U1-S2G weights (`u1_s2g_weights.pth`) by centering the 1x1 kernel at (1, 1)
    and zero-initializing spatial neighbors, guaranteeing exact numerical identity at t=0 (|Delta alpha| < 1e-4).
  * Maintain exact frozen strategy and training hyperparameters (8 epochs, AdamW lr=1e-3, wd=1e-2, CosineAnnealingLR).

Invariants:
- Total trainable parameters: EXACTLY 83,073 parameters:
  * Conv1 (3x3): 288 * 32 * 3 * 3 + 32 = 82,976
  * BatchNorm2d: 32 weight + 32 bias = 64
  * Conv2 (1x1): 32 * 1 * 1 * 1 + 1 = 33
  * Total = 83,073
- Backbone, SAGE Routers, and Decoder blocks 100% strictly FROZEN.
- Pretrained base checkpoint: P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth
  (SHA256: 147f784021414efd0db514aa6dae94585fece820e88f584e436fc65de851fb66) NEVER MUTATED ON DISK.
- Sealed Test Set: N=1124 strictly untouched.
- Setting A canonical validation: N=348 samples, 448x448 tiling, stride 448, reflect pad, tau=0.5.
- Training & Tiling Inference: STRICT FP32 ONLY (no torch.amp.autocast to avoid NaN logits).
- Mandatory Dedicated Report: Recovery tracking of the 11 worsened cases from U1-S2G.
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
# Standalone Dataset and Loss (Strict FP32)
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


class Crack500TrainDataset(Dataset):
    """
    Crack500 In-Memory Random Crop Training Dataset (448x448, FP32).
    """
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
            pad_h = max(crop_sz - H, 0)
            pad_w = max(crop_sz - W, 0)
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            H, W = img.shape[:2]

        max_y = H - crop_sz
        max_x = W - crop_sz
        top = np.random.randint(0, max_y + 1) if max_y > 0 else 0
        left = np.random.randint(0, max_x + 1) if max_x > 0 else 0

        crop_img = img[top : top + crop_sz, left : left + crop_sz]
        crop_mask = mask[top : top + crop_sz, left : left + crop_sz]

        # Random horizontal & vertical flips
        if np.random.rand() > 0.5:
            crop_img = np.fliplr(crop_img).copy()
            crop_mask = np.fliplr(crop_mask).copy()
        if np.random.rand() > 0.5:
            crop_img = np.flipud(crop_img).copy()
            crop_mask = np.flipud(crop_mask).copy()

        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        crop_img = (crop_img - self.mean) / self.std
        crop_img = crop_img.transpose(2, 0, 1)
        crop_mask = (crop_mask > 127).astype(np.float32)[np.newaxis, :, :]

        return {
            "image": torch.from_numpy(crop_img).float(),
            "mask": torch.from_numpy(crop_mask).float(),
            "stem": stem,
        }


# -----------------------------------------------------------------------------
# Evaluation Helper Functions
# -----------------------------------------------------------------------------

def compute_file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_boundary_iou(pred_bin: np.ndarray, target_bin: np.ndarray, dilation: int = 2) -> float:
    if np.sum(target_bin) == 0 and np.sum(pred_bin) == 0:
        return 1.0
    if np.sum(target_bin) == 0 or np.sum(pred_bin) == 0:
        return 0.0

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilation + 1, 2 * dilation + 1))
    pred_boundary = cv2.dilate(pred_bin.astype(np.uint8), k) - cv2.erode(pred_bin.astype(np.uint8), k)
    gt_boundary = cv2.dilate(target_bin.astype(np.uint8), k) - cv2.erode(target_bin.astype(np.uint8), k)

    inter = np.sum((pred_boundary > 0) & (gt_boundary > 0))
    union = np.sum((pred_boundary > 0) | (gt_boundary > 0))
    return float(inter / (union + 1e-8))


def compute_thin_crack_dice(pred_bin: np.ndarray, target_bin: np.ndarray, max_thickness: int = 3) -> float:
    if np.sum(target_bin) == 0:
        return np.nan

    dt = distance_transform_edt(target_bin > 0)
    thin_mask = (target_bin > 0) & (dt <= (max_thickness / 2.0 + 0.5))
    if np.sum(thin_mask) == 0:
        return np.nan

    tp = np.sum((pred_bin == 1) & (thin_mask == 1))
    fp = np.sum((pred_bin == 1) & (target_bin == 0))
    fn = np.sum((pred_bin == 0) & (thin_mask == 1))
    return float((2.0 * tp) / (2.0 * tp + fp + fn + 1e-8))


def predict_full_image_tiling_setting_a_fp32(
    model: nn.Module,
    image: np.ndarray,
    device: torch.device,
    tile_size: int = 448,
    batch_size: int = 8,
) -> np.ndarray:
    """
    Crack500 Setting A Tiling in STRICT FP32 (NO torch.amp.autocast to avoid NaN logits).
    """
    H, W = image.shape[:2]

    pad_h = (tile_size - (H % tile_size)) % tile_size
    pad_w = (tile_size - (W % tile_size)) % tile_size

    padded_img = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    pH, pW = padded_img.shape[:2]

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    patches = []
    coords = []
    for y in range(0, pH, tile_size):
        for x in range(0, pW, tile_size):
            patch = padded_img[y : y + tile_size, x : x + tile_size]
            patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
            patch_tensor = (patch_tensor - mean) / std
            patches.append(patch_tensor)
            coords.append((y, x))

    pred_logits = np.zeros((pH, pW), dtype=np.float32)

    for i in range(0, len(patches), batch_size):
        batch = torch.stack(patches[i : i + batch_size]).to(device, non_blocking=True)
        with torch.no_grad():
            logits = model(batch)  # Strict FP32 inference
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
# Training Routine (U1-S2G-v2)
# -----------------------------------------------------------------------------

def train_u1_s2g_v2(
    model: nn.Module,
    s2_gate: S2GateModule,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 8,
    out_dir: str = "results/diagnostics/phase6_u1_s2g_v2",
) -> List[Dict[str, Any]]:
    print("\n" + "=" * 80)
    print("TRAINING U1-S2G-v2: S2-Gate Conv3x3 Modulation on Skip S1 (56x56)")
    print(f"Epochs: {epochs}, Physical Batch: 14, Precision: FP32 ONLY (Strict)")
    print("Parameter Group: S2GateModule(k=3) (83,073 params), lr = 1e-3, wd = 1e-2")
    total_trainable = sum(p.numel() for p in s2_gate.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_trainable} (Expected: 83,073)")
    print("=" * 80)

    optimizer = torch.optim.AdamW(s2_gate.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    logs = []

    for ep in range(1, epochs + 1):
        model.eval()   # Keep entire base model frozen in eval
        s2_gate.train()

        ep_loss_total = 0.0
        ep_loss_bce = 0.0
        ep_loss_dice = 0.0

        pbar = tqdm(train_loader, desc=f"U1-S2G-v2 Ep {ep}/{epochs}", ncols=90)
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

        # Monitor gate stats
        with torch.no_grad():
            w1_norm = float(torch.norm(s2_gate.gate_net[0].weight).item())
            w2_norm = float(torch.norm(s2_gate.gate_net[3].weight).item())
            b2_val = float(s2_gate.gate_net[3].bias.item())
            last_alpha = model.decoder.last_gate_alpha
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
            "last_batch_alpha_mean": alpha_mean,
            "last_batch_alpha_min": alpha_min,
            "last_batch_alpha_max": alpha_max,
        }
        logs.append(ep_log)
        print(f"Epoch {ep:02d} Summary: Loss={ep_log['loss_total']:.4f} (BCE={ep_log['loss_bce']:.4f}, Dice={ep_log['loss_dice']:.4f}) | Gate W1 Norm={w1_norm:.4f}, W2 Norm={w2_norm:.4f}, Bias={b2_val:.4f} | Alpha=[{alpha_min:.3f}, {alpha_mean:.3f}, {alpha_max:.3f}]")

    return logs


# -----------------------------------------------------------------------------
# Spatial Gate Alpha Audit
# -----------------------------------------------------------------------------

def audit_gate_alpha_spatial(
    model: nn.Module,
    val_files: List[Tuple[str, str]],
    df_118: Optional[pd.DataFrame],
    device: torch.device,
    out_dir: str,
) -> pd.DataFrame:
    """
    Evaluates whether S2Gate v2 alpha is lower in bridge corridors vs real cracks.
    """
    print("\n" + "=" * 80)
    print("AUDITING S2-GATE-v2 SPATIAL MODULATION (Alpha at Corridor vs Real Crack)")
    print("=" * 80)

    model.eval()
    if df_118 is None or len(df_118) == 0:
        return pd.DataFrame()

    results = []
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    for _, ev_row in tqdm(df_118.iterrows(), total=len(df_118), desc="S2-Gate-v2 Alpha Audit", ncols=90):
        ev_id = int(ev_row["event_id"])
        case_name = ev_row["case_name"]
        d_gap = float(ev_row["d_gap_px"])
        is_wider = bool(d_gap > 5.0)
        gA = int(ev_row["primary_gt_A"])
        gB = int(ev_row["primary_gt_B"])

        # Find img & mask
        img_p = None
        mask_p = None
        for img_cand, mask_cand in val_files:
            if os.path.splitext(os.path.basename(img_cand))[0] == case_name:
                img_p = img_cand
                mask_p = mask_cand
                break

        if img_p is None:
            continue

        img_bgr = cv2.imread(img_p)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        target_bin = (mask > 127).astype(np.uint8)
        H, W = target_bin.shape

        num_gt_cc, gt_labels = cv2.connectedComponents(target_bin, connectivity=8)

        r = max(int(np.ceil(d_gap / 2.0)) + 2, 3)
        k_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        dil_A = cv2.dilate((gt_labels == gA).astype(np.uint8), k_elem)
        dil_B = cv2.dilate((gt_labels == gB).astype(np.uint8), k_elem)
        corridor_mask = (dil_A > 0) & (dil_B > 0) & (target_bin == 0)
        crack_mask = target_bin > 0

        # Tiling inference and collect gate alpha map at full resolution
        tile_size = 448
        pad_h = (tile_size - (H % tile_size)) % tile_size
        pad_w = (tile_size - (W % tile_size)) % tile_size
        img_padded = cv2.copyMakeBorder(img_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        pH, pW = img_padded.shape[:2]

        alpha_full = np.ones((pH, pW), dtype=np.float32)

        with torch.no_grad():
            for py in range(0, pH, tile_size):
                for px in range(0, pW, tile_size):
                    tile = img_padded[py : py + tile_size, px : px + tile_size]
                    tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    tile_norm = (tile_rgb - mean) / std
                    tile_t = torch.from_numpy(tile_norm.transpose(2, 0, 1)).unsqueeze(0).to(device)

                    _ = model(tile_t)
                    gate_alpha = model.decoder.last_gate_alpha  # [1, 1, 56, 56]
                    if gate_alpha is not None:
                        alpha_up = F.interpolate(gate_alpha, size=(tile_size, tile_size), mode="bilinear", align_corners=False)
                        alpha_full[py : py + tile_size, px : px + tile_size] = alpha_up.squeeze().cpu().numpy()

        alpha_crop = alpha_full[:H, :W]

        alpha_corridor = float(np.mean(alpha_crop[corridor_mask])) if np.sum(corridor_mask) > 0 else float("nan")
        alpha_crack = float(np.mean(alpha_crop[crack_mask])) if np.sum(crack_mask) > 0 else float("nan")
        delta_alpha = (alpha_crack - alpha_corridor) if not np.isnan(alpha_crack) and not np.isnan(alpha_corridor) else float("nan")

        results.append({
            "event_id": ev_id,
            "case_name": case_name,
            "d_gap_px": d_gap,
            "is_wider_gap": is_wider,
            "alpha_corridor": alpha_corridor,
            "alpha_crack": alpha_crack,
            "delta_alpha_crack_minus_corridor": delta_alpha,
        })

    df_alpha = pd.DataFrame(results)
    alpha_csv = os.path.join(out_dir, "s2g_v2_alpha_spatial_audit.csv")
    df_alpha.to_csv(alpha_csv, index=False)
    print(f"S2-Gate-v2 Alpha spatial audit saved to {alpha_csv}")

    if len(df_alpha) > 0:
        wider_df = df_alpha[df_alpha["is_wider_gap"] == True]
        print(f"  Overall Mean Alpha on Crack:     {df_alpha['alpha_crack'].mean():.4f}")
        print(f"  Overall Mean Alpha in Corridor:  {df_alpha['alpha_corridor'].mean():.4f}")
        print(f"  43 Wider Mean Alpha in Corridor: {wider_df['alpha_corridor'].mean():.4f}")
        print(f"  43 Wider Delta (Crack - Corr):   {wider_df['delta_alpha_crack_minus_corridor'].mean():+.4f}")

    return df_alpha


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 6 U1-S2G-v2: S2-Gate Conv3x3 Modulation Experiment")
    parser.add_argument("--config", type=str, default="configs/p3_ablation/b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml")
    parser.add_argument("--checkpoint", type=str, default="results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth")
    parser.add_argument("--v1_weights", type=str, default="results/diagnostics/phase6_u1_s2g/u1_s2g_weights.pth")
    parser.add_argument("--data_root", type=str, default="datasets/Crack500_ready")
    parser.add_argument("--out_dir", type=str, default="results/diagnostics/phase6_u1_s2g_v2")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_train", action="store_true", help="Skip training if weights already exist")
    args = parser.parse_args()

    # Automatically resolve config path
    if not os.path.exists(args.config):
        alt_config = os.path.join("SAGE_LITE", args.config)
        if os.path.exists(alt_config):
            args.config = alt_config

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    # Automatically resolve checkpoint path
    if not os.path.exists(args.checkpoint):
        possible_ckpt_paths = [
            os.path.join("/content", args.checkpoint.lstrip("/")),
            os.path.join("/content/SAGE_LITE", args.checkpoint.lstrip("/")),
            os.path.join("/content", "P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth"),
            os.path.join("/content/SAGE_LITE", "P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth"),
            os.path.join(project_root, args.checkpoint),
            os.path.join(project_root, "..", args.checkpoint),
            os.path.join("/content", "results", "checkpoints", "P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth"),
            os.path.join("/content/SAGE_LITE", "results", "checkpoints", "P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth"),
        ]
        for cand in possible_ckpt_paths:
            if os.path.exists(cand):
                print(f"Discovered checkpoint at: {cand}")
                args.checkpoint = cand
                break

    # Automatically resolve v1 weights path
    if not os.path.exists(args.v1_weights):
        possible_v1_paths = [
            os.path.join("/content", args.v1_weights.lstrip("/")),
            os.path.join("/content/SAGE_LITE", args.v1_weights.lstrip("/")),
            os.path.join(project_root, args.v1_weights),
            os.path.join(project_root, "..", args.v1_weights),
            os.path.join("/content", "u1_s2g_weights.pth"),
            os.path.join("/content/SAGE_LITE", "u1_s2g_weights.pth"),
        ]
        for cand in possible_v1_paths:
            if os.path.exists(cand):
                print(f"Discovered v1 weights at: {cand}")
                args.v1_weights = cand
                break

    # -----------------------------------------------------------------------
    # Step 0: Checkpoint & Pretrained Invariant Verification
    # -----------------------------------------------------------------------
    expected_ckpt_sha256 = "147f784021414efd0db514aa6dae94585fece820e88f584e436fc65de851fb66"
    actual_ckpt_sha256 = compute_file_hash(args.checkpoint)
    print(f"Base Checkpoint SHA256: {actual_ckpt_sha256}")
    assert actual_ckpt_sha256 == expected_ckpt_sha256, f"Checkpoint SHA256 mismatch! Got {actual_ckpt_sha256}"
    print(">> PASS: Checkpoint integrity verified.")

    # Load baseline model
    print("Loading baseline model from checkpoint...")
    model = load_model_from_checkpoint(args.config, args.checkpoint, device=device)
    model.eval()

    # Instantiate S2GateModule with Conv3x3 (kernel_size=3)
    s2_gate = S2GateModule(s2_channels=192, skip_channels=96, kernel_size=3).to(device)

    # Warm-start from U1-S2G (v1) weights
    if os.path.exists(args.v1_weights):
        print(f"Warm-starting S2Gate v2 from trained v1 weights: {args.v1_weights}")
        v1_ckpt = torch.load(args.v1_weights, map_location=device, weights_only=False)
        v1_state = v1_ckpt["s2_gate_state"] if "s2_gate_state" in v1_ckpt else v1_ckpt
        s2_gate.warm_start_from_v1(v1_state)
        print(">> PASS: Warm-start initialization from v1 weights completed.")
    else:
        print(f"[Warning] v1 weights not found at {args.v1_weights}. Initializing from scratch.")

    model.decoder.s2_gate = s2_gate
    model.decoder.use_s2_gate = True

    # Strict parameter freezing: ONLY s2_gate is trainable
    for p in model.parameters():
        p.requires_grad = False
    for p in s2_gate.parameters():
        p.requires_grad = True

    trainable_whitelist = [p for p in model.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable_whitelist)
    print(f"Trainable parameters count: {total_trainable}")
    assert total_trainable == 83073, f"Expected 83,073 trainable parameters, got {total_trainable}"
    print(">> PASS: Exactly 83,073 trainable parameters whitelisted.")

    # Preflight Identity & Gradient Flow Check
    dummy_x = torch.randn(2, 3, 448, 448, device=device)
    dummy_out = model(dummy_x)
    assert dummy_out.shape == (2, 1, 448, 448), f"Unexpected shape {dummy_out.shape}"
    dummy_loss = dummy_out.sum()
    dummy_loss.backward()

    for name, p in s2_gate.named_parameters():
        assert p.grad is not None, f"Gradient missing on S2Gate parameter {name}"
    assert model.decoder.decoder_blocks[0].conv1[0].weight.grad is None
    assert model.decoder.decoder_blocks[1].conv1[0].weight.grad is None
    assert model.decoder.segmentation_head[0].weight.grad is None
    for p in model.backbone.parameters():
        assert p.grad is None
    print(">> PASS: Preflight backward gradient isolation confirmed.")

    model.zero_grad()

    # -----------------------------------------------------------------------
    # Datasets
    # -----------------------------------------------------------------------
    cand_data_roots = [
        args.data_root,
        "/content/dataset/Crack500",
        os.path.join(project_root, args.data_root),
        os.path.join(project_root, "..", "data"),
        os.path.join(project_root, "..", "dataset", "Crack500"),
        "/content/SAGE_LITE/datasets/Crack500_ready",
        "/content/datasets/Crack500_ready",
        "datasets/Crack500_ready",
    ]
    resolved_data_root = None
    for cand in cand_data_roots:
        cand_val_img = os.path.join(cand, "val", "images")
        if os.path.exists(cand_val_img):
            resolved_data_root = cand
            break

    if resolved_data_root is not None:
        args.data_root = resolved_data_root
        print(f"Discovered and resolved data_root at: {args.data_root}")
    else:
        print(f"[Warning] Could not find 'val/images' under search roots. Using requested: {args.data_root}")

    train_img_dir = os.path.join(args.data_root, "train", "images")
    train_mask_dir = os.path.join(args.data_root, "train", "masks")
    val_img_dir = os.path.join(args.data_root, "val", "images")
    val_mask_dir = os.path.join(args.data_root, "val", "masks")

    val_files = []
    for f in sorted(os.listdir(val_img_dir)):
        if f.lower().endswith((".jpg", ".png")):
            base = os.path.splitext(f)[0]
            mp = os.path.join(val_mask_dir, base + ".png")
            if not os.path.exists(mp):
                mp = os.path.join(val_mask_dir, base + ".jpg")
            if os.path.exists(mp):
                val_files.append((os.path.join(val_img_dir, f), mp))

    assert len(val_files) == 348, f"Expected 348 validation samples, got {len(val_files)}"
    print(f"Found {len(val_files)} canonical validation samples.")

    train_dataset = Crack500TrainDataset(train_img_dir, train_mask_dir, img_size=448, seed=args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=14,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Output artifact paths
    weights_path = os.path.join(args.out_dir, "s2g_v2_weights.pth")
    training_log_path = os.path.join(args.out_dir, "s2g_v2_training_log.csv")
    val_s2g_csv = os.path.join(args.out_dir, "validation_s2g_v2_per_sample.csv")
    wider_43_csv = os.path.join(args.out_dir, "s2g_v2_43wider_events.csv")
    all_118_csv = os.path.join(args.out_dir, "s2g_v2_118all_events.csv")
    worsened_11_csv = os.path.join(args.out_dir, "s2g_v2_11worsened_recovery.csv")
    summary_json_path = os.path.join(args.out_dir, "u1_s2g_v2_vs_candidate_b_summary.json")

    # -----------------------------------------------------------------------
    # Step 1: Training or Loading Weights
    # -----------------------------------------------------------------------
    if os.path.exists(weights_path) and args.skip_train:
        print(f"\nLoading existing trained weights from {weights_path}...")
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        s2_gate.load_state_dict(ckpt["s2_gate_state"])
        print(">> Trained weights loaded successfully.")
    else:
        print("\nStarting U1-S2G-v2 Training...")
        train_logs = train_u1_s2g_v2(
            model=model,
            s2_gate=s2_gate,
            train_loader=train_loader,
            device=device,
            epochs=args.epochs,
            out_dir=args.out_dir,
        )
        pd.DataFrame(train_logs).to_csv(training_log_path, index=False)
        print(f"Training logs saved to {training_log_path}")

        # Save weights
        save_dict = {
            "s2_gate_state": s2_gate.state_dict(),
            "trainable_param_count": total_trainable,
            "kernel_size": 3,
            "epochs": args.epochs,
        }
        torch.save(save_dict, weights_path)
        print(f"Saved U1-S2G-v2 weights to {weights_path}")

    # -----------------------------------------------------------------------
    # Step 2: Canonical Setting A Validation (N=348)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SETTING A CANONICAL EVALUATION (Val N=348, tile 448x448, tau=0.5, FP32 Strict)")
    print("=" * 80)

    model.eval()
    val_records = []
    logits_cache = {}
    runtimes = []

    for img_path, mask_path in tqdm(val_files, desc="Setting A Validation (U1-S2G-v2)", ncols=90):
        case_name = os.path.splitext(os.path.basename(img_path))[0]
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        target_bin = (mask > 127).astype(np.uint8)

        t0 = time.time()
        logits = predict_full_image_tiling_setting_a_fp32(model, img_rgb, device, tile_size=448, batch_size=8)
        dt_ms = (time.time() - t0) * 1000.0
        runtimes.append(dt_ms)
        logits_cache[case_name] = logits

        pred_bin = (logits > 0.0).astype(np.uint8)

        # Basic segmentation metrics
        tp = int(np.sum((pred_bin == 1) & (target_bin == 1)))
        fp = int(np.sum((pred_bin == 1) & (target_bin == 0)))
        fn = int(np.sum((pred_bin == 0) & (target_bin == 1)))
        tn = int(np.sum((pred_bin == 0) & (target_bin == 0)))

        dice = float(2 * tp / (2 * tp + fp + fn + 1e-8))
        iou = float(tp / (tp + fp + fn + 1e-8))
        prec = float(tp / (tp + fp + 1e-8))
        rec = float(tp / (tp + fn + 1e-8))

        # Advanced topology metrics
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
            "fragmented_gt_components": topo["fragmented_gt_components"],
            "break_status": 1 if topo["fragmented_gt_components"] > 0 else 0,
            "spurious_islands": topo.get("spurious_island_count", topo.get("spurious_islands", 0)),
            "gt_cc": topo["gt_cc"],
            "pred_cc": topo["pred_cc"],
            "runtime_ms": dt_ms,
        })

    df_s2g = pd.DataFrame(val_records)
    df_s2g.to_csv(val_s2g_csv, index=False)
    print(f"Validation per-sample results saved to {val_s2g_csv}")

    # Summary metrics
    s2g_summary = {
        "N": len(df_s2g),
        "dice": float(df_s2g["dice"].mean()),
        "iou": float(df_s2g["iou"].mean()),
        "recall": float(df_s2g["recall"].mean()),
        "precision": float(df_s2g["precision"].mean()),
        "cldice": float(df_s2g["cldice"].mean()),
        "boundary_iou": float(df_s2g["boundary_iou"].mean()),
        "hd95": float(df_s2g["hd95"].dropna().mean()),
        "bf1": float(df_s2g["bf1"].dropna().mean()),
        "thin_dice": float(df_s2g["thin_dice"].dropna().mean()),
        "total_bridge_events": int(df_s2g["bridge_events"].sum()),
        "bridge_images": int(df_s2g["bridge_status"].sum()),
        "total_break_events": int(df_s2g["fragmented_gt_components"].sum()),
        "break_images": int(df_s2g["break_status"].sum()),
        "spurious_islands": int(df_s2g["spurious_islands"].sum()),
        "mean_runtime_ms": float(np.mean(runtimes)),
    }

    # -----------------------------------------------------------------------
    # Step 3: Cohort Analysis on 43 Wider-Gap & 118 All Events
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("COHORT EVALUATION: 43 Wider-Gap Events & 118 All Events")
    print("=" * 80)

    events_csv = "results/diagnostics/phase6_bottleneck_path/bottleneck_path_118events.csv"
    for cand in [events_csv, os.path.join("..", events_csv), os.path.join("/content", events_csv), os.path.join("/content/SAGE_LITE", events_csv), os.path.join(project_root, events_csv)]:
        if os.path.exists(cand):
            events_csv = cand
            break

    cgsr_43_csv = "results/Phase6D_CGSR/evaluation_phase6d/cgsr_evaluation_43wider_events.csv"
    for cand in [cgsr_43_csv, os.path.join("..", cgsr_43_csv), os.path.join("/content", cgsr_43_csv), os.path.join("/content/SAGE_LITE", cgsr_43_csv), os.path.join(project_root, cgsr_43_csv)]:
        if os.path.exists(cand):
            cgsr_43_csv = cand
            break

    v1_43_csv = "results/diagnostics/phase6_u1_s2g/s2g_43wider_events.csv"
    for cand in [v1_43_csv, os.path.join("..", v1_43_csv), os.path.join("/content", v1_43_csv), os.path.join("/content/SAGE_LITE", v1_43_csv), os.path.join(project_root, v1_43_csv)]:
        if os.path.exists(cand):
            v1_43_csv = cand
            break

    cgsr_ref = pd.read_csv(cgsr_43_csv).set_index("event_id") if os.path.exists(cgsr_43_csv) else None
    v1_ref = pd.read_csv(v1_43_csv).set_index("event_id") if os.path.exists(v1_43_csv) else None

    wider_43_records = []
    all_118_records = []
    df_118 = None

    if os.path.exists(events_csv):
        df_118 = pd.read_csv(events_csv)
        for _, ev_row in df_118.iterrows():
            ev_id = int(ev_row["event_id"])
            case_name = ev_row["case_name"]
            d_gap = float(ev_row["d_gap_px"])
            is_wider = bool(d_gap > 5.0)
            gA = int(ev_row["primary_gt_A"])
            gB = int(ev_row["primary_gt_B"])

            mp = os.path.join(val_mask_dir, case_name + ".png")
            if not os.path.exists(mp):
                mp = os.path.join(val_mask_dir, case_name + ".jpg")
            target = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            target_bin = (target > 127).astype(np.uint8)

            logits = logits_cache[case_name]
            pred_bin = (logits > 0.0).astype(np.uint8)

            num_gt_cc, gt_labels = cv2.connectedComponents(target_bin, connectivity=8)
            num_pred_cc, pred_labels = cv2.connectedComponents(pred_bin, connectivity=8)

            r = max(int(np.ceil(d_gap / 2.0)) + 2, 3)
            k_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            dil_A = cv2.dilate((gt_labels == gA).astype(np.uint8), k_elem)
            dil_B = cv2.dilate((gt_labels == gB).astype(np.uint8), k_elem)
            corridor_mask = (dil_A > 0) & (dil_B > 0) & (target_bin == 0)

            z_bridge = float(np.mean(logits[corridor_mask])) if np.sum(corridor_mask) > 0 else float("nan")

            is_merged = False
            for p in range(1, num_pred_cc):
                gt_overlap = np.unique(gt_labels[pred_labels == p])
                if gA in gt_overlap and gB in gt_overlap:
                    is_merged = True
                    break

            is_cured = (not is_merged) or (z_bridge < 0.0)

            z_ctrl = float(cgsr_ref.loc[ev_id, "z_ctrl_bridge"]) if (cgsr_ref is not None and ev_id in cgsr_ref.index) else float("nan")
            delta_z = (z_ctrl - z_bridge) if not np.isnan(z_ctrl) and not np.isnan(z_bridge) else float("nan")

            z_v1 = float(v1_ref.loc[ev_id, "z_s2g_bridge"]) if (v1_ref is not None and ev_id in v1_ref.index) else float("nan")
            delta_z_v1 = float(v1_ref.loc[ev_id, "delta_z_bridge"]) if (v1_ref is not None and ev_id in v1_ref.index) else float("nan")
            cured_v1 = bool(v1_ref.loc[ev_id, "is_cured"]) if (v1_ref is not None and ev_id in v1_ref.index) else False

            rec = {
                "event_id": ev_id,
                "case_name": case_name,
                "d_gap": d_gap,
                "is_wider_gap": is_wider,
                "z_ctrl_bridge": z_ctrl,
                "z_s2g_v1_bridge": z_v1,
                "delta_z_v1_bridge": delta_z_v1,
                "cured_v1": cured_v1,
                "z_s2g_v2_bridge": z_bridge,
                "delta_z_v2_bridge": delta_z,
                "delta_z_v2_vs_v1": (z_v1 - z_bridge) if not np.isnan(z_v1) and not np.isnan(z_bridge) else float("nan"),
                "is_merged": is_merged,
                "is_cured": is_cured,
            }

            all_118_records.append(rec)
            if is_wider:
                wider_43_records.append(rec)

        pd.DataFrame(all_118_records).to_csv(all_118_csv, index=False)
        pd.DataFrame(wider_43_records).to_csv(wider_43_csv, index=False)

    df_wider = pd.DataFrame(wider_43_records)
    cured_43 = int(df_wider["is_cured"].sum()) if len(df_wider) > 0 else 0
    cure_rate_43 = float(cured_43 / len(df_wider) * 100.0) if len(df_wider) > 0 else 0.0

    df_all_ev = pd.DataFrame(all_118_records)
    cured_118 = int(df_all_ev["is_cured"].sum()) if len(df_all_ev) > 0 else 0
    cure_rate_118 = float(cured_118 / len(df_all_ev) * 100.0) if len(df_all_ev) > 0 else 0.0

    # -----------------------------------------------------------------------
    # Step 4: Dedicated 11 Worsened Events Recovery Audit
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DEDICATED RECOVERY AUDIT: The 11 Worsened Cases from U1-S2G (v1)")
    print("=" * 80)

    # Identifiers of 11 worsened cases from s2g_43wider_events.csv
    worsened_event_ids = [2, 20, 21, 22, 23, 33, 47, 52, 54, 69, 111]
    worsened_records = []

    for rec in wider_43_records:
        if rec["event_id"] in worsened_event_ids:
            ev_id = rec["event_id"]
            z_ctrl = rec["z_ctrl_bridge"]
            z_v1 = rec["z_s2g_v1_bridge"]
            z_v2 = rec["z_s2g_v2_bridge"]
            delta_z_v2 = rec["delta_z_v2_bridge"]
            # Flipped/improved definition: delta_z_v2 > 0 (suppressed below ctrl) OR z_v2 < z_v1 (better than v1)
            improved_vs_v1 = bool(z_v2 < z_v1) if not np.isnan(z_v2) and not np.isnan(z_v1) else False
            flipped_vs_ctrl = bool(delta_z_v2 > 0.0) if not np.isnan(delta_z_v2) else False

            worsened_records.append({
                "event_id": ev_id,
                "case_name": rec["case_name"],
                "d_gap": rec["d_gap"],
                "z_ctrl": z_ctrl,
                "z_v1": z_v1,
                "delta_z_v1": rec["delta_z_v1_bridge"],
                "z_v2": z_v2,
                "delta_z_v2": delta_z_v2,
                "delta_z_v2_vs_v1": rec["delta_z_v2_vs_v1"],
                "improved_vs_v1": improved_vs_v1,
                "flipped_vs_ctrl": flipped_vs_ctrl,
                "is_cured_v2": rec["is_cured"],
            })

    df_worsened = pd.DataFrame(worsened_records)
    df_worsened.to_csv(worsened_11_csv, index=False)
    print(f"Dedicated 11 Worsened Recovery Audit saved to {worsened_11_csv}")

    num_improved_vs_v1 = int(df_worsened["improved_vs_v1"].sum()) if len(df_worsened) > 0 else 0
    num_flipped_vs_ctrl = int(df_worsened["flipped_vs_ctrl"].sum()) if len(df_worsened) > 0 else 0
    num_cured_in_11 = int(df_worsened["is_cured_v2"].sum()) if len(df_worsened) > 0 else 0

    print(f"\n11 WORSENED CASES RECOVERY SUMMARY:")
    print(f"  Improved vs v1 (z_v2 < z_v1):         {num_improved_vs_v1}/11 ({num_improved_vs_v1/11*100:.1f}%)")
    print(f"  Flipped vs Ctrl (delta_z_v2 > 0):     {num_flipped_vs_ctrl}/11 ({num_flipped_vs_ctrl/11*100:.1f}%)")
    print(f"  Completely Cured in 11:               {num_cured_in_11}/11 ({num_cured_in_11/11*100:.1f}%)")
    if len(df_worsened) > 0:
        print(f"  Mean z_ctrl: {df_worsened['z_ctrl'].mean():.3f} | Mean z_v1: {df_worsened['z_v1'].mean():.3f} | Mean z_v2: {df_worsened['z_v2'].mean():.3f}")

    # -----------------------------------------------------------------------
    # Step 5: Spatial Gate Alpha Audit
    # -----------------------------------------------------------------------
    audit_gate_alpha_spatial(
        model=model,
        val_files=val_files,
        df_118=df_118,
        device=device,
        out_dir=args.out_dir,
    )

    # -----------------------------------------------------------------------
    # Step 6: Master Comparison & Summary JSON
    # -----------------------------------------------------------------------
    base_summary = {
        "N": 348,
        "dice": 0.7640881806771141,
        "recall": 0.8476884187188468,
        "precision": 0.7336947741984694,
        "cldice": 0.8498515829576173,
        "boundary_iou": 0.24175755623551817,
        "hd95": 53.648256920776745,
        "bf1": 0.38201035372136555,
        "thin_dice": 0.4230063161265414,
        "total_bridge_events": 118,
        "bridge_images": 110,
        "total_break_events": 35,
        "break_images": 34,
        "spurious_islands": 111,
        "mean_runtime_ms": 147.89956328512608,
    }

    u1_s2g_v1_summary = {
        "N": 348,
        "dice": 0.7570,
        "cldice": 0.8415,
        "boundary_iou": 0.2319,
        "hd95": 54.04,
        "total_bridge_events": 115,
        "total_break_events": 43,
        "cured_43": 12,
        "cure_rate_43": 27.91,
        "cured_118": 15,
        "cure_rate_118": 12.71,
        "worsened_count": 11,
        "mean_delta_z_43": 0.8508,
        "median_delta_z_43": 0.6031,
    }

    u0_c3_summary = {
        "N": 348,
        "dice": 0.7346,
        "cldice": 0.8173,
        "boundary_iou": 0.2116,
        "total_bridge_events": 105,
        "total_break_events": 76,
        "cured_43": 17,
        "cure_rate_43": 39.53,
    }

    full_report = {
        "Candidate_B_Baseline": base_summary,
        "Candidate_B_U0_C3_DCInit_Ref": u0_c3_summary,
        "Candidate_B_U1_S2G_v1": u1_s2g_v1_summary,
        "Candidate_B_U1_S2G_v2": s2g_summary,
        "Delta_S2G_v2_minus_Base": {
            "delta_dice": s2g_summary["dice"] - base_summary["dice"],
            "delta_cldice": s2g_summary["cldice"] - base_summary["cldice"],
            "delta_boundary_iou": s2g_summary["boundary_iou"] - base_summary["boundary_iou"],
            "delta_hd95": s2g_summary["hd95"] - base_summary["hd95"],
            "delta_bridge_events": s2g_summary["total_bridge_events"] - base_summary["total_bridge_events"],
            "delta_break_events": s2g_summary["total_break_events"] - base_summary["total_break_events"],
        },
        "Delta_S2G_v2_minus_v1": {
            "delta_dice": s2g_summary["dice"] - u1_s2g_v1_summary["dice"],
            "delta_cldice": s2g_summary["cldice"] - u1_s2g_v1_summary["cldice"],
            "delta_boundary_iou": s2g_summary["boundary_iou"] - u1_s2g_v1_summary["boundary_iou"],
            "delta_hd95": s2g_summary["hd95"] - u1_s2g_v1_summary["hd95"],
            "delta_bridge_events": s2g_summary["total_bridge_events"] - u1_s2g_v1_summary["total_bridge_events"],
            "delta_break_events": s2g_summary["total_break_events"] - u1_s2g_v1_summary["total_break_events"],
        },
        "Cohort_43_Wider_Gap": {
            "N": len(df_wider),
            "cured_count_v2": cured_43,
            "cure_rate_pct_v2": cure_rate_43,
            "cured_count_v1": 12,
            "cure_rate_pct_v1": 27.91,
            "u0_c3_cured_count": 17,
            "u0_c3_cure_rate_pct": 39.53,
            "median_delta_z_bridge_v2": float(df_wider["delta_z_v2_bridge"].dropna().median()) if len(df_wider) > 0 else 0.0,
            "mean_delta_z_bridge_v2": float(df_wider["delta_z_v2_bridge"].dropna().mean()) if len(df_wider) > 0 else 0.0,
        },
        "Cohort_11_Worsened_Recovery": {
            "N": len(df_worsened),
            "improved_vs_v1_count": num_improved_vs_v1,
            "improved_vs_v1_pct": float(num_improved_vs_v1 / 11.0 * 100.0) if len(df_worsened) > 0 else 0.0,
            "flipped_vs_ctrl_count": num_flipped_vs_ctrl,
            "flipped_vs_ctrl_pct": float(num_flipped_vs_ctrl / 11.0 * 100.0) if len(df_worsened) > 0 else 0.0,
            "cured_count": num_cured_in_11,
            "cured_pct": float(num_cured_in_11 / 11.0 * 100.0) if len(df_worsened) > 0 else 0.0,
        },
        "Cohort_All_118_Events": {
            "N": len(df_all_ev),
            "cured_count_v2": cured_118,
            "cure_rate_pct_v2": cure_rate_118,
            "cured_count_v1": 15,
            "cure_rate_pct_v1": 12.71,
            "median_delta_z_bridge_v2": float(df_all_ev["delta_z_v2_bridge"].dropna().median()) if len(df_all_ev) > 0 else 0.0,
            "mean_delta_z_bridge_v2": float(df_all_ev["delta_z_v2_bridge"].dropna().mean()) if len(df_all_ev) > 0 else 0.0,
        },
    }

    with open(summary_json_path, "w") as f:
        json.dump(full_report, f, indent=2)

    print(f"\nFinal Summary Report saved to {summary_json_path}")
    print("\n" + "=" * 80)
    print("EXPERIMENT U1-S2G-v2 RESULTS SUMMARY:")
    print(f"  Val Dice:          {s2g_summary['dice']:.4f} (Base: {base_summary['dice']:.4f}, v1: {u1_s2g_v1_summary['dice']:.4f})")
    print(f"  Val clDice:        {s2g_summary['cldice']:.4f} (Base: {base_summary['cldice']:.4f}, v1: {u1_s2g_v1_summary['cldice']:.4f})")
    print(f"  Val HD95:          {s2g_summary['hd95']:.2f} px (Base: {base_summary['hd95']:.2f} px, v1: {u1_s2g_v1_summary['hd95']:.2f} px)")
    print(f"  Val Bridge Events: {s2g_summary['total_bridge_events']} (Base: {base_summary['total_bridge_events']}, v1: {u1_s2g_v1_summary['total_bridge_events']})")
    print(f"  Val Break Events:  {s2g_summary['total_break_events']} (Base: {base_summary['total_break_events']}, v1: {u1_s2g_v1_summary['total_break_events']})")
    print(f"  43 Wider Cured:    {cured_43}/{len(df_wider)} ({cure_rate_43:.1f}%) [v1: 12/43 = 27.9%, C3: 17/43 = 39.5%]")
    print(f"  11 Worsened Flipped: {num_flipped_vs_ctrl}/11 ({num_flipped_vs_ctrl/11*100:.1f}%) | Improved vs v1: {num_improved_vs_v1}/11 ({num_improved_vs_v1/11*100:.1f}%)")
    print(f"  118 All Cured:     {cured_118}/{len(df_all_ev)} ({cure_rate_118:.1f}%) [v1: 15/118 = 12.7%]")
    print("=" * 80)


if __name__ == "__main__":
    main()
