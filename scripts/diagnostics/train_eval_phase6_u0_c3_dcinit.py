#!/usr/bin/env python3
"""
scripts/diagnostics/train_eval_phase6_u0_c3_dcinit.py

Phase 6: Upstream Genesis / DC-Init Stem Experiment (U0-C3)
============================================================

Scientific Objective:
- Test the causal hypothesis U0-C3:
  Does factorizing the ConvNeXt-V2-Femto stem Conv2d(3, 48, 4, 4, stride=4) into:
    - DC pathway (AvgPool2d(4,4) -> Conv2d(3, 48, 1, 1)) initialized with pretrained spatial sum weights
    - AC residual (Conv2d(3, 48, 4, 4, stride=4, bias=False)) initialized strictly to ZERO
    - LayerNorm2d initialized from pretrained stem norm
  and training ONLY the Stem (dc_proj, ac_delta, norm) + Stage 0 LayerNorm affine parameters
  (2,784 trainable parameters total) eliminate false bridges while preserving crack continuity?

Invariants:
- Total trainable parameters: EXACTLY 2,784 parameters.
  - dc_proj: 48*3*1*1 + 48 = 192
  - ac_delta: 48*3*4*4 = 2,304
  - stem norm: 48 + 48 = 96
  - Stage 0 LayerNorms: 2 blocks * (48 weight + 48 bias) = 192
  - Sum: 192 + 2304 + 96 + 192 = 2,784
- Downstream network (Stages 1-3, ViT, SAGE Routers, Decoder) 100% strictly FROZEN.
- Pretrained base checkpoint: P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth
  (SHA256: 147f784021414efd0db514aa6dae94585fece820e88f584e436fc65de851fb66) NEVER MUTATED ON DISK.
- Sealed Test Set: N=1124 strictly untouched.
- Setting A validation protocol: N=348 samples, 448x448 tiling, stride 448, reflect pad, tau=0.5.
- Training: FP32 ONLY, physical batch size 14, seed 42, 8 epochs, CosineAnnealingLR.
  - Optimizer groups:
    * dc_proj: lr = 1e-4, wd = 1e-2
    * ac_delta: lr = 1e-3, wd = 1e-4 (zero-init needs higher learning rate to grow clean AC filters)
    * stem norm: lr = 1e-4, wd = 1e-2
    * stage 0 LayerNorms: lr = 5e-5, wd = 1e-2
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

# Self-contained path resolution: works both inside /content/SAGE_LITE and in root SpecialSubjectTTNT
current_dir = os.path.dirname(os.path.abspath(__file__))
cand_roots = [
    os.path.abspath(os.path.join(current_dir, "..", "..")),
    os.path.abspath(os.path.join(current_dir, "..")),
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
from scripts.evaluate_crack_official import predict_full_image_tiling_setting_a
from sage.networks.dc_init_stem import DCInitStem, init_dc_stem_from_pretrained


# -----------------------------------------------------------------------------
# Standalone Dataset and Loss (Fully inlined for zero-dependency execution)
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
    def __init__(self, img_dir: str, mask_dir: str, img_size: int = 448, seed: int = 42):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.rng = np.random.RandomState(seed)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_p = self.img_paths[idx]
        stem = os.path.splitext(os.path.basename(img_p))[0]
        mask_p = os.path.join(self.mask_dir, f"{stem}.png")
        if not os.path.exists(mask_p):
            mask_p = os.path.join(self.mask_dir, f"{stem}.jpg")

        img = cv2.imread(img_p)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_p}")
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask: {mask_p}")

        H, W = mask.shape
        if H < self.img_size or W < self.img_size:
            pad_h = max(0, self.img_size - H)
            pad_w = max(0, self.img_size - W)
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            H, W = mask.shape

        y1 = self.rng.randint(0, H - self.img_size + 1)
        x1 = self.rng.randint(0, W - self.img_size + 1)

        crop_img = img[y1 : y1 + self.img_size, x1 : x1 + self.img_size]
        crop_mask = mask[y1 : y1 + self.img_size, x1 : x1 + self.img_size]

        if self.rng.rand() > 0.5:
            crop_img = np.fliplr(crop_img).copy()
            crop_mask = np.fliplr(crop_mask).copy()
        if self.rng.rand() > 0.5:
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
# Standalone ROI Isolation Functions
# -----------------------------------------------------------------------------

def isolate_bridged_pairs_and_rois(
    pred_bin: np.ndarray,
    target_bin: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    H, W = target_bin.shape
    num_gt_cc, gt_labels = cv2.connectedComponents(target_bin.astype(np.uint8), connectivity=8)
    num_pred_cc, pred_labels = cv2.connectedComponents(pred_bin.astype(np.uint8), connectivity=8)
    pred_cc = int(max(num_pred_cc - 1, 0))
    gt_cc = int(max(num_gt_cc - 1, 0))

    neck_mask = np.zeros((H, W), dtype=np.uint8)
    crack_mask = np.zeros((H, W), dtype=np.uint8)
    corridor_union = np.zeros((H, W), dtype=np.uint8)
    pair_records = []

    if gt_cc < 2 or pred_cc < 1:
        return neck_mask, crack_mask, neck_mask, pair_records

    for p_id in range(1, pred_cc + 1):
        cc_pred = (pred_labels == p_id)
        overlapping_gt = np.unique(gt_labels[cc_pred])
        overlapping_gt = overlapping_gt[overlapping_gt > 0]

        if len(overlapping_gt) >= 2:
            cc_fp = cc_pred & (target_bin == 0)
            for g_id in overlapping_gt:
                crack_mask = crack_mask | (gt_labels == g_id).astype(np.uint8)

            for i in range(len(overlapping_gt)):
                for j in range(i + 1, len(overlapping_gt)):
                    gi = overlapping_gt[i]
                    gj = overlapping_gt[j]
                    mask_i = (gt_labels == gi)
                    mask_j = (gt_labels == gj)

                    dt_i = distance_transform_edt(~mask_i)
                    gap_dist = float(np.min(dt_i[mask_j]))
                    radius = max(int(np.ceil(gap_dist / 2.0)) + 2, 3)
                    ksize = 2 * radius + 1
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

                    dil_i = cv2.dilate(mask_i.astype(np.uint8), kernel)
                    dil_j = cv2.dilate(mask_j.astype(np.uint8), kernel)

                    corridor = (dil_i > 0) & (dil_j > 0)
                    corridor_union = corridor_union | corridor.astype(np.uint8)
                    neck = cc_fp & corridor
                    neck_mask = neck_mask | neck.astype(np.uint8)

                    pair_records.append({
                        "gt_i": int(gi),
                        "gt_j": int(gj),
                        "gap_distance_px": gap_dist,
                    })

    if len(pair_records) > 0 and np.sum(neck_mask) == 0:
        for p_id in range(1, pred_cc + 1):
            cc_pred = (pred_labels == p_id)
            overlapping_gt = np.unique(gt_labels[cc_pred])
            overlapping_gt = overlapping_gt[overlapping_gt > 0]
            if len(overlapping_gt) >= 2:
                neck_mask = neck_mask | (cc_pred & (target_bin == 0)).astype(np.uint8)

    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    dilated_corridor = cv2.dilate(corridor_union, kernel_bg)
    nearby_bg_mask = (dilated_corridor > 0) & (target_bin == 0) & (pred_bin == 0)

    if np.sum(nearby_bg_mask) < 50:
        kernel_bg_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
        dilated_corridor_large = cv2.dilate(corridor_union, kernel_bg_large)
        nearby_bg_mask = (dilated_corridor_large > 0) & (target_bin == 0) & (pred_bin == 0)

    return neck_mask, crack_mask, nearby_bg_mask.astype(np.uint8), pair_records


def isolate_clean_pairs_and_rois(
    target_bin: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    H, W = target_bin.shape
    num_gt_cc, gt_labels = cv2.connectedComponents(target_bin.astype(np.uint8), connectivity=8)

    min_gap = 1e9
    best_pair = None
    for i in range(1, num_gt_cc):
        m_i = (gt_labels == i)
        dt_i = distance_transform_edt(~m_i)
        for j in range(i + 1, num_gt_cc):
            m_j = (gt_labels == j)
            d = float(np.min(dt_i[m_j]))
            if d < min_gap:
                min_gap = d
                best_pair = (i, j)

    if best_pair is None:
        return np.zeros((H, W), dtype=np.uint8), target_bin.astype(np.uint8), np.ones((H, W), dtype=np.uint8), []

    i, j = best_pair
    m_i = (gt_labels == i)
    m_j = (gt_labels == j)
    radius = max(int(np.ceil(min_gap / 2.0)) + 2, 3)
    ksize = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dil_i = cv2.dilate(m_i.astype(np.uint8), kernel)
    dil_j = cv2.dilate(m_j.astype(np.uint8), kernel)
    corridor = (dil_i > 0) & (dil_j > 0)

    gap_neck = corridor & (target_bin == 0)
    crack_mask = (m_i | m_j).astype(np.uint8)

    k_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    dil_corridor = cv2.dilate(corridor.astype(np.uint8), k_bg)
    nearby_bg = (dil_corridor > 0) & (target_bin == 0) & (~gap_neck)

    if np.sum(nearby_bg) < 50:
        k_bg_l = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
        dil_corridor_l = cv2.dilate(corridor.astype(np.uint8), k_bg_l)
        nearby_bg = (dil_corridor_l > 0) & (target_bin == 0) & (~gap_neck)

    return gap_neck.astype(np.uint8), crack_mask, nearby_bg.astype(np.uint8), [{"gap_distance_px": min_gap}]


def compute_file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
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


def extract_stem_and_stage0_energy_and_features(
    model: nn.Module,
    tile_t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    act_stem = model.backbone.convnext.stem(tile_t)
    energy_stem = torch.norm(act_stem, p=2, dim=1).squeeze(0)

    stage0 = model.backbone.convnext.stages[0]
    act_stage0 = stage0(act_stem)
    if isinstance(act_stage0, tuple):
        act_stage0 = act_stage0[0]
    energy_stage0 = torch.norm(act_stage0, p=2, dim=1).squeeze(0)

    return act_stem, energy_stem, act_stage0, energy_stage0


def evaluate_dual_representation(
    model: nn.Module,
    cohort_stems: List[Tuple[str, str]],
    val_img_dir: str,
    val_mask_dir: str,
    device: torch.device,
    model_tag: str,
    tile_size: int = 448,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model.eval()
    rows_stem = []
    rows_stage0 = []

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    for stem_name, cohort in tqdm(cohort_stems, desc=f"Dual Repr Audit [{model_tag}]", ncols=90):
        img_p = os.path.join(val_img_dir, f"{stem_name}.jpg")
        mask_p = os.path.join(val_mask_dir, f"{stem_name}.png")
        if not os.path.exists(mask_p):
            mask_p = os.path.join(val_mask_dir, f"{stem_name}.jpg")

        img = cv2.imread(img_p)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue

        gt_binary = (mask > 127).astype(np.uint8)
        logits = predict_full_image_tiling_setting_a(model, cv2.cvtColor(img, cv2.COLOR_BGR2RGB), device, tile_size=tile_size, batch_size=8)
        pred_binary = (logits > 0.0).astype(np.uint8)

        if cohort != "Clean_Control":
            neck_mask, crack_mask, bg_mask, _ = isolate_bridged_pairs_and_rois(pred_binary, gt_binary)
        else:
            neck_mask, crack_mask, bg_mask, _ = isolate_clean_pairs_and_rois(gt_binary)

        n_neck_px = int(np.sum(neck_mask))
        n_crack_px = int(np.sum(crack_mask))
        n_bg_px = int(np.sum(bg_mask))

        H, W = gt_binary.shape
        pad_h = (tile_size - (H % tile_size)) % tile_size
        pad_w = (tile_size - (W % tile_size)) % tile_size
        img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        pH, pW = img_padded.shape[:2]

        energy_stem_full = np.zeros((pH, pW), dtype=np.float32)
        energy_stage0_full = np.zeros((pH, pW), dtype=np.float32)

        f_neck_stem = []
        f_crack_stem = []
        f_neck_stage0 = []
        f_crack_stage0 = []

        with torch.no_grad():
            for py in range(0, pH, tile_size):
                for px in range(0, pW, tile_size):
                    tile = img_padded[py : py + tile_size, px : px + tile_size]
                    tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    tile_norm = (tile_rgb - mean) / std
                    tile_t = torch.from_numpy(tile_norm.transpose(2, 0, 1)).unsqueeze(0).to(device)

                    act_s, eng_s, act_0, eng_0 = extract_stem_and_stage0_energy_and_features(model, tile_t)

                    eng_s_448 = F.interpolate(eng_s.unsqueeze(0).unsqueeze(0), size=(tile_size, tile_size), mode="bilinear", align_corners=False).squeeze().cpu().numpy()
                    eng_0_448 = F.interpolate(eng_0.unsqueeze(0).unsqueeze(0), size=(tile_size, tile_size), mode="bilinear", align_corners=False).squeeze().cpu().numpy()

                    energy_stem_full[py : py + tile_size, px : px + tile_size] = eng_s_448
                    energy_stage0_full[py : py + tile_size, px : px + tile_size] = eng_0_448

                    p_neck = neck_mask[py : py + tile_size, px : px + tile_size] if py < H and px < W else None
                    p_crack = crack_mask[py : py + tile_size, px : px + tile_size] if py < H and px < W else None

                    if p_neck is not None and np.sum(p_neck) > 0:
                        m_n = (F.interpolate(torch.from_numpy(p_neck).float().unsqueeze(0).unsqueeze(0), size=(112, 112), mode="nearest").squeeze() > 0)
                        if m_n.any():
                            f_neck_stem.append(act_s[0, :, m_n].mean(dim=1).cpu())
                            f_neck_stage0.append(act_0[0, :, m_n].mean(dim=1).cpu())

                    if p_crack is not None and np.sum(p_crack) > 0:
                        m_c = (F.interpolate(torch.from_numpy(p_crack).float().unsqueeze(0).unsqueeze(0), size=(112, 112), mode="nearest").squeeze() > 0)
                        if m_c.any():
                            f_crack_stem.append(act_s[0, :, m_c].mean(dim=1).cpu())
                            f_crack_stage0.append(act_0[0, :, m_c].mean(dim=1).cpu())

        # Stem metrics
        eng_stem_crop = energy_stem_full[:H, :W]
        en_s = float(np.mean(eng_stem_crop[neck_mask == 1])) if n_neck_px > 0 else 0.0
        ec_s = float(np.mean(eng_stem_crop[crack_mask == 1])) if n_crack_px > 0 else 0.0
        eb_s = float(np.mean(eng_stem_crop[bg_mask == 1])) if n_bg_px > 0 else 0.0
        sep_s = float(1.0 - (en_s / (ec_s + 1e-6)))
        denom_s = ec_s - eb_s
        rn_s = float((en_s - eb_s) / denom_s) if abs(denom_s) > 1e-5 else 0.0
        cos_s = 0.0
        if len(f_neck_stem) > 0 and len(f_crack_stem) > 0:
            vn = torch.stack(f_neck_stem).mean(dim=0)
            vc = torch.stack(f_crack_stem).mean(dim=0)
            cos_s = float(F.cosine_similarity(vn.unsqueeze(0), vc.unsqueeze(0)).item())

        rows_stem.append({
            "image_id": stem_name,
            "cohort": cohort,
            "model": model_tag,
            "level": "Stem",
            "n_neck_px": n_neck_px,
            "n_crack_px": n_crack_px,
            "e_neck": en_s,
            "e_crack": ec_s,
            "e_bg": eb_s,
            "sep_margin": sep_s,
            "r_norm": rn_s,
            "cos_neck_crack": cos_s,
        })

        # Stage 0 metrics
        eng_stage0_crop = energy_stage0_full[:H, :W]
        en_0 = float(np.mean(eng_stage0_crop[neck_mask == 1])) if n_neck_px > 0 else 0.0
        ec_0 = float(np.mean(eng_stage0_crop[crack_mask == 1])) if n_crack_px > 0 else 0.0
        eb_0 = float(np.mean(eng_stage0_crop[bg_mask == 1])) if n_bg_px > 0 else 0.0
        sep_0 = float(1.0 - (en_0 / (ec_0 + 1e-6)))
        denom_0 = ec_0 - eb_0
        rn_0 = float((en_0 - eb_0) / denom_0) if abs(denom_0) > 1e-5 else 0.0
        cos_0 = 0.0
        if len(f_neck_stage0) > 0 and len(f_crack_stage0) > 0:
            vn = torch.stack(f_neck_stage0).mean(dim=0)
            vc = torch.stack(f_crack_stage0).mean(dim=0)
            cos_0 = float(F.cosine_similarity(vn.unsqueeze(0), vc.unsqueeze(0)).item())

        rows_stage0.append({
            "image_id": stem_name,
            "cohort": cohort,
            "model": model_tag,
            "level": "Stage0",
            "n_neck_px": n_neck_px,
            "n_crack_px": n_crack_px,
            "e_neck": en_0,
            "e_crack": ec_0,
            "e_bg": eb_0,
            "sep_margin": sep_0,
            "r_norm": rn_0,
            "cos_neck_crack": cos_0,
        })

    return pd.DataFrame(rows_stem), pd.DataFrame(rows_stage0)


def train_u0_c3(
    model: nn.Module,
    dc_stem: DCInitStem,
    stage0_ln_params: List[nn.Parameter],
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 8,
    out_dir: str = "results/diagnostics/phase6_u0_c3_dcinit",
) -> List[Dict[str, Any]]:
    print("\n" + "=" * 80)
    print("TRAINING U0-C3: DC-Init Stem + Stage-0 LayerNorm Affine Adaptation")
    print(f"Epochs: {epochs}, Physical Batch: 14, Precision: FP32 ONLY (Strict)")
    print("Parameter Groups:")
    print("  1. dc_stem.dc_proj  (192 params): lr = 1e-4, wd = 1e-2")
    print("  2. dc_stem.ac_delta (2304 params): lr = 1e-3, wd = 1e-4")
    print("  3. dc_stem.norm     (96 params):   lr = 1e-4, wd = 1e-2")
    print("  4. stage0_ln        (192 params):  lr = 5e-5, wd = 1e-2")
    print(f"Total Trainable Parameters: {sum(p.numel() for p in dc_stem.parameters()) + sum(p.numel() for p in stage0_ln_params)} (Expected: 2,784)")
    print("=" * 80)

    param_groups = [
        {"params": list(dc_stem.dc_proj.parameters()), "lr": 1e-4, "weight_decay": 1e-2, "name": "dc_proj"},
        {"params": list(dc_stem.ac_delta.parameters()), "lr": 1e-3, "weight_decay": 1e-4, "name": "ac_delta"},
        {"params": list(dc_stem.norm.parameters()), "lr": 1e-4, "weight_decay": 1e-2, "name": "stem_norm"},
        {"params": stage0_ln_params, "lr": 5e-5, "weight_decay": 1e-2, "name": "stage0_ln"},
    ]

    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    logs = []

    stage0 = model.backbone.convnext.stages[0]
    stage0_convnext = stage0.main_block.module if hasattr(stage0, "main_block") else stage0
    ln0 = stage0_convnext.blocks[0].norm
    ln1 = stage0_convnext.blocks[1].norm

    for ep in range(1, epochs + 1):
        model.eval()  # Keep all batchnorm/dropout/SAGE in eval
        dc_stem.train()
        ln0.train()
        ln1.train()

        ep_loss_total = 0.0
        ep_loss_bce = 0.0
        ep_loss_dice = 0.0

        pbar = tqdm(train_loader, desc=f"U0-C3 Ep {ep}/{epochs}", ncols=90)
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
            ac_norm = float(torch.norm(dc_stem.ac_delta.weight).item())
            ac_max = float(torch.max(torch.abs(dc_stem.ac_delta.weight)).item())
            dc_norm = float(torch.norm(dc_stem.dc_proj.weight).item())
            stem_norm_w = float(torch.norm(dc_stem.norm.weight).item())
            ln0_w_norm = float(torch.norm(ln0.weight).item())
            ln0_b_norm = float(torch.norm(ln0.bias).item())
            ln1_w_norm = float(torch.norm(ln1.weight).item())
            ln1_b_norm = float(torch.norm(ln1.bias).item())

        n_batches = len(train_loader)
        ep_log = {
            "epoch": ep,
            "loss_total": ep_loss_total / n_batches,
            "loss_bce": ep_loss_bce / n_batches,
            "loss_dice": ep_loss_dice / n_batches,
            "lr_dc": optimizer.param_groups[0]["lr"],
            "lr_ac": optimizer.param_groups[1]["lr"],
            "lr_stem_norm": optimizer.param_groups[2]["lr"],
            "lr_stage0_ln": optimizer.param_groups[3]["lr"],
            "ac_weight_norm": ac_norm,
            "ac_weight_max": ac_max,
            "dc_weight_norm": dc_norm,
            "stem_norm_weight": stem_norm_w,
            "stage0_b0_norm_w": ln0_w_norm,
            "stage0_b0_norm_b": ln0_b_norm,
            "stage0_b1_norm_w": ln1_w_norm,
            "stage0_b1_norm_b": ln1_b_norm,
        }
        logs.append(ep_log)
        print(f"Epoch {ep:02d} Summary: Loss={ep_log['loss_total']:.4f} (BCE={ep_log['loss_bce']:.4f}, Dice={ep_log['loss_dice']:.4f}) | AC Norm={ac_norm:.4f}, Max={ac_max:.4f} | Stage0 LN Norms=({ln0_w_norm:.4f}, {ln1_w_norm:.4f})")

    return logs


def main():
    parser = argparse.ArgumentParser(description="Phase 6 U0-C3: DC-Init Stem Experiment")
    parser.add_argument("--config", type=str, default="configs/p3_ablation/b2_candidate_b_no_asdw_adaptive_fusion.yaml")
    parser.add_argument("--checkpoint", type=str, default="results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth")
    parser.add_argument("--data_root", type=str, default="datasets/Crack500_ready")
    parser.add_argument("--out_dir", type=str, default="results/diagnostics/phase6_u0_c3_dcinit")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_train", action="store_true", help="Skip training if weights already exist")
    args = parser.parse_args()

    # Automatically resolve config path if running inside SAGE_LITE or outside
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

    # Automatically resolve checkpoint path across possible directory structures
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
        found = False
        for cand in possible_ckpt_paths:
            if os.path.exists(cand):
                print(f"Discovered checkpoint at: {cand}")
                args.checkpoint = cand
                found = True
                break
        if not found:
            # Try to search recursively in /content
            print(f"Checkpoint not found at '{args.checkpoint}'. Searching /content for *.pth...")
            matched_ckpts = glob.glob("/content/**/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth", recursive=True)
            if matched_ckpts:
                args.checkpoint = matched_ckpts[0]
                print(f"Found candidate checkpoint: {args.checkpoint}")
            else:
                matched_all_pth = glob.glob("/content/**/*.pth", recursive=True)
                print(f"Available .pth files found in /content: {matched_all_pth}")

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

    orig_stem = model.backbone.convnext.stem
    dc_stem = DCInitStem().to(device)
    init_dc_stem_from_pretrained(dc_stem, orig_stem)
    model.backbone.convnext.stem = dc_stem

    # Freeze entire model
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze DCInitStem parameters
    for p in dc_stem.parameters():
        p.requires_grad = True

    # Unfreeze Stage 0 LayerNorm affine parameters
    stage0 = model.backbone.convnext.stages[0]
    stage0_convnext = stage0.main_block.module if hasattr(stage0, "main_block") else stage0
    stage0_ln_params = []
    for block in stage0_convnext.blocks:
        if hasattr(block, "norm"):
            if block.norm.weight is not None:
                block.norm.weight.requires_grad = True
                stage0_ln_params.append(block.norm.weight)
            if block.norm.bias is not None:
                block.norm.bias.requires_grad = True
                stage0_ln_params.append(block.norm.bias)

    trainable_whitelist = [p for p in model.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable_whitelist)
    print(f"Trainable parameters count: {total_trainable}")
    assert total_trainable == 2784, f"Expected 2,784 trainable parameters, got {total_trainable}"
    print(">> PASS: Exactly 2,784 trainable parameters whitelisted.")

    # Preflight Dummy Forward & Backward Gradient Invariant Check
    dummy_x = torch.randn(2, 3, 448, 448, device=device)
    dummy_out = model(dummy_x)
    assert dummy_out.shape == (2, 1, 448, 448), f"Unexpected shape {dummy_out.shape}"
    dummy_loss = dummy_out.sum()
    dummy_loss.backward()

    assert dc_stem.dc_proj.weight.grad is not None
    assert dc_stem.ac_delta.weight.grad is not None
    assert dc_stem.norm.weight.grad is not None
    assert stage0_convnext.blocks[0].norm.weight.grad is not None
    assert stage0_convnext.blocks[0].conv_dw.weight.grad is None
    stage1 = model.backbone.convnext.stages[1]
    stage1_convnext = stage1.main_block.module if hasattr(stage1, "main_block") else stage1
    assert stage1_convnext.blocks[0].norm.weight.grad is None
    assert model.decoder.segmentation_head[0].weight.grad is None
    print(">> PASS: Preflight backward gradient isolation confirmed.")

    model.zero_grad()

    # -----------------------------------------------------------------------
    # Datasets & Cohorts
    # -----------------------------------------------------------------------
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

    # Cohort stems for dual representation audit
    master_path = "results/diagnostics/phase6_d2_topology/topology_7models_master_paired.csv"
    if not os.path.exists(master_path):
        alt_master = os.path.join("..", master_path)
        if os.path.exists(alt_master):
            master_path = alt_master

    if os.path.exists(master_path):
        df_master = pd.read_csv(master_path)
        models_list = ["Base", "A1", "A2", "B1", "C1", "D1", "D2"]
        bcols = [f"{m}_bridge_events" for m in models_list]
        c7_stems = set(df_master[(df_master[bcols] > 0).all(axis=1)]["case_name"].tolist())
        clean_stems = set(df_master[(df_master[bcols] == 0).all(axis=1) & (df_master["gt_cc"] >= 2)]["case_name"].tolist())
    else:
        c7_stems = set()
        clean_stems = set()

    ac_atten_csv = "results/diagnostics/phase6_stem_ac_attenuation/ac_attenuation_per_sample_alpha.csv"
    if not os.path.exists(ac_atten_csv):
        alt_ac = os.path.join("..", ac_atten_csv)
        if os.path.exists(alt_ac):
            ac_atten_csv = alt_ac

    if os.path.exists(ac_atten_csv):
        df_prev = pd.read_csv(ac_atten_csv)
        c7a0 = df_prev[(df_prev["cohort"] == "Consensus_7of7") & (df_prev["alpha"] == 0.0)]
        resistant_stems = set(c7a0[c7a0["has_bridge"] == True]["image_id"].tolist())
        sensitive_stems = set(c7a0[c7a0["has_bridge"] == False]["image_id"].tolist())
    else:
        resistant_stems = set()
        sensitive_stems = set()

    cohort_stems = []
    for s in sorted(list(resistant_stems)):
        cohort_stems.append((s, "Resistant_82"))
    for s in sorted(list(sensitive_stems)):
        cohort_stems.append((s, "Sensitive_19"))
    for s in sorted(list(clean_stems)):
        cohort_stems.append((s, "Clean_Control"))

    # Output artifact paths
    weights_path = os.path.join(args.out_dir, "u0_c3_weights.pth")
    training_log_path = os.path.join(args.out_dir, "c3_training_log.csv")
    val_c3_csv = os.path.join(args.out_dir, "validation_c3_per_sample.csv")
    wider_43_csv = os.path.join(args.out_dir, "c3_43wider_events.csv")
    all_118_csv = os.path.join(args.out_dir, "c3_118all_events.csv")
    summary_json_path = os.path.join(args.out_dir, "u0_c3_vs_candidate_b_summary.json")

    # -----------------------------------------------------------------------
    # Step 1: Training or Loading Weights
    # -----------------------------------------------------------------------
    if os.path.exists(weights_path) and args.skip_train:
        print(f"\nLoading existing trained weights from {weights_path}...")
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        dc_stem.load_state_dict(ckpt["dc_stem_state"])
        for idx, block in enumerate(stage0_convnext.blocks):
            if hasattr(block, "norm") and f"stage0_block_{idx}_norm" in ckpt:
                block.norm.load_state_dict(ckpt[f"stage0_block_{idx}_norm"])
        print(">> Trained weights loaded successfully.")
    else:
        print("\nStarting U0-C3 Training...")
        train_logs = train_u0_c3(
            model=model,
            dc_stem=dc_stem,
            stage0_ln_params=stage0_ln_params,
            train_loader=train_loader,
            device=device,
            epochs=args.epochs,
            out_dir=args.out_dir,
        )
        pd.DataFrame(train_logs).to_csv(training_log_path, index=False)
        print(f"Training logs saved to {training_log_path}")

        # Save weights
        save_dict = {
            "dc_stem_state": dc_stem.state_dict(),
            "trainable_param_count": total_trainable,
            "epochs": args.epochs,
        }
        for idx, block in enumerate(stage0_convnext.blocks):
            if hasattr(block, "norm"):
                save_dict[f"stage0_block_{idx}_norm"] = block.norm.state_dict()
        torch.save(save_dict, weights_path)
        print(f"Saved U0-C3 weights to {weights_path}")

    # -----------------------------------------------------------------------
    # Step 2: Canonical Setting A Validation (N=348)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SETTING A CANONICAL EVALUATION (Val N=348, tile 448x448, tau=0.5)")
    print("=" * 80)

    model.eval()
    val_records = []
    logits_cache = {}
    runtimes = []

    for img_path, mask_path in tqdm(val_files, desc="Setting A Validation (U0-C3)", ncols=90):
        case_name = os.path.splitext(os.path.basename(img_path))[0]
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        target_bin = (mask > 127).astype(np.uint8)

        t0 = time.time()
        logits = predict_full_image_tiling_setting_a(model, img_rgb, device, tile_size=448, batch_size=8)
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
            "spurious_islands": topo["spurious_islands"],
            "gt_cc": topo["gt_cc"],
            "pred_cc": topo["pred_cc"],
            "runtime_ms": dt_ms,
        })

    df_c3 = pd.DataFrame(val_records)
    df_c3.to_csv(val_c3_csv, index=False)
    print(f"Validation per-sample results saved to {val_c3_csv}")

    # Summary metrics
    c3_summary = {
        "N": len(df_c3),
        "dice": float(df_c3["dice"].mean()),
        "iou": float(df_c3["iou"].mean()),
        "recall": float(df_c3["recall"].mean()),
        "precision": float(df_c3["precision"].mean()),
        "cldice": float(df_c3["cldice"].mean()),
        "boundary_iou": float(df_c3["boundary_iou"].mean()),
        "hd95": float(df_c3["hd95"].dropna().mean()),
        "bf1": float(df_c3["bf1"].dropna().mean()),
        "thin_dice": float(df_c3["thin_dice"].dropna().mean()),
        "total_bridge_events": int(df_c3["bridge_events"].sum()),
        "bridge_images": int(df_c3["bridge_status"].sum()),
        "total_break_events": int(df_c3["fragmented_gt_components"].sum()),
        "break_images": int(df_c3["break_status"].sum()),
        "spurious_islands": int(df_c3["spurious_islands"].sum()),
        "mean_runtime_ms": float(np.mean(runtimes)),
    }

    # -----------------------------------------------------------------------
    # Step 3: Cohort Analysis on 43 Wider-Gap & 118 All Events
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("COHORT EVALUATION: 43 Wider-Gap Events & 118 All Events")
    print("=" * 80)

    events_csv = "results/diagnostics/phase6_bottleneck_path/bottleneck_path_118events.csv"
    if not os.path.exists(events_csv):
        alt_ev = os.path.join("..", events_csv)
        if os.path.exists(alt_ev):
            events_csv = alt_ev

    cgsr_43_csv = "results/Phase6D_CGSR/evaluation_phase6d/cgsr_evaluation_43wider_events.csv"
    if not os.path.exists(cgsr_43_csv):
        alt_cgsr = os.path.join("..", cgsr_43_csv)
        if os.path.exists(alt_cgsr):
            cgsr_43_csv = alt_cgsr

    cgsr_ref = pd.read_csv(cgsr_43_csv).set_index("event_id") if os.path.exists(cgsr_43_csv) else None

    wider_43_records = []
    all_118_records = []

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

            rec = {
                "event_id": ev_id,
                "case_name": case_name,
                "d_gap": d_gap,
                "is_wider_gap": is_wider,
                "z_ctrl_bridge": z_ctrl,
                "z_c3_bridge": z_bridge,
                "delta_z_bridge": delta_z,
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
    # Step 4: Dual Representation Audit (Stem SepMargin)
    # -----------------------------------------------------------------------
    repr_stem_df, repr_stage0_df = evaluate_dual_representation(
        model=model,
        cohort_stems=cohort_stems,
        val_img_dir=val_img_dir,
        val_mask_dir=val_mask_dir,
        device=device,
        model_tag="U0-C3_DCInit",
    )
    repr_stem_csv = os.path.join(args.out_dir, "representation_stem_comparison.csv")
    repr_stage0_csv = os.path.join(args.out_dir, "representation_stage0_comparison.csv")
    repr_stem_df.to_csv(repr_stem_csv, index=False)
    repr_stage0_df.to_csv(repr_stage0_csv, index=False)

    # -----------------------------------------------------------------------
    # Step 5: Master Comparison & Summary JSON
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

    full_report = {
        "Candidate_B_Baseline": base_summary,
        "Candidate_B_U0_C3_DCInit": c3_summary,
        "Delta_C3_minus_Base": {
            "delta_dice": c3_summary["dice"] - base_summary["dice"],
            "delta_cldice": c3_summary["cldice"] - base_summary["cldice"],
            "delta_boundary_iou": c3_summary["boundary_iou"] - base_summary["boundary_iou"],
            "delta_hd95": c3_summary["hd95"] - base_summary["hd95"],
            "delta_bridge_events": c3_summary["total_bridge_events"] - base_summary["total_bridge_events"],
            "delta_break_events": c3_summary["total_break_events"] - base_summary["total_break_events"],
        },
        "Cohort_43_Wider_Gap": {
            "N": len(df_wider),
            "cured_count": cured_43,
            "cure_rate_pct": cure_rate_43,
            "cgsr_baseline_cured_count": 5,
            "cgsr_baseline_cure_rate_pct": 11.63,
            "median_delta_z_bridge": float(df_wider["delta_z_bridge"].dropna().median()) if len(df_wider) > 0 else 0.0,
            "mean_delta_z_bridge": float(df_wider["delta_z_bridge"].dropna().mean()) if len(df_wider) > 0 else 0.0,
        },
        "Cohort_All_118_Events": {
            "N": len(df_all_ev),
            "cured_count": cured_118,
            "cure_rate_pct": cure_rate_118,
            "cgsr_baseline_cured_count": 6,
            "cgsr_baseline_cure_rate_pct": 5.08,
            "median_delta_z_bridge": float(df_all_ev["delta_z_bridge"].dropna().median()) if len(df_all_ev) > 0 else 0.0,
            "mean_delta_z_bridge": float(df_all_ev["delta_z_bridge"].dropna().mean()) if len(df_all_ev) > 0 else 0.0,
        },
    }

    with open(summary_json_path, "w") as f:
        json.dump(full_report, f, indent=2)

    print(f"\nFinal Summary Report saved to {summary_json_path}")
    print("\n" + "=" * 80)
    print("EXPERIMENT U0-C3 RESULTS SUMMARY:")
    print(f"  Val Dice:          {c3_summary['dice']:.4f} (Base: {base_summary['dice']:.4f}, Delta: {full_report['Delta_C3_minus_Base']['delta_dice']:+.4f})")
    print(f"  Val clDice:        {c3_summary['cldice']:.4f} (Base: {base_summary['cldice']:.4f}, Delta: {full_report['Delta_C3_minus_Base']['delta_cldice']:+.4f})")
    print(f"  Val HD95:          {c3_summary['hd95']:.2f} px (Base: {base_summary['hd95']:.2f} px, Delta: {full_report['Delta_C3_minus_Base']['delta_hd95']:+.2f} px)")
    print(f"  Val Bridge Events: {c3_summary['total_bridge_events']} (Base: {base_summary['total_bridge_events']}, Delta: {full_report['Delta_C3_minus_Base']['delta_bridge_events']:+d})")
    print(f"  Val Break Events:  {c3_summary['total_break_events']} (Base: {base_summary['total_break_events']}, Delta: {full_report['Delta_C3_minus_Base']['delta_break_events']:+d})")
    print(f"  43 Wider Cured:    {cured_43}/{len(df_wider)} ({cure_rate_43:.1f}%) [CGSR Ref: 5/43 = 11.6%]")
    print(f"  118 All Cured:     {cured_118}/{len(df_all_ev)} ({cure_rate_118:.1f}%) [CGSR Ref: 6/118 = 5.1%]")
    print("=" * 80)


if __name__ == "__main__":
    main()
