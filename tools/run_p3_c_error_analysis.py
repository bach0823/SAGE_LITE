#!/usr/bin/env python3
"""
tools/run_p3_c_error_analysis.py

Comprehensive Error Analysis and Error x Routing Diagnostics for Canonical P3-C Checkpoint.
Evaluates Crack500 validation split (348 samples) in canonical Setting A.
Aligns per-sample pixel metrics, crack morphology proxies, and routing behavior.

Outputs:
- per_sample_metrics.csv
- worst_cases.csv
- best_cases.csv
- error_summary.json
- routing_vs_error.csv
- morphology_vs_error.csv
- figures/ (dice_distribution, error_vs_routing_entropy, error_vs_cnn_vit_fraction, error_vs_expert_concentration)
- qualitative/ (gallery_worst_cases, gallery_median_cases, gallery_best_cases)
"""

import argparse
import glob
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple
import yaml

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import torch
from tqdm import tqdm

# Ensure SAGE_LITE project root is on sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sage.networks import create_b2_unet
from scripts.evaluate_crack_official import predict_full_image_tiling_setting_a, get_image_mask_pairs

# Use clean styling for publication figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12


def compute_sample_morphology(target: np.ndarray) -> Dict[str, Any]:
    """Computes quantitative morphology proxies for a single binary crack mask."""
    gt_area = int(np.sum(target > 0))
    if gt_area == 0:
        return {
            'gt_area': 0,
            'num_gt_cc': 0,
            'mean_component_size': 0.0,
            'perimeter': 0,
            'boundary_to_area_ratio': 0.0,
            'thinness_score': 0.0,
        }

    # Connected components
    num_cc, labels = cv2.connectedComponents(target.astype(np.uint8))
    num_gt_cc = max(num_cc - 1, 0)
    mean_cc_size = gt_area / max(num_gt_cc, 1)

    # Boundary and perimeter via morphological gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    boundary = cv2.morphologyEx(target.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
    perimeter = int(np.sum(boundary > 0))
    boundary_to_area = perimeter / max(gt_area, 1)
    thinness = perimeter / (2.0 * max(gt_area, 1))

    return {
        'gt_area': gt_area,
        'num_gt_cc': num_gt_cc,
        'mean_component_size': float(mean_cc_size),
        'perimeter': perimeter,
        'boundary_to_area_ratio': float(boundary_to_area),
        'thinness_score': float(thinness),
    }


def compute_routing_features(routing_record: Dict[str, Any]) -> Dict[str, Any]:
    """Computes per-sample routing entropy, CNN/ViT fraction, HHI, and gs."""
    routers = routing_record.get('routing', [])
    num_routers = len(routers)
    if num_routers == 0:
        return {
            'mean_routing_entropy': 0.0,
            'cnn_expert_fraction': 0.0,
            'vit_expert_fraction': 0.0,
            'expert_hhi_concentration': 0.0,
            'mean_gs': 0.0,
            'cnn_expert_selections': 0,
            'vit_expert_selections': 0,
        }

    entropies = []
    gs_values = []
    all_selected_experts = []

    for r in routers:
        scores = np.array(r.get('selected_scores', []), dtype=np.float64)
        if len(scores) > 0 and np.sum(scores) > 0:
            p = scores / np.sum(scores)
            p = p[p > 0]
            ent = -np.sum(p * np.log2(p + 1e-12))
            entropies.append(ent)
        else:
            entropies.append(0.0)

        gs_values.append(float(r.get('g_s', 0.5)))
        all_selected_experts.extend(r.get('selected_experts', []))

    total_selections = len(all_selected_experts)
    cnn_count = sum(1 for e in all_selected_experts if e in [0, 1, 2, 3])
    vit_count = total_selections - cnn_count
    cnn_fraction = cnn_count / max(total_selections, 1)
    vit_fraction = vit_count / max(total_selections, 1)

    # Herfindahl-Hirschman Index (HHI) across all 16 experts
    expert_counts = np.bincount(all_selected_experts, minlength=16)
    p_experts = expert_counts / max(total_selections, 1)
    hhi = float(np.sum(p_experts ** 2))

    return {
        'mean_routing_entropy': float(np.mean(entropies)),
        'cnn_expert_fraction': float(cnn_fraction),
        'vit_expert_fraction': float(vit_fraction),
        'expert_hhi_concentration': float(hhi),
        'mean_gs': float(np.mean(gs_values)),
        'cnn_expert_selections': int(cnn_count),
        'vit_expert_selections': int(vit_count),
    }


def classify_error_taxonomy(row: pd.Series) -> List[str]:
    """Reproducible, quantitative error taxonomy classification."""
    tags = []
    dice = row['dice']
    precision = row['precision']
    recall = row['recall']
    gt_area = row['gt_area']
    pred_area = row['pred_area']
    thinness = row.get('thinness_score', 0.0)
    num_gt_cc = row.get('num_gt_cc', 0)
    frag_ratio = row.get('fragmentation_ratio', 1.0)

    # 1. Missed crack (Severe False Negative)
    if recall < 0.35 and gt_area > 500:
        tags.append('missed_crack_high_fn')

    # 2. False crack (Severe False Positive)
    if precision < 0.35 and pred_area > 500:
        tags.append('false_crack_high_fp')

    # 3. Thin / Low-area failure
    if gt_area < 8000 and dice < 0.55:
        tags.append('thin_low_area_failure')

    # 4. Over-segmentation (Predicted area vastly exceeds GT area)
    if pred_area > 2.0 * max(gt_area, 1) and precision < 0.60:
        tags.append('over_segmentation')

    # 5. Fragmented prediction
    if frag_ratio > 3.0 and dice < 0.70:
        tags.append('fragmented_prediction')

    # 6. Complex morphology / multiple branches
    if num_gt_cc >= 5:
        tags.append('complex_topology')

    # 7. Boundary / marginal error (High Dice but imperfect boundary)
    if 0.65 <= dice < 0.85 and precision > 0.60 and recall > 0.60:
        tags.append('boundary_margin_error')

    # 8. High quality prediction
    if dice >= 0.85:
        tags.append('high_quality')

    if len(tags) == 0:
        tags.append('moderate_general_error')

    return tags


def create_error_map(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Creates RGB Error Map:
    - True Positive (TP): Green [0, 220, 0]
    - False Positive (FP): Red [230, 30, 30]
    - False Negative (FN): Blue [30, 100, 240]
    - True Negative (TN): Dark gray/black [20, 20, 20]
    """
    H, W = target.shape[:2]
    error_map = np.zeros((H, W, 3), dtype=np.uint8)
    error_map[:] = (25, 25, 25)

    tp_mask = (pred == 1) & (target == 1)
    fp_mask = (pred == 1) & (target == 0)
    fn_mask = (pred == 0) & (target == 1)

    error_map[tp_mask] = (0, 220, 0)     # TP Green
    error_map[fp_mask] = (235, 40, 40)   # FP Red
    error_map[fn_mask] = (40, 120, 245)  # FN Blue

    return error_map


def create_overlay(image_rgb: np.ndarray, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Overlays GT (green outline) and Prediction (magenta alpha mask) onto RGB image."""
    overlay = image_rgb.copy()
    H, W = overlay.shape[:2]
    pred_res = pred[:H, :W]
    target_res = target[:H, :W]

    # Red/Magenta overlay for prediction
    pred_color = np.array([255, 60, 60], dtype=np.uint8)
    mask_bool = pred_res > 0
    overlay[mask_bool] = (0.55 * overlay[mask_bool] + 0.45 * pred_color).astype(np.uint8)

    # Green contour for GT
    contours, _ = cv2.findContours(target_res.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    return overlay


def build_gallery_figure(
    samples_df: pd.DataFrame,
    images_dict: Dict[str, np.ndarray],
    targets_dict: Dict[str, np.ndarray],
    preds_dict: Dict[str, np.ndarray],
    title: str,
    output_path: str,
):
    """Builds a multi-panel visual gallery: Input | GT | Pred | Error Map | Overlay."""
    num_samples = len(samples_df)
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 3.8 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)

    col_titles = [
        "1. Input Image",
        "2. Ground Truth",
        "3. P3-C Prediction",
        "4. Error Map\n(TP:Green, FP:Red, FN:Blue)",
        "5. Overlay\n(Pred:Red, GT:Green Edge)",
    ]
    for c_idx, ct in enumerate(col_titles):
        axes[0, c_idx].set_title(ct, fontsize=12, fontweight='semibold', pad=10)

    for r_idx, (_, row) in enumerate(samples_df.iterrows()):
        case_name = row['case_name']
        img = images_dict[case_name]
        tgt = targets_dict[case_name]
        prd = preds_dict[case_name]

        H, W = tgt.shape[:2]
        img_cropped = img[:H, :W]
        prd_cropped = prd[:H, :W]

        err_map = create_error_map(prd_cropped, tgt)
        ovl = create_overlay(img_cropped, prd_cropped, tgt)

        axes[r_idx, 0].imshow(img_cropped)
        axes[r_idx, 1].imshow(tgt, cmap='gray')
        axes[r_idx, 2].imshow(prd_cropped, cmap='gray')
        axes[r_idx, 3].imshow(err_map)
        axes[r_idx, 4].imshow(ovl)

        # Label metadata on left column
        meta_str = (
            f"Case: {case_name}\n"
            f"Dice: {row['dice']:.4f} | IoU: {row['iou']:.4f}\n"
            f"Prec: {row['precision']:.4f} | Rec: {row['recall']:.4f}\n"
            f"GT Area: {int(row['gt_area']):,} | Pred: {int(row['pred_area']):,}"
        )
        axes[r_idx, 0].set_ylabel(meta_str, fontsize=9, fontweight='medium', rotation=0, labelpad=90, va='center')

        for c in range(5):
            axes[r_idx, c].set_xticks([])
            axes[r_idx, c].set_yticks([])

    plt.tight_layout(rect=[0.05, 0.01, 0.98, 0.98])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved Gallery: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="P3-C Comprehensive Error Analysis & Routing Cross-Analysis")
    parser.add_argument("--config", type=str, default="configs/p3_ablation/b2_p3_run_c.yaml", help="Path to P3-C config")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/P3_C_Routing_Diagnostics/checkpoints/best_model_b2_global.pth",
        help="Path to canonical P3-C checkpoint",
    )
    parser.add_argument(
        "--routing-json",
        type=str,
        default="results/P3_C_Routing_Diagnostics/full_val/routing_statistics.json",
        help="Path to full-val routing statistics JSON",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../datasets/Crack500_ready",
        help="Path to local Crack500_ready dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/P3_C_Routing_Diagnostics/error_analysis",
        help="Output directory for error analysis artifacts",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for Setting A inference")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    figures_dir = os.path.join(args.output_dir, "figures")
    qualitative_dir = os.path.join(args.output_dir, "qualitative")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(qualitative_dir, exist_ok=True)

    print("=" * 80)
    print("STARTING COMPREHENSIVE ERROR ANALYSIS FOR CANONICAL P3-C (ASDW)")
    print("=" * 80)
    print(f"Config:          {args.config}")
    print(f"Checkpoint:      {args.checkpoint}")
    print(f"Routing JSON:    {args.routing_json}")
    print(f"Data Root:       {args.data_root}")
    print(f"Output Dir:      {args.output_dir}")
    print(f"Device:          {args.device}")

    # 1. Load routing statistics JSON
    with open(args.routing_json, "r", encoding="utf-8") as f:
        routing_json_data = json.load(f)
    routing_records_list = routing_json_data.get("sample_routing_records", [])
    routing_dict = {rec["case_name"]: rec for rec in routing_records_list}
    print(f"Loaded {len(routing_dict)} per-sample routing records from {args.routing_json}.")

    # 2. Discover Validation Pairs
    val_img_dir = os.path.join(args.data_root, "val", "images")
    val_mask_dir = os.path.join(args.data_root, "val", "masks")
    img_paths = sorted(glob.glob(os.path.join(val_img_dir, "*")))
    img_paths = [p for p in img_paths if os.path.splitext(p)[1].lower() in [".jpg", ".jpeg", ".png"]]

    val_pairs = []
    for ip in img_paths:
        stem = os.path.splitext(os.path.basename(ip))[0]
        for ext in [".png", ".jpg", ".jpeg"]:
            mp = os.path.join(val_mask_dir, stem + ext)
            if os.path.exists(mp):
                val_pairs.append((ip, mp, stem))
                break

    print(f"Discovered {len(val_pairs)} validation image-mask pairs in {args.data_root}/val.")
    assert len(val_pairs) == 348, f"Expected 348 validation samples, found {len(val_pairs)}!"

    # 3. Load Model and Checkpoint
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.enabled = False  # stability on GTX 1650/1660

    # Load config to dynamically configure depth and p3_mode
    model_cfg = {}
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            model_cfg = yaml.safe_load(f) or {}

    num_layers = int(model_cfg.get("num_transformer_layers", 12))
    p3_mode = model_cfg.get("p3_mode", "C")
    img_size = int(model_cfg.get("img_size", 448))

    print(f"Initializing B2 UNet (Depth={num_layers}, p3_mode='{p3_mode}', img_size={img_size})...")
    model = create_b2_unet(num_transformer_layers=num_layers, p3_mode=p3_mode, img_size=img_size).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Successfully loaded checkpoint (recorded best_dice={ckpt.get('best_dice'):.4f}, epoch={ckpt.get('epoch')}).")

    # 4. Check if per_sample_metrics.csv already exists to enable fast-resume
    per_sample_csv = os.path.join(args.output_dir, "per_sample_metrics.csv")
    images_cache = {}
    targets_cache = {}
    preds_cache = {}

    if os.path.exists(per_sample_csv) and len(pd.read_csv(per_sample_csv)) == 348:
        print(f"\n[Fast-Path] Found complete per_sample_metrics.csv ({per_sample_csv}). Loading directly...")
        df = pd.read_csv(per_sample_csv)
    else:
        print("\nRunning Canonical Setting A Inference on all 348 samples...")
        sample_rows = []
        start_time = time.time()
        for img_path, mask_path, case_name in tqdm(val_pairs, desc="Inference [Setting A]"):
            img_bgr = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            target = (mask > 0).astype(np.uint8)

            logits_np = predict_full_image_tiling_setting_a(
                model, img_rgb, device, tile_size=448, batch_size=args.batch_size
            )
            pred = (logits_np > 0.0).astype(np.uint8)
            H, W = target.shape[:2]
            pred = pred[:H, :W]

            tp = int(np.sum((pred == 1) & (target == 1)))
            fp = int(np.sum((pred == 1) & (target == 0)))
            fn = int(np.sum((pred == 0) & (target == 1)))
            tn = int(np.sum((pred == 0) & (target == 0)))

            precision = tp / (tp + fp + 1e-5)
            recall = tp / (tp + fn + 1e-5)
            dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-5)
            iou = tp / (tp + fp + fn + 1e-5)

            pred_area = int(np.sum(pred == 1))
            gt_area = int(np.sum(target == 1))
            fp_area_ratio = fp / max(pred_area, 1)
            fn_area_ratio = fn / max(gt_area, 1)

            morph = compute_sample_morphology(target)
            num_pred_cc, _ = cv2.connectedComponents(pred.astype(np.uint8))
            num_pred_cc = max(num_pred_cc - 1, 0)
            frag_ratio = num_pred_cc / max(morph['num_gt_cc'], 1)

            routing_rec = routing_dict.get(case_name, {})
            routing_feats = compute_routing_features(routing_rec)

            row = {
                'case_name': case_name,
                'dice': float(dice),
                'iou': float(iou),
                'precision': float(precision),
                'recall': float(recall),
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn,
                'gt_area': gt_area,
                'pred_area': pred_area,
                'fp_area_ratio': float(fp_area_ratio),
                'fn_area_ratio': float(fn_area_ratio),
                'num_gt_cc': morph['num_gt_cc'],
                'num_pred_cc': num_pred_cc,
                'mean_component_size': morph['mean_component_size'],
                'perimeter': morph['perimeter'],
                'boundary_to_area_ratio': morph['boundary_to_area_ratio'],
                'thinness_score': morph['thinness_score'],
                'fragmentation_ratio': float(frag_ratio),
                'mean_routing_entropy': routing_feats['mean_routing_entropy'],
                'cnn_expert_fraction': routing_feats['cnn_expert_fraction'],
                'vit_expert_fraction': routing_feats['vit_expert_fraction'],
                'expert_hhi_concentration': routing_feats['expert_hhi_concentration'],
                'mean_gs': routing_feats['mean_gs'],
                'cnn_expert_selections': routing_feats['cnn_expert_selections'],
                'vit_expert_selections': routing_feats['vit_expert_selections'],
            }
            sample_rows.append(row)

        total_time = time.time() - start_time
        print(f"Inference completed in {total_time:.2f} s ({total_time / 348:.3f} s/sample).")
        df = pd.DataFrame(sample_rows)
        df.to_csv(per_sample_csv, index=False)
        print(f"Saved: {per_sample_csv}")

    # Classify error taxonomy per sample
    df['error_taxonomy'] = df.apply(classify_error_taxonomy, axis=1)
    df['primary_error_category'] = df['error_taxonomy'].apply(lambda tags: tags[0])

    # Assign Performance Tier (Low: <25%, Medium: 25-75%, High: >=75%)
    q25 = df['dice'].quantile(0.25)
    q75 = df['dice'].quantile(0.75)
    def assign_tier(dice_val):
        if dice_val < q25:
            return 'Low_Dice (<25%)'
        elif dice_val <= q75:
            return 'Medium_Dice (25-75%)'
        else:
            return 'High_Dice (>75%)'
    df['performance_tier'] = df['dice'].apply(assign_tier)

    # Global Summary Verification
    mean_val_dice = df['dice'].mean()
    mean_val_iou = df['iou'].mean()
    mean_val_prec = df['precision'].mean()
    mean_val_rec = df['recall'].mean()
    print(f"\n--> Global Val Dice: {mean_val_dice:.4f} (Expected: ~0.7557)")
    print(f"--> Global Val IoU:  {mean_val_iou:.4f}")
    print(f"--> Global Val Prec: {mean_val_prec:.4f}")
    print(f"--> Global Val Rec:  {mean_val_rec:.4f}")

    # 5. Save per_sample_metrics.csv
    per_sample_csv = os.path.join(args.output_dir, "per_sample_metrics.csv")
    df.to_csv(per_sample_csv, index=False)
    print(f"Saved: {per_sample_csv}")

    # 6. Save worst_cases.csv (Top 20) and best_cases.csv (Top 10)
    worst_20 = df.sort_values(by='dice', ascending=True).head(20)
    best_10 = df.sort_values(by='dice', ascending=False).head(10)

    worst_csv = os.path.join(args.output_dir, "worst_cases.csv")
    best_csv = os.path.join(args.output_dir, "best_cases.csv")
    worst_20.to_csv(worst_csv, index=False)
    best_10.to_csv(best_csv, index=False)
    print(f"Saved: {worst_csv} (20 samples, min Dice={worst_20['dice'].iloc[0]:.4f})")
    print(f"Saved: {best_csv} (10 samples, max Dice={best_10['dice'].iloc[0]:.4f})")

    # 7. Comprehensive Distribution Statistics & JSON Summary
    def get_stats(series: pd.Series) -> Dict[str, float]:
        return {
            'mean': float(series.mean()),
            'std': float(series.std()),
            'median': float(series.median()),
            'min': float(series.min()),
            'max': float(series.max()),
            'p10': float(series.quantile(0.10)),
            'p25': float(series.quantile(0.25)),
            'p75': float(series.quantile(0.75)),
            'p90': float(series.quantile(0.90)),
            'p95': float(series.quantile(0.95)),
            'iqr': float(series.quantile(0.75) - series.quantile(0.25)),
        }

    # Taxonomy frequency counts
    import ast
    all_taxonomy_tags = []
    for item in df['error_taxonomy']:
        if isinstance(item, list):
            all_taxonomy_tags.extend(item)
        elif isinstance(item, str):
            try:
                parsed = ast.literal_eval(item)
                if isinstance(parsed, list):
                    all_taxonomy_tags.extend(parsed)
                else:
                    all_taxonomy_tags.append(str(parsed))
            except Exception:
                all_taxonomy_tags.append(item)
    taxonomy_counts = {str(k): int(v) for k, v in pd.Series(all_taxonomy_tags).value_counts().items()}

    # Tier statistics breakdown
    tier_stats = {}
    for tier_name, grp in df.groupby('performance_tier'):
        tier_stats[tier_name] = {
            'count': int(len(grp)),
            'dice_mean': float(grp['dice'].mean()),
            'dice_median': float(grp['dice'].median()),
            'precision_mean': float(grp['precision'].mean()),
            'recall_mean': float(grp['recall'].mean()),
            'gt_area_mean': float(grp['gt_area'].mean()),
            'gt_area_median': float(grp['gt_area'].median()),
            'thinness_mean': float(grp['thinness_score'].mean()),
            'boundary_ratio_mean': float(grp['boundary_to_area_ratio'].mean()),
            'routing_entropy_mean': float(grp['mean_routing_entropy'].mean()),
            'cnn_fraction_mean': float(grp['cnn_expert_fraction'].mean()),
            'vit_fraction_mean': float(grp['vit_expert_fraction'].mean()),
            'hhi_mean': float(grp['expert_hhi_concentration'].mean()),
            'gs_mean': float(grp['mean_gs'].mean()),
        }

    # Pearson and Spearman correlations
    corr_vars = [
        'gt_area', 'num_gt_cc', 'thinness_score', 'boundary_to_area_ratio',
        'mean_routing_entropy', 'cnn_expert_fraction', 'vit_expert_fraction',
        'expert_hhi_concentration', 'mean_gs'
    ]
    correlations = {}
    for var in corr_vars:
        p_r, p_p = stats.pearsonr(df[var], df['dice'])
        s_r, s_p = stats.spearmanr(df[var], df['dice'])
        correlations[var] = {
            'pearson_r': float(p_r),
            'pearson_pvalue': float(p_p),
            'spearman_rho': float(s_r),
            'spearman_pvalue': float(s_p),
        }

    summary_json_data = {
        'metadata': {
            'model': 'B2',
            'p3_mode': 'C',
            'architecture': 'ConvNeXt-Femto + 12 ViT + ASDW (1x7, 7x1, 3x3) + PE28',
            'checkpoint': os.path.abspath(args.checkpoint),
            'total_samples': len(df),
            'evaluation_protocol': 'setting_a',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'overall_metrics': {
            'dice': get_stats(df['dice']),
            'iou': get_stats(df['iou']),
            'precision': get_stats(df['precision']),
            'recall': get_stats(df['recall']),
            'gt_area': get_stats(df['gt_area']),
            'pred_area': get_stats(df['pred_area']),
            'thinness_score': get_stats(df['thinness_score']),
            'boundary_to_area_ratio': get_stats(df['boundary_to_area_ratio']),
            'routing_entropy': get_stats(df['mean_routing_entropy']),
            'cnn_expert_fraction': get_stats(df['cnn_expert_fraction']),
            'vit_expert_fraction': get_stats(df['vit_expert_fraction']),
            'expert_hhi_concentration': get_stats(df['expert_hhi_concentration']),
            'mean_gs': get_stats(df['mean_gs']),
        },
        'error_taxonomy_distribution': taxonomy_counts,
        'performance_tiers': tier_stats,
        'correlations_with_dice': correlations,
    }

    def json_serializer(o):
        if isinstance(o, (np.integer, np.int64)):
            return int(o)
        if isinstance(o, (np.floating, np.float64)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    summary_json_path = os.path.join(args.output_dir, "error_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json_data, f, indent=2, default=json_serializer)
    print(f"Saved: {summary_json_path}")

    # 8. Save routing_vs_error.csv and morphology_vs_error.csv
    routing_comparison_rows = []
    for tier_name, grp in df.groupby('performance_tier'):
        routing_comparison_rows.append({
            'Performance_Tier': tier_name,
            'Sample_Count': len(grp),
            'Mean_Dice': round(float(grp['dice'].mean()), 4),
            'Median_Dice': round(float(grp['dice'].median()), 4),
            'Mean_Routing_Entropy': round(float(grp['mean_routing_entropy'].mean()), 4),
            'Std_Routing_Entropy': round(float(grp['mean_routing_entropy'].std()), 4),
            'Mean_CNN_Fraction': round(float(grp['cnn_expert_fraction'].mean()), 4),
            'Mean_ViT_Fraction': round(float(grp['vit_expert_fraction'].mean()), 4),
            'Mean_HHI_Concentration': round(float(grp['expert_hhi_concentration'].mean()), 4),
            'Mean_gs_Gate': round(float(grp['mean_gs'].mean()), 4),
        })
    pd.DataFrame(routing_comparison_rows).to_csv(os.path.join(args.output_dir, "routing_vs_error.csv"), index=False)
    print(f"Saved: {os.path.join(args.output_dir, 'routing_vs_error.csv')}")

    morph_comparison_rows = []
    for tier_name, grp in df.groupby('performance_tier'):
        morph_comparison_rows.append({
            'Performance_Tier': tier_name,
            'Sample_Count': len(grp),
            'Mean_Dice': round(float(grp['dice'].mean()), 4),
            'Median_GT_Area': round(float(grp['gt_area'].median()), 1),
            'Mean_GT_Area': round(float(grp['gt_area'].mean()), 1),
            'Mean_Num_GT_Components': round(float(grp['num_gt_cc'].mean()), 2),
            'Mean_Component_Size': round(float(grp['mean_component_size'].mean()), 1),
            'Mean_Thinness_Score': round(float(grp['thinness_score'].mean()), 4),
            'Mean_Boundary_To_Area': round(float(grp['boundary_to_area_ratio'].mean()), 4),
            'Mean_Fragmentation_Ratio': round(float(grp['fragmentation_ratio'].mean()), 2),
        })
    pd.DataFrame(morph_comparison_rows).to_csv(os.path.join(args.output_dir, "morphology_vs_error.csv"), index=False)
    print(f"Saved: {os.path.join(args.output_dir, 'morphology_vs_error.csv')}")

    # 9. Generate Figures
    print("\nGenerating Statistical Visualizations in figures/ ...")

    # Figure 1: Metric Distributions (Dice, IoU, Precision, Recall)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics_info = [
        ('dice', 'Val Dice Distribution', 'royalblue', axes[0, 0]),
        ('iou', 'Val IoU Distribution', 'darkgreen', axes[0, 1]),
        ('precision', 'Val Precision Distribution', 'darkorange', axes[1, 0]),
        ('recall', 'Val Recall Distribution', 'purple', axes[1, 1]),
    ]
    for col_name, title, color, ax in metrics_info:
        sns.histplot(df[col_name], kde=True, color=color, ax=ax, bins=30, stat="density", alpha=0.6)
        mean_v = df[col_name].mean()
        med_v = df[col_name].median()
        ax.axvline(mean_v, color='red', linestyle='--', linewidth=1.5, label=f"Mean: {mean_v:.4f}")
        ax.axvline(med_v, color='black', linestyle=':', linewidth=1.5, label=f"Median: {med_v:.4f}")
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(col_name.capitalize())
        ax.legend(loc='upper left')

    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, "dice_distribution.png")
    plt.savefig(fig1_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig1_path}")

    # Figure 2: Error vs Routing Entropy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.scatterplot(
        data=df, x='mean_routing_entropy', y='dice', hue='performance_tier',
        palette={'Low_Dice (<25%)': 'crimson', 'Medium_Dice (25-75%)': 'goldenrod', 'High_Dice (>75%)': 'forestgreen'},
        alpha=0.75, s=45, ax=ax1
    )
    sns.regplot(data=df, x='mean_routing_entropy', y='dice', scatter=False, ax=ax1, color='black', line_kws={'linestyle': '--', 'linewidth': 1.5})
    r_ent = correlations['mean_routing_entropy']['pearson_r']
    p_ent = correlations['mean_routing_entropy']['pearson_pvalue']
    ax1.set_title(f"Val Dice vs Mean Routing Entropy (r = {r_ent:+.3f}, p = {p_ent:.3f})", fontweight='bold')
    ax1.set_xlabel("Mean Shannon Routing Entropy (bits)")
    ax1.set_ylabel("Validation Dice")

    sns.boxplot(
        data=df, x='performance_tier', y='mean_routing_entropy',
        palette={'Low_Dice (<25%)': 'crimson', 'Medium_Dice (25-75%)': 'goldenrod', 'High_Dice (>75%)': 'forestgreen'},
        ax=ax2, width=0.45
    )
    ax2.set_title("Routing Entropy across Performance Tiers", fontweight='bold')
    ax2.set_xlabel("Performance Tier")
    ax2.set_ylabel("Mean Routing Entropy")

    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, "error_vs_routing_entropy.png")
    plt.savefig(fig2_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig2_path}")

    # Figure 3: Error vs CNN / ViT Expert Fraction
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.scatterplot(
        data=df, x='cnn_expert_fraction', y='dice', hue='performance_tier',
        palette={'Low_Dice (<25%)': 'crimson', 'Medium_Dice (25-75%)': 'goldenrod', 'High_Dice (>75%)': 'forestgreen'},
        alpha=0.75, s=45, ax=ax1
    )
    r_cnn = correlations['cnn_expert_fraction']['pearson_r']
    p_cnn = correlations['cnn_expert_fraction']['pearson_pvalue']
    ax1.set_title(f"Val Dice vs CNN Expert Fraction (r = {r_cnn:+.3f}, p = {p_cnn:.3f})", fontweight='bold')
    ax1.set_xlabel("CNN Expert Fraction (Shared Stages 0..3)")
    ax1.set_ylabel("Validation Dice")

    sns.boxplot(
        data=df, x='performance_tier', y='cnn_expert_fraction',
        palette={'Low_Dice (<25%)': 'crimson', 'Medium_Dice (25-75%)': 'goldenrod', 'High_Dice (>75%)': 'forestgreen'},
        ax=ax2, width=0.45
    )
    ax2.set_title("CNN Expert Fraction across Performance Tiers", fontweight='bold')
    ax2.set_xlabel("Performance Tier")
    ax2.set_ylabel("CNN Expert Fraction")

    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, "error_vs_cnn_vit_fraction.png")
    plt.savefig(fig3_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig3_path}")

    # Figure 4: Error vs Expert Concentration (HHI)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.scatterplot(
        data=df, x='expert_hhi_concentration', y='dice', hue='performance_tier',
        palette={'Low_Dice (<25%)': 'crimson', 'Medium_Dice (25-75%)': 'goldenrod', 'High_Dice (>75%)': 'forestgreen'},
        alpha=0.75, s=45, ax=ax1
    )
    r_hhi = correlations['expert_hhi_concentration']['pearson_r']
    p_hhi = correlations['expert_hhi_concentration']['pearson_pvalue']
    ax1.set_title(f"Val Dice vs Expert Concentration (HHI) (r = {r_hhi:+.3f}, p = {p_hhi:.3f})", fontweight='bold')
    ax1.set_xlabel("Expert Concentration Index (HHI)")
    ax1.set_ylabel("Validation Dice")

    sns.boxplot(
        data=df, x='performance_tier', y='expert_hhi_concentration',
        palette={'Low_Dice (<25%)': 'crimson', 'Medium_Dice (25-75%)': 'goldenrod', 'High_Dice (>75%)': 'forestgreen'},
        ax=ax2, width=0.45
    )
    ax2.set_title("Expert Concentration (HHI) across Performance Tiers", fontweight='bold')
    ax2.set_xlabel("Performance Tier")
    ax2.set_ylabel("Expert HHI")

    plt.tight_layout()
    fig4_path = os.path.join(figures_dir, "error_vs_expert_concentration.png")
    plt.savefig(fig4_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig4_path}")

    # Figure 5: Error vs Crack Morphology (Area & Thinness)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.scatterplot(
        data=df, x='gt_area', y='dice', hue='performance_tier',
        palette={'Low_Dice (<25%)': 'crimson', 'Medium_Dice (25-75%)': 'goldenrod', 'High_Dice (>75%)': 'forestgreen'},
        alpha=0.75, s=45, ax=ax1
    )
    ax1.set_xscale('log')
    r_area = correlations['gt_area']['spearman_rho']
    p_area = correlations['gt_area']['spearman_pvalue']
    ax1.set_title(f"Val Dice vs Ground Truth Crack Area (Spearman rho = {r_area:+.3f})", fontweight='bold')
    ax1.set_xlabel("GT Crack Area in Pixels (Log Scale)")
    ax1.set_ylabel("Validation Dice")

    sns.scatterplot(
        data=df, x='thinness_score', y='dice', hue='performance_tier',
        palette={'Low_Dice (<25%)': 'crimson', 'Medium_Dice (25-75%)': 'goldenrod', 'High_Dice (>75%)': 'forestgreen'},
        alpha=0.75, s=45, ax=ax2
    )
    r_thin = correlations['thinness_score']['spearman_rho']
    p_thin = correlations['thinness_score']['spearman_pvalue']
    ax2.set_title(f"Val Dice vs Crack Thinness Score (Spearman rho = {r_thin:+.3f})", fontweight='bold')
    ax2.set_xlabel("Thinness Score (Perimeter / 2*Area)")
    ax2.set_ylabel("Validation Dice")

    plt.tight_layout()
    fig5_path = os.path.join(figures_dir, "morphology_vs_error.png")
    plt.savefig(fig5_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig5_path}")

    # 10. Generate Qualitative Visual Galleries
    print("\nGenerating Qualitative Visual Galleries in qualitative/ ...")
    worst_10_df = df.sort_values(by='dice', ascending=True).head(10)
    med_dice = df['dice'].median()
    df_sorted = df.sort_values(by='dice').reset_index(drop=True)
    med_idx = len(df_sorted) // 2
    median_5_df = df_sorted.iloc[med_idx - 2 : med_idx + 3]
    best_5_df = df.sort_values(by='dice', ascending=False).head(5)

    needed_cases = set(worst_10_df['case_name']) | set(median_5_df['case_name']) | set(best_5_df['case_name'])
    pairs_dict = {stem: (ip, mp) for ip, mp, stem in val_pairs}
    missing_cases = [c for c in needed_cases if c not in images_cache]
    if missing_cases:
        print(f"Loading/predicting on {len(missing_cases)} target gallery cases...")
        for cname in tqdm(missing_cases, desc="Gallery Inference"):
            ip, mp = pairs_dict[cname]
            img_bgr = cv2.imread(ip)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            target = (mask > 0).astype(np.uint8)
            logits_np = predict_full_image_tiling_setting_a(
                model, img_rgb, device, tile_size=448, batch_size=args.batch_size
            )
            pred = (logits_np > 0.0).astype(np.uint8)[:target.shape[0], :target.shape[1]]
            images_cache[cname] = img_rgb
            targets_cache[cname] = target
            preds_cache[cname] = pred

    # A. 10 Worst Cases
    build_gallery_figure(
        worst_10_df, images_cache, targets_cache, preds_cache,
        title="P3-C Canonical Error Analysis: Top 10 Lowest-Performing Validation Cases",
        output_path=os.path.join(qualitative_dir, "gallery_worst_cases.png"),
    )

    # B. 5 Representative Median Cases (around 50th percentile)
    med_dice = df['dice'].median()
    df_sorted = df.sort_values(by='dice').reset_index(drop=True)
    med_idx = len(df_sorted) // 2
    median_5_df = df_sorted.iloc[med_idx - 2 : med_idx + 3]
    build_gallery_figure(
        median_5_df, images_cache, targets_cache, preds_cache,
        title=f"P3-C Canonical Error Analysis: 5 Representative Median Validation Cases (Median Dice: {med_dice:.4f})",
        output_path=os.path.join(qualitative_dir, "gallery_median_cases.png"),
    )

    # C. 5 Highest Quality Cases
    best_5_df = df.sort_values(by='dice', ascending=False).head(5)
    build_gallery_figure(
        best_5_df, images_cache, targets_cache, preds_cache,
        title="P3-C Canonical Error Analysis: Top 5 Highest-Quality Validation Cases",
        output_path=os.path.join(qualitative_dir, "gallery_best_cases.png"),
    )

    print("\n" + "=" * 80)
    print("ALL P3-C ERROR ANALYSIS ARTIFACTS SUCCESSFULLY GENERATED")
    print("=" * 80)
    print(f"Outputs written to: {os.path.abspath(args.output_dir)}")
    print(f"  - per_sample_metrics.csv ({len(df)} samples)")
    print(f"  - worst_cases.csv (Top 20)")
    print(f"  - best_cases.csv (Top 10)")
    print(f"  - error_summary.json")
    print(f"  - routing_vs_error.csv")
    print(f"  - morphology_vs_error.csv")
    print(f"  - figures/ (5 statistical figures)")
    print(f"  - qualitative/ (3 visual gallery panels)")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
