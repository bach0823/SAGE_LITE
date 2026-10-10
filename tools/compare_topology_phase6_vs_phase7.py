#!/usr/bin/env python3
"""
tools/compare_topology_phase6_vs_phase7.py

So sánh trực tiếp các chỉ số Tô-pô hình thái giữa:
1. Phase 6 (Single S2-Gate: Chỉ Decoder Block 1 Conv 3x3)
2. Phase 7 (Dual S2-Gate: Decoder Block 1 Conv 3x3 + Decoder Block 2 Conv 3x3 Dilation 2)

Đánh giá trên tập Crack500 Validation Split (348 mẫu) theo giao thức Setting A Tiling (448x448).
Các chỉ số đo đạc:
- Breakage events (Số sự kiện gãy đứt vết nứt GT) & Tỷ lệ ảnh bị đứt (Break image rate)
- Bridge events (Số sự kiện nối dính giả tạo giữa các vết nứt) & Tỷ lệ ảnh bị dính (Bridge image rate)
- Spurious islands (Số lượng đảo nhiễu giả mạo không thuộc vết nứt)
- clDice (Centerline Dice đo độ liên tục trục giữa khung xương)
- HD95 (Khoảng cách Hausdorff phân vị 95)
- Boundary IoU (Độ chính xác đường viền biên)
"""

import argparse
import glob
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple
import yaml

import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
import torch
import torch.nn as nn
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sage.networks import create_b2_unet
from scripts.evaluate_crack_official import predict_full_image_tiling_setting_a
from sage.utils.advanced_metrics import calculate_hd95_bf1


def compute_boundary_iou(pred: np.ndarray, target: np.ndarray, d: int = 2) -> float:
    if np.sum(target) == 0 and np.sum(pred) == 0:
        return 1.0
    if np.sum(target) == 0 or np.sum(pred) == 0:
        return 0.0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * d + 1, 2 * d + 1))
    gt_boundary = cv2.morphologyEx(target.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    pred_boundary = cv2.morphologyEx(pred.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0

    gt_region = gt_boundary & (target > 0)
    pred_region = pred_boundary & (pred > 0)

    intersection = np.sum(gt_region & pred_region)
    union = np.sum(gt_region | pred_region)
    return float(intersection / (union + 1e-7))


def compute_sample_topology(pred_bin: np.ndarray, target_bin: np.ndarray) -> Dict[str, Any]:
    assert pred_bin.shape == target_bin.shape
    H, W = target_bin.shape[:2]

    num_gt_cc, gt_labels = cv2.connectedComponents(target_bin.astype(np.uint8), connectivity=8)
    num_pred_cc, pred_labels = cv2.connectedComponents(pred_bin.astype(np.uint8), connectivity=8)

    gt_cc = int(max(num_gt_cc - 1, 0))
    pred_cc = int(max(num_pred_cc - 1, 0))

    tp = int(np.sum((pred_bin == 1) & (target_bin == 1)))
    fp = int(np.sum((pred_bin == 1) & (target_bin == 0)))
    fn = int(np.sum((pred_bin == 0) & (target_bin == 1)))
    dice = float((2.0 * tp) / (2.0 * tp + fp + fn + 1e-8))
    precision = float(tp / (tp + fp + 1e-8))
    recall = float(tp / (tp + fn + 1e-8))
    iou = float(tp / (tp + fp + fn + 1e-8))

    # 1. False Bridge / Merge
    bridge_events = 0
    merged_gt_set = set()
    for p_id in range(1, pred_cc + 1):
        overlapping_gt = np.unique(gt_labels[pred_labels == p_id])
        overlapping_gt = overlapping_gt[overlapping_gt > 0]
        if len(overlapping_gt) >= 2:
            bridge_events += 1
            for g_id in overlapping_gt:
                merged_gt_set.add(int(g_id))

    # 2. Fragmentation / Breakage
    break_events = 0
    extra_pred_fragments = 0
    for g_id in range(1, gt_cc + 1):
        overlapping_pred = np.unique(pred_labels[gt_labels == g_id])
        overlapping_pred = overlapping_pred[overlapping_pred > 0]
        if len(overlapping_pred) >= 2:
            break_events += 1
            extra_pred_fragments += int(len(overlapping_pred) - 1)

    # 3. Spurious Islands
    spurious_islands = 0
    for p_id in range(1, pred_cc + 1):
        overlapping_gt = np.unique(gt_labels[pred_labels == p_id])
        overlapping_gt = overlapping_gt[overlapping_gt > 0]
        if len(overlapping_gt) == 0:
            spurious_islands += 1

    # 4. Centerline Dice (clDice)
    s_gt = skeletonize(target_bin > 0)
    s_pred = skeletonize(pred_bin > 0)
    len_s_gt = int(np.sum(s_gt))
    len_s_pred = int(np.sum(s_pred))

    if len_s_pred == 0:
        tprec = 1.0 if len_s_gt == 0 else 0.0
    else:
        tprec = float(np.sum(s_pred & (target_bin > 0))) / float(len_s_pred)

    if len_s_gt == 0:
        tsens = 1.0 if len_s_pred == 0 else 0.0
    else:
        tsens = float(np.sum(s_gt & (pred_bin > 0))) / float(len_s_gt)

    cldice = 0.0 if (tprec + tsens) == 0.0 else float(2.0 * tprec * tsens / (tprec + tsens))
    b_iou = compute_boundary_iou(pred_bin, target_bin, d=2)
    hd, _ = calculate_hd95_bf1(pred_bin, target_bin)

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "cldice": cldice,
        "boundary_iou": b_iou,
        "hd95": hd,
        "gt_cc": gt_cc,
        "pred_cc": pred_cc,
        "bridge_events": bridge_events,
        "has_bridge": 1 if bridge_events > 0 else 0,
        "break_events": break_events,
        "has_break": 1 if break_events > 0 else 0,
        "spurious_islands": spurious_islands,
        "has_island": 1 if spurious_islands > 0 else 0,
    }


def load_model(config_path: str, checkpoint_path: str, device: torch.device) -> nn.Module:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = create_b2_unet(
        num_classes=1,
        img_size=cfg.get("img_size", 448),
        num_transformer_layers=cfg.get("num_transformer_layers", 4),
        pretrained=False,
        sage_config=cfg.get("sage_config", {}),
        p3_mode=cfg.get("p3_mode", None),
        use_s2_gate=cfg.get("use_s2_gate", False),
        s2_gate_kernel_size=cfg.get("s2_gate_kernel_size", 3),
        use_s2_gate_block2=cfg.get("use_s2_gate_block2", False),
        s2_gate_block2_kernel_size=cfg.get("s2_gate_block2_kernel_size", 3),
        s2_gate_block2_dilation=cfg.get("s2_gate_block2_dilation", 1),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    state_dict = {k: v for k, v in state_dict.items() if "p3_refinement" not in k}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def evaluate_model_topology(model: nn.Module, pairs: List[Tuple[str, str, str]], device: torch.device, tag: str) -> pd.DataFrame:
    records = []
    for img_path, mask_path, stem in tqdm(pairs, desc=f"Evaluating {tag}"):
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        target = (mask > 0).astype(np.uint8)

        with torch.no_grad():
            pred_logits = predict_full_image_tiling_setting_a(model, img_rgb, device, tile_size=448)
        pred_bin = (pred_logits > 0.0).astype(np.uint8)

        m = compute_sample_topology(pred_bin, target)
        m["case_name"] = stem
        records.append(m)

    df = pd.DataFrame(records)
    return df


def main():
    parser = argparse.ArgumentParser(description="Topology Comparison: Phase 6 vs Phase 7")
    parser.add_argument("--data-root", type=str, default="/content/dataset/Crack500", help="Crack500 dataset root")
    parser.add_argument("--output-dir", type=str, default="results/diagnostics/topology_comparison_p6_vs_p7", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--phase6-config", type=str, default="configs/p3_ablation/phase6_full_s1/a1_s2g_end_to_end.yaml")
    parser.add_argument("--phase6-ckpt", type=str, default="checkpoints/best_model_b2_phase6_s2g.pth")
    parser.add_argument("--phase7-config", type=str, default="configs/p3_ablation/phase7/phase7_b2_s2g_b1_b2_end_to_end.yaml")
    parser.add_argument("--phase7-ckpt", type=str, default="results/phase7_combination/phase7_s2g_b1_b2_end_to_end/best_model_b2_global.pth")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Resolve Validation pairs
    val_img_dir = os.path.join(args.data_root, "val", "images")
    val_mask_dir = os.path.join(args.data_root, "val", "masks")
    img_files = sorted(glob.glob(os.path.join(val_img_dir, "*")))
    pairs = []
    for ip in img_files:
        stem = os.path.splitext(os.path.basename(ip))[0]
        for ext in [".png", ".jpg", ".jpeg"]:
            mp = os.path.join(val_mask_dir, stem + ext)
            if os.path.exists(mp):
                pairs.append((ip, mp, stem))
                break

    print("=" * 80)
    print("SO SÁNH TÔ PÔ HÌNH THÁI CHÍNH THỨC: PHASE 6 vs PHASE 7")
    print(f"Tổng số mẫu Val: {len(pairs)}")
    print(f"Phase 6 Checkpoint: {args.phase6_ckpt}")
    print(f"Phase 7 Checkpoint: {args.phase7_ckpt}")
    print("=" * 80)

    # 1. Evaluate Phase 6
    print("\n[1/2] Nạp và đánh giá Phase 6 (Single S2-Gate Block 1)...")
    model_p6 = load_model(args.phase6_config, args.phase6_ckpt, device)
    df_p6 = evaluate_model_topology(model_p6, pairs, device, "Phase 6 (Single S2G)")

    # 2. Evaluate Phase 7
    print("\n[2/2] Nạp và đánh giá Phase 7 (Dual S2-Gate Block 1 + Block 2)...")
    model_p7 = load_model(args.phase7_config, args.phase7_ckpt, device)
    df_p7 = evaluate_model_topology(model_p7, pairs, device, "Phase 7 (Dual S2G)")

    # 3. Aggregate Comparison Summary
    summary = {
        "Phase 6 (Single S2G)": {
            "val_dice": float(df_p6["dice"].mean()),
            "val_iou": float(df_p6["iou"].mean()),
            "cldice": float(df_p6["cldice"].mean()),
            "boundary_iou": float(df_p6["boundary_iou"].mean()),
            "hd95": float(df_p6["hd95"].dropna().mean()),
            "total_break_events": int(df_p6["break_events"].sum()),
            "break_images": int(df_p6["has_break"].sum()),
            "break_image_pct": float(df_p6["has_break"].mean() * 100),
            "total_bridge_events": int(df_p6["bridge_events"].sum()),
            "bridge_images": int(df_p6["has_bridge"].sum()),
            "bridge_image_pct": float(df_p6["has_bridge"].mean() * 100),
            "total_spurious_islands": int(df_p6["spurious_islands"].sum()),
            "precision": float(df_p6["precision"].mean()),
            "recall": float(df_p6["recall"].mean()),
        },
        "Phase 7 (Dual S2G)": {
            "val_dice": float(df_p7["dice"].mean()),
            "val_iou": float(df_p7["iou"].mean()),
            "cldice": float(df_p7["cldice"].mean()),
            "boundary_iou": float(df_p7["boundary_iou"].mean()),
            "hd95": float(df_p7["hd95"].dropna().mean()),
            "total_break_events": int(df_p7["break_events"].sum()),
            "break_images": int(df_p7["has_break"].sum()),
            "break_image_pct": float(df_p7["has_break"].mean() * 100),
            "total_bridge_events": int(df_p7["bridge_events"].sum()),
            "bridge_images": int(df_p7["has_bridge"].sum()),
            "bridge_image_pct": float(df_p7["has_bridge"].mean() * 100),
            "total_spurious_islands": int(df_p7["spurious_islands"].sum()),
            "precision": float(df_p7["precision"].mean()),
            "recall": float(df_p7["recall"].mean()),
        }
    }

    # Save CSVs & JSON
    df_p6.to_csv(os.path.join(args.output_dir, "phase6_val_topology_per_sample.csv"), index=False)
    df_p7.to_csv(os.path.join(args.output_dir, "phase7_val_topology_per_sample.csv"), index=False)
    with open(os.path.join(args.output_dir, "topology_comparison_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Generate Markdown Table
    p6_s = summary["Phase 6 (Single S2G)"]
    p7_s = summary["Phase 7 (Dual S2G)"]

    d_break = p7_s["total_break_events"] - p6_s["total_break_events"]
    d_bridge = p7_s["total_bridge_events"] - p6_s["total_bridge_events"]
    d_cldice = p7_s["cldice"] - p6_s["cldice"]
    d_hd95 = p7_s["hd95"] - p6_s["hd95"]
    d_dice = p7_s["val_dice"] - p6_s["val_dice"]

    md_report = f"""# Báo Cáo Đối Soát Tô-Pô Hình Thái: Phase 6 vs Phase 7

| Chỉ số Tô-pô & Hình thái | Phase 6 (Chỉ có S2G Block 1) | Phase 7 (Dual S2G: Block 1 + Block 2) | Chênh lệch ($\Delta$ P7 - P6) | Nhận định Giả thuyết |
| :--- | :---: | :---: | :---: | :--- |
| **Validation Dice** | **{p6_s['val_dice']:.4f}** | **{p7_s['val_dice']:.4f}** | `{d_dice:+.4f}` | {"Tăng" if d_dice > 0 else "Giảm"} |
| **Centerline Dice (clDice)** | **{p6_s['cldice']:.4f}** | **{p7_s['cldice']:.4f}** | `{d_cldice:+.4f}` | {"Cải thiện độ liền trục" if d_cldice > 0 else "Suy giảm độ liền trục"} |
| **Boundary IoU** | **{p6_s['boundary_iou']:.4f}** | **{p7_s['boundary_iou']:.4f}** | `{p7_s['boundary_iou'] - p6_s['boundary_iou']:+.4f}` | - |
| **HD95 (px, thấp hơn là tốt)**| **{p6_s['hd95']:.2f} px** | **{p7_s['hd95']:.2f} px** | `{d_hd95:+.2f} px` | {"Rút ngắn khoảng cách biên" if d_hd95 < 0 else "Biên sai lệch lớn hơn"} |
| **Số sự kiện Gãy đứt (Break events)** | **{p6_s['total_break_events']}** | **{p7_s['total_break_events']}** | `{d_break:+d}` | {"ĐẠT: Giảm đứt đoạn" if d_break < 0 else "KHÔNG ĐẠT: Tăng đứt đoạn"} |
| **Ảnh bị gãy đứt vết nứt** | **{p6_s['break_images']} ({p6_s['break_image_pct']:.1f}%)** | **{p7_s['break_images']} ({p7_s['break_image_pct']:.1f}%)** | `{p7_s['break_images'] - p6_s['break_images']:+d}` | - |
| **Số sự kiện Nối dính giả (Bridge events)** | **{p6_s['total_bridge_events']}** | **{p7_s['total_bridge_events']}** | `{d_bridge:+d}` | {"ĐẠT: Giảm cầu nối giả" if d_bridge < 0 else "KHÔNG ĐẠT: Tăng cầu nối giả"} |
| **Ảnh bị dính cầu nối giả** | **{p6_s['bridge_images']} ({p6_s['bridge_image_pct']:.1f}%)** | **{p7_s['bridge_images']} ({p7_s['bridge_image_pct']:.1f}%)** | `{p7_s['bridge_images'] - p6_s['bridge_images']:+d}` | - |
| **Đảo nhiễu giả (Spurious islands)** | **{p6_s['total_spurious_islands']}** | **{p7_s['total_spurious_islands']}** | `{p7_s['total_spurious_islands'] - p6_s['total_spurious_islands']:+d}` | - |
| **Precision** | **{p6_s['precision']:.4f}** | **{p7_s['precision']:.4f}** | `{p7_s['precision'] - p6_s['precision']:+.4f}` | - |
| **Recall** | **{p6_s['recall']:.4f}** | **{p7_s['recall']:.4f}** | `{p7_s['recall'] - p6_s['recall']:+.4f}` | - |

"""
    report_file = os.path.join(args.output_dir, "TOPOLOGY_COMPARISON_REPORT.md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(md_report)

    print("\n" + "=" * 80)
    print(md_report)
    print(f"Đã lưu báo cáo chi tiết vào: {report_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
