#!/usr/bin/env python3
"""
tools/evaluate_final_test.py

PRE-REGISTERED OFFICIAL SINGLE-SHOT FINAL TEST EVALUATOR FOR CRACK500 (1,124 SAMPLES).

Strict Scientific Invariants:
1. Target Split: HARDCODED TO 'test' SPLIT ONLY (1,124 images).
2. Zero Exploration / Zero Tuning: Fixed probability threshold = 0.5 (logits > 0.0).
3. Inference Protocol: Standard Deterministic Inference (TTA = False, exploration_noise = False).
4. Dual Official Protocol:
   - Setting A: Non-overlapping 448x448 tiling (canonical baseline protocol).
   - Setting B: 50% overlapping tiling (stride = 224, probability blending).
5. Comprehensive Metrics:
   - Macro Dice, IoU, Precision, Recall, Crack-Present IoU.
   - Global Pixel-level IoU.
   - Boundary IoU (dilation radius d=2).
   - Hausdorff Distance 95 (HD95).
6. Anti-Leakage Guard:
   - Prevents accidental re-runs if final test artifacts already exist without explicit override.

Author: Special Subject AI Team
Date: September 2026
Branch: crack500-audit
"""

import argparse
import csv
import datetime
import glob
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sage.networks import create_b0_unet, create_b1_unet, create_b2_unet
from sage.utils.advanced_metrics import calculate_hd95_bf1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("final_test_eval")


def compute_boundary_iou(pred: np.ndarray, target: np.ndarray, d: int = 2) -> float:
    """Computes Boundary IoU (Cheng et al., 2021) with boundary dilation radius d."""
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
    if union == 0:
        return 1.0 if np.sum(pred) == 0 and np.sum(target) == 0 else 0.0
    return float(intersection / (union + 1e-5))


def predict_tiling_setting_a(model: torch.nn.Module, image: np.ndarray, device: torch.device, tile_size: int = 448, batch_size: int = 8) -> np.ndarray:
    """Setting A: Non-overlapping tiling (stride = tile_size)."""
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
            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                logits = model(batch)
            logits_np = logits.squeeze(1).cpu().numpy()
        for j, logit in enumerate(logits_np):
            y, x = coords[i + j]
            pred_logits[y : y + tile_size, x : x + tile_size] = logit

    return pred_logits[:H, :W]


def predict_tiling_setting_b(model: torch.nn.Module, image: np.ndarray, device: torch.device, tile_size: int = 448, stride: int = 224, batch_size: int = 8) -> np.ndarray:
    """Setting B: 50% overlapping tiling with probability blending."""
    H, W = image.shape[:2]
    if H < tile_size:
        pad_h = tile_size - H
    else:
        rem_h = (H - tile_size) % stride
        pad_h = (stride - rem_h) % stride if rem_h != 0 else 0

    if W < tile_size:
        pad_w = tile_size - W
    else:
        rem_w = (W - tile_size) % stride
        pad_w = (stride - rem_w) % stride if rem_w != 0 else 0

    padded_img = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    pH, pW = padded_img.shape[:2]

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    patches = []
    coords = []
    for y in range(0, pH - tile_size + 1, stride):
        for x in range(0, pW - tile_size + 1, stride):
            patch = padded_img[y : y + tile_size, x : x + tile_size]
            patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
            patch_tensor = (patch_tensor - mean) / std
            patches.append(patch_tensor)
            coords.append((y, x))

    prob_accum = np.zeros((pH, pW), dtype=np.float32)
    weight_accum = np.zeros((pH, pW), dtype=np.float32)

    for i in range(0, len(patches), batch_size):
        batch = torch.stack(patches[i : i + batch_size]).to(device, non_blocking=True)
        with torch.no_grad():
            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                logits = model(batch)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        for j, prob in enumerate(probs):
            y, x = coords[i + j]
            prob_accum[y : y + tile_size, x : x + tile_size] += prob
            weight_accum[y : y + tile_size, x : x + tile_size] += 1.0

    blended_probs = prob_accum / np.maximum(weight_accum, 1.0)
    return blended_probs[:H, :W]


def discover_test_pairs(config: Dict[str, Any], data_root_override: str = None) -> List[Tuple[str, str, str]]:
    """Strictly discovers test/images and test/masks pairs."""
    root_dir = data_root_override or config.get("root_dir", "/content/dataset/Crack500")
    test_cfg = config.get("test", {"images": "test/images", "masks": "test/masks"})

    img_dir = test_cfg["images"]
    mask_dir = test_cfg["masks"]
    if not os.path.isabs(img_dir):
        img_dir = os.path.join(root_dir, img_dir)
    if not os.path.isabs(mask_dir):
        mask_dir = os.path.join(root_dir, mask_dir)

    img_paths = sorted(glob.glob(os.path.join(img_dir, "*")))
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_paths = [p for p in img_paths if os.path.splitext(p)[1].lower() in valid_exts]

    pairs = []
    mask_suffix = config.get("mask_suffix", "")
    for ip in img_paths:
        stem = os.path.splitext(os.path.basename(ip))[0]
        for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
            mp = os.path.join(mask_dir, stem + mask_suffix + ext)
            if os.path.exists(mp):
                pairs.append((ip, mp, stem))
                break

    return pairs


def load_model(config: Dict[str, Any], checkpoint_path: str, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Instantiates the exact model architecture and loads checkpoint."""
    model_type = config.get("model", "B2")
    img_size = int(config.get("img_size", 448))
    vit_depth = int(config.get("num_transformer_layers", 4))
    p3_mode = config.get("p3_mode", "C")
    sage_cfg = config.get("sage_config", {})

    logger.info(f"Instantiating model {model_type} (vit_depth={vit_depth}, p3_mode='{p3_mode}', top_k={sage_cfg.get('top_k', 2)})...")
    if model_type == "B2":
        model = create_b2_unet(
            num_classes=1,
            img_size=img_size,
            num_transformer_layers=vit_depth,
            pretrained=False,
            sage_config=sage_cfg,
            p3_mode=p3_mode,
            use_s2_gate=config.get("use_s2_gate", False),
            s2_gate_kernel_size=config.get("s2_gate_kernel_size", 3),
            use_s2_gate_block2=config.get("use_s2_gate_block2", False),
            s2_gate_block2_kernel_size=config.get("s2_gate_block2_kernel_size", 3),
            s2_gate_block2_dilation=config.get("s2_gate_block2_dilation", 1),
        ).to(device)
    elif model_type == "B1":
        model = create_b1_unet(num_transformer_layers=vit_depth, pretrained=False).to(device)
    elif model_type == "B0":
        model = create_b0_unet(pretrained=False).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    logger.info(f"Loading checkpoint weights from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    return model, ckpt


def evaluate_protocol(model: torch.nn.Module, pairs: List[Tuple[str, str, str]], protocol: str, device: torch.device, tile_size: int = 448, batch_size: int = 8) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Runs single evaluation pass across all pairs for the designated protocol."""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    metrics = {
        "dice": [],
        "iou": [],
        "precision": [],
        "recall": [],
        "crack_iou": [],
        "boundary_iou": [],
        "hd95": [],
    }
    per_sample_rows = []

    for ip, mp, case_name in tqdm(pairs, desc=f"Evaluating {protocol.upper()}"):
        img_bgr = cv2.imread(ip)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        target = (mask > 0).astype(np.uint8)

        if protocol == "setting_a":
            logits = predict_tiling_setting_a(model, img_rgb, device, tile_size=tile_size, batch_size=batch_size)
            pred = (logits > 0.0).astype(np.uint8)
        elif protocol == "setting_b":
            probs = predict_tiling_setting_b(model, img_rgb, device, tile_size=tile_size, stride=tile_size // 2, batch_size=batch_size)
            pred = (probs >= 0.5).astype(np.uint8)
        else:
            raise ValueError(f"Unknown protocol: {protocol}")

        tp = int(np.sum((pred == 1) & (target == 1)))
        fp = int(np.sum((pred == 1) & (target == 0)))
        fn = int(np.sum((pred == 0) & (target == 1)))
        tn = int(np.sum((pred == 0) & (target == 0)))

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        precision = tp / (tp + fp + 1e-5)
        recall = tp / (tp + fn + 1e-5)
        dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-5)
        iou = tp / (tp + fp + fn + 1e-5)

        b_iou = compute_boundary_iou(pred, target, d=2)
        hd, _ = calculate_hd95_bf1(pred, target)

        metrics["dice"].append(dice)
        metrics["iou"].append(iou)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["boundary_iou"].append(b_iou)
        metrics["hd95"].append(hd)

        if np.sum(target) > 0:
            metrics["crack_iou"].append(iou)

        per_sample_rows.append({
            "case_name": case_name,
            "protocol": protocol,
            "dice": round(float(dice), 5),
            "iou": round(float(iou), 5),
            "precision": round(float(precision), 5),
            "recall": round(float(recall), 5),
            "boundary_iou": round(float(b_iou), 5),
            "hd95": round(float(hd), 3),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        })

    summary = {
        "mean_dice": float(np.mean(metrics["dice"])),
        "median_dice": float(np.median(metrics["dice"])),
        "std_dice": float(np.std(metrics["dice"])),
        "mean_iou": float(np.mean(metrics["iou"])),
        "median_iou": float(np.median(metrics["iou"])),
        "mean_precision": float(np.mean(metrics["precision"])),
        "mean_recall": float(np.mean(metrics["recall"])),
        "crack_present_iou": float(np.mean(metrics["crack_iou"])),
        "global_pixel_iou": float(total_tp / (total_tp + total_fp + total_fn + 1e-5)),
        "mean_boundary_iou": float(np.mean(metrics["boundary_iou"])),
        "mean_hd95": float(np.mean(metrics["hd95"])),
        "evaluated_samples": len(pairs),
    }

    return summary, per_sample_rows


def main():
    parser = argparse.ArgumentParser(description="Official Single-Shot Final Test Evaluator for Crack500")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to finalized model checkpoint")
    parser.add_argument("--data-root", type=str, default=None, help="Dataset root directory")
    parser.add_argument("--output-dir", type=str, default="results/test_eval", help="Output directory for test results")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for tiling inference")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force-rerun", action="store_true", help="Force re-run even if final test output exists")
    parser.add_argument("--rerun-reason", type=str, default=None, help="Mandatory justification reason if --force-rerun is used")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    final_json_path = os.path.join(args.output_dir, "final_test_metrics.json")
    audit_log_path = os.path.join(args.output_dir, "test_evaluation_audit_trail.log")

    is_rerun = os.path.exists(final_json_path)
    if is_rerun:
        if not args.force_rerun:
            logger.error(
                f"PRE-REGISTRATION LOCK ENGAGED: '{final_json_path}' already exists! "
                "To prevent multiple testing bias on the Test set, re-evaluating is blocked. "
                "To execute an intentional re-run, both --force-rerun and --rerun-reason '<justification>' are strictly required."
            )
            sys.exit(1)
        if not args.rerun_reason or len(args.rerun_reason.strip()) < 5:
            logger.error(
                "AUDIT INTEGRITY VIOLATION: --force-rerun was supplied without a valid --rerun-reason! "
                "You must document a clear justification for re-evaluating the pristine Test set."
            )
            sys.exit(1)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.enabled = False

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. Discover Test Split (Strictly 1,124 images)
    test_pairs = discover_test_pairs(config, args.data_root)
    logger.info(f"Discovered {len(test_pairs)} test image-mask pairs.")
    assert len(test_pairs) == 1124, f"Expected exactly 1,124 test samples in Crack500, but discovered {len(test_pairs)}!"

    # 2. Instantiate Model and Load Checkpoint
    model, ckpt_meta = load_model(config, args.checkpoint, device)

    # 3. Evaluate Setting A and Setting B
    logger.info("Executing Primary Official Protocol: Setting A (Non-Overlapping Tiling)...")
    summary_a, samples_a = evaluate_protocol(model, test_pairs, "setting_a", device, tile_size=config.get("img_size", 448), batch_size=args.batch_size)

    logger.info("Executing Secondary Reference Protocol: Setting B (50% Overlapping Tiling)...")
    summary_b, samples_b = evaluate_protocol(model, test_pairs, "setting_b", device, tile_size=config.get("img_size", 448), batch_size=args.batch_size)

    # 4. Save Final Machine-Readable JSON
    final_output = {
        "metadata": {
            "model": config.get("model", "B2"),
            "p3_mode": config.get("p3_mode", None),
            "vit_depth": config.get("num_transformer_layers", 4),
            "primary_official_protocol": "setting_a",
            "secondary_reference_protocol": "setting_b",
            "config": os.path.abspath(args.config),
            "checkpoint": os.path.abspath(args.checkpoint),
            "checkpoint_metadata": {
                "epoch": ckpt_meta.get("epoch"),
                "stage": ckpt_meta.get("stage"),
                "val_dice": ckpt_meta.get("val_dice"),
                "best_dice": ckpt_meta.get("best_dice"),
            },
            "timestamp": datetime.datetime.now().isoformat(),
            "total_test_samples": len(test_pairs),
            "evaluation_guard": "FORCE_RERUN" if is_rerun else "INITIAL_RUN",
            "rerun_reason": args.rerun_reason if is_rerun else None,
            "preregistration_invariants": {
                "threshold": 0.5,
                "tta": False,
                "setting_a_stride": 448,
                "setting_b_stride": 224,
            },
        },
        "setting_a_official": summary_a,
        "setting_b_reference": summary_b,
    }

    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    logger.info(f"Saved final test metrics to: {final_json_path}")

    # Write to Audit Trail Log
    audit_entry = (
        f"[{datetime.datetime.now().isoformat()}] EVENT: {'FORCE_RERUN' if is_rerun else 'INITIAL_RUN'}\n"
        f"  Checkpoint: {os.path.abspath(args.checkpoint)}\n"
        f"  Config:     {os.path.abspath(args.config)}\n"
        f"  Reason:     {args.rerun_reason if is_rerun else 'Official Preregistered First Execution'}\n"
        f"  Results (Setting A): Dice={summary_a['mean_dice']:.4f}, IoU={summary_a['mean_iou']:.4f}, Boundary_IoU={summary_a['mean_boundary_iou']:.4f}\n"
        f"  Results (Setting B): Dice={summary_b['mean_dice']:.4f}, IoU={summary_b['mean_iou']:.4f}, Boundary_IoU={summary_b['mean_boundary_iou']:.4f}\n"
        "--------------------------------------------------------------------------------\n"
    )
    with open(audit_log_path, "a", encoding="utf-8") as f:
        f.write(audit_entry)
    logger.info(f"Appended audit trail entry to: {audit_log_path}")

    # 5. Save Per-Sample CSV
    csv_path = os.path.join(args.output_dir, "final_test_per_sample.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(samples_a[0].keys()))
        writer.writeheader()
        writer.writerows(samples_a)
        writer.writerows(samples_b)
    logger.info(f"Saved per-sample test metrics to: {csv_path}")

    # 6. Generate Summary Markdown
    md_path = os.path.join(args.output_dir, "final_test_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"""# Crack500 Official Final Test Set Evaluation Report

> **Single-Shot Final Test Evaluation (Preregistered Protocol)**
> - **Evaluation Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> - **Test Samples:** {len(test_pairs)} images (Crack500 Test Split)
> - **Evaluated Checkpoint:** `{os.path.basename(args.checkpoint)}`
> - **Architecture:** {config.get('model', 'B2')} (Depth={config.get('num_transformer_layers', 4)}, P3={config.get('p3_mode', 'C')})

---

## 1. Official Test Benchmark Results

| Protocol | Mean Dice | Median Dice | Macro IoU | Global Pixel IoU | Crack-Present IoU | Precision | Recall | Boundary IoU | HD95 (px) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Setting A** (Non-overlapping) | **{summary_a['mean_dice']:.4f}** | **{summary_a['median_dice']:.4f}** | **{summary_a['mean_iou']:.4f}** | **{summary_a['global_pixel_iou']:.4f}** | **{summary_a['crack_present_iou']:.4f}** | **{summary_a['mean_precision']:.4f}** | **{summary_a['mean_recall']:.4f}** | **{summary_a['mean_boundary_iou']:.4f}** | **{summary_a['mean_hd95']:.2f}** |
| **Setting B** (50% Overlapping) | **{summary_b['mean_dice']:.4f}** | **{summary_b['median_dice']:.4f}** | **{summary_b['mean_iou']:.4f}** | **{summary_b['global_pixel_iou']:.4f}** | **{summary_b['crack_present_iou']:.4f}** | **{summary_b['mean_precision']:.4f}** | **{summary_b['mean_recall']:.4f}** | **{summary_b['mean_boundary_iou']:.4f}** | **{summary_b['mean_hd95']:.2f}** |

---

*Report automatically generated by `tools/evaluate_final_test.py`.*
""")
    logger.info(f"Saved final test summary to: {md_path}")
    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION COMPLETE")
    print(f"Setting A Mean Dice: {summary_a['mean_dice']:.4f} | Macro IoU: {summary_a['mean_iou']:.4f} | Boundary IoU: {summary_a['mean_boundary_iou']:.4f}")
    print(f"Setting B Mean Dice: {summary_b['mean_dice']:.4f} | Macro IoU: {summary_b['mean_iou']:.4f} | Boundary IoU: {summary_b['mean_boundary_iou']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
