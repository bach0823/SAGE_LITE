#!/usr/bin/env python3
"""
scripts/run_phase6_combination_pipeline.py

Phase 6 Final Pipeline Combination & Candidate C Synthesizer.
Orchestrates the 3-stage controlled experimental ladder:
  - Stage 1A: AB-BPL (B1) Lambda Sweep (3 runs: v1, v2, v3 + v0 ref)
  - Stage 1B: SoftBIoU (A1) Lambda Sweep (3 runs: v1, v2, v3 + v0 ref)
  - Stage 2:  Combination A1(lambda*) + B1(lambda*) (Run 2A + optional 2B)
  - Stage 3:  Stack S2-Gate-v2 (Conv3x3) -> Candidate C (final_candidate_c.pth)

Usage Examples:
  # Run entire pipeline from start to finish:
  python scripts/run_phase6_combination_pipeline.py --stage all

  # Run only Stage 1A (B1 Sweep):
  python scripts/run_phase6_combination_pipeline.py --stage 1a

  # Run only Stage 1B (A1 Sweep):
  python scripts/run_phase6_combination_pipeline.py --stage 1b

  # Run only Stage 2 (Combination using best discovered Stage 1 params):
  python scripts/run_phase6_combination_pipeline.py --stage stage2

  # Run only Stage 3 (Stack S2-Gate-v2 onto best combined base):
  python scripts/run_phase6_combination_pipeline.py --stage stage3
"""

import os
import sys
import json
import time
import argparse
import subprocess
import urllib.request
from typing import Dict, Any, Tuple, Optional, List
import yaml
import numpy as np
import cv2
import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt

# Add project root to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sage.networks import create_b0_unet, create_b1_unet, create_b2_unet
from scripts.evaluate_crack_official import get_image_mask_pairs
from scripts.diagnostics.train_eval_phase6_u1_s2g_v2 import (
    compute_thin_crack_dice,
    predict_full_image_tiling_setting_a_fp32,
)

# Reference URLs for Checkpoints & Pre-trained weights (Zero-Drive Colab)
URL_CANDIDATE_B_STAGE1 = (
    "https://raw.githubusercontent.com/bach0823/chuyendettnt/main/results/checkpoints/"
    "P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_stage1.pth"
)
URL_U1_S2G_V1_WEIGHTS = (
    "https://raw.githubusercontent.com/bach0823/chuyendettnt/main/results/diagnostics/"
    "phase6_u1_s2g/u1_s2g_weights.pth"
)

# Ex-Ante Reference Baselines (Known Ground-Truth)
BASE_CANDIDATE_B = {
    "dice": 0.7641,
    "precision": 0.7337,
    "recall": 0.8477,
    "thin_dice": 0.423006,
}
REF_B1_V0 = {
    "run_id": "b1_v0",
    "lambda": 0.040,
    "r": 2,
    "dice": 0.7685,
    "precision": 0.7376,
    "recall": 0.8458,
    "epoch": 28,
}
REF_A1_V0 = {
    "run_id": "a1_v0",
    "lambda": 0.50,
    "d": 2,
    "dice": 0.7684,
    "precision": 0.7416,
    "recall": 0.8413,
    "thin_dice": 0.4350,  # Improved thin crack dice
    "epoch": 31,
}


def download_with_progress(url: str, dest_path: str):
    """Download file from url to dest_path with size verification."""
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024:
        print(f"[Cache Hit] File already exists: {dest_path} ({os.path.getsize(dest_path):,} bytes)")
        return
    print(f"[Downloading] {url} -> {dest_path}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"[Done] Downloaded {os.path.getsize(dest_path):,} bytes.")
    except Exception as e:
        print(f"[Download Error] Failed to download {url}: {e}")
        raise


def run_command(cmd: List[str], desc: str) -> None:
    """Run command via subprocess and stream output."""
    print("\n" + "=" * 80)
    print(f"RUNNING: {desc}")
    print(f"COMMAND: {' '.join(cmd)}")
    print("=" * 80 + "\n")
    start_t = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - start_t
    if res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}: {' '.join(cmd)}")
    print(f"\n[Completed] {desc} in {elapsed:.1f}s (Exit code 0)\n")


def evaluate_thin_crack_for_checkpoint(
    checkpoint_path: str,
    config_path: str,
    data_root: str,
    device: torch.device,
) -> float:
    """
    Evaluates Setting A FP32 thin crack Dice (thickness <= 3px) on canonical validation set (N=348).
    """
    print(f"[Eval Thin Crack] Evaluating thin crack Dice for {os.path.basename(checkpoint_path)}...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config["root_dir"] = data_root

    model = create_b2_unet(
        num_classes=1,
        img_size=config.get("img_size", 448),
        num_transformer_layers=int(config.get("num_transformer_layers", 4)),
        pretrained=False,
        sage_config=config.get("sage_config", {}),
        p3_mode=config.get("p3_mode", "C"),
        use_plu_head=config.get("use_plu_head", False),
        use_s2_gate=config.get("use_s2_gate", False),
        s2_gate_kernel_size=int(config.get("s2_gate_kernel_size", 3)),
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    val_pairs = get_image_mask_pairs(config, "val")
    if len(val_pairs) == 0:
        raise FileNotFoundError(f"No validation pairs found under {data_root}")

    thin_scores = []
    with torch.no_grad():
        for img_p, mask_p in val_pairs:
            img = cv2.imread(img_p)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            target_bin = (mask > 127).astype(np.uint8)

            logits = predict_full_image_tiling_setting_a_fp32(model, img_rgb, device, tile_size=448, batch_size=8)
            pred_bin = (logits > 0.0).astype(np.uint8)

            t_dice = compute_thin_crack_dice(pred_bin, target_bin, max_thickness=3)
            if not np.isnan(t_dice):
                thin_scores.append(t_dice)

    mean_thin = float(np.mean(thin_scores)) if len(thin_scores) > 0 else 0.0
    print(f"[Eval Thin Crack] Mean Thin Crack Dice: {mean_thin:.4f} (over {len(thin_scores)} valid samples)")
    return mean_thin


RUN_CONFIG_MAP = {
    "b1_v1": {"type": "B1", "lambda": 0.020, "param_name": "r", "param_val": 2, "cfg": "configs/p3_ablation/phase6_combination/b1_v1_l002_r2.yaml"},
    "b1_v2": {"type": "B1", "lambda": 0.080, "param_name": "r", "param_val": 2, "cfg": "configs/p3_ablation/phase6_combination/b1_v2_l008_r2.yaml"},
    "b1_v3": {"type": "B1", "lambda": 0.040, "param_name": "r", "param_val": 1, "cfg": "configs/p3_ablation/phase6_combination/b1_v3_l004_r1.yaml"},
    "a1_v1": {"type": "A1", "lambda": 0.250, "param_name": "d", "param_val": 2, "cfg": "configs/p3_ablation/phase6_combination/a1_v1_l025_d2.yaml"},
    "a1_v2": {"type": "A1", "lambda": 0.750, "param_name": "d", "param_val": 2, "cfg": "configs/p3_ablation/phase6_combination/a1_v2_l075_d2.yaml"},
    "a1_v3": {"type": "A1", "lambda": 0.500, "param_name": "d", "param_val": 3, "cfg": "configs/p3_ablation/phase6_combination/a1_v3_l050_d3.yaml"},
    "a1_a2_v1": {
        "type": "A1+A2",
        "lambda": 0.250,
        "param_name": "d",
        "param_val": 2,
        "cfg": "configs/p3_ablation/phase6_a1_a2/a1_a2_v1_l025_d2.yaml",
        "ckpt_name": "P3_C_Phase6_A2_Pure_PLU_D4_K2_H64_best_model_b2_stage1.pth",
        "ckpt_url": "https://raw.githubusercontent.com/bach0823/chuyendettnt/main/results/checkpoints/P3_C_Phase6_A2_Pure_PLU_D4_K2_H64_best_model_b2_stage1.pth",
    },
    "a1_a2_v2": {
        "type": "A1+A2",
        "lambda": 0.500,
        "param_name": "d",
        "param_val": 2,
        "cfg": "configs/p3_ablation/phase6_a1_a2/a1_a2_v2_l050_d2.yaml",
        "ckpt_name": "P3_C_Phase6_A2_Pure_PLU_D4_K2_H64_best_model_b2_stage1.pth",
        "ckpt_url": "https://raw.githubusercontent.com/bach0823/chuyendettnt/main/results/checkpoints/P3_C_Phase6_A2_Pure_PLU_D4_K2_H64_best_model_b2_stage1.pth",
    },
    "a1_a2_v3": {
        "type": "A1+A2",
        "lambda": 0.750,
        "param_name": "d",
        "param_val": 2,
        "cfg": "configs/p3_ablation/phase6_a1_a2/a1_a2_v3_l075_d2.yaml",
        "ckpt_name": "P3_C_Phase6_A2_Pure_PLU_D4_K2_H64_best_model_b2_stage1.pth",
        "ckpt_url": "https://raw.githubusercontent.com/bach0823/chuyendettnt/main/results/checkpoints/P3_C_Phase6_A2_Pure_PLU_D4_K2_H64_best_model_b2_stage1.pth",
    },
    "a1_full_s1_l050": {
        "type": "A1_Full_S1",
        "lambda": 0.500,
        "param_name": "d",
        "param_val": 2,
        "cfg": "configs/p3_ablation/phase6_full_s1/a1_full_s1_l050_d2.yaml",
        "is_full_s1": True,
    },
    "a1_full_s1_l075": {
        "type": "A1_Full_S1",
        "lambda": 0.750,
        "param_name": "d",
        "param_val": 2,
        "cfg": "configs/p3_ablation/phase6_full_s1/a1_full_s1_l075_d2.yaml",
        "is_full_s1": True,
    },
    "a1_a2_v1_full_s1": {
        "type": "A1+A2_Full_S1",
        "lambda": 0.250,
        "param_name": "d",
        "param_val": 2,
        "cfg": "configs/p3_ablation/phase6_full_s1/a1_a2_v1_full_s1.yaml",
        "is_full_s1": True,
    },
    "a1_s2g_end_to_end": {
        "type": "A1+S2Gate_EndToEnd",
        "lambda": 0.500,
        "param_name": "d",
        "param_val": 2,
        "cfg": "configs/p3_ablation/phase6_full_s1/a1_s2g_end_to_end.yaml",
        "is_full_s1": True,
    },
    "a1_s2g_stage2": {
        "type": "A1+S2Gate_Stage2Only",
        "lambda": 0.500,
        "param_name": "d",
        "param_val": 2,
        "cfg": "configs/p3_ablation/phase6_full_s1/a1_s2g_end_to_end.yaml",
        "ckpt_path": "results/phase6_combination/phase6_comb_a1_s2g_end_to_end/best_model_b2_stage1.pth",
    },
}


def run_custom_runs(
    args: argparse.Namespace,
    run_ids: List[str],
    stage1_ckpt: str,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Runs a specific subset of Stage 1 runs (ideal for multi-Colab parallel execution)."""
    print("\n" + "#" * 80)
    print(f"RUNNING CUSTOM PARALLEL RUNS: {run_ids}")
    print("#" * 80)

    results = []
    for run_id in run_ids:
        if run_id not in RUN_CONFIG_MAP:
            raise ValueError(f"Unknown run ID: {run_id}. Valid choices: {list(RUN_CONFIG_MAP.keys())}")

        item = RUN_CONFIG_MAP[run_id]
        cfg_path = os.path.join(PROJECT_ROOT, item["cfg"])
        run_out_dir = os.path.join(args.output_dir, f"phase6_comb_{run_id}")
        completion_file = os.path.join(run_out_dir, "stage2_completion.json")
        is_full = item.get("is_full_s1", False)

        target_ckpt = stage1_ckpt
        if item.get("ckpt_path"):
            target_ckpt = os.path.join(PROJECT_ROOT, item["ckpt_path"]) if not os.path.isabs(item["ckpt_path"]) else item["ckpt_path"]
        elif item.get("ckpt_name") and item.get("ckpt_url"):
            target_ckpt = os.path.join(args.checkpoint_dir, item["ckpt_name"])
            if not os.path.exists(target_ckpt):
                download_with_progress(item["ckpt_url"], target_ckpt)

        if args.skip_completed and os.path.exists(completion_file):
            print(f"[{run_id}] Found existing stage2_completion.json. Skipping training.")
        else:
            cmd = [
                sys.executable,
                os.path.join(PROJECT_ROOT, "scripts", "train_crack.py"),
                "--config", cfg_path,
                "--output-dir", run_out_dir,
                "--data-root", args.data_root,
            ]
            if not is_full:
                cmd.extend(["--stage2-only", "--checkpoint", target_ckpt])

            run_command(cmd, f"Executing Run {run_id} ({item['type']}, lambda={item['lambda']}, {item['param_name']}={item['param_val']}, Full S1={is_full})")

        stats = {}
        if os.path.exists(completion_file):
            with open(completion_file, "r") as f:
                stats = json.load(f)
        else:
            # Fallback for full 2-stage runs if stage2_completion isn't written directly
            summary_path = os.path.join(run_out_dir, "training_summary.json")
            if os.path.exists(summary_path):
                with open(summary_path, "r") as f:
                    stats = json.load(f)

        dice = stats.get("best_dice", stats.get("dice", 0.0))
        prec = stats.get("precision", 0.0)
        rec = stats.get("recall", 0.0)
        best_model_path = os.path.join(run_out_dir, "best_model_b2_stage2.pth")
        if not os.path.exists(best_model_path):
            best_model_path = os.path.join(run_out_dir, "best_model_b2_global.pth")

        entry = {
            "run_id": run_id,
            "type": item["type"],
            "lambda": item["lambda"],
            item["param_name"]: item["param_val"],
            "dice": dice,
            "precision": prec,
            "recall": rec,
            "best_epoch": stats.get("best_epoch", 0),
            "output_dir": run_out_dir,
            "best_model": best_model_path,
        }

        if item["type"] == "B1":
            score = dice * rec
            c_pass = rec >= 0.840
            entry["score"] = score
            entry["constraint_pass"] = c_pass
        else:
            thin_dice = evaluate_thin_crack_for_checkpoint(best_model_path, cfg_path, args.data_root, device)
            score = dice + 2.0 * (thin_dice - 0.4230)
            entry["thin_dice"] = thin_dice
            entry["score"] = score

        results.append(entry)

    print("\n" + "=" * 80)
    print("PARALLEL RUNS SUMMARY TABLE:")
    print("-" * 80)
    for r in results:
        if r["type"] == "B1":
            print(f"  [{r['run_id']}] Dice: {r['dice']:.4f} | Prec: {r['precision']:.4f} | Recall: {r['recall']:.4f} | Score (Dice*Rec): {r['score']:.4f} | Constraint (Rec>=0.840): {'PASS' if r['constraint_pass'] else 'FAIL'}")
        else:
            print(f"  [{r['run_id']}] Dice: {r['dice']:.4f} | Prec: {r['precision']:.4f} | Recall: {r['recall']:.4f} | ThinDice: {r.get('thin_dice', 0):.4f} | Objective Score: {r['score']:.4f}")
    print("=" * 80 + "\n")

    summary_file = os.path.join(args.output_dir, f"summary_{'_'.join(run_ids)}.json")
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary saved to {summary_file}")
    return results


# =============================================================================
# STAGE 1A: B1 LAMBDA SWEEP
# =============================================================================
def run_stage_1a(args: argparse.Namespace, stage1_ckpt: str) -> Dict[str, Any]:
    print("\n" + "#" * 80)
    print("STAGE 1A: AB-BPL (B1) LAMBDA SWEEP")
    print("Goal: Find optimal lambda*_B1 that maximizes (Dice * Recall) s.t. Recall >= 0.840")
    print("#" * 80)

    sweep_configs = [
        {"run_id": "b1_v1", "lambda": 0.020, "r": 2, "cfg": "configs/p3_ablation/phase6_combination/b1_v1_l002_r2.yaml"},
        {"run_id": "b1_v2", "lambda": 0.080, "r": 2, "cfg": "configs/p3_ablation/phase6_combination/b1_v2_l008_r2.yaml"},
        {"run_id": "b1_v3", "lambda": 0.040, "r": 1, "cfg": "configs/p3_ablation/phase6_combination/b1_v3_l004_r1.yaml"},
    ]

    all_results = [REF_B1_V0.copy()]
    REF_B1_V0["score"] = REF_B1_V0["dice"] * REF_B1_V0["recall"]
    REF_B1_V0["constraint_pass"] = REF_B1_V0["recall"] >= 0.840

    for item in sweep_configs:
        run_id = item["run_id"]
        cfg_path = os.path.join(PROJECT_ROOT, item["cfg"])
        run_out_dir = os.path.join(args.output_dir, f"phase6_comb_{run_id}")
        completion_file = os.path.join(run_out_dir, "stage2_completion.json")

        if args.skip_completed and os.path.exists(completion_file):
            print(f"[{run_id}] Found existing stage2_completion.json. Skipping training.")
        else:
            cmd = [
                sys.executable,
                os.path.join(PROJECT_ROOT, "scripts", "train_crack.py"),
                "--config", cfg_path,
                "--stage2-only",
                "--checkpoint", stage1_ckpt,
                "--output-dir", run_out_dir,
                "--data-root", args.data_root,
            ]
            run_command(cmd, f"Stage 1A Run {run_id} (lambda={item['lambda']}, r={item['r']})")

        with open(completion_file, "r") as f:
            stats = json.load(f)

        dice = stats.get("best_dice", stats.get("dice", 0.0))
        prec = stats.get("precision", 0.0)
        rec = stats.get("recall", 0.0)
        score = dice * rec
        c_pass = rec >= 0.840

        res_entry = {
            "run_id": run_id,
            "lambda": item["lambda"],
            "r": item["r"],
            "dice": dice,
            "precision": prec,
            "recall": rec,
            "score": score,
            "constraint_pass": c_pass,
            "best_epoch": stats.get("best_epoch", 0),
            "output_dir": run_out_dir,
            "best_model": os.path.join(run_out_dir, "best_model_b2_stage2.pth"),
        }
        all_results.append(res_entry)

    # Selection: Maximize Dice * Recall s.t. Recall >= 0.840
    valid_candidates = [r for r in all_results if r["constraint_pass"]]
    if len(valid_candidates) > 0:
        best_b1 = max(valid_candidates, key=lambda x: x["score"])
    else:
        print("[Warning] No B1 run met hard constraint Recall >= 0.840! Selecting highest Dice * Recall.")
        best_b1 = max(all_results, key=lambda x: x["score"])

    print("\n" + "=" * 80)
    print("STAGE 1A: B1 (AB-BPL) SWEEP SUMMARY TABLE")
    print(f"{'Run':<8} | {'Lambda':<8} | {'r':<4} | {'Dice':<8} | {'Precision':<10} | {'Recall':<8} | {'Dice*Recall':<12} | {'Constraint (>=0.840)'}")
    print("-" * 80)
    for r in all_results:
        flag = "PASS" if r["constraint_pass"] else "FAIL"
        is_sel = " <-- [SELECTED lambda*]" if r["run_id"] == best_b1["run_id"] else ""
        print(f"{r['run_id']:<8} | {r['lambda']:<8.3f} | {r['r']:<4} | {r['dice']:<8.4f} | {r['precision']:<10.4f} | {r['recall']:<8.4f} | {r['score']:<12.4f} | {flag}{is_sel}")
    print("=" * 80)

    summary_file = os.path.join(args.output_dir, "stage1a_b1_sweep_results.json")
    with open(summary_file, "w") as f:
        json.dump({"runs": all_results, "selected_best_b1": best_b1}, f, indent=2)
    print(f"Stage 1A results saved to {summary_file}")
    return best_b1


# =============================================================================
# STAGE 1B: A1 LAMBDA SWEEP
# =============================================================================
def run_stage_1b(args: argparse.Namespace, stage1_ckpt: str, device: torch.device) -> Dict[str, Any]:
    print("\n" + "#" * 80)
    print("STAGE 1B: SoftBIoU (A1) LAMBDA SWEEP")
    print("Goal: Find optimal lambda*_A1 that maximizes Dice + 2 * (ThinCrackDice - 0.4230)")
    print("#" * 80)

    sweep_configs = [
        {"run_id": "a1_v1", "lambda": 0.25, "d": 2, "cfg": "configs/p3_ablation/phase6_combination/a1_v1_l025_d2.yaml"},
        {"run_id": "a1_v2", "lambda": 0.75, "d": 2, "cfg": "configs/p3_ablation/phase6_combination/a1_v2_l075_d2.yaml"},
        {"run_id": "a1_v3", "lambda": 0.50, "d": 3, "cfg": "configs/p3_ablation/phase6_combination/a1_v3_l050_d3.yaml"},
    ]

    all_results = [REF_A1_V0.copy()]
    REF_A1_V0["score"] = REF_A1_V0["dice"] + 2.0 * (REF_A1_V0["thin_dice"] - 0.4230)

    for item in sweep_configs:
        run_id = item["run_id"]
        cfg_path = os.path.join(PROJECT_ROOT, item["cfg"])
        run_out_dir = os.path.join(args.output_dir, f"phase6_comb_{run_id}")
        completion_file = os.path.join(run_out_dir, "stage2_completion.json")

        if args.skip_completed and os.path.exists(completion_file):
            print(f"[{run_id}] Found existing stage2_completion.json. Skipping training.")
        else:
            cmd = [
                sys.executable,
                os.path.join(PROJECT_ROOT, "scripts", "train_crack.py"),
                "--config", cfg_path,
                "--stage2-only",
                "--checkpoint", stage1_ckpt,
                "--output-dir", run_out_dir,
                "--data-root", args.data_root,
            ]
            run_command(cmd, f"Stage 1B Run {run_id} (lambda={item['lambda']}, d={item['d']})")

        with open(completion_file, "r") as f:
            stats = json.load(f)

        dice = stats.get("best_dice", stats.get("dice", 0.0))
        prec = stats.get("precision", 0.0)
        rec = stats.get("recall", 0.0)
        best_model_path = os.path.join(run_out_dir, "best_model_b2_stage2.pth")

        # Evaluate Thin Crack Dice on canonical validation set
        thin_dice = evaluate_thin_crack_for_checkpoint(best_model_path, cfg_path, args.data_root, device)
        score = dice + 2.0 * (thin_dice - 0.4230)

        res_entry = {
            "run_id": run_id,
            "lambda": item["lambda"],
            "d": item["d"],
            "dice": dice,
            "precision": prec,
            "recall": rec,
            "thin_dice": thin_dice,
            "score": score,
            "best_epoch": stats.get("best_epoch", 0),
            "output_dir": run_out_dir,
            "best_model": best_model_path,
        }
        all_results.append(res_entry)

    # Selection: Maximize Dice + 2 * (ThinCrackDice - 0.4230)
    best_a1 = max(all_results, key=lambda x: x["score"])

    print("\n" + "=" * 80)
    print("STAGE 1B: A1 (SoftBIoU) SWEEP SUMMARY TABLE")
    print(f"{'Run':<8} | {'Lambda':<8} | {'d':<4} | {'Dice':<8} | {'Precision':<10} | {'Recall':<8} | {'ThinDice':<10} | {'Objective Score'}")
    print("-" * 80)
    for r in all_results:
        is_sel = " <-- [SELECTED lambda*]" if r["run_id"] == best_a1["run_id"] else ""
        print(f"{r['run_id']:<8} | {r['lambda']:<8.3f} | {r['d']:<4} | {r['dice']:<8.4f} | {r['precision']:<10.4f} | {r['recall']:<8.4f} | {r['thin_dice']:<10.4f} | {r['score']:<12.4f}{is_sel}")
    print("=" * 80)

    summary_file = os.path.join(args.output_dir, "stage1b_a1_sweep_results.json")
    with open(summary_file, "w") as f:
        json.dump({"runs": all_results, "selected_best_a1": best_a1}, f, indent=2)
    print(f"Stage 1B results saved to {summary_file}")
    return best_a1


# =============================================================================
# STAGE 2: COMBINATION (A1* + B1*)
# =============================================================================
def generate_combined_config(
    lambda_b1: float,
    r_b1: int,
    lambda_a1: float,
    d_a1: int,
    output_cfg_path: str,
    output_run_dir: str,
    data_root: str,
) -> str:
    """Generates YAML configuration for combined A1* + B1* training."""
    cfg_content = {
        "model": "B2",
        "num_transformer_layers": 4,
        "p3_mode": "C",
        "seed": 42,
        "img_size": 448,
        "batch_size": 14,
        "num_workers": 2,
        "epochs": 35,
        "stage1_epochs": 17,
        "warmup_epochs": 3,
        "patience": 8,
        "two_stage": True,
        "lr": 1e-4,
        "sage_lr": 2e-4,
        "p3_lr": 1e-4,
        "gamma_init": 0.01,
        "stage2_base_lr": 1e-4,
        "stage2_shared_lr": 1e-4,
        "stage2_p3_lr": 1e-4,
        "stage2_sage_lr": 1e-4,
        "output_dir": output_run_dir,
        "mask_suffix": "",
        "smart_filter": True,
        "crop_mode": "random",
        "ab_bpl_weight": float(lambda_b1),
        "ab_bpl_dilation": int(r_b1),
        "boundary_iou_weight": float(lambda_a1),
        "boundary_iou_dilation": int(d_a1),
        "sage_config": {
            "top_k": 2,
            "gating_type": "sigmoid",
            "shared_expert_indices": [0, 1, 2, 3],
            "router_hidden_dim": 64,
            "load_balance_factor": 0.01,
            "logit_modulation": True,
            "expert_dropout": 0.1,
            "fusion_type": "residual",
            "residual_scale": 0.1,
        },
        "root_dir": data_root,
        "train": {"images": "train/images", "masks": "train/masks"},
        "val": {"images": "val/images", "masks": "val/masks"},
        "test": {"images": "test/images", "masks": "test/masks"},
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_cfg_path)), exist_ok=True)
    with open(output_cfg_path, "w") as f:
        yaml.dump(cfg_content, f, default_flow_style=False, sort_keys=False)
    print(f"[Config Generated] Saved combined configuration to {output_cfg_path}")
    return output_cfg_path


def run_stage_2(
    args: argparse.Namespace,
    best_b1: Dict[str, Any],
    best_a1: Dict[str, Any],
    stage1_ckpt: str,
) -> Tuple[str, str, bool]:
    lam_a1 = args.lambda_a1 if getattr(args, "lambda_a1", None) is not None else best_a1["lambda"]
    dil_a1 = args.dilation_a1 if getattr(args, "dilation_a1", None) is not None else best_a1.get("d", 2)
    lam_b1 = args.lambda_b1 if getattr(args, "lambda_b1", None) is not None else best_b1["lambda"]
    dil_b1 = args.dilation_b1 if getattr(args, "dilation_b1", None) is not None else best_b1.get("r", 2)

    print("\n" + "#" * 80)
    print("STAGE 2: COMBINATION EXPERIMENT (Run 2A: A1* + B1*)")
    print(f"Optimal A1*: lambda={lam_a1}, d={dil_a1} (Base reference: {best_a1.get('run_id', 'a1_v0')})")
    print(f"Optimal B1*: lambda={lam_b1}, r={dil_b1} (Base reference: {best_b1.get('run_id', 'b1_v0')})")
    print("#" * 80)

    best_standalone_dice = max(best_a1.get("dice", 0.7685), best_b1.get("dice", 0.7685))
    print(f"Baseline to beat (Best Standalone Dice): {best_standalone_dice:.4f}")

    run_dir = os.path.join(args.output_dir, "phase6_comb_stage2a_a1_b1")
    cfg_path = os.path.join(args.output_dir, "configs", "comb_a1_b1.yaml")
    generate_combined_config(
        lambda_b1=lam_b1,
        r_b1=dil_b1,
        lambda_a1=lam_a1,
        d_a1=dil_a1,
        output_cfg_path=cfg_path,
        output_run_dir=run_dir,
        data_root=args.data_root,
    )

    completion_file = os.path.join(run_dir, "stage2_completion.json")
    if args.skip_completed and os.path.exists(completion_file):
        print("[Run 2A] Found existing stage2_completion.json. Skipping training.")
    else:
        cmd = [
            sys.executable,
            os.path.join(PROJECT_ROOT, "scripts", "train_crack.py"),
            "--config", cfg_path,
            "--stage2-only",
            "--checkpoint", stage1_ckpt,
            "--output-dir", run_dir,
            "--data-root", args.data_root,
        ]
        run_command(cmd, "Stage 2A Combination Training [A1* + B1*]")

    with open(completion_file, "r") as f:
        stats = json.load(f)

    comb_dice = stats.get("best_dice", stats.get("dice", 0.0))
    comb_prec = stats.get("precision", 0.0)
    comb_rec = stats.get("recall", 0.0)
    best_stage2_ckpt = os.path.join(run_dir, "best_model_b2_stage2.pth")

    # Invariant Evaluation:
    # PASS: Dice >= max(A1, B1) standalone AND Recall >= 0.838
    is_pass = (comb_dice >= best_standalone_dice) and (comb_rec >= 0.838)

    print("\n" + "=" * 80)
    print("STAGE 2A EVALUATION VERDICT:")
    print(f"  Combined Dice:      {comb_dice:.4f} (Best Standalone: {best_standalone_dice:.4f})")
    print(f"  Combined Precision: {comb_prec:.4f}")
    print(f"  Combined Recall:    {comb_rec:.4f} (Constraint: >= 0.838)")
    print(f"  VERDICT:            {'>>> PASS <<<' if is_pass else '>>> FAIL (Conflict detected) <<<'}")
    print("=" * 80)

    summary_file = os.path.join(args.output_dir, "stage2_combination_results.json")
    with open(summary_file, "w") as f:
        json.dump({
            "stage2a": {
                "dice": comb_dice,
                "precision": comb_prec,
                "recall": comb_rec,
                "best_standalone_dice": best_standalone_dice,
                "passed": is_pass,
                "checkpoint": best_stage2_ckpt,
                "config": cfg_path,
            }
        }, f, indent=2)

    best_combined_base = os.path.join(args.output_dir, "best_combined_base.pth")
    if is_pass:
        import shutil
        shutil.copyfile(best_stage2_ckpt, best_combined_base)
        print(f"[Saved Base] Exported {best_combined_base} from Run 2A.")
        chosen_config = cfg_path
    else:
        # Fallback to best standalone base model
        chosen_standalone = best_b1 if best_b1["dice"] >= best_a1["dice"] else best_a1
        fallback_model = chosen_standalone.get("best_model", None)
        if fallback_model and os.path.exists(fallback_model):
            import shutil
            shutil.copyfile(fallback_model, best_combined_base)
            print(f"[Fallback Base] Exported {best_combined_base} from best standalone {chosen_standalone['run_id']}.")
        else:
            # If standalone was reference run v0, use global Candidate B checkpoint
            cand_b_global = os.path.join(args.checkpoint_dir, "P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth")
            if not os.path.exists(cand_b_global):
                download_with_progress(
                    "https://raw.githubusercontent.com/bach0823/chuyendettnt/main/results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth",
                    cand_b_global
                )
            import shutil
            shutil.copyfile(cand_b_global, best_combined_base)
            print(f"[Fallback Base] Using Candidate B global checkpoint as base.")
        chosen_config = os.path.join(PROJECT_ROOT, "configs", "p3_ablation", "b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml")

    return best_combined_base, chosen_config, is_pass


# =============================================================================
# STAGE 3: STACK S2-GATE-V2 (CONV3X3) -> CANDIDATE C
# =============================================================================
def run_stage_3(
    args: argparse.Namespace,
    base_model_path: str,
    base_config_path: str,
    v1_weights_path: str,
) -> None:
    print("\n" + "#" * 80)
    print("STAGE 3: STACK S2-GATE-V2 ON TOP OF BEST COMBINED BASE")
    print(f"Base Model:       {base_model_path}")
    print(f"Base Config:      {base_config_path}")
    print(f"Warm-Start Wts:   {v1_weights_path}")
    print("#" * 80)

    s3_out_dir = args.stage3_out_dir if getattr(args, "stage3_out_dir", None) else os.path.join(args.output_dir, "final_candidate_c")
    os.makedirs(s3_out_dir, exist_ok=True)

    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "scripts", "diagnostics", "train_eval_phase6_u1_s2g_v2.py"),
        "--config", base_config_path,
        "--checkpoint", base_model_path,
        "--v1_weights", v1_weights_path,
        "--data_root", args.data_root,
        "--out_dir", s3_out_dir,
        "--epochs", "8",
        "--seed", "42",
    ]
    run_command(cmd, "Stage 3: S2-Gate-v2 Spatial Modulation -> Candidate C Synthesis")

    final_ckpt = os.path.join(s3_out_dir, "final_candidate_c.pth")
    if os.path.exists(final_ckpt):
        print("\n" + "*" * 80)
        print("PIPELINE COMBINATION EXPERIMENT COMPLETED SUCCESSFULLY!")
        print(f"FINAL CANDIDATE C MODEL SAVED AT: {final_ckpt}")
        print("*" * 80 + "\n")


# =============================================================================
# MAIN CLI DISPATCHER
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="SAGE-Lite Phase 6 Final Pipeline Combination")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["1a", "1b", "stage1", "stage2", "stage3", "all"],
        help="Pipeline stage to execute (default: all)",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=None,
        choices=list(RUN_CONFIG_MAP.keys()),
        help="Specify specific run IDs to execute in parallel across multiple Colabs (e.g. --runs a1_a2_v1 a1_a2_v2)",
    )
    parser.add_argument("--data_root", type=str, default="datasets/Crack500_ready", help="Path to Crack500 dataset")
    parser.add_argument("--output_dir", type=str, default="results/phase6_combination", help="Root directory for outputs")
    parser.add_argument("--checkpoint_dir", type=str, default="results/checkpoints", help="Directory containing base checkpoints")
    parser.add_argument("--candidate_b_stage1", type=str, default=None, help="Explicit path to Candidate B stage 1 checkpoint")
    parser.add_argument("--s2g_v1_weights", type=str, default=None, help="Explicit path to U1-S2G v1 weights")
    parser.add_argument("--skip_completed", action="store_true", help="Skip runs that have already produced stage2_completion.json")
    parser.add_argument("--lambda_a1", type=float, default=None, help="Override SoftBIoU lambda for Stage 2 (default: from sweep best or 0.50)")
    parser.add_argument("--dilation_a1", type=int, default=None, help="Override SoftBIoU dilation kernel d for Stage 2 (default: 2)")
    parser.add_argument("--lambda_b1", type=float, default=None, help="Override AB-BPL lambda for Stage 2 (default: from sweep best or 0.04)")
    parser.add_argument("--dilation_b1", type=int, default=None, help="Override AB-BPL dilation radius r for Stage 2 (default: 2)")
    parser.add_argument("--base_model", type=str, default=None, help="Explicit base model checkpoint path for Stage 3")
    parser.add_argument("--base_config", type=str, default=None, help="Explicit base model config path for Stage 3")
    parser.add_argument("--stage3_out_dir", type=str, default=None, help="Explicit output directory for Stage 3")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using device: {device}")

    # Auto-detect Crack500 location if not present at requested data_root
    if not os.path.exists(os.path.join(args.data_root, "val", "images")):
        candidates = [
            "/content/dataset/Crack500",
            "/content/Crack500",
            "/content/Crack500_ready",
            os.path.join(PROJECT_ROOT, "datasets", "Crack500_ready"),
            os.path.join(PROJECT_ROOT, "datasets", "Crack500"),
            os.path.join(PROJECT_ROOT, "..", "datasets", "Crack500_ready"),
            os.path.join(PROJECT_ROOT, "..", "dataset", "Crack500"),
        ]
        for c in candidates:
            if os.path.exists(os.path.join(c, "val", "images")):
                print(f"[Auto-detect Data Root] Switched data_root: {args.data_root} -> {c}")
                args.data_root = c
                break

    # Ensure Candidate B Stage 1 Checkpoint exists
    stage1_ckpt = args.candidate_b_stage1
    if not stage1_ckpt:
        stage1_ckpt = os.path.join(args.checkpoint_dir, "P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_stage1.pth")
    if not os.path.exists(stage1_ckpt):
        download_with_progress(URL_CANDIDATE_B_STAGE1, stage1_ckpt)

    # Ensure U1-S2G v1 weights exist
    v1_weights = args.s2g_v1_weights
    if not v1_weights:
        v1_weights = os.path.join(PROJECT_ROOT, "results", "diagnostics", "phase6_u1_s2g", "u1_s2g_weights.pth")
    if not os.path.exists(v1_weights):
        download_with_progress(URL_U1_S2G_V1_WEIGHTS, v1_weights)

    # Multi-Colab parallel run mode: if --runs provided, execute those and exit
    if args.runs:
        run_custom_runs(args, args.runs, stage1_ckpt, device)
        return

    best_b1 = REF_B1_V0
    best_a1 = REF_A1_V0

    # Execute Stages
    if args.stage in ["1a", "stage1", "all"]:
        best_b1 = run_stage_1a(args, stage1_ckpt)

    if args.stage in ["1b", "stage1", "all"]:
        best_a1 = run_stage_1b(args, stage1_ckpt, device)

    best_base_path = os.path.join(args.output_dir, "best_combined_base.pth")
    base_config_path = os.path.join(args.output_dir, "configs", "comb_a1_b1.yaml")

    if args.stage in ["stage2", "all"]:
        # Load best Stage 1 parameters if available
        s1a_file = os.path.join(args.output_dir, "stage1a_b1_sweep_results.json")
        if os.path.exists(s1a_file):
            with open(s1a_file, "r") as f:
                best_b1 = json.load(f)["selected_best_b1"]

        s1b_file = os.path.join(args.output_dir, "stage1b_a1_sweep_results.json")
        if os.path.exists(s1b_file):
            with open(s1b_file, "r") as f:
                best_a1 = json.load(f)["selected_best_a1"]

        best_base_path, base_config_path, passed = run_stage_2(args, best_b1, best_a1, stage1_ckpt)

    if args.stage in ["stage3", "all"]:
        if getattr(args, "base_model", None):
            best_base_path = args.base_model
            base_config_path = args.base_config if getattr(args, "base_config", None) else os.path.join(PROJECT_ROOT, "configs", "p3_ablation", "b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml")
        elif not os.path.exists(best_base_path):
            print(f"[Info] best_combined_base.pth not found at {best_base_path}. Using Candidate B global checkpoint.")
            cand_b_global = os.path.join(args.checkpoint_dir, "P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth")
            if not os.path.exists(cand_b_global):
                download_with_progress(
                    "https://raw.githubusercontent.com/bach0823/chuyendettnt/main/results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth",
                    cand_b_global
                )
            best_base_path = cand_b_global
            base_config_path = os.path.join(PROJECT_ROOT, "configs", "p3_ablation", "b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml")

        run_stage_3(args, best_base_path, base_config_path, v1_weights)


if __name__ == "__main__":
    main()
