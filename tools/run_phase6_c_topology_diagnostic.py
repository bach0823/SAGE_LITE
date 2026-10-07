#!/usr/bin/env python3
"""
tools/run_phase6_c_topology_diagnostic.py

Phase 6-C Pre-intervention Diagnostic: Topology, Connectivity & Morphological Failure Phenotyping.
Evaluates all 348 validation samples of Crack500 in canonical Setting A across 4 models:
1. Candidate B (Base Canonical)
2. Phase 6-A.1 (BoundaryIoU Objective Probe)
3. Phase 6-A.2 (Pure PLU Representation Probe)
4. Phase 6-B.1 (AB-BPL Boundary Margin Probe)

Measures:
1. Connected Component profile: gt_cc, pred_cc, signed_cc_error, abs_cc_error
2. False-bridge / merge events: bridge_events, merged_gt_components, false_bridge_rate
3. Fragmentation / breakage events: fragmented_gt_components, extra_pred_fragments, breakage_rate
4. Spurious islands: spurious_island_count, spurious_island_rate, spurious_area
5. Centerline Dice (clDice): hard skeletonization on GT and Pred, tprec, tsens, cldice
6. Sample Phenotype classification: clean, merge, fragment, island, mixed
7. Cross-tabulation with Candidate B error taxonomy (BM_127, Thin_66, Thin-low-area_5, etc.)

Outputs:
- results/diagnostics/phase6_c_topology/topology_per_sample_paired.csv
- results/diagnostics/phase6_c_topology/topology_summary.json
- results/diagnostics/phase6_c_topology/topology_cross_tabulation.csv
- results/diagnostics/phase6_c_topology/TOPOLOGY_PRE_INTERVENTION_REPORT.md
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

# Ensure SAGE_LITE project root is on sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Also ensure SpecialSubjectTTNT is accessible
ss_root = os.path.abspath(os.path.join(project_root, ".."))
if ss_root not in sys.path:
    sys.path.insert(0, ss_root)

from sage.networks import create_b2_unet
from scripts.evaluate_crack_official import predict_full_image_tiling_setting_a


def compute_topology_metrics(pred_bin: np.ndarray, target_bin: np.ndarray) -> Dict[str, Any]:
    """
    Computes rigorous topology and connectivity metrics for a single prediction and target mask.
    Convention: 8-connectivity via cv2.connectedComponents.
    """
    assert pred_bin.shape == target_bin.shape, f"Shape mismatch: {pred_bin.shape} vs {target_bin.shape}"
    H, W = target_bin.shape[:2]

    num_gt_cc, gt_labels = cv2.connectedComponents(target_bin.astype(np.uint8), connectivity=8)
    num_pred_cc, pred_labels = cv2.connectedComponents(pred_bin.astype(np.uint8), connectivity=8)

    gt_cc = int(max(num_gt_cc - 1, 0))
    pred_cc = int(max(num_pred_cc - 1, 0))
    signed_cc_error = int(pred_cc - gt_cc)
    abs_cc_error = int(abs(signed_cc_error))

    # Basic pixel metrics
    tp = int(np.sum((pred_bin == 1) & (target_bin == 1)))
    fp = int(np.sum((pred_bin == 1) & (target_bin == 0)))
    fn = int(np.sum((pred_bin == 0) & (target_bin == 1)))
    dice = float((2.0 * tp) / (2.0 * tp + fp + fn + 1e-8))
    precision = float(tp / (tp + fp + 1e-8))
    recall = float(tp / (tp + fn + 1e-8))
    gt_area = int(np.sum(target_bin == 1))
    pred_area = int(np.sum(pred_bin == 1))

    # 1. False-Bridge / Merge Phenotype
    bridge_events = 0
    merged_gt_set = set()
    for p_id in range(1, pred_cc + 1):
        overlapping_gt = np.unique(gt_labels[pred_labels == p_id])
        overlapping_gt = overlapping_gt[overlapping_gt > 0]
        if len(overlapping_gt) >= 2:
            bridge_events += 1
            for g_id in overlapping_gt:
                merged_gt_set.add(int(g_id))
    merged_gt_components = len(merged_gt_set)
    false_bridge_rate = float(merged_gt_components / max(gt_cc, 1))

    # 2. Fragmentation / Breakage Phenotype
    fragmented_gt_components = 0
    extra_pred_fragments = 0
    for g_id in range(1, gt_cc + 1):
        overlapping_pred = np.unique(pred_labels[gt_labels == g_id])
        overlapping_pred = overlapping_pred[overlapping_pred > 0]
        if len(overlapping_pred) >= 2:
            fragmented_gt_components += 1
            extra_pred_fragments += int(len(overlapping_pred) - 1)
    breakage_rate = float(fragmented_gt_components / max(gt_cc, 1))

    # 3. Spurious Islands
    spurious_island_count = 0
    spurious_area = 0
    for p_id in range(1, pred_cc + 1):
        overlapping_gt = np.unique(gt_labels[pred_labels == p_id])
        overlapping_gt = overlapping_gt[overlapping_gt > 0]
        if len(overlapping_gt) == 0:
            spurious_island_count += 1
            spurious_area += int(np.sum(pred_labels == p_id))
    spurious_island_rate = float(spurious_island_count / max(pred_cc, 1))

    # 4. Centerline Dice (clDice) with hard skeletonization
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

    if tprec + tsens == 0.0:
        cldice = 0.0
    else:
        cldice = float(2.0 * tprec * tsens / (tprec + tsens))

    # 5. Phenotype Classification
    has_merge = (bridge_events > 0)
    has_fragment = (fragmented_gt_components > 0)
    has_island = (spurious_island_count > 0)

    active_count = sum([has_merge, has_fragment, has_island])
    if active_count == 0:
        phenotype = "clean"
    elif active_count > 1:
        phenotype = "mixed"
    elif has_merge:
        phenotype = "merge"
    elif has_fragment:
        phenotype = "fragment"
    else:
        phenotype = "island"

    return {
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "gt_area": gt_area,
        "pred_area": pred_area,
        "gt_cc": gt_cc,
        "pred_cc": pred_cc,
        "signed_cc_error": signed_cc_error,
        "abs_cc_error": abs_cc_error,
        "bridge_events": bridge_events,
        "merged_gt_components": merged_gt_components,
        "false_bridge_rate": false_bridge_rate,
        "fragmented_gt_components": fragmented_gt_components,
        "extra_pred_fragments": extra_pred_fragments,
        "breakage_rate": breakage_rate,
        "spurious_island_count": spurious_island_count,
        "spurious_island_rate": spurious_island_rate,
        "spurious_area": spurious_area,
        "cldice": cldice,
        "tprec": tprec,
        "tsens": tsens,
        "has_merge": has_merge,
        "has_fragment": has_fragment,
        "has_island": has_island,
        "phenotype": phenotype,
    }


def load_model_from_checkpoint(config_path: str, checkpoint_path: str, device: torch.device, turn_off_asdw: bool = True):
    """
    Initializes B2 UNet and loads checkpoint weights.
    By default (turn_off_asdw=True), ASDW refinement on Stage 0 and Stage 1 is bypassed
    with nn.Identity(), following the official deprecation of P3-C (Interim ASDW-OFF Candidate B).
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    num_layers = int(cfg.get("num_transformer_layers", 4))
    p3_mode = cfg.get("p3_mode", "C")
    img_size = int(cfg.get("img_size", 448))
    sage_cfg = cfg.get("sage_config", {})
    use_plu_head = cfg.get("use_plu_head", False)
    use_cgsr = cfg.get("use_cgsr", cfg.get("cgsr", False))
    cgsr_init_bias = float(cfg.get("cgsr_init_bias", 3.0))
    use_point_rend = cfg.get("use_point_rend", False)
    point_rend_mid_channels = int(cfg.get("point_rend_mid_channels", 128))
    point_rend_train_points = int(cfg.get("point_rend_train_points", 2048))
    point_rend_subdivision_points = int(cfg.get("point_rend_subdivision_points", 8192))
    use_oriented_strip_pooling = cfg.get("use_oriented_strip_pooling", False)
    use_tangent_head = cfg.get("use_tangent_head", False) or (float(cfg.get("tangent_weight", 0.0)) > 0.0)
    use_s2_gate = cfg.get("use_s2_gate", False)
    s2_gate_kernel_size = int(cfg.get("s2_gate_kernel_size", 3))
    use_s2_gate_block2 = cfg.get("use_s2_gate_block2", False)
    s2_gate_block2_kernel_size = int(cfg.get("s2_gate_block2_kernel_size", 3))

    model = create_b2_unet(
        num_transformer_layers=num_layers,
        p3_mode=p3_mode,
        img_size=img_size,
        sage_config=sage_cfg,
        use_plu_head=use_plu_head,
        use_cgsr=use_cgsr,
        cgsr_init_bias=cgsr_init_bias,
        use_point_rend=use_point_rend,
        point_rend_mid_channels=point_rend_mid_channels,
        point_rend_train_points=point_rend_train_points,
        point_rend_subdivision_points=point_rend_subdivision_points,
        use_oriented_strip_pooling=use_oriented_strip_pooling,
        use_tangent_head=use_tangent_head,
        use_s2_gate=use_s2_gate,
        s2_gate_kernel_size=s2_gate_kernel_size,
        use_s2_gate_block2=use_s2_gate_block2,
        s2_gate_block2_kernel_size=s2_gate_block2_kernel_size,
        pretrained=False,
    ).to(device)

    if not os.path.exists(checkpoint_path):
        print(f"[Warning] Checkpoint {checkpoint_path} not found.")
        # Auto-fallback candidates for Colab/fresh environments
        fallback_cand_b = "/content/results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth"
        fallback_u1 = "/content/SAGE_LITE/results/diagnostics/phase6_u1_s2g/u1_s2g_weights.pth"
        if os.path.exists(fallback_cand_b):
            print(f"[Fallback] Loading base weights from Candidate B: {fallback_cand_b}")
            ckpt_b = torch.load(fallback_cand_b, map_location=device, weights_only=False)
            sd_b = ckpt_b["model_state_dict"] if "model_state_dict" in ckpt_b else ckpt_b
            model.load_state_dict(sd_b, strict=False)
            if use_s2_gate and model.decoder.s2_gate is not None and os.path.exists(fallback_u1):
                print(f"[Fallback] Initializing S2-Gate Block 1 from {fallback_u1}")
                u1_dict = torch.load(fallback_u1, map_location=device, weights_only=False)
                u1_sd = u1_dict["s2_gate_state"] if "s2_gate_state" in u1_dict else u1_dict
                model.decoder.s2_gate.warm_start_from_v1(u1_sd)
            checkpoint_path = fallback_cand_b
        else:
            raise FileNotFoundError(f"Neither {checkpoint_path} nor fallback {fallback_cand_b} exists!")

    if os.path.exists(checkpoint_path) and "P3_C_D4_K2_H64" not in checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)

    if turn_off_asdw and hasattr(model.backbone, "convnext"):
        for stage in model.backbone.convnext.stages[:2]:
            if hasattr(stage, "p3_refinement") and stage.p3_refinement is not None:
                stage.p3_refinement = nn.Identity()

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Phase 6-C Pre-intervention Topology Diagnostic")
    parser.add_argument("--data-root", type=str, default="datasets/Crack500_ready", help="Path to Crack500_ready")
    parser.add_argument("--output-dir", type=str, default="results/diagnostics/phase6_c_topology", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for Setting A inference")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of validation samples for quick verification")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.enabled = False

    print("=" * 80)
    print("PHASE 6-C PRE-INTERVENTION TOPOLOGY DIAGNOSTIC (4 MODELS PAIRED)")
    print("=" * 80)
    print(f"Data Root:   {args.data_root}")
    print(f"Output Dir:  {args.output_dir}")
    print(f"Device:      {args.device}")

    # Discover Validation Pairs
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

    print(f"Discovered {len(val_pairs)} validation image-mask pairs.")
    assert len(val_pairs) == 348, f"Expected 348 validation samples, found {len(val_pairs)}!"

    if args.limit is not None and args.limit > 0:
        val_pairs = val_pairs[:args.limit]
        print(f"[TEST MODE] Limited to first {len(val_pairs)} samples.")

    # Load Candidate B Baseline Taxonomy & Per-Sample Metrics
    candidate_b_csv = "results/P3_C_Routing_Diagnostics_D4_K2_H64_Phase5_SAGELR2e-4/diagnostics/error_analysis/per_sample_metrics.csv"
    if not os.path.exists(candidate_b_csv):
        candidate_b_csv = "results/P3_C_Routing_Diagnostics_D4/error_analysis/per_sample_metrics.csv"
    df_candidate_b = pd.read_csv(candidate_b_csv)
    b_taxonomy_map = dict(zip(df_candidate_b["case_name"], df_candidate_b["primary_error_category"]))
    b_thinness_map = dict(zip(df_candidate_b["case_name"], df_candidate_b["thinness_score"]))
    b_gt_area_map = dict(zip(df_candidate_b["case_name"], df_candidate_b["gt_area"]))

    models_meta = [
        {
            "tag": "Base",
            "name": "Candidate B (Base)",
            "config": "results/configs/b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml",
            "ckpt": "results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth",
        },
        {
            "tag": "A1",
            "name": "Phase 6-A.1 (BoundaryIoU)",
            "config": "results/configs/b2_p3_run_c_d4_k2_h64_phase6_a1_boundary_iou.yaml",
            "ckpt": "results/checkpoints/P3_C_Phase6_A1_BoundaryIoU_D4_K2_H64_best_model_b2_global.pth",
        },
        {
            "tag": "A2",
            "name": "Phase 6-A.2 (Pure PLU)",
            "config": "results/configs/b2_p3_run_c_d4_k2_h64_phase6_a2_pure_plu.yaml",
            "ckpt": "results/checkpoints/P3_C_Phase6_A2_Pure_PLU_D4_K2_H64_best_model_b2_global.pth",
        },
        {
            "tag": "B1",
            "name": "Phase 6-B.1 (AB-BPL)",
            "config": "results/configs/b2_p3_run_c_d4_k2_h64_phase6_b1_ab_bpl.yaml",
            "ckpt": "results/checkpoints/P3_C_Phase6_B1_AB_BPL_D4_K2_H64_best_model_b2_global.pth",
        },
    ]

    # Pre-load targets into memory to ensure 100% exact same reference
    print("\nPre-loading 348 validation targets...")
    targets_dict = {}
    images_dict = {}
    for img_path, mask_path, case_name in val_pairs:
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        target = (mask > 0).astype(np.uint8)
        targets_dict[case_name] = target
        images_dict[case_name] = img_rgb

    # Run inference and topology analysis for each model
    per_model_results = {}
    for m in models_meta:
        tag = m["tag"]
        name = m["name"]
        print(f"\nEvaluating Model: {name} (tag='{tag}')...")
        model = load_model_from_checkpoint(m["config"], m["ckpt"], device)

        m_rows = []
        t0 = time.time()
        for img_path, mask_path, case_name in tqdm(val_pairs, desc=f"Inference [{tag}]"):
            img_rgb = images_dict[case_name]
            target = targets_dict[case_name]

            with torch.no_grad():
                logits_np = predict_full_image_tiling_setting_a(
                    model, img_rgb, device, tile_size=448, batch_size=args.batch_size
                )
            pred = (logits_np > 0.0).astype(np.uint8)
            H, W = target.shape[:2]
            pred = pred[:H, :W]

            metrics = compute_topology_metrics(pred, target)
            metrics["case_name"] = case_name
            m_rows.append(metrics)

        t1 = time.time()
        print(f"Completed {name} in {t1 - t0:.1f}s (avg {(t1 - t0) / 348:.3f}s/img).")
        per_model_results[tag] = pd.DataFrame(m_rows)
        # Release model GPU memory
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Build Paired Joined DataFrame
    print("\nConstructing Paired Master DataFrame...")
    paired_rows = []
    base_df = per_model_results["Base"]

    for idx in range(len(base_df)):
        case_name = base_df.loc[idx, "case_name"]
        gt_cc = int(base_df.loc[idx, "gt_cc"])
        cat = b_taxonomy_map.get(case_name, "moderate_general_error")
        thinness = float(b_thinness_map.get(case_name, 0.0))
        gt_area = int(b_gt_area_map.get(case_name, 0))

        is_thin_66 = bool(thinness > 0.20)
        is_thin_low_area_5 = bool(cat == "thin_low_area_failure")
        is_bm_127 = bool(cat == "boundary_margin_error")

        row = {
            "case_name": case_name,
            "candidate_b_category": cat,
            "thinness_score": thinness,
            "gt_area": gt_area,
            "is_thin_66": is_thin_66,
            "is_thin_low_area_5": is_thin_low_area_5,
            "is_bm_127": is_bm_127,
            "gt_cc": gt_cc,
        }

        for tag in ["Base", "A1", "A2", "B1"]:
            m_df = per_model_results[tag]
            m_row = m_df[m_df["case_name"] == case_name].iloc[0]

            row[f"{tag}_dice"] = float(m_row["dice"])
            row[f"{tag}_precision"] = float(m_row["precision"])
            row[f"{tag}_recall"] = float(m_row["recall"])
            row[f"{tag}_pred_area"] = int(m_row["pred_area"])
            row[f"{tag}_pred_cc"] = int(m_row["pred_cc"])
            row[f"{tag}_signed_cc_error"] = int(m_row["signed_cc_error"])
            row[f"{tag}_abs_cc_error"] = int(m_row["abs_cc_error"])
            row[f"{tag}_bridge_events"] = int(m_row["bridge_events"])
            row[f"{tag}_merged_gt_components"] = int(m_row["merged_gt_components"])
            row[f"{tag}_false_bridge_rate"] = float(m_row["false_bridge_rate"])
            row[f"{tag}_fragmented_gt_components"] = int(m_row["fragmented_gt_components"])
            row[f"{tag}_extra_pred_fragments"] = int(m_row["extra_pred_fragments"])
            row[f"{tag}_breakage_rate"] = float(m_row["breakage_rate"])
            row[f"{tag}_spurious_island_count"] = int(m_row["spurious_island_count"])
            row[f"{tag}_spurious_island_rate"] = float(m_row["spurious_island_rate"])
            row[f"{tag}_spurious_area"] = int(m_row["spurious_area"])
            row[f"{tag}_cldice"] = float(m_row["cldice"])
            row[f"{tag}_tprec"] = float(m_row["tprec"])
            row[f"{tag}_tsens"] = float(m_row["tsens"])
            row[f"{tag}_phenotype"] = str(m_row["phenotype"])

        paired_rows.append(row)

    paired_df = pd.DataFrame(paired_rows)
    paired_csv_path = os.path.join(args.output_dir, "topology_per_sample_paired.csv")
    paired_df.to_csv(paired_csv_path, index=False)
    print(f"Saved Paired Per-Sample CSV: {paired_csv_path}")

    # Build Global & Stratified Aggregations
    summary_dict = {}
    tags = ["Base", "A1", "A2", "B1"]

    for tag in tags:
        summary_dict[tag] = {
            "global": {
                "dice_mean": float(paired_df[f"{tag}_dice"].mean()),
                "precision_mean": float(paired_df[f"{tag}_precision"].mean()),
                "recall_mean": float(paired_df[f"{tag}_recall"].mean()),
                "cldice_mean": float(paired_df[f"{tag}_cldice"].mean()),
                "cldice_median": float(paired_df[f"{tag}_cldice"].median()),
                "tprec_mean": float(paired_df[f"{tag}_tprec"].mean()),
                "tsens_mean": float(paired_df[f"{tag}_tsens"].mean()),
                "pred_cc_mean": float(paired_df[f"{tag}_pred_cc"].mean()),
                "gt_cc_mean": float(paired_df["gt_cc"].mean()),
                "abs_cc_error_mean": float(paired_df[f"{tag}_abs_cc_error"].mean()),
                "pred_lt_gt_count": int((paired_df[f"{tag}_pred_cc"] < paired_df["gt_cc"]).sum()),
                "pred_lt_gt_pct": float((paired_df[f"{tag}_pred_cc"] < paired_df["gt_cc"]).mean() * 100),
                "pred_eq_gt_count": int((paired_df[f"{tag}_pred_cc"] == paired_df["gt_cc"]).sum()),
                "pred_eq_gt_pct": float((paired_df[f"{tag}_pred_cc"] == paired_df["gt_cc"]).mean() * 100),
                "pred_gt_gt_count": int((paired_df[f"{tag}_pred_cc"] > paired_df["gt_cc"]).sum()),
                "pred_gt_gt_pct": float((paired_df[f"{tag}_pred_cc"] > paired_df["gt_cc"]).mean() * 100),
                "bridge_events_mean": float(paired_df[f"{tag}_bridge_events"].mean()),
                "bridge_events_total": int(paired_df[f"{tag}_bridge_events"].sum()),
                "merged_gt_components_total": int(paired_df[f"{tag}_merged_gt_components"].sum()),
                "false_bridge_rate_mean": float(paired_df[f"{tag}_false_bridge_rate"].mean()),
                "samples_with_bridge_count": int((paired_df[f"{tag}_bridge_events"] > 0).sum()),
                "samples_with_bridge_pct": float((paired_df[f"{tag}_bridge_events"] > 0).mean() * 100),
                "fragmented_gt_components_total": int(paired_df[f"{tag}_fragmented_gt_components"].sum()),
                "extra_pred_fragments_total": int(paired_df[f"{tag}_extra_pred_fragments"].sum()),
                "breakage_rate_mean": float(paired_df[f"{tag}_breakage_rate"].mean()),
                "samples_with_breakage_count": int((paired_df[f"{tag}_fragmented_gt_components"] > 0).sum()),
                "samples_with_breakage_pct": float((paired_df[f"{tag}_fragmented_gt_components"] > 0).mean() * 100),
                "spurious_island_count_total": int(paired_df[f"{tag}_spurious_island_count"].sum()),
                "spurious_island_count_mean": float(paired_df[f"{tag}_spurious_island_count"].mean()),
                "spurious_area_mean": float(paired_df[f"{tag}_spurious_area"].mean()),
                "samples_with_island_count": int((paired_df[f"{tag}_spurious_island_count"] > 0).sum()),
                "samples_with_island_pct": float((paired_df[f"{tag}_spurious_island_count"] > 0).mean() * 100),
                "phenotype_distribution": paired_df[f"{tag}_phenotype"].value_counts().to_dict(),
            }
        }

    summary_json_path = os.path.join(args.output_dir, "topology_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"Saved Topology Summary JSON: {summary_json_path}")

    # Build Cross-Tabulation Matrix
    # Phenotype vs Category for each model
    cross_tab_list = []
    categories = [
        "boundary_margin_error",
        "high_quality",
        "complex_topology",
        "moderate_general_error",
        "false_crack_high_fp",
        "missed_crack_high_fn",
        "over_segmentation",
        "thin_low_area_failure",
        "ALL_SAMPLES",
        "THIN_CRACKS_66",
    ]

    for cat in categories:
        if cat == "ALL_SAMPLES":
            sub_df = paired_df
        elif cat == "THIN_CRACKS_66":
            sub_df = paired_df[paired_df["is_thin_66"]]
        else:
            sub_df = paired_df[paired_df["candidate_b_category"] == cat]

        n_sub = len(sub_df)
        if n_sub == 0:
            continue

        for tag in tags:
            counts = sub_df[f"{tag}_phenotype"].value_counts().to_dict()
            clean_c = counts.get("clean", 0)
            merge_c = counts.get("merge", 0)
            frag_c = counts.get("fragment", 0)
            island_c = counts.get("island", 0)
            mixed_c = counts.get("mixed", 0)

            bridge_samples = int((sub_df[f"{tag}_bridge_events"] > 0).sum())
            frag_samples = int((sub_df[f"{tag}_fragmented_gt_components"] > 0).sum())
            island_samples = int((sub_df[f"{tag}_spurious_island_count"] > 0).sum())

            cross_tab_list.append({
                "stratum": cat,
                "model": tag,
                "n_samples": n_sub,
                "dice_mean": float(sub_df[f"{tag}_dice"].mean()),
                "cldice_mean": float(sub_df[f"{tag}_cldice"].mean()),
                "cldice_median": float(sub_df[f"{tag}_cldice"].median()),
                "pred_cc_mean": float(sub_df[f"{tag}_pred_cc"].mean()),
                "gt_cc_mean": float(sub_df["gt_cc"].mean()),
                "clean_count": clean_c,
                "clean_pct": float(clean_c / n_sub * 100),
                "merge_only_count": merge_c,
                "merge_only_pct": float(merge_c / n_sub * 100),
                "fragment_only_count": frag_c,
                "fragment_only_pct": float(frag_c / n_sub * 100),
                "island_only_count": island_c,
                "island_only_pct": float(island_c / n_sub * 100),
                "mixed_count": mixed_c,
                "mixed_pct": float(mixed_c / n_sub * 100),
                "total_samples_with_bridge": bridge_samples,
                "bridge_sample_pct": float(bridge_samples / n_sub * 100),
                "total_samples_with_breakage": frag_samples,
                "breakage_sample_pct": float(frag_samples / n_sub * 100),
                "total_samples_with_island": island_samples,
                "island_sample_pct": float(island_samples / n_sub * 100),
            })

    cross_tab_df = pd.DataFrame(cross_tab_list)
    cross_tab_csv_path = os.path.join(args.output_dir, "topology_cross_tabulation.csv")
    cross_tab_df.to_csv(cross_tab_csv_path, index=False)
    print(f"Saved Cross-Tabulation CSV: {cross_tab_csv_path}")

    # Generate Markdown Report
    report_path = os.path.join(args.output_dir, "TOPOLOGY_PRE_INTERVENTION_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 6-C Pre-Intervention Topology & Connectivity Diagnostic Report\n\n")
        f.write("**Evaluation Split:** Crack500 Official Validation Split ($N=348$) under Setting A Tiling Protocol\n\n")
        f.write("## 1. Global Topology Profile across 4 Models\n\n")
        f.write("| Metric | Candidate B (Base) | Phase 6-A.1 (B-IoU) | Phase 6-A.2 (Pure PLU) | Phase 6-B.1 (AB-BPL) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")

        g_b = summary_dict["Base"]["global"]
        g_a1 = summary_dict["A1"]["global"]
        g_a2 = summary_dict["A2"]["global"]
        g_b1 = summary_dict["B1"]["global"]

        f.write(f"| **Global Dice** | {g_b['dice_mean']:.4f} | {g_a1['dice_mean']:.4f} | {g_a2['dice_mean']:.4f} | {g_b1['dice_mean']:.4f} |\n")
        f.write(f"| **Centerline Dice (clDice Mean)** | {g_b['cldice_mean']:.4f} | {g_a1['cldice_mean']:.4f} | {g_a2['cldice_mean']:.4f} | {g_b1['cldice_mean']:.4f} |\n")
        f.write(f"| **clDice Median** | {g_b['cldice_median']:.4f} | {g_a1['cldice_median']:.4f} | {g_a2['cldice_median']:.4f} | {g_b1['cldice_median']:.4f} |\n")
        f.write(f"| **Skeleton Precision (Tprec)** | {g_b['tprec_mean']:.4f} | {g_a1['tprec_mean']:.4f} | {g_a2['tprec_mean']:.4f} | {g_b1['tprec_mean']:.4f} |\n")
        f.write(f"| **Skeleton Sensitivity (Tsens)**| {g_b['tsens_mean']:.4f} | {g_a1['tsens_mean']:.4f} | {g_a2['tsens_mean']:.4f} | {g_b1['tsens_mean']:.4f} |\n")
        f.write(f"| **Mean GT Connected Components**| {g_b['gt_cc_mean']:.3f} | {g_a1['gt_cc_mean']:.3f} | {g_a2['gt_cc_mean']:.3f} | {g_b1['gt_cc_mean']:.3f} |\n")
        f.write(f"| **Mean Pred Connected Components**| {g_b['pred_cc_mean']:.3f} | {g_a1['pred_cc_mean']:.3f} | {g_a2['pred_cc_mean']:.3f} | {g_b1['pred_cc_mean']:.3f} |\n")
        f.write(f"| **Mean Absolute CC Error** | {g_b['abs_cc_error_mean']:.3f} | {g_a1['abs_cc_error_mean']:.3f} | {g_a2['abs_cc_error_mean']:.3f} | {g_b1['abs_cc_error_mean']:.3f} |\n")
        f.write(f"| **Pred CC < GT CC (Merged/Missed)** | {g_b['pred_lt_gt_count']} ({g_b['pred_lt_gt_pct']:.1f}%) | {g_a1['pred_lt_gt_count']} ({g_a1['pred_lt_gt_pct']:.1f}%) | {g_a2['pred_lt_gt_count']} ({g_a2['pred_lt_gt_pct']:.1f}%) | {g_b1['pred_lt_gt_count']} ({g_b1['pred_lt_gt_pct']:.1f}%) |\n")
        f.write(f"| **Pred CC == GT CC (Exact Match)** | {g_b['pred_eq_gt_count']} ({g_b['pred_eq_gt_pct']:.1f}%) | {g_a1['pred_eq_gt_count']} ({g_a1['pred_eq_gt_pct']:.1f}%) | {g_a2['pred_eq_gt_count']} ({g_a2['pred_eq_gt_pct']:.1f}%) | {g_b1['pred_eq_gt_count']} ({g_b1['pred_eq_gt_pct']:.1f}%) |\n")
        f.write(f"| **Pred CC > GT CC (Fragment/Island)**| {g_b['pred_gt_gt_count']} ({g_b['pred_gt_gt_pct']:.1f}%) | {g_a1['pred_gt_gt_count']} ({g_a1['pred_gt_gt_pct']:.1f}%) | {g_a2['pred_gt_gt_count']} ({g_a2['pred_gt_gt_pct']:.1f}%) | {g_b1['pred_gt_gt_count']} ({g_b1['pred_gt_gt_pct']:.1f}%) |\n\n")

        f.write("## 2. Disentangled Topological Failure Events\n\n")
        f.write("| Event Type | Candidate B (Base) | Phase 6-A.1 (B-IoU) | Phase 6-A.2 (Pure PLU) | Phase 6-B.1 (AB-BPL) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **Samples with False Bridge** | {g_b['samples_with_bridge_count']} ({g_b['samples_with_bridge_pct']:.1f}%) | {g_a1['samples_with_bridge_count']} ({g_a1['samples_with_bridge_pct']:.1f}%) | {g_a2['samples_with_bridge_count']} ({g_a2['samples_with_bridge_pct']:.1f}%) | {g_b1['samples_with_bridge_count']} ({g_b1['samples_with_bridge_pct']:.1f}%) |\n")
        f.write(f"| **Total Bridge Events** | {g_b['bridge_events_total']} | {g_a1['bridge_events_total']} | {g_a2['bridge_events_total']} | {g_b1['bridge_events_total']} |\n")
        f.write(f"| **Total Merged GT Components** | {g_b['merged_gt_components_total']} | {g_a1['merged_gt_components_total']} | {g_a2['merged_gt_components_total']} | {g_b1['merged_gt_components_total']} |\n")
        f.write(f"| **Samples with Breakage/Frag**| {g_b['samples_with_breakage_count']} ({g_b['samples_with_breakage_pct']:.1f}%) | {g_a1['samples_with_breakage_count']} ({g_a1['samples_with_breakage_pct']:.1f}%) | {g_a2['samples_with_breakage_count']} ({g_a2['samples_with_breakage_pct']:.1f}%) | {g_b1['samples_with_breakage_count']} ({g_b1['samples_with_breakage_pct']:.1f}%) |\n")
        f.write(f"| **Total Fragmented GT Components**| {g_b['fragmented_gt_components_total']} | {g_a1['fragmented_gt_components_total']} | {g_a2['fragmented_gt_components_total']} | {g_b1['fragmented_gt_components_total']} |\n")
        f.write(f"| **Total Extra Fragments** | {g_b['extra_pred_fragments_total']} | {g_a1['extra_pred_fragments_total']} | {g_a2['extra_pred_fragments_total']} | {g_b1['extra_pred_fragments_total']} |\n")
        f.write(f"| **Samples with Spurious Islands** | {g_b['samples_with_island_count']} ({g_b['samples_with_island_pct']:.1f}%) | {g_a1['samples_with_island_count']} ({g_a1['samples_with_island_pct']:.1f}%) | {g_a2['samples_with_island_count']} ({g_a2['samples_with_island_pct']:.1f}%) | {g_b1['samples_with_island_count']} ({g_b1['samples_with_island_pct']:.1f}%) |\n")
        f.write(f"| **Total Spurious Islands Count** | {g_b['spurious_island_count_total']} | {g_a1['spurious_island_count_total']} | {g_a2['spurious_island_count_total']} | {g_b1['spurious_island_count_total']} |\n")
        f.write(f"| **Mean Spurious Island Area (px)**| {g_b['spurious_area_mean']:.1f} | {g_a1['spurious_area_mean']:.1f} | {g_a2['spurious_area_mean']:.1f} | {g_b1['spurious_area_mean']:.1f} |\n\n")

        f.write("## 3. Disentangled Phenotype Distribution (Mutual Exclusive Categories)\n\n")
        f.write("| Phenotype | Candidate B (Base) | Phase 6-A.1 (B-IoU) | Phase 6-A.2 (Pure PLU) | Phase 6-B.1 (AB-BPL) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for pheno in ["clean", "merge", "fragment", "island", "mixed"]:
            c_b = g_b["phenotype_distribution"].get(pheno, 0)
            c_a1 = g_a1["phenotype_distribution"].get(pheno, 0)
            c_a2 = g_a2["phenotype_distribution"].get(pheno, 0)
            c_b1 = g_b1["phenotype_distribution"].get(pheno, 0)
            f.write(f"| **{pheno.capitalize()}** | {c_b} ({c_b/3.48:.1f}%) | {c_a1} ({c_a1/3.48:.1f}%) | {c_a2} ({c_a2/3.48:.1f}%) | {c_b1} ({c_b1/3.48:.1f}%) |\n")

    print(f"Generated Markdown Report: {report_path}")
    print("\n[SUCCESS] Phase 6-C Topology Pre-intervention Diagnostic Completed Successfully!")


if __name__ == "__main__":
    main()
