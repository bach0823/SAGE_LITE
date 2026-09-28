#!/usr/bin/env python3
"""
scripts/diagnostics/analyze_gap_routing_signal.py

Diagnostic-Only Study:
Evaluating whether Global Average Pooling (GAP) is a routing-signal bottleneck for Crack500.

Targets:
- Canonical D4-P3-C Standalone Model (ViT depth=4, 8 experts=4 CNN + 4 ViT, top_k=4).
- Strictly Crack500 validation set (348 samples).
- Eval mode (deterministic, exploration noise OFF).
- Zero model parameter changes, zero architecture modifications.
- Strict checkpoint integrity audit (fails loudly on any missing/unexpected keys).
- Dynamic source code audit of original SAGE router implementation.

Author: Special Subject AI Team
Date: September 2026
Branch: crack500-audit
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import yaml
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sage.components.router import SageRouter
from sage.components.sage_layer import SageLayer
from sage.networks.b2_unet import create_b2_unet, B2ConvNeXtViTUNet
from sage.utils.dataloader import get_dataset_from_config
from sage.utils.training_utils import set_seed, seed_worker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("analyze_gap_routing")


# -----------------------------------------------------------------------------
# Morphology Computation (Reused verbatim from tools/run_p3_c_error_analysis.py)
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Math / Information Theoretic Utilities
# -----------------------------------------------------------------------------

def compute_jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Computes Jensen-Shannon Divergence in bits between two discrete probability distributions."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p_sum = np.sum(p)
    q_sum = np.sum(q)
    if p_sum == 0 or q_sum == 0:
        return 0.0
    p = p / p_sum
    q = q / q_sum
    m = 0.5 * (p + q)

    def kl_div(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2((a[mask] + eps) / (b[mask] + eps)))

    jsd = 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)
    return float(np.clip(jsd, 0.0, 1.0))


def compute_sample_routing_entropy(gating_weights: np.ndarray) -> float:
    """Computes Shannon entropy (base 2) of selected gating weights for a single sample."""
    w = np.asarray(gating_weights, dtype=np.float64)
    w_sum = np.sum(w)
    if w_sum <= 0:
        return 0.0
    p = w / w_sum
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


# -----------------------------------------------------------------------------
# Router Pooled Feature Collector (Non-invasive Hook)
# -----------------------------------------------------------------------------

class RouterPooledFeatureCollector:
    """
    Non-invasive forward hook collector for a SageRouter instance.
    Captures:
    - aggregated_features: immediately after GAP / mean pooling
    - g_s: shared expert gate score
    - base_logits: SAR base affinity scores
    - top_k_indices: selected expert indices
    - gating_weights: gating weights
    """

    def __init__(self, name: str, router_module: SageRouter, router_index: int, layer_type: str):
        self.name = name
        self.router = router_module
        self.router_index = router_index
        self.layer_type = layer_type
        self.hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None

        self.features: List[np.ndarray] = []
        self.gs: List[np.ndarray] = []
        self.base_logits: List[np.ndarray] = []
        self.top_k_indices: List[np.ndarray] = []
        self.gating_weights: List[np.ndarray] = []

    def register(self):
        self.hook_handle = self.router.register_forward_hook(self._hook)

    def remove(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def _hook(self, module: SageRouter, inp: Tuple[torch.Tensor, ...], out: Tuple[Any, ...]):
        x = inp[0]
        # Exact extraction matching user specification:
        if x.dim() == 4:  # CNN router
            aggregated = module.feature_aggregator(x).squeeze(-1).squeeze(-1)
        elif x.dim() == 3:  # Transformer router
            aggregated = x.mean(dim=1)
        else:
            aggregated = x.flatten(1)

        with torch.no_grad():
            g_s = torch.sigmoid(module.shared_expert_gate(aggregated))
            query = module.query_projection(aggregated)
            base_logits = torch.matmul(query, module.expert_keys.T) / module.temperature
            top_k_indices, gating_weights, _ = out

            self.features.append(aggregated.detach().cpu().float().numpy())
            self.gs.append(g_s.detach().cpu().float().numpy())
            self.base_logits.append(base_logits.detach().cpu().float().numpy())
            self.top_k_indices.append(top_k_indices.detach().cpu().numpy())
            self.gating_weights.append(gating_weights.detach().cpu().float().numpy())

    def get_arrays(self) -> Dict[str, np.ndarray]:
        return {
            'features': np.concatenate(self.features, axis=0) if self.features else np.empty((0,)),
            'gs': np.concatenate(self.gs, axis=0).squeeze(-1) if self.gs else np.empty((0,)),
            'base_logits': np.concatenate(self.base_logits, axis=0) if self.base_logits else np.empty((0,)),
            'top_k_indices': np.concatenate(self.top_k_indices, axis=0) if self.top_k_indices else np.empty((0,)),
            'gating_weights': np.concatenate(self.gating_weights, axis=0) if self.gating_weights else np.empty((0,)),
        }


# -----------------------------------------------------------------------------
# Router & Model Discovery
# -----------------------------------------------------------------------------

def discover_routers(model: B2ConvNeXtViTUNet) -> List[Dict[str, Any]]:
    """Dynamically discovers all 8 routers in the D4 model."""
    routers = []
    idx = 0
    if hasattr(model, "backbone") and hasattr(model.backbone, "convnext"):
        for s_idx, stage in enumerate(model.backbone.convnext.stages):
            if isinstance(stage, SageLayer):
                routers.append({
                    "name": f"convnext.stage_{s_idx}",
                    "alias": f"S{s_idx}",
                    "layer_type": "CNN",
                    "router_index": idx,
                    "stage_or_block_idx": s_idx,
                    "router": stage.router,
                })
                idx += 1

    if hasattr(model, "backbone") and hasattr(model.backbone, "transformer_blocks"):
        for b_idx, blk in enumerate(model.backbone.transformer_blocks):
            if isinstance(blk, SageLayer):
                routers.append({
                    "name": f"transformer.block_{b_idx}",
                    "alias": f"B{b_idx}",
                    "layer_type": "ViT",
                    "router_index": idx,
                    "stage_or_block_idx": b_idx,
                    "router": blk.router,
                })
                idx += 1

    return routers


def load_canonical_d4_model(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
    data_root_override: Optional[str] = None,
) -> Tuple[B2ConvNeXtViTUNet, Dict[str, Any], Dict[str, Any]]:
    """
    Loads canonical D4 model and checkpoint safely with strict integrity checks.
    Fails loudly with RuntimeError if there are any missing or unexpected keys.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if data_root_override:
        config["root_dir"] = data_root_override

    vit_depth = int(config.get("num_transformer_layers", 4))
    p3_mode = config.get("p3_mode", "C")
    sage_cfg = config.get("sage_config", {})
    img_size = int(config.get("img_size", 448))

    model = create_b2_unet(
        num_classes=1,
        img_size=img_size,
        num_transformer_layers=vit_depth,
        pretrained=False,
        sage_config=sage_cfg,
        p3_mode=p3_mode,
    ).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    raw_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    ckpt_metadata = {}
    if isinstance(raw_checkpoint, dict):
        for field in ["stage", "epoch", "best_dice", "best_loss", "p3_mode", "num_transformer_layers", "model_type"]:
            if field in raw_checkpoint:
                val = raw_checkpoint[field]
                if isinstance(val, (torch.Tensor, np.generic)):
                    val = val.item()
                ckpt_metadata[field] = val
        state_dict = raw_checkpoint.get("model_state_dict", raw_checkpoint)
    else:
        state_dict = raw_checkpoint

    cleaned_state_dict = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }

    initial_params_count = sum(p.numel() for p in model.parameters())
    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    load_res = model.load_state_dict(cleaned_state_dict, strict=False)

    best_dice_str = f"{ckpt_metadata['best_dice']:.4f}" if ckpt_metadata.get('best_dice') is not None else "N/A"
    logger.info(f"Checkpoint path: {checkpoint_path}")
    logger.info(f"Checkpoint metadata: Epoch={ckpt_metadata.get('epoch')}, Best Dice={best_dice_str}")
    logger.info(f"State dict audit: missing_keys={len(load_res.missing_keys)}, unexpected_keys={len(load_res.unexpected_keys)}")
    logger.info(f"Model parameters: Total={initial_params_count:,}, Trainable={trainable_params_count:,}")

    # STRICT INTEGRITY CHECK: Fail loudly if any key is missing or unexpected
    if load_res.missing_keys or load_res.unexpected_keys:
        err_msg = (
            f"STRICT INTEGRITY CHECK FAILURE: Checkpoint parameter mismatch detected!\n"
            f"Checkpoint path: {checkpoint_path}\n"
            f"Missing keys ({len(load_res.missing_keys)}): {load_res.missing_keys[:10]}\n"
            f"Unexpected keys ({len(load_res.unexpected_keys)}): {load_res.unexpected_keys[:10]}\n"
            f"Aborting evaluation. Diagnostics cannot proceed with mismatched parameters."
        )
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    model.eval()
    return model, config, ckpt_metadata


# -----------------------------------------------------------------------------
# Test 1: Foreground Area Ratio
# -----------------------------------------------------------------------------

def run_test1_foreground_ratio(
    sample_morphologies: List[Dict[str, Any]],
    output_dir: str,
) -> Dict[str, Any]:
    """Test 1 — Quantitative Foreground Area Ratio Evaluation."""
    logger.info("--- Executing Test 1: Foreground Area Ratio ---")
    os.makedirs(output_dir, exist_ok=True)

    r_values = np.array([m['foreground_ratio'] for m in sample_morphologies], dtype=np.float64)
    N = len(r_values)

    q1 = float(np.percentile(r_values, 25))
    q2 = float(np.percentile(r_values, 50))  # median
    q3 = float(np.percentile(r_values, 75))
    q4 = float(np.percentile(r_values, 100))  # max

    pct_below_1 = float(np.mean(r_values < 0.01) * 100.0)
    pct_below_2 = float(np.mean(r_values < 0.02) * 100.0)
    pct_below_5 = float(np.mean(r_values < 0.05) * 100.0)
    pct_below_10 = float(np.mean(r_values < 0.10) * 100.0)

    hist_counts, hist_edges = np.histogram(r_values, bins=10)
    histogram = [
        {
            "bin_min": float(hist_edges[i]),
            "bin_max": float(hist_edges[i + 1]),
            "count": int(hist_counts[i]),
            "percentage": float(hist_counts[i] / N * 100.0)
        }
        for i in range(len(hist_counts))
    ]

    summary = {
        "N": N,
        "mean": float(np.mean(r_values)),
        "std": float(np.std(r_values, ddof=1 if N > 1 else 0)),
        "median": q2,
        "min": float(np.min(r_values)),
        "max": q4,
        "Q1": q1,
        "Q2": q2,
        "Q3": q3,
        "Q4": q4,
        "percentage_samples_below_1pct": pct_below_1,
        "percentage_samples_below_2pct": pct_below_2,
        "percentage_samples_below_5pct": pct_below_5,
        "percentage_samples_below_10pct": pct_below_10,
        "histogram_10bins": histogram,
    }

    # Save CSV
    csv_path = os.path.join(output_dir, "foreground_ratio.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "sample_id", "gt_area", "total_pixels", "foreground_ratio", "perimeter", "thinness_score"])
        for i, m in enumerate(sample_morphologies):
            writer.writerow([
                i,
                m["sample_id"],
                m["gt_area"],
                m["total_pixels"],
                f"{m['foreground_ratio']:.6f}",
                m["perimeter"],
                f"{m['thinness_score']:.6f}",
            ])

    # Save JSON
    json_path = os.path.join(output_dir, "foreground_ratio_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Test 1 Complete: N={N}, Mean={summary['mean']:.4f}, Median={summary['median']:.4f}, "
                f"<1%: {pct_below_1:.1f}%, <2%: {pct_below_2:.1f}%, <5%: {pct_below_5:.1f}%")
    return summary


# -----------------------------------------------------------------------------
# Test 2: Capture Router Pooled Features
# -----------------------------------------------------------------------------

def run_test2_save_features(
    collectors: List[RouterPooledFeatureCollector],
    sample_ids: List[str],
    output_dir: str,
) -> Dict[str, Any]:
    """Test 2 — Save captured router pooled features, g_s, base_logits, and top_k indices."""
    logger.info("--- Executing Test 2: Saving Router Pooled Features (.npz) ---")
    os.makedirs(output_dir, exist_ok=True)

    npz_data = {"sample_ids": np.array(sample_ids)}
    meta = {"sample_count": len(sample_ids), "routers": {}}

    for c in collectors:
        arrays = c.get_arrays()
        prefix = c.name
        npz_data[f"{prefix}_features"] = arrays["features"]
        npz_data[f"{prefix}_gs"] = arrays["gs"]
        npz_data[f"{prefix}_base_logits"] = arrays["base_logits"]
        npz_data[f"{prefix}_topk_indices"] = arrays["top_k_indices"]
        npz_data[f"{prefix}_gating_weights"] = arrays["gating_weights"]

        meta["routers"][c.name] = {
            "alias": getattr(c, "alias", c.name),
            "layer_type": c.layer_type,
            "feature_dim": int(arrays["features"].shape[1]) if arrays["features"].ndim == 2 else 0,
            "num_experts": int(arrays["base_logits"].shape[1]) if arrays["base_logits"].ndim == 2 else 0,
            "top_k": int(arrays["top_k_indices"].shape[1]) if arrays["top_k_indices"].ndim == 2 else 0,
        }

    npz_path = os.path.join(output_dir, "router_pooled_features.npz")
    np.savez_compressed(npz_path, **npz_data)

    meta_path = os.path.join(output_dir, "router_pooled_features_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Test 2 Complete: Saved pooled features to {npz_path} ({os.path.getsize(npz_path) / 1024:.1f} KB)")
    return meta


# -----------------------------------------------------------------------------
# Test 3: Linear Probe (Ridge Regression 5-fold CV)
# -----------------------------------------------------------------------------

def run_test3_linear_probe(
    collectors: List[RouterPooledFeatureCollector],
    sample_morphologies: List[Dict[str, Any]],
    output_dir: str,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Test 3 — Linear Probe on Pooled Features:
    Predicts (1) gt_area (foreground ratio) and (2) thinness_score using Ridge regression.
    Protocol: 5-fold CV with StandardScaler fitted strictly inside each training fold.
    """
    logger.info("--- Executing Test 3: Linear Probe (Ridge Regression 5-fold CV) ---")
    os.makedirs(output_dir, exist_ok=True)

    N = len(sample_morphologies)
    actual_splits = min(n_splits, N) if N >= 2 else 1
    kf = KFold(n_splits=actual_splits, shuffle=True, random_state=seed)

    targets = {
        "gt_area": np.array([m["foreground_ratio"] for m in sample_morphologies], dtype=np.float64),
        "thinness": np.array([m["thinness_score"] for m in sample_morphologies], dtype=np.float64),
    }

    probe_results = {}
    csv_rows = []

    for c in collectors:
        arrays = c.get_arrays()
        X = arrays["features"]  # [N, D]
        router_alias = getattr(c, "alias", c.name)
        probe_results[router_alias] = {
            "name": c.name,
            "layer_type": c.layer_type,
            "feature_dim": X.shape[1],
            "targets": {},
        }

        for target_name, y in targets.items():
            r2_scores = []
            mae_scores = []
            rmse_scores = []

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)

                reg = Ridge(alpha=1.0)
                reg.fit(X_train_scaled, y_train)
                y_pred = reg.predict(X_val_scaled)

                r2 = r2_score(y_val, y_pred) if len(y_val) > 1 else 0.0
                mae = mean_absolute_error(y_val, y_pred)
                rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))

                r2_scores.append(r2)
                mae_scores.append(mae)
                rmse_scores.append(rmse)

            target_summary = {
                "r2_mean": float(np.mean(r2_scores)),
                "r2_std": float(np.std(r2_scores, ddof=1 if len(r2_scores) > 1 else 0)),
                "mae_mean": float(np.mean(mae_scores)),
                "mae_std": float(np.std(mae_scores, ddof=1 if len(mae_scores) > 1 else 0)),
                "rmse_mean": float(np.mean(rmse_scores)),
                "rmse_std": float(np.std(rmse_scores, ddof=1 if len(rmse_scores) > 1 else 0)),
                "folds": [
                    {"fold": i + 1, "r2": float(r2_scores[i]), "mae": float(mae_scores[i]), "rmse": float(rmse_scores[i])}
                    for i in range(len(r2_scores))
                ]
            }

            probe_results[router_alias]["targets"][target_name] = target_summary

            csv_rows.append([
                c.name,
                router_alias,
                c.layer_type,
                X.shape[1],
                target_name,
                f"{target_summary['r2_mean']:.4f}",
                f"{target_summary['r2_std']:.4f}",
                f"{target_summary['mae_mean']:.6f}",
                f"{target_summary['mae_std']:.6f}",
                f"{target_summary['rmse_mean']:.6f}",
                f"{target_summary['rmse_std']:.6f}",
            ])

    # Save CSV
    csv_path = os.path.join(output_dir, "linear_probe_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "router_name", "router_alias", "layer_type", "feature_dim",
            "target", "r2_mean", "r2_std", "mae_mean", "mae_std", "rmse_mean", "rmse_std"
        ])
        writer.writerows(csv_rows)

    # Save JSON
    json_path = os.path.join(output_dir, "linear_probe_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(probe_results, f, indent=2)

    logger.info("Test 3 Complete: Linear probe evaluated across all 8 routers for gt_area and thinness.")
    return probe_results


# -----------------------------------------------------------------------------
# Test 4: Routing Sensitivity according to Morphology Quartiles
# -----------------------------------------------------------------------------

def run_test4_routing_sensitivity(
    collectors: List[RouterPooledFeatureCollector],
    sample_morphologies: List[Dict[str, Any]],
    output_dir: str,
) -> Dict[str, Any]:
    """
    Test 4 — Routing Sensitivity by Morphology Quartile:
    Divides validation set into quartiles by (1) thinness_score and (2) gt_area (foreground ratio).
    Reports for each quartile & router:
    - expert selection frequency [0..7]
    - family routing (CNN vs ViT fraction)
    - mean / std of g_s
    - mean routing entropy
    - absolute difference & Jensen-Shannon Divergence between Q1 and Q4
    """
    logger.info("--- Executing Test 4: Routing Sensitivity by Morphology Quartiles ---")
    os.makedirs(output_dir, exist_ok=True)

    N = len(sample_morphologies)
    morph_types = {
        "thinness": np.array([m["thinness_score"] for m in sample_morphologies], dtype=np.float64),
        "gt_area": np.array([m["foreground_ratio"] for m in sample_morphologies], dtype=np.float64),
    }

    sensitivity_results = {}
    csv_rows = []

    for morph_name, values in morph_types.items():
        # Compute quartile thresholds
        q25 = float(np.percentile(values, 25))
        q50 = float(np.percentile(values, 50))
        q75 = float(np.percentile(values, 75))

        # Assign each sample to Q1, Q2, Q3, Q4
        quartile_indices: Dict[str, List[int]] = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}
        for i, val in enumerate(values):
            if val <= q25:
                quartile_indices["Q1"].append(i)
            elif val <= q50:
                quartile_indices["Q2"].append(i)
            elif val <= q75:
                quartile_indices["Q3"].append(i)
            else:
                quartile_indices["Q4"].append(i)

        sensitivity_results[morph_name] = {
            "thresholds": {"Q1_max": q25, "Q2_max": q50, "Q3_max": q75},
            "quartile_counts": {k: len(v) for k, v in quartile_indices.items()},
            "routers": {},
        }

        for c in collectors:
            arrays = c.get_arrays()
            top_k_indices = arrays["top_k_indices"]  # [N, K]
            gating_weights = arrays["gating_weights"]  # [N, K]
            gs_values = arrays["gs"]  # [N]
            num_experts = int(arrays["base_logits"].shape[1]) if arrays["base_logits"].ndim == 2 else 8
            router_alias = getattr(c, "alias", c.name)

            router_data = {
                "name": c.name,
                "layer_type": c.layer_type,
                "quartiles": {},
            }

            quartile_distributions = {}

            for q_name in ["Q1", "Q2", "Q3", "Q4"]:
                idxs = quartile_indices[q_name]
                q_count = len(idxs)
                if q_count == 0:
                    continue

                q_topk = top_k_indices[idxs]  # [q_count, K]
                q_weights = gating_weights[idxs]  # [q_count, K]
                q_gs = gs_values[idxs]  # [q_count]

                # Expert selection frequency across all calls in this quartile
                expert_counts = np.zeros(num_experts, dtype=np.float64)
                for e in q_topk.flatten():
                    if 0 <= e < num_experts:
                        expert_counts[e] += 1.0

                total_selections = np.sum(expert_counts)
                expert_freq = expert_counts / total_selections if total_selections > 0 else expert_counts
                quartile_distributions[q_name] = expert_freq

                # Family routing
                cnn_experts_count = sum(expert_counts[e] for e in range(min(4, num_experts)))
                vit_experts_count = sum(expert_counts[e] for e in range(4, num_experts))
                cnn_fraction = float(cnn_experts_count / total_selections) if total_selections > 0 else 0.0
                vit_fraction = float(vit_experts_count / total_selections) if total_selections > 0 else 0.0

                # Mean / std g_s
                mean_gs = float(np.mean(q_gs))
                std_gs = float(np.std(q_gs, ddof=1 if q_count > 1 else 0))

                # Mean routing entropy
                entropies = [compute_sample_routing_entropy(q_weights[i]) for i in range(q_count)]
                mean_entropy = float(np.mean(entropies))

                router_data["quartiles"][q_name] = {
                    "sample_count": q_count,
                    "expert_frequencies": [float(f) for f in expert_freq],
                    "cnn_expert_fraction": cnn_fraction,
                    "vit_expert_fraction": vit_fraction,
                    "mean_gs": mean_gs,
                    "std_gs": std_gs,
                    "mean_routing_entropy": mean_entropy,
                }

                csv_rows.append([
                    morph_name,
                    c.name,
                    router_alias,
                    c.layer_type,
                    q_name,
                    q_count,
                    f"{cnn_fraction:.4f}",
                    f"{vit_fraction:.4f}",
                    f"{mean_gs:.4f}",
                    f"{std_gs:.4f}",
                    f"{mean_entropy:.4f}",
                    ";".join(f"{f:.4f}" for f in expert_freq),
                ])

            # Q1 vs Q4 Distance
            if "Q1" in quartile_distributions and "Q4" in quartile_distributions:
                p_q1 = quartile_distributions["Q1"]
                p_q4 = quartile_distributions["Q4"]
                abs_diff_per_expert = np.abs(p_q1 - p_q4)
                total_abs_diff = float(np.sum(abs_diff_per_expert))
                max_abs_diff = float(np.max(abs_diff_per_expert))
                jsd_q1_q4 = compute_jsd(p_q1, p_q4)

                router_data["q1_vs_q4_comparison"] = {
                    "jsd_bits": jsd_q1_q4,
                    "total_absolute_difference": total_abs_diff,
                    "max_absolute_difference": max_abs_diff,
                    "per_expert_absolute_difference": [float(d) for d in abs_diff_per_expert],
                }

            sensitivity_results[morph_name]["routers"][router_alias] = router_data

    # Save CSV
    csv_path = os.path.join(output_dir, "routing_sensitivity_by_quartile.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "morphology_metric", "router_name", "router_alias", "layer_type", "quartile",
            "sample_count", "cnn_fraction", "vit_fraction", "mean_gs", "std_gs", "mean_entropy", "expert_frequencies"
        ])
        writer.writerows(csv_rows)

    # Save JSON
    json_path = os.path.join(output_dir, "routing_sensitivity_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sensitivity_results, f, indent=2)

    logger.info("Test 4 Complete: Routing sensitivity analyzed across morphology quartiles.")
    return sensitivity_results


# -----------------------------------------------------------------------------
# Test 5: Empirical Magnitude of g_s Modulation
# -----------------------------------------------------------------------------

def run_test5_gs_modulation_magnitude(
    collectors: List[RouterPooledFeatureCollector],
    output_dir: str,
) -> Dict[str, Any]:
    """
    Test 5 — Magnitude of g_s Modulation:
    Computes empirically for each sample & router:
        Delta_family = log(g_s / (1 - g_s))
    Reports min, max, mean, std, percentiles 5, 50, 95 for both Delta_family and g_s,
    as well as base logits statistics for direct scale comparison.
    """
    logger.info("--- Executing Test 5: Magnitude of g_s Modulation ---")
    os.makedirs(output_dir, exist_ok=True)

    eps = 1e-5
    modulation_results = {}
    csv_rows = []

    for c in collectors:
        arrays = c.get_arrays()
        gs = arrays["gs"]  # [N]
        base_logits = arrays["base_logits"]  # [N, M]
        router_alias = getattr(c, "alias", c.name)

        gs_clamped = np.clip(gs, eps, 1.0 - eps)
        delta_family = np.log(gs_clamped / (1.0 - gs_clamped))

        def get_stats(arr: np.ndarray) -> Dict[str, float]:
            return {
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1 if len(arr) > 1 else 0)),
                "p5": float(np.percentile(arr, 5)),
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
            }

        delta_stats = get_stats(delta_family)
        gs_stats = get_stats(gs)

        # Base logits statistics
        base_logits_flat = base_logits.flatten()
        base_logits_stats = get_stats(base_logits_flat)
        base_logits_range = base_logits_stats["max"] - base_logits_stats["min"]

        router_mod_summary = {
            "name": c.name,
            "layer_type": c.layer_type,
            "sample_count": len(gs),
            "delta_family_stats": delta_stats,
            "gs_stats": gs_stats,
            "base_logits_stats": base_logits_stats,
            "base_logits_range": float(base_logits_range),
            "delta_to_base_std_ratio": float(delta_stats["std"] / (base_logits_stats["std"] + 1e-8)),
        }

        modulation_results[router_alias] = router_mod_summary

        csv_rows.append([
            c.name,
            router_alias,
            c.layer_type,
            len(gs),
            f"{delta_stats['min']:.4f}",
            f"{delta_stats['max']:.4f}",
            f"{delta_stats['mean']:.4f}",
            f"{delta_stats['std']:.4f}",
            f"{delta_stats['p5']:.4f}",
            f"{delta_stats['p50']:.4f}",
            f"{delta_stats['p95']:.4f}",
            f"{gs_stats['min']:.4f}",
            f"{gs_stats['max']:.4f}",
            f"{gs_stats['mean']:.4f}",
            f"{gs_stats['std']:.4f}",
            f"{base_logits_stats['mean']:.4f}",
            f"{base_logits_stats['std']:.4f}",
            f"{base_logits_range:.4f}",
        ])

    # Save CSV
    csv_path = os.path.join(output_dir, "gs_modulation_magnitude.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "router_name", "router_alias", "layer_type", "sample_count",
            "delta_min", "delta_max", "delta_mean", "delta_std", "delta_p5", "delta_p50", "delta_p95",
            "gs_min", "gs_max", "gs_mean", "gs_std",
            "base_logits_mean", "base_logits_std", "base_logits_range"
        ])
        writer.writerows(csv_rows)

    # Save JSON
    json_path = os.path.join(output_dir, "gs_modulation_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(modulation_results, f, indent=2)

    logger.info("Test 5 Complete: g_s modulation magnitude statistics computed.")
    return modulation_results


# -----------------------------------------------------------------------------
# Test 6: Original SAGE Repository Dynamic Code Audit
# -----------------------------------------------------------------------------

def run_test6_original_sage_audit(
    sage_original_path: Optional[str],
    output_dir: str,
) -> Dict[str, Any]:
    """
    Test 6 — Original SAGE Codebase Inspection:
    Dynamically opens and parses router.py in the original SAGE repository.
    Verifies:
    1. AdaptiveAvgPool2d((1, 1))
    2. x.mean(dim=1)
    3. shared_expert_gate(aggregated)
    4. query_projection(aggregated)
    Marks as 'not_available' if file does not exist.
    """
    logger.info("--- Executing Test 6: Original SAGE Codebase Dynamic Audit ---")
    os.makedirs(output_dir, exist_ok=True)

    resolved_path = None
    if sage_original_path and os.path.exists(sage_original_path):
        resolved_path = sage_original_path
    else:
        # Check standard default candidates
        candidates = [
            "d:/truong/SpecialSubjectTTNT/SAGE/sage/components/router.py",
            "D:/truong/SpecialSubjectTTNT/SAGE/sage/components/router.py",
            "../SAGE/sage/components/router.py",
            "../../SAGE/sage/components/router.py",
        ]
        for c in candidates:
            if os.path.exists(c):
                resolved_path = c
                break

    if resolved_path is None or not os.path.exists(resolved_path):
        audit_info = {
            "status": "not_available",
            "target_file": sage_original_path or "None specified",
            "exists": False,
            "message": "SAGE original router.py not found at specified path. Test 6 marked as not_available.",
            "has_gap_or_mean_pooling": False,
            "matched_patterns": {},
            "cnn_pooling_evidence": None,
            "transformer_pooling_evidence": None,
            "downstream_coupling_evidence": None,
        }
        logger.warning(f"Test 6: SAGE original router.py not found (searched '{sage_original_path}'). Status: not_available.")
    else:
        logger.info(f"Test 6: Opening and dynamically parsing {resolved_path}...")
        with open(resolved_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Dynamic search for target patterns
        target_patterns = {
            "adaptive_pool_def": "AdaptiveAvgPool2d((1, 1))",
            "cnn_pooling_call": "feature_aggregator(x).squeeze(-1).squeeze(-1)",
            "transformer_mean_call": "x.mean(dim=1)",
            "shared_gate_call": "shared_expert_gate(aggregated",
            "query_proj_call": "query_projection(aggregated",
        }

        matches = {}
        for key, pat in target_patterns.items():
            matches[key] = [
                {"line_number": i + 1, "code": line.strip()}
                for i, line in enumerate(lines)
                if pat in line
            ]

        has_cnn_gap = len(matches["adaptive_pool_def"]) > 0 or len(matches["cnn_pooling_call"]) > 0
        has_transformer_mean = len(matches["transformer_mean_call"]) > 0
        has_gap_or_mean = has_cnn_gap and has_transformer_mean

        has_shared_gate_coupling = len(matches["shared_gate_call"]) > 0
        has_query_proj_coupling = len(matches["query_proj_call"]) > 0

        audit_info = {
            "status": "verified_present" if has_gap_or_mean else "pattern_mismatch",
            "target_file": os.path.abspath(resolved_path),
            "exists": True,
            "total_lines": len(lines),
            "has_gap_or_mean_pooling": has_gap_or_mean,
            "matched_patterns": matches,
            "cnn_pooling_evidence": {
                "verified": has_cnn_gap,
                "pool_definition": matches["adaptive_pool_def"],
                "pool_execution": matches["cnn_pooling_call"],
                "mechanism": "nn.AdaptiveAvgPool2d((1, 1)) followed by squeeze(-1).squeeze(-1) -> Global Average Pooling to (B, C)",
            },
            "transformer_pooling_evidence": {
                "verified": has_transformer_mean,
                "pool_execution": matches["transformer_mean_call"],
                "mechanism": "x.mean(dim=1) -> Global Mean Pooling across token sequence to (B, D)",
            },
            "downstream_coupling_evidence": {
                "shared_gate_coupled_to_pooled_feature": has_shared_gate_coupling,
                "query_projection_coupled_to_pooled_feature": has_query_proj_coupling,
                "gate_call_matches": matches["shared_gate_call"],
                "query_call_matches": matches["query_proj_call"],
            },
        }
        logger.info(f"Test 6: Parse complete. has_gap_or_mean_pooling={has_gap_or_mean}. Found {sum(len(v) for v in matches.values())} pattern occurrences.")

    # Save JSON
    audit_json_path = os.path.join(output_dir, "test6_sage_original_audit.json")
    with open(audit_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_info, f, indent=2)

    return audit_info


# -----------------------------------------------------------------------------
# Master Consolidated Markdown Report
# -----------------------------------------------------------------------------

def generate_consolidated_markdown_report(
    test1_summary: Dict[str, Any],
    test2_meta: Dict[str, Any],
    test3_summary: Dict[str, Any],
    test4_summary: Dict[str, Any],
    test5_summary: Dict[str, Any],
    test6_audit: Dict[str, Any],
    output_dir: str,
) -> str:
    """Generates a complete, publication-grade markdown report of the GAP routing study."""
    report_path = os.path.join(output_dir, "GAP_ROUTING_STUDY_REPORT.md")

    lines = [
        "# SAGE-Lite Empirical Diagnostics: Global Average Pooling (GAP) Routing-Signal Bottleneck Study",
        "",
        "*Date: September 2026*",
        "*Architecture Target: Canonical D4 P3-C (ASDW) Standalone Model*",
        "*Dataset Split: Crack500 Validation Split (N = 348 samples)*",
        "*Execution Protocol: Deterministic Evaluation Pass (model.eval(), torch.no_grad(), exploration noise OFF)*",
        "",
        "---",
        "",
        "## 1. Test 1: Validation Set Foreground Area Ratio Distribution",
        "",
        "$$r_i = \\frac{\\#\\text{positive pixels}}{H \\times W} = \\frac{\\text{gt\\_area}}{448 \\times 448}$$",
        "",
        f"- **Total Samples (N)**: `{test1_summary['N']}`",
        f"- **Mean Foreground Ratio**: `{test1_summary['mean']:.4%}`",
        f"- **Standard Deviation**: `{test1_summary['std']:.4%}`",
        f"- **Median**: `{test1_summary['median']:.4%}`",
        f"- **Range [Min, Max]**: `[{test1_summary['min']:.4%}, {test1_summary['max']:.4%}]`",
        f"- **Quartiles [Q1, Q2, Q3, Q4]**: `[{test1_summary['Q1']:.4%}, {test1_summary['Q2']:.4%}, {test1_summary['Q3']:.4%}, {test1_summary['Q4']:.4%}]`",
        "",
        "### Extreme Sparsity Tiers",
        f"- **Samples with Foreground < 1%**: `{test1_summary['percentage_samples_below_1pct']:.2f}%`",
        f"- **Samples with Foreground < 2%**: `{test1_summary['percentage_samples_below_2pct']:.2f}%`",
        f"- **Samples with Foreground < 5%**: `{test1_summary['percentage_samples_below_5pct']:.2f}%`",
        f"- **Samples with Foreground < 10%**: `{test1_summary['percentage_samples_below_10pct']:.2f}%`",
        "",
        "| Bin Interval | Pixel Ratio Range | Sample Count | Percentage |",
        "|:---:|:---:|:---:|:---:|",
    ]

    for b in test1_summary["histogram_10bins"]:
        lines.append(f"| Bin | `[{b['bin_min']:.4%}, {b['bin_max']:.4%}]` | {b['count']} | {b['percentage']:.2f}% |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Test 2: Captured Router Pooled Features Metadata",
        "",
        f"- **Stored Array Archive**: `router_pooled_features.npz` ({test2_meta['sample_count']} samples)",
        f"- **Metadata File**: `router_pooled_features_meta.json`",
        "",
        "| Router | Alias | Layer Type | Feature Dim (Pooled) | Experts Pool | Top-k Selection |",
        "|:---|:---:|:---:|:---:|:---:|:---:|",
    ])

    for r_name, r_info in test2_meta["routers"].items():
        lines.append(f"| `{r_name}` | **{r_info['alias']}** | {r_info['layer_type']} | `{r_info['feature_dim']}` | `{r_info['num_experts']}` | `{r_info['top_k']}` |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Test 3: Linear Probe (Ridge Regression 5-fold CV)",
        "",
        "Evaluation of morphology information retained in the pooled features:",
        "$$\\text{pooled feature } (\\mathbf{h}) \\xrightarrow{\\text{Ridge Regression (5-fold CV)}} \\text{target morphology}$$",
        "",
        "| Router | Type | Feature Dim | Target: `gt_area` ($r_i$) $R^2$ | `gt_area` MAE | Target: `thinness` $R^2$ | `thinness` MAE |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for r_alias, r_probe in test3_summary.items():
        gt_res = r_probe["targets"]["gt_area"]
        thin_res = r_probe["targets"]["thinness"]
        lines.append(
            f"| **{r_alias}** (`{r_probe['name']}`) | {r_probe['layer_type']} | {r_probe['feature_dim']} | "
            f"`{gt_res['r2_mean']:+.4f} ± {gt_res['r2_std']:.4f}` | `{gt_res['mae_mean']:.6f}` | "
            f"`{thin_res['r2_mean']:+.4f} ± {thin_res['r2_std']:.4f}` | `{thin_res['mae_mean']:.4f}` |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Test 4: Routing Sensitivity by Morphology Quartiles",
        "",
        "Distribution shift comparison between Q1 (least pronounced) vs Q4 (most pronounced):",
        "",
        "### 4.1. Sensitivity Across Thinness Quartiles",
        "",
        "| Router | Q1 vs Q4 JSD (bits) | Q1 vs Q4 Total Abs Diff | Q1 CNN / ViT Fraction | Q4 CNN / ViT Fraction | Q1 Mean $g_s$ | Q4 Mean $g_s$ |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    thin_routers = test4_summary["thinness"]["routers"]
    for r_alias, r_sens in thin_routers.items():
        comp = r_sens.get("q1_vs_q4_comparison", {})
        q1_data = r_sens["quartiles"]["Q1"]
        q4_data = r_sens["quartiles"]["Q4"]
        lines.append(
            f"| **{r_alias}** | `{comp.get('jsd_bits', 0.0):.4f}` | `{comp.get('total_abs_diff', 0.0):.4f}` | "
            f"`{q1_data['cnn_expert_fraction']:.1%} / {q1_data['vit_expert_fraction']:.1%}` | "
            f"`{q4_data['cnn_expert_fraction']:.1%} / {q4_data['vit_expert_fraction']:.1%}` | "
            f"`{q1_data['mean_gs']:.4f}` | `{q4_data['mean_gs']:.4f}` |"
        )

    lines.extend([
        "",
        "### 4.2. Sensitivity Across Crack Area Quartiles (`gt_area`)",
        "",
        "| Router | Q1 vs Q4 JSD (bits) | Q1 vs Q4 Total Abs Diff | Q1 CNN / ViT Fraction | Q4 CNN / ViT Fraction | Q1 Mean $g_s$ | Q4 Mean $g_s$ |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    area_routers = test4_summary["gt_area"]["routers"]
    for r_alias, r_sens in area_routers.items():
        comp = r_sens.get("q1_vs_q4_comparison", {})
        q1_data = r_sens["quartiles"]["Q1"]
        q4_data = r_sens["quartiles"]["Q4"]
        lines.append(
            f"| **{r_alias}** | `{comp.get('jsd_bits', 0.0):.4f}` | `{comp.get('total_abs_diff', 0.0):.4f}` | "
            f"`{q1_data['cnn_expert_fraction']:.1%} / {q1_data['vit_expert_fraction']:.1%}` | "
            f"`{q4_data['cnn_expert_fraction']:.1%} / {q4_data['vit_expert_fraction']:.1%}` | "
            f"`{q1_data['mean_gs']:.4f}` | `{q4_data['mean_gs']:.4f}` |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Test 5: Empirical Magnitude of $g_s$ Logit Modulation",
        "",
        "$$\\Delta_{\\text{family}} = \\log\\frac{g_s}{1 - g_s}$$",
        "",
        "| Router | Alias | Mean $\\Delta_{\\text{family}}$ | Std $\\Delta_{\\text{family}}$ | $\\Delta$ Range [Min, Max] | $\\Delta$ [P5, P50, P95] | Mean $g_s$ | Base Logits Std | Base Logits Range |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for r_alias, r_mod in test5_summary.items():
        d_st = r_mod["delta_family_stats"]
        gs_st = r_mod["gs_stats"]
        b_st = r_mod["base_logits_stats"]
        lines.append(
            f"| `{r_mod['name']}` | **{r_alias}** | `{d_st['mean']:+.4f}` | `{d_st['std']:.4f}` | "
            f"`[{d_st['min']:+.4f}, {d_st['max']:+.4f}]` | `[{d_st['p5']:+.4f}, {d_st['p50']:+.4f}, {d_st['p95']:+.4f}]` | "
            f"`{gs_st['mean']:.4f}` | `{b_st['std']:.4f}` | `{r_mod['base_logits_range']:.4f}` |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 6. Test 6: Original SAGE Codebase Audit",
        "",
    ])

    if test6_audit.get("status") == "not_available":
        lines.extend([
            "> [!NOTE]",
            f"> SAGE original router.py was not found at `{test6_audit['target_file']}` in this runtime environment.",
            "> Test 6 is marked as **not_available** (no fabricated audit).",
            "",
        ])
    else:
        lines.extend([
            f"- **Source File**: `{test6_audit['target_file']}` (Exists: True, Total Lines: {test6_audit.get('total_lines')})",
            f"- **Audit Status**: `{test6_audit['status']}`",
            f"- **Has GAP or Mean Pooling**: `{test6_audit['has_gap_or_mean_pooling']}`",
            "",
            "### Verified Evidence from Source Code",
        ])
        cnn_ev = test6_audit.get("cnn_pooling_evidence", {})
        if cnn_ev and cnn_ev.get("pool_definition"):
            lines.append(f"1. **CNN Pooling Definition (Line {cnn_ev['pool_definition'][0]['line_number']})**:")
            lines.append(f"   ```python\n   {cnn_ev['pool_definition'][0]['code']}\n   ```")
        if cnn_ev and cnn_ev.get("pool_execution"):
            lines.append(f"   Execution (Line {cnn_ev['pool_execution'][0]['line_number']}):")
            lines.append(f"   ```python\n   {cnn_ev['pool_execution'][0]['code']}\n   ```")

        tf_ev = test6_audit.get("transformer_pooling_evidence", {})
        if tf_ev and tf_ev.get("pool_execution"):
            lines.append(f"2. **Transformer Mean Pooling Execution (Line {tf_ev['pool_execution'][0]['line_number']})**:")
            lines.append(f"   ```python\n   {tf_ev['pool_execution'][0]['code']}\n   ```")

        ds_ev = test6_audit.get("downstream_coupling_evidence", {})
        if ds_ev and ds_ev.get("gate_call_matches"):
            lines.append(f"3. **Shared Expert Gate $g_s$ Coupling (Line {ds_ev['gate_call_matches'][0]['line_number']})**:")
            lines.append(f"   ```python\n   {ds_ev['gate_call_matches'][0]['code']}\n   ```")
        if ds_ev and ds_ev.get("query_call_matches"):
            lines.append(f"4. **Query Projection SAR Coupling (Line {ds_ev['query_call_matches'][0]['line_number']})**:")
            lines.append(f"   ```python\n   {ds_ev['query_call_matches'][0]['code']}\n   ```")

    lines.extend([
        "",
        "---",
        "",
        "## 7. Artifact Manifest",
        "- `foreground_ratio.csv`: Individual foreground ratio per sample",
        "- `foreground_ratio_summary.json`: Detailed distribution summary",
        "- `router_pooled_features.npz`: Compressed arrays of all captured pooled features",
        "- `router_pooled_features_meta.json`: Metadata for pooled feature arrays",
        "- `linear_probe_results.csv`: 5-fold CV R², MAE, RMSE per router and target",
        "- `linear_probe_summary.json`: Complete fold-level linear probe data",
        "- `routing_sensitivity_by_quartile.csv`: Quartile distributions and frequencies",
        "- `routing_sensitivity_summary.json`: Quartile data and JSD metrics",
        "- `gs_modulation_magnitude.csv`: Modulation magnitude metrics",
        "- `gs_modulation_summary.json`: Statistical bounds on g_s and Delta_family",
        "- `test6_sage_original_audit.json`: Source code analysis of original SAGE",
        "- `GAP_ROUTING_STUDY_REPORT.md`: This consolidated technical report",
        "",
    ])

    report_content = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info(f"Consolidated study report generated at: {report_path}")
    return report_path


# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Empirical Diagnostic Study: GAP Routing Signal Bottleneck")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to canonical checkpoint (.pth / .pt)")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default="diagnostics/gap_study", help="Output directory")
    parser.add_argument("--data-root", "--data_root", dest="data_root", type=str, default=None, help="Dataset root directory override")
    parser.add_argument("--split", type=str, default="val", choices=["val"], help="Dataset split (strictly 'val' allowed)")
    parser.add_argument("--max-samples", "--max_samples", dest="max_samples", type=int, default=None, help="Max samples for smoke testing (e.g. 16)")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--sage-original-path", "--sage_original_path",
        dest="sage_original_path",
        type=str,
        default="d:/truong/SpecialSubjectTTNT/SAGE/sage/components/router.py",
        help="Path to original SAGE router.py file for Test 6 audit"
    )
    args = parser.parse_args()

    # 1. Deterministic Setup
    set_seed(args.seed)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        dev_name = torch.cuda.get_device_name(0)
        if "1650" in dev_name or "1660" in dev_name:
            torch.backends.cudnn.enabled = False
            logger.info(f"Detected {dev_name}: disabled cuDNN for numerical stability.")
    logger.info(f"Target execution device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Load Canonical D4 Model & Checkpoint (Strict integrity audit)
    model, config, ckpt_meta = load_canonical_d4_model(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=device,
        data_root_override=args.data_root,
    )

    # Sanity Check: Verify model is strictly in eval mode and parameter count unchanged
    assert not model.training, "Sanity Check Failure: Model must be in eval mode!"
    initial_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 3. Discover Routers & Register Non-invasive Hooks
    routers_meta = discover_routers(model)
    if len(routers_meta) != 8:
        logger.warning(f"Expected 8 routers for D4, found {len(routers_meta)} routers!")

    collectors: List[RouterPooledFeatureCollector] = []
    for r_info in routers_meta:
        collector = RouterPooledFeatureCollector(
            name=r_info["name"],
            router_module=r_info["router"],
            router_index=r_info["router_index"],
            layer_type=r_info["layer_type"],
        )
        collector.alias = r_info["alias"]
        collector.register()
        collectors.append(collector)

    # 4. Load Validation Dataset
    img_size = int(config.get("img_size", 448))
    dataset = get_dataset_from_config(config, split="val", image_size=img_size)
    total_val = len(dataset)
    logger.info(f"Loaded validation split: {total_val} samples available.")

    if args.max_samples is not None and args.max_samples < total_val:
        indices = list(range(args.max_samples))
        dataset = Subset(dataset, indices)
        logger.info(f"Smoke test mode: Evaluating subset of {len(dataset)} samples (--max-samples={args.max_samples})")

    g = torch.Generator()
    g.manual_seed(args.seed)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0 if device.type == "cpu" else 2,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=False,
    )

    # 5. Execute Evaluation Pass
    logger.info("Executing evaluation pass (torch.no_grad(), exploration noise OFF)...")
    sample_morphologies: List[Dict[str, Any]] = []
    sample_ids: List[str] = []
    sample_counter = 0

    try:
        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].cpu().numpy()
                case_names = batch.get("case_name", [f"sample_{sample_counter + i}" for i in range(images.size(0))])

                for b in range(images.size(0)):
                    c_name = case_names[b]
                    lbl = labels[b]
                    morph = compute_sample_morphology(lbl)
                    total_px = int(lbl.size)
                    r_i = float(morph["gt_area"] / total_px) if total_px > 0 else 0.0
                    morph["sample_id"] = c_name
                    morph["total_pixels"] = total_px
                    morph["foreground_ratio"] = r_i
                    sample_morphologies.append(morph)
                    sample_ids.append(c_name)

                sample_counter += images.size(0)

                # Forward pass executes router hooks without creating parameters
                _ = model(images)
    finally:
        for c in collectors:
            c.remove()

    logger.info(f"Evaluation complete for {len(sample_ids)} samples.")

    # Sanity Check: Ensure parameter count remained unchanged throughout forward pass
    final_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert initial_trainable_params == final_trainable_params, (
        f"Sanity Check Failure: Trainable parameters changed! Initial={initial_trainable_params}, Final={final_trainable_params}"
    )

    # 6. Execute Diagnostic Tests
    test1_summary = run_test1_foreground_ratio(sample_morphologies, args.output_dir)
    test2_meta = run_test2_save_features(collectors, sample_ids, args.output_dir)
    test3_summary = run_test3_linear_probe(collectors, sample_morphologies, args.output_dir, seed=args.seed)
    test4_summary = run_test4_routing_sensitivity(collectors, sample_morphologies, args.output_dir)
    test5_summary = run_test5_gs_modulation_magnitude(collectors, args.output_dir)
    test6_audit = run_test6_original_sage_audit(args.sage_original_path, args.output_dir)

    # 7. Generate Master Consolidated Markdown Report
    report_path = generate_consolidated_markdown_report(
        test1_summary=test1_summary,
        test2_meta=test2_meta,
        test3_summary=test3_summary,
        test4_summary=test4_summary,
        test5_summary=test5_summary,
        test6_audit=test6_audit,
        output_dir=args.output_dir,
    )

    logger.info("================================================================================")
    logger.info(f"GAP Routing Diagnostic Study Complete! Artifacts stored in: {args.output_dir}")
    logger.info(f"Master technical report available at: {report_path}")
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
