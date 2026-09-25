#!/usr/bin/env python3
"""
tools/analyze_routing.py

Dedicated Post-Training Routing Diagnostics Evaluator for SAGE-Lite Models.
Specifically designed for the Canonical P3-C (ASDW) Standalone Model and B2 UNet.

Features:
- Dynamic discovery of routers and expert pool from the actual model (no hard-coded L0-L27).
- Deterministic evaluation pass (model.eval(), torch.no_grad(), exploration noise OFF).
- Captures full per-sample routing decisions, gating weights, SAR base affinity scores, and g_s gate.
- Generates 6 machine-readable CSV reports + consolidated JSON report + figures/ directory.
- Rigorous mathematical consistency assertions (sample_count * top_k per router).

Author: Special Subject AI Team
Date: September 2026
Branch: crack500-audit
"""

import argparse
import csv
import datetime
import hashlib
import json
import logging
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import yaml

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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
logger = logging.getLogger("analyze_routing")


class RouterDataCollector:
    """
    Non-invasive forward hook collector for an individual SageRouter instance.
    Captures:
    - top_k_indices: [B, K]
    - gating_weights: [B, K]
    - base_logits: [B, M] (SAR affinity scores before modulation)
    - sigmoid_affinity: [B, M] (Sigmoid of base affinity logits)
    - g_s: [B] (Shared vs Dynamic gate value)
    """

    def __init__(self, name: str, layer_type: str, router_index: int, router_module: SageRouter):
        self.name = name
        self.layer_type = layer_type
        self.router_index = router_index
        self.router = router_module
        self.hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None

        # Storage for evaluated samples
        self.top_k_indices: List[List[int]] = []
        self.gating_weights: List[List[float]] = []
        self.base_logits: List[List[float]] = []
        self.sigmoid_affinity: List[List[float]] = []
        self.g_s_values: List[float] = []

    def register(self):
        """Register the forward hook on the router."""
        self.hook_handle = self.router.register_forward_hook(self._hook_fn)

    def remove(self):
        """Remove the forward hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def _hook_fn(self, module: SageRouter, inputs: Tuple[torch.Tensor, ...], output: Tuple[Any, ...]):
        """
        Hook function executed after module.forward.
        inputs[0]: input_tensor [B, C, H, W] or [B, N, D]
        output: (top_k_indices, gating_weights, routing_info)
        """
        input_tensor = inputs[0]
        top_k_indices, gating_weights, _ = output

        with torch.no_grad():
            # Step 1: Feature aggregation and adaptation (matching SageRouter.forward)
            agg = module._aggregate_features_with_adaptation(input_tensor)

            # Step 2: Shared expert gate g_s [B, 1]
            g_s = torch.sigmoid(module.shared_expert_gate(agg))

            # Step 3: Base query-key affinity logits [B, expert_pool_size]
            query = module.query_projection(agg)
            base_logits = torch.matmul(query, module.expert_keys.T) / module.temperature
            sigmoid_affinity = torch.sigmoid(base_logits)

            B = input_tensor.shape[0]
            top_k_cpu = top_k_indices.cpu().numpy()
            weights_cpu = gating_weights.cpu().numpy()
            base_logits_cpu = base_logits.cpu().numpy()
            sigmoid_affinity_cpu = sigmoid_affinity.cpu().numpy()
            g_s_cpu = g_s.squeeze(-1).cpu().numpy()

            for b in range(B):
                self.top_k_indices.append(top_k_cpu[b].tolist())
                self.gating_weights.append(weights_cpu[b].tolist())
                self.base_logits.append(base_logits_cpu[b].tolist())
                self.sigmoid_affinity.append(sigmoid_affinity_cpu[b].tolist())
                self.g_s_values.append(float(g_s_cpu[b]) if g_s_cpu.ndim > 0 else float(g_s_cpu))


def discover_routers(model: B2ConvNeXtViTUNet) -> List[Dict[str, Any]]:
    """
    Dynamically discover all active SageRouter instances in the model.
    Returns ordered list of metadata dicts:
    - name: str (e.g. convnext.stage_0, transformer.block_5)
    - layer_type: 'CNN' or 'ViT'
    - router_index: int
    - stage_or_block_idx: int
    - router: SageRouter instance
    - layer: SageLayer instance
    """
    routers: List[Dict[str, Any]] = []
    router_idx = 0

    # 1. Discover ConvNeXt CNN stages
    if hasattr(model, "backbone") and hasattr(model.backbone, "convnext"):
        for s_idx, stage in enumerate(model.backbone.convnext.stages):
            if isinstance(stage, SageLayer):
                routers.append({
                    "name": f"convnext.stage_{s_idx}",
                    "layer_type": "CNN",
                    "router_index": router_idx,
                    "stage_or_block_idx": s_idx,
                    "router": stage.router,
                    "layer": stage,
                })
                router_idx += 1

    # 2. Discover Transformer ViT blocks
    if hasattr(model, "backbone") and hasattr(model.backbone, "transformer_blocks"):
        for b_idx, blk in enumerate(model.backbone.transformer_blocks):
            if isinstance(blk, SageLayer):
                routers.append({
                    "name": f"transformer.block_{b_idx}",
                    "layer_type": "ViT",
                    "router_index": router_idx,
                    "stage_or_block_idx": b_idx,
                    "router": blk.router,
                    "layer": blk,
                })
                router_idx += 1

    logger.info(f"Dynamically discovered {len(routers)} routers: "
                f"{sum(1 for r in routers if r['layer_type'] == 'CNN')} CNN stages + "
                f"{sum(1 for r in routers if r['layer_type'] == 'ViT')} ViT blocks.")
    return routers


def discover_experts(model: B2ConvNeXtViTUNet, sage_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Discover expert pool structure and metadata from actual model attributes.
    Derives expert type (CNN vs ViT) dynamically from expert module metadata (expert_type)
    or router expert_infos rather than assuming index 0..3 is CNN and index 4.. is ViT.
    """
    expert_pool = getattr(model, "expert_pool", None)
    num_experts = len(expert_pool) if expert_pool is not None else 16
    shared_indices = set(sage_cfg.get("shared_expert_indices", [0, 1, 2, 3]))

    # Retrieve expert_infos from any available router if present
    router_expert_infos = None
    for module in model.modules():
        if isinstance(module, SageRouter) and getattr(module, "expert_infos", None):
            router_expert_infos = module.expert_infos
            break

    experts: List[Dict[str, Any]] = []
    cnn_count = 0
    vit_count = 0

    for e_idx in range(num_experts):
        expert_type_attr = None
        expert_name_attr = None

        # 1. Query expert_pool module attributes
        if expert_pool is not None and e_idx < len(expert_pool):
            exp_mod = expert_pool[e_idx]
            expert_type_attr = getattr(exp_mod, "expert_type", None)
            expert_name_attr = getattr(exp_mod, "expert_name", None)

        # 2. Query router.expert_infos metadata if not found on module
        if (expert_type_attr is None or expert_name_attr is None) and router_expert_infos and e_idx < len(router_expert_infos):
            info = router_expert_infos[e_idx]
            if expert_type_attr is None:
                expert_type_attr = info.get("type", None)
            if expert_name_attr is None:
                expert_name_attr = info.get("name", None)

        # Determine normalized family: 'CNN' vs 'ViT'
        if expert_type_attr is not None:
            raw_str = str(expert_type_attr).strip().lower()
            if "cnn" in raw_str or "conv" in raw_str:
                family = "CNN"
            elif "transformer" in raw_str or "vit" in raw_str or "attn" in raw_str:
                family = "ViT"
            else:
                family = raw_str.upper()
        else:
            family = "CNN" if e_idx < 4 else "ViT"

        # Determine canonical reporting name
        if family == "CNN":
            e_name = f"CNN_Stage_{cnn_count}"
            cnn_count += 1
        elif family == "ViT":
            e_name = f"ViT_Block_{vit_count}"
            vit_count += 1
        else:
            e_name = expert_name_attr or f"Expert_{e_idx}"

        experts.append({
            "index": e_idx,
            "column_name": f"E{e_idx}",
            "name": e_name,
            "type": family,
            "is_shared": (e_idx in shared_indices),
        })

    logger.info(f"Discovered expert pool size: {len(experts)} ({sum(1 for e in experts if e['type'] == 'CNN')} CNN, "
                f"{sum(1 for e in experts if e['type'] == 'ViT')} ViT, Shared: {sorted(list(shared_indices))}).")
    return experts


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_model_from_checkpoint(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
    data_root_override: Optional[str] = None,
) -> Tuple[B2ConvNeXtViTUNet, Dict[str, Any], Dict[str, Any]]:
    """
    Safely instantiate the model matching config and load trained checkpoint.
    Includes rigorous SHA256 computation, metadata compatibility validation,
    and strict parameter completeness verification.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if data_root_override:
        config["root_dir"] = data_root_override

    model_type = config.get("model", "B2")
    if model_type != "B2":
        raise ValueError(f"analyze_routing currently targets B2 / P3-C models, got model={model_type}")

    img_size = int(config.get("img_size", 448))
    vit_depth = int(config.get("num_transformer_layers", 12))
    p3_mode = config.get("p3_mode", None)
    sage_cfg = config.get("sage_config", {})

    logger.info(f"Instantiating B2 UNet: Depth={vit_depth}, BS={config.get('batch_size', 14)}, "
                f"p3_mode='{p3_mode}', img_size={img_size}")

    model = create_b2_unet(
        num_classes=1,
        img_size=img_size,
        num_transformer_layers=vit_depth,
        pretrained=False,
        sage_config=sage_cfg,
        p3_mode=p3_mode,
    ).to(device)

    # 1. Checkpoint existence & SHA256 verification
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint_sha256 = compute_file_sha256(checkpoint_path)
    logger.info(f"Loading checkpoint from: {checkpoint_path} (SHA256: {checkpoint_sha256})")
    raw_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 2. Extract and verify checkpoint metadata
    ckpt_metadata: Dict[str, Any] = {"sha256": checkpoint_sha256}
    if isinstance(raw_checkpoint, dict):
        for field in ["stage", "epoch", "best_dice", "best_loss", "p3_mode", "num_transformer_layers", "model_type"]:
            if field in raw_checkpoint:
                val = raw_checkpoint[field]
                if isinstance(val, (torch.Tensor, np.generic)):
                    val = val.item()
                ckpt_metadata[field] = val

    # Verify metadata compatibility against configuration
    if "p3_mode" in ckpt_metadata and ckpt_metadata["p3_mode"] is not None:
        ckpt_p3 = str(ckpt_metadata["p3_mode"]).upper()
        cfg_p3 = str(p3_mode).upper() if p3_mode is not None else None
        if cfg_p3 is not None and ckpt_p3 != cfg_p3:
            raise ValueError(
                f"Checkpoint metadata mismatch: p3_mode in checkpoint is '{ckpt_p3}', "
                f"but config specifies '{cfg_p3}'!"
            )
        if cfg_p3 == "C" and ckpt_p3 != "C":
            raise ValueError(
                f"Canonical P3-C audit requires p3_mode='C', but checkpoint metadata has '{ckpt_p3}'!"
            )

    if "num_transformer_layers" in ckpt_metadata and ckpt_metadata["num_transformer_layers"] is not None:
        ckpt_depth = int(ckpt_metadata["num_transformer_layers"])
        if ckpt_depth != vit_depth:
            raise ValueError(
                f"Checkpoint metadata mismatch: num_transformer_layers in checkpoint is {ckpt_depth}, "
                f"but config specifies {vit_depth}!"
            )
        if vit_depth == 12 and ckpt_depth != 12:
            raise ValueError(
                f"Canonical P3-C audit requires num_transformer_layers=12, but checkpoint metadata has {ckpt_depth}!"
            )

    # 3. Clean and verify state dict
    if isinstance(raw_checkpoint, dict):
        if "model_state_dict" in raw_checkpoint:
            state_dict = raw_checkpoint["model_state_dict"]
        elif "state_dict" in raw_checkpoint:
            state_dict = raw_checkpoint["state_dict"]
        else:
            state_dict = raw_checkpoint
    else:
        state_dict = raw_checkpoint

    # Strip potential 'module.' prefixes from DataParallel if present
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        clean_k = k[7:] if k.startswith("module.") else k
        cleaned_state_dict[clean_k] = v

    # Model compatibility audit
    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    ckpt_keys = set(cleaned_state_dict.keys())
    missing_keys = model_keys - ckpt_keys
    unexpected_keys = ckpt_keys - model_keys

    # Check for structurally important parameters and fail loudly
    critical_submodules = ("router", "backbone", "decoder", "p3_refinement", "sa_hub")
    missing_critical = [k for k in missing_keys if any(sub in k for sub in critical_submodules)]

    if missing_critical:
        missing_router = [k for k in missing_critical if "router" in k]
        if missing_router:
            raise ValueError(
                f"CRITICAL AUDIT FAILURE: Checkpoint is missing {len(missing_router)} SAGE router parameter(s)! "
                f"Missing keys include: {missing_router[:5]}. Aborting evaluation."
            )
        raise ValueError(
            f"CRITICAL AUDIT FAILURE: Checkpoint is missing {len(missing_critical)} structurally important parameter(s)! "
            f"Missing keys include: {missing_critical[:5]}. Aborting evaluation."
        )

    if missing_keys:
        missing_numel = sum(model_state[k].numel() for k in missing_keys)
        total_numel = sum(p.numel() for p in model.parameters())
        missing_ratio = missing_numel / total_numel if total_numel > 0 else 1.0
        if missing_ratio > 0.01:
            raise ValueError(
                f"CRITICAL AUDIT FAILURE: Checkpoint is missing substantial portion of parameters: "
                f"{len(missing_keys)} keys ({missing_numel:,} parameters, {missing_ratio:.2%})! "
                f"Missing keys include: {sorted(list(missing_keys))[:5]}. Aborting evaluation."
            )
        logger.warning(
            f"Checkpoint has {len(missing_keys)} non-critical missing keys ({missing_ratio:.4%}): {sorted(list(missing_keys))[:5]}"
        )

    if unexpected_keys:
        logger.warning(f"Checkpoint contains {len(unexpected_keys)} unexpected keys: {sorted(list(unexpected_keys))[:5]}...")

    # Load state dict
    load_res = model.load_state_dict(cleaned_state_dict, strict=False)
    logger.info(f"Model state dict loaded. Missing: {len(load_res.missing_keys)}, Unexpected: {len(load_res.unexpected_keys)}")

    model.eval()
    return model, config, ckpt_metadata


def compute_entropy_and_concentration(distribution: np.ndarray) -> Dict[str, float]:
    """
    Computes Shannon entropy (base 2 and natural log), normalized entropy, HHI, and effective number of experts.
    distribution: 1D probability distribution summing to 1.0.
    """
    p = np.asarray(distribution, dtype=np.float64)
    p = p[p > 0]  # Avoid log(0)

    if len(p) == 0:
        return {
            "entropy_bits": 0.0,
            "normalized_entropy": 0.0,
            "max_expert_share": 0.0,
            "hhi_concentration": 1.0,
            "effective_num_experts": 1.0,
        }

    # Base-2 Shannon entropy (in bits)
    h2 = -float(np.sum(p * np.log2(p)))

    # Natural-log entropy for effective number of experts: exp(H_e)
    he = -float(np.sum(p * np.log(p)))
    effective_num_experts = float(np.exp(he))

    # Normalized entropy against uniform distribution of size M
    M = len(distribution)
    max_h2 = math.log2(M) if M > 1 else 1.0
    normalized_entropy = float(h2 / max_h2) if max_h2 > 0 else 0.0

    # Max expert share
    max_share = float(np.max(distribution))

    # Herfindahl-Hirschman Index: sum(p_i^2)
    hhi = float(np.sum(distribution**2))

    return {
        "entropy_bits": h2,
        "normalized_entropy": normalized_entropy,
        "max_expert_share": max_share,
        "hhi_concentration": hhi,
        "effective_num_experts": effective_num_experts,
    }


def compute_normalized_gating_weight_entropy(gating_weights_list: List[List[float]]) -> float:
    """
    Computes mean normalized selected gating weight entropy across evaluated samples for top-k selections.
    Normalized by log2(K). Note: SageRouter gating_weights are sigmoid(top_k_logits) after modulation,
    representing the normalized distribution over selected experts.
    """
    entropies = []
    for weights in gating_weights_list:
        w = np.asarray(weights, dtype=np.float64)
        w_sum = np.sum(w)
        if w_sum > 0:
            norm_w = w / w_sum
            norm_w = norm_w[norm_w > 0]
            if len(norm_w) > 0:
                h = -np.sum(norm_w * np.log2(norm_w))
                K = len(weights)
                norm_h = h / math.log2(K) if K > 1 else 0.0
                entropies.append(norm_h)
    return float(np.mean(entropies)) if len(entropies) > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Dedicated SAGE-Lite Routing Diagnostics Evaluator")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pth / .pt)")
    parser.add_argument("--data_root", "--data-root", dest="data_root", type=str, default=None, help="Dataset root directory")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split to evaluate (default: val)")
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", type=str, default="results/routing_diagnostics", help="Output directory")
    parser.add_argument("--max_samples", "--max-samples", dest="max_samples", type=int, default=None, help="Maximum samples to evaluate (default: all)")
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=4, help="Batch size for evaluation (default: 4)")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader num_workers (default: 2)")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed (default: 42)")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
    args = parser.parse_args()

    # 1. Deterministic setup
    set_seed(args.seed)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        dev_name = torch.cuda.get_device_name(0)
        if "1650" in dev_name or "1660" in dev_name:
            torch.backends.cudnn.enabled = False
            logger.info(f"Detected {dev_name}: set torch.backends.cudnn.enabled = False for FP16 stability.")
    logger.info(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # 2. Instantiate model and load checkpoint
    model, config, ckpt_meta = load_model_from_checkpoint(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=device,
        data_root_override=args.data_root,
    )

    sage_cfg = config.get("sage_config", {})
    top_k = int(sage_cfg.get("top_k", 4))
    img_size = int(config.get("img_size", 448))

    # 3. Dynamic Discovery of Routers and Experts
    routers_meta = discover_routers(model)
    if not routers_meta:
        raise RuntimeError("No SAGE routers discovered in model!")

    experts_meta = discover_experts(model, sage_cfg)
    num_experts = len(experts_meta)
    num_routers = len(routers_meta)

    # 4. Attach Forward Hook Collectors to every router
    collectors: List[RouterDataCollector] = []
    for r_meta in routers_meta:
        collector = RouterDataCollector(
            name=r_meta["name"],
            layer_type=r_meta["layer_type"],
            router_index=r_meta["router_index"],
            router_module=r_meta["router"],
        )
        collector.register()
        collectors.append(collector)

    # 5. Load Dataset and DataLoader
    logger.info(f"Loading '{args.split}' dataset from config...")
    dataset = get_dataset_from_config(config, split=args.split, image_size=img_size)
    total_available_samples = len(dataset)
    logger.info(f"Total samples available in '{args.split}' split: {total_available_samples}")

    if args.max_samples is not None and args.max_samples < total_available_samples:
        indices = list(range(args.max_samples))
        dataset = Subset(dataset, indices)
        logger.info(f"Evaluating subset of {len(dataset)} samples (--max_samples={args.max_samples})")

    evaluated_samples_count = len(dataset)
    if evaluated_samples_count == 0:
        raise ValueError(f"No samples to evaluate in split '{args.split}'!")

    g = torch.Generator()
    g.manual_seed(args.seed)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers if device.type == "cuda" else 0,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # 6. Run Evaluation Pass (Deterministic, Exploration Noise OFF)
    logger.info("Executing deterministic evaluation forward pass...")
    sample_case_names: List[str] = []
    sample_index = 0

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                images = batch["image"].to(device, non_blocking=True)
                case_names = batch.get("case_name", [f"sample_{sample_index + i}" for i in range(images.size(0))])
                sample_case_names.extend(case_names)
                sample_index += images.size(0)

                # Forward pass triggers router hooks
                _ = model(images)
    finally:
        # Always remove hooks cleanly
        for c in collectors:
            c.remove()

    logger.info(f"Inference completed. Evaluated {len(sample_case_names)} samples across {num_routers} routers.")

    # 7. Consistency Checks (Fail Loudly on Violations)
    expected_selections_per_router = evaluated_samples_count * top_k
    for c in collectors:
        actual_count = len(c.top_k_indices)
        if actual_count != evaluated_samples_count:
            raise AssertionError(
                f"Sample count mismatch on router '{c.name}': collected {actual_count} samples vs expected {evaluated_samples_count}!"
            )
        total_selections = sum(len(indices) for indices in c.top_k_indices)
        if total_selections != expected_selections_per_router:
            raise AssertionError(
                f"Selection count mismatch on router '{c.name}': collected {total_selections} vs expected {expected_selections_per_router}!"
            )

    # 8. Aggregation A: Global Expert Usage
    logger.info("Computing Aggregation A: Global Expert Usage...")
    expert_selection_counts = {e_idx: 0 for e_idx in range(num_experts)}
    for c in collectors:
        for indices in c.top_k_indices:
            for e_idx in indices:
                expert_selection_counts[e_idx] += 1

    total_global_selections = num_routers * evaluated_samples_count * top_k
    actual_global_selections = sum(expert_selection_counts.values())
    assert actual_global_selections == total_global_selections, (
        f"Global selection count mismatch: {actual_global_selections} != {total_global_selections}"
    )

    expected_uniform_ratio = top_k / num_experts

    expert_usage_rows = []
    for exp in experts_meta:
        e_idx = exp["index"]
        sel_count = expert_selection_counts[e_idx]
        sel_ratio = sel_count / total_global_selections if total_global_selections > 0 else 0.0
        rel_uniform = sel_ratio / expected_uniform_ratio if expected_uniform_ratio > 0 else 0.0

        expert_usage_rows.append({
            "Expert": exp["column_name"],
            "Name": exp["name"],
            "Type": exp["type"],
            "Shared": "Yes" if exp["is_shared"] else "No",
            "Selection_Count": sel_count,
            "Selection_Ratio": f"{sel_ratio:.6f}",
            "Expected_Uniform_Ratio": f"{expected_uniform_ratio:.6f}",
            "Relative_To_Uniform": f"{rel_uniform:.4f}",
        })

    expert_usage_csv_path = os.path.join(args.output_dir, "expert_usage.csv")
    with open(expert_usage_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Expert", "Name", "Type", "Shared", "Selection_Count", "Selection_Ratio", "Expected_Uniform_Ratio", "Relative_To_Uniform"]
        )
        writer.writeheader()
        writer.writerows(expert_usage_rows)
    logger.info(f"Saved: {expert_usage_csv_path}")

    # 9. Aggregation B: Per-Router Expert Usage
    logger.info("Computing Aggregation B: Per-Router Expert Usage...")
    per_router_rows = []
    per_router_distribution_map: Dict[str, np.ndarray] = {}

    for c in collectors:
        r_counts = np.zeros(num_experts, dtype=np.int64)
        for indices in c.top_k_indices:
            for e_idx in indices:
                r_counts[e_idx] += 1

        r_total = evaluated_samples_count * top_k
        r_dist = r_counts / r_total if r_total > 0 else np.zeros(num_experts, dtype=np.float64)
        per_router_distribution_map[c.name] = r_dist

        stats = compute_entropy_and_concentration(r_dist)

        row = {
            "Router": c.name,
            "Layer_Type": c.layer_type,
        }
        for e_idx in range(num_experts):
            row[f"E{e_idx}"] = f"{r_dist[e_idx]:.6f}"

        row["Entropy"] = f"{stats['entropy_bits']:.4f}"
        row["Normalized_Entropy"] = f"{stats['normalized_entropy']:.4f}"
        row["Max_Expert_Share"] = f"{stats['max_expert_share']:.4f}"
        row["Concentration"] = f"{stats['hhi_concentration']:.4f}"
        row["Effective_Num_Experts"] = f"{stats['effective_num_experts']:.2f}"

        per_router_rows.append(row)

    expert_cols = [f"E{i}" for i in range(num_experts)]
    per_router_csv_path = os.path.join(args.output_dir, "per_router_usage.csv")
    with open(per_router_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["Router", "Layer_Type"] + expert_cols + ["Entropy", "Normalized_Entropy", "Max_Expert_Share", "Concentration", "Effective_Num_Experts"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_router_rows)
    logger.info(f"Saved: {per_router_csv_path}")

    # 10. Aggregation C: Family-Level Routing (CNN vs ViT Experts)
    logger.info("Computing Aggregation C: Family-Level Routing...")
    family_rows = []
    cnn_expert_indices = {e["index"] for e in experts_meta if e["type"] == "CNN"}
    vit_expert_indices = {e["index"] for e in experts_meta if e["type"] == "ViT"}

    for c in collectors:
        cnn_selections = 0
        vit_selections = 0
        for indices in c.top_k_indices:
            for e_idx in indices:
                if e_idx in cnn_expert_indices:
                    cnn_selections += 1
                elif e_idx in vit_expert_indices:
                    vit_selections += 1

        total_selections = cnn_selections + vit_selections
        assert total_selections == evaluated_samples_count * top_k, "Family selection sum mismatch!"

        cnn_ratio = cnn_selections / total_selections if total_selections > 0 else 0.0
        vit_ratio = vit_selections / total_selections if total_selections > 0 else 0.0

        family_rows.append({
            "Router": c.name,
            "Layer_Type": c.layer_type,
            "CNN_Expert_Selections": cnn_selections,
            "CNN_Expert_Selection_Ratio": f"{cnn_ratio:.6f}",
            "ViT_Expert_Selections": vit_selections,
            "ViT_Expert_Selection_Ratio": f"{vit_ratio:.6f}",
            "Total_Selections": total_selections,
        })

    family_csv_path = os.path.join(args.output_dir, "family_routing.csv")
    with open(family_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["Router", "Layer_Type", "CNN_Expert_Selections", "CNN_Expert_Selection_Ratio", "ViT_Expert_Selections", "ViT_Expert_Selection_Ratio", "Total_Selections"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(family_rows)
    logger.info(f"Saved: {family_csv_path}")

    # 11. Aggregation D: Entropy & Concentration
    logger.info("Computing Aggregation D: Entropy and Concentration...")
    entropy_rows = []
    for c in collectors:
        r_dist = per_router_distribution_map[c.name]
        stats = compute_entropy_and_concentration(r_dist)
        norm_gating_entropy = compute_normalized_gating_weight_entropy(c.gating_weights)

        entropy_rows.append({
            "Router": c.name,
            "Layer_Type": c.layer_type,
            "Entropy_Bits": f"{stats['entropy_bits']:.4f}",
            "Normalized_Entropy": f"{stats['normalized_entropy']:.4f}",
            "Max_Expert_Share": f"{stats['max_expert_share']:.4f}",
            "HHI_Concentration": f"{stats['hhi_concentration']:.4f}",
            "Effective_Num_Experts": f"{stats['effective_num_experts']:.2f}",
            "Normalized_Selected_Gating_Weight_Entropy": f"{norm_gating_entropy:.4f}",
        })

    entropy_csv_path = os.path.join(args.output_dir, "entropy_concentration.csv")
    with open(entropy_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["Router", "Layer_Type", "Entropy_Bits", "Normalized_Entropy", "Max_Expert_Share", "HHI_Concentration", "Effective_Num_Experts", "Normalized_Selected_Gating_Weight_Entropy"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(entropy_rows)
    logger.info(f"Saved: {entropy_csv_path}")

    # 12. Aggregation E: Affinity Matrix (Mean and Std of Sigmoid Affinity)
    logger.info("Computing Aggregation E: Affinity Matrix...")
    affinity_rows = []
    mean_cols = [f"E{i}_mean" for i in range(num_experts)]
    std_cols = [f"E{i}_std" for i in range(num_experts)]

    for c in collectors:
        # c.sigmoid_affinity is list of [B, M] arrays
        aff_array = np.asarray(c.sigmoid_affinity, dtype=np.float64)  # Shape [S, M]
        means = np.mean(aff_array, axis=0)
        stds = np.std(aff_array, axis=0)

        row = {
            "Router": c.name,
            "Layer_Type": c.layer_type,
        }
        for e_idx in range(num_experts):
            row[f"E{e_idx}_mean"] = f"{means[e_idx]:.6f}"
            row[f"E{e_idx}_std"] = f"{stds[e_idx]:.6f}"

        affinity_rows.append(row)

    affinity_csv_path = os.path.join(args.output_dir, "affinity_matrix.csv")
    with open(affinity_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["Router", "Layer_Type"] + mean_cols + std_cols
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(affinity_rows)
    logger.info(f"Saved: {affinity_csv_path}")

    # 13. Aggregation F: g_s Statistics (Shared vs Dynamic Gate)
    logger.info("Computing Aggregation F: g_s Gate Statistics...")
    gs_rows = []
    all_cnn_gs = []
    all_vit_gs = []

    for c in collectors:
        gs_vals = np.asarray(c.g_s_values, dtype=np.float64)
        if c.layer_type == "CNN":
            all_cnn_gs.extend(gs_vals.tolist())
        else:
            all_vit_gs.extend(gs_vals.tolist())

        gs_rows.append({
            "Router": c.name,
            "Layer_Type": c.layer_type,
            "Count": len(gs_vals),
            "Mean": f"{np.mean(gs_vals):.4f}",
            "Std": f"{np.std(gs_vals):.4f}",
            "Min": f"{np.min(gs_vals):.4f}",
            "Max": f"{np.max(gs_vals):.4f}",
            "Median": f"{np.median(gs_vals):.4f}",
        })

    # Summary rows
    all_gs = all_cnn_gs + all_vit_gs
    if all_cnn_gs:
        gs_rows.append({
            "Router": "SUMMARY_CNN_ROUTERS",
            "Layer_Type": "CNN",
            "Count": len(all_cnn_gs),
            "Mean": f"{np.mean(all_cnn_gs):.4f}",
            "Std": f"{np.std(all_cnn_gs):.4f}",
            "Min": f"{np.min(all_cnn_gs):.4f}",
            "Max": f"{np.max(all_cnn_gs):.4f}",
            "Median": f"{np.median(all_cnn_gs):.4f}",
        })
    if all_vit_gs:
        gs_rows.append({
            "Router": "SUMMARY_VIT_ROUTERS",
            "Layer_Type": "ViT",
            "Count": len(all_vit_gs),
            "Mean": f"{np.mean(all_vit_gs):.4f}",
            "Std": f"{np.std(all_vit_gs):.4f}",
            "Min": f"{np.min(all_vit_gs):.4f}",
            "Max": f"{np.max(all_vit_gs):.4f}",
            "Median": f"{np.median(all_vit_gs):.4f}",
        })
    if all_gs:
        gs_rows.append({
            "Router": "SUMMARY_ALL_ROUTERS",
            "Layer_Type": "ALL",
            "Count": len(all_gs),
            "Mean": f"{np.mean(all_gs):.4f}",
            "Std": f"{np.std(all_gs):.4f}",
            "Min": f"{np.min(all_gs):.4f}",
            "Max": f"{np.max(all_gs):.4f}",
            "Median": f"{np.median(all_gs):.4f}",
        })

    gs_csv_path = os.path.join(args.output_dir, "gs_statistics.csv")
    with open(gs_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["Router", "Layer_Type", "Count", "Mean", "Std", "Min", "Max", "Median"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(gs_rows)
    logger.info(f"Saved: {gs_csv_path}")

    # 14. Consolidated Machine-Readable JSON Report
    logger.info("Building consolidated routing_statistics.json...")
    compact_sample_records = []
    # Save compact per-sample routing records
    for s_idx in range(evaluated_samples_count):
        case_name = sample_case_names[s_idx]
        sample_routing = []
        for c in collectors:
            sample_routing.append({
                "router": c.name,
                "layer_type": c.layer_type,
                "selected_experts": c.top_k_indices[s_idx],
                "selected_scores": [round(float(w), 4) for w in c.gating_weights[s_idx]],
                "g_s": round(float(c.g_s_values[s_idx]), 4),
            })
        compact_sample_records.append({
            "sample_idx": s_idx,
            "case_name": case_name,
            "routing": sample_routing,
        })

    full_json_data = {
        "metadata": {
            "model": config.get("model", "B2"),
            "p3_mode": config.get("p3_mode", None),
            "vit_depth": config.get("num_transformer_layers", 12),
            "num_routers": num_routers,
            "num_experts": num_experts,
            "top_k": top_k,
            "split": args.split,
            "num_samples": evaluated_samples_count,
            "checkpoint": os.path.abspath(args.checkpoint),
            "checkpoint_sha256": ckpt_meta.get("sha256", ""),
            "checkpoint_metadata": {
                "stage": ckpt_meta.get("stage"),
                "epoch": ckpt_meta.get("epoch"),
                "best_dice": ckpt_meta.get("best_dice"),
                "best_loss": ckpt_meta.get("best_loss"),
                "p3_mode": ckpt_meta.get("p3_mode"),
                "num_transformer_layers": ckpt_meta.get("num_transformer_layers"),
            },
            "config": os.path.abspath(args.config),
            "data_root": config.get("root_dir", ""),
            "evaluation_mode": "eval",
            "exploration_noise_active": False,
            "timestamp": datetime.datetime.now().isoformat(),
        },
        "routers": [
            {"index": r["router_index"], "name": r["name"], "layer_type": r["layer_type"]}
            for r in routers_meta
        ],
        "experts": experts_meta,
        "global_expert_usage": expert_usage_rows,
        "per_router_usage": per_router_rows,
        "family_routing": family_rows,
        "entropy_concentration": entropy_rows,
        "gs_statistics": gs_rows,
        "consistency_checks": {
            "per_router_total_selections_match": True,
            "global_total_selections_match": True,
            "evaluated_samples": evaluated_samples_count,
            "expected_selections_per_router": expected_selections_per_router,
            "total_global_selections": total_global_selections,
        },
        "sample_routing_records": compact_sample_records,
    }

    json_report_path = os.path.join(args.output_dir, "routing_statistics.json")
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(full_json_data, f, indent=2)
    logger.info(f"Saved: {json_report_path}")

    # Summary Output
    print("\n" + "=" * 80)
    print("ROUTING DIAGNOSTICS EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Model:                {config.get('model', 'B2')} (p3_mode='{config.get('p3_mode')}', ViT Depth={config.get('num_transformer_layers')})")
    print(f"Split Evaluated:      {args.split} ({evaluated_samples_count} samples)")
    print(f"Routers Evaluated:    {num_routers} ({sum(1 for r in routers_meta if r['layer_type'] == 'CNN')} CNN + {sum(1 for r in routers_meta if r['layer_type'] == 'ViT')} ViT)")
    print(f"Expert Pool Size:     {num_experts} (top_k={top_k})")
    print(f"Total Selections:     {total_global_selections}")
    print(f"Output Directory:     {os.path.abspath(args.output_dir)}")
    print(f"Artifacts Generated:")
    print(f"  - {os.path.basename(json_report_path)}")
    print(f"  - {os.path.basename(expert_usage_csv_path)}")
    print(f"  - {os.path.basename(per_router_csv_path)}")
    print(f"  - {os.path.basename(family_csv_path)}")
    print(f"  - {os.path.basename(entropy_csv_path)}")
    print(f"  - {os.path.basename(affinity_csv_path)}")
    print(f"  - {os.path.basename(gs_csv_path)}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
