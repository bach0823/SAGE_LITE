#!/usr/bin/env python3
"""
scripts/diagnostics/evaluate_routing_and_asdw_ablation.py

Zero-retraining evaluation on best_model_b2_stage2.pth checkpoint:
1. Routing Modes:
   - Adaptive Routing (Default top-k=2 router decisions)
   - Static Top-K Routing (Fixed top-k=2 experts from checkpoint usage)
   - Random Top-K Routing (Uniformly sampled top-k=2 experts)
2. ASDW Ablation:
   - ASDW ON (Original learned gamma: gamma ~ 0.0086, 0.0067)
   - ASDW OFF (Bypass: gamma=0.0 -> exact identity x + 0*f = x)

Matrix of evaluations across all 348 validation images in Setting A:
  Condition 1: Adaptive Routing + ASDW ON (Baseline)
  Condition 2: Adaptive Routing + ASDW OFF (Ablate ASDW)
  Condition 3: Static Routing + ASDW ON
  Condition 4: Static Routing + ASDW OFF
  Condition 5: Random Routing + ASDW ON
  Condition 6: Random Routing + ASDW OFF
"""

import argparse
import copy
import glob
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import yaml

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sage.components.router import SageRouter
from sage.networks import create_b2_unet
from scripts.evaluate_crack_official import predict_full_image_tiling_setting_a

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate_routing_and_asdw")


def compute_modulated_logits(router: SageRouter, input_tensor: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        agg = router._aggregate_features_with_adaptation(input_tensor)
        g_s = torch.sigmoid(router.shared_expert_gate(agg))
        query = router.query_projection(agg)
        base_logits = torch.matmul(query, router.expert_keys.T) / router.temperature

        if router.logit_modulation:
            eps = 1e-5
            orig_dtype = base_logits.dtype
            g_s_clamped = torch.clamp(g_s.float(), min=eps, max=1.0 - eps)
            log_g_s = torch.log(g_s_clamped)
            log_one_minus_g_s = torch.log(1.0 - g_s_clamped)
            mask_f32 = router.shared_mask.float()
            modulated = (
                base_logits.float() +
                mask_f32 * log_g_s +
                (1.0 - mask_f32) * log_one_minus_g_s
            ).to(orig_dtype)
        else:
            modulated = base_logits

        return modulated


class RoutingInterventionContext:
    def __init__(
        self,
        router_modules: Dict[str, SageRouter],
        mode: str,
        static_policy: Dict[str, List[int]] = None,
        random_seed: int = 42,
    ):
        self.router_modules = router_modules
        self.mode = mode.lower()
        self.static_policy = static_policy or {}
        self.random_seed = random_seed
        self.hook_handles = []
        self.generator = torch.Generator(device="cpu").manual_seed(random_seed)

    def _make_hook(self, alias: str, router: SageRouter):
        def hook(module: SageRouter, inputs: Tuple[torch.Tensor, ...], output: Tuple[Any, ...]):
            if self.mode == "adaptive":
                return output

            input_tensor = inputs[0]
            B = input_tensor.shape[0]
            device = input_tensor.device
            M = module.expert_pool_size
            K = module.top_k

            mod_logits = compute_modulated_logits(module, input_tensor)

            if self.mode == "static":
                fixed_k = self.static_policy.get(alias)
                if not fixed_k or len(fixed_k) != K:
                    raise ValueError(f"Static policy missing or invalid for router '{alias}': {fixed_k}")
                sel_indices = torch.tensor(fixed_k, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)
            elif self.mode == "random":
                batch_perms = []
                for _ in range(B):
                    perm = torch.randperm(M, generator=self.generator)[:K]
                    batch_perms.append(perm)
                sel_indices = torch.stack(batch_perms, dim=0).to(device)
            else:
                raise ValueError(f"Unknown mode {self.mode}")

            sel_logits = mod_logits.gather(dim=-1, index=sel_indices)
            if module.gating_type == "softmax":
                gating_weights = F.softmax(sel_logits, dim=-1)
            else:
                gating_weights = torch.sigmoid(sel_logits)

            orig_indices, orig_weights, orig_info = output
            new_info = dict(orig_info)
            new_info["intervention_mode"] = self.mode
            new_info["intervened_indices"] = sel_indices
            return sel_indices, gating_weights, new_info

        return hook

    def __enter__(self):
        self.hook_handles = []
        for alias, router in self.router_modules.items():
            h = router.register_forward_hook(self._make_hook(alias, router))
            self.hook_handles.append(h)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()


def discover_routers(model: nn.Module) -> Dict[str, SageRouter]:
    routers = {}
    if hasattr(model, "backbone"):
        if hasattr(model.backbone, "convnext") and hasattr(model.backbone.convnext, "stages"):
            for idx, stage in enumerate(model.backbone.convnext.stages):
                if hasattr(stage, "router") and isinstance(stage.router, SageRouter):
                    routers[f"S{idx}"] = stage.router
        if hasattr(model.backbone, "transformer_blocks"):
            for idx, blk in enumerate(model.backbone.transformer_blocks):
                if hasattr(blk, "router") and isinstance(blk.router, SageRouter):
                    routers[f"B{idx}"] = blk.router
    return routers


def extract_static_policy(ckpt_sd: Dict[str, Any], router_keys: List[str], k: int = 2) -> Dict[str, List[int]]:
    policy = {}
    for alias in ["S0", "S1", "S2", "S3", "B0", "B1", "B2", "B3"]:
        if alias.startswith("S"):
            idx = alias[1]
            buf_name = f"backbone.convnext.stages.{idx}.router.expert_usage_count"
        else:
            idx = alias[1]
            buf_name = f"backbone.transformer_blocks.{idx}.router.expert_usage_count"
        if buf_name in ckpt_sd:
            counts = ckpt_sd[buf_name].cpu().numpy()
            top_k = counts.argsort()[::-1][:k].tolist()
            policy[alias] = top_k
        else:
            policy[alias] = list(range(k))
    return policy


def predict_tiling_fp32(model, image, device, tile_size=448, batch_size=8):
    H, W = image.shape[:2]
    pad_h = (tile_size - (H % tile_size)) % tile_size
    pad_w = (tile_size - (W % tile_size)) % tile_size
    padded_img = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    pH, pW = padded_img.shape[:2]

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    patches = []
    coords = []
    for y in range(0, pH, tile_size):
        for x in range(0, pW, tile_size):
            patch = padded_img[y:y+tile_size, x:x+tile_size]
            patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
            patches.append(patch_tensor)
            coords.append((y, x))

    pred_logits = np.zeros((pH, pW), dtype=np.float32)
    for i in range(0, len(patches), batch_size):
        batch = torch.stack(patches[i:i+batch_size]).to(device, non_blocking=True)
        batch = (batch - mean) / std
        with torch.no_grad():
            logits = model(batch)
            logits_np = logits.squeeze(1).cpu().numpy()
        for j, logit in enumerate(logits_np):
            y, x = coords[i+j]
            pred_logits[y:y+tile_size, x:x+tile_size] = logit

    return pred_logits[:H, :W]


def evaluate_condition(
    model: nn.Module,
    cached_images: Dict[str, Tuple[np.ndarray, np.ndarray]],
    device: torch.device,
) -> Dict[str, float]:
    dices, ious, precs, recs = [], [], [], []
    with torch.no_grad():
        for stem, (img_rgb, target) in cached_images.items():
            logits_np = predict_tiling_fp32(
                model, img_rgb, device, tile_size=448, batch_size=8
            )
            pred = (logits_np > 0.0).astype(np.uint8)
            H, W = target.shape[:2]
            pred = pred[:H, :W]

            tp = int(np.sum((pred == 1) & (target == 1)))
            fp = int(np.sum((pred == 1) & (target == 0)))
            fn = int(np.sum((pred == 0) & (target == 1)))

            dice = float((2.0 * tp) / (2.0 * tp + fp + fn)) if (2.0 * tp + fp + fn) > 0 else (1.0 if (tp + fp + fn) == 0 else 0.0)
            iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else (1.0 if (tp + fp + fn) == 0 else 0.0)
            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)

            dices.append(dice)
            ious.append(iou)
            precs.append(prec)
            recs.append(rec)

    return {
        "dice_mean": float(np.mean(dices)),
        "dice_std": float(np.std(dices)),
        "dice_median": float(np.median(dices)),
        "iou_mean": float(np.mean(ious)),
        "precision_mean": float(np.mean(precs)),
        "recall_mean": float(np.mean(recs)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/p3_ablation/phase6_full_s1/a1_s2g_end_to_end.yaml")
    parser.add_argument("--checkpoint", default="results/phase6_combination/phase6_comb_a1_s2g_end_to_end/best_model_b2_stage2.pth")
    parser.add_argument("--data-root", default="datasets/Crack500_ready")
    parser.add_argument("--output-dir", default="results/diagnostics/routing_and_asdw_ablation/a1_s2g_end_to_end")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not os.path.exists(args.data_root):
        alt = os.path.join("..", args.data_root)
        if os.path.exists(alt):
            args.data_root = alt

    logger.info(f"Using data_root: {os.path.abspath(args.data_root)}")
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # 1. Load config & build model
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = create_b2_unet(
        num_classes=1,
        img_size=cfg.get("img_size", 448),
        num_transformer_layers=cfg.get("num_transformer_layers", 4),
        pretrained=False,
        sage_config=cfg.get("sage_config", {}),
        p3_mode=cfg.get("p3_mode", "C"),
        use_plu_head=cfg.get("use_plu_head", False),
        use_s2_gate=cfg.get("use_s2_gate", False),
        s2_gate_kernel_size=cfg.get("s2_gate_kernel_size", 3),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(ckpt_sd, strict=True)
    model.eval()

    # 2. Discover routers and extract static policy
    router_map = discover_routers(model)
    static_policy = extract_static_policy(ckpt_sd, list(router_map.keys()), k=cfg.get("sage_config", {}).get("top_k", 2))
    logger.info(f"Discovered {len(router_map)} routers. Static top-2 policy: {static_policy}")

    # 3. Cache ASDW original gammas
    asdw_modules = []
    original_gammas = []
    for name, mod in model.named_modules():
        if hasattr(mod, "gamma") and isinstance(mod.gamma, torch.Tensor):
            asdw_modules.append((name, mod))
            original_gammas.append(mod.gamma.data.clone())
            logger.info(f"Found ASDW module: {name} (learned gamma = {mod.gamma.item():.6f})")

    # 4. Load validation dataset
    val_img_dir = os.path.join(args.data_root, "val", "images")
    val_mask_dir = os.path.join(args.data_root, "val", "masks")
    img_paths = sorted(glob.glob(os.path.join(val_img_dir, "*")))
    img_paths = [p for p in img_paths if os.path.splitext(p)[1].lower() in [".jpg", ".jpeg", ".png"]]

    cached_images = {}
    for ip in img_paths:
        stem = os.path.splitext(os.path.basename(ip))[0]
        for ext in [".png", ".jpg", ".jpeg"]:
            mp = os.path.join(val_mask_dir, stem + ext)
            if os.path.exists(mp):
                img_bgr = cv2.imread(ip)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                cached_images[stem] = (img_rgb, (mask > 0).astype(np.uint8))
                break

    logger.info(f"Cached {len(cached_images)} validation images.")

    # 5. Run Matrix of 6 Experiments
    matrix_experiments = [
        ("Adaptive_Routing", "ASDW_ON"),
        ("Adaptive_Routing", "ASDW_OFF"),
        ("Static_Routing",   "ASDW_ON"),
        ("Static_Routing",   "ASDW_OFF"),
        ("Random_Routing",   "ASDW_ON"),
        ("Random_Routing",   "ASDW_OFF"),
    ]

    results_table = []
    for routing_mode, asdw_status in matrix_experiments:
        logger.info(f"\n=======================================================")
        logger.info(f"EVALUATING: Routing={routing_mode} | ASDW={asdw_status}")
        logger.info(f"=======================================================")

        # Configure ASDW
        if asdw_status == "ASDW_OFF":
            for name, mod in asdw_modules:
                mod.gamma.data.zero_()
        else:
            for (name, mod), g_orig in zip(asdw_modules, original_gammas):
                mod.gamma.data.copy_(g_orig)

        # Configure Routing Context
        r_mode = routing_mode.split("_")[0].lower()
        with RoutingInterventionContext(router_map, mode=r_mode, static_policy=static_policy, random_seed=42):
            t0 = time.time()
            metrics = evaluate_condition(model, cached_images, device)
            dur = time.time() - t0

        row = {
            "routing_mode": routing_mode,
            "asdw_status": asdw_status,
            "dice_mean": metrics["dice_mean"],
            "dice_std": metrics["dice_std"],
            "dice_median": metrics["dice_median"],
            "iou_mean": metrics["iou_mean"],
            "precision_mean": metrics["precision_mean"],
            "recall_mean": metrics["recall_mean"],
            "elapsed_sec": dur,
        }
        results_table.append(row)
        logger.info(f"Result: Dice={metrics['dice_mean']:.4f} (median={metrics['dice_median']:.4f}), IoU={metrics['iou_mean']:.4f}, Prec={metrics['precision_mean']:.4f}, Rec={metrics['recall_mean']:.4f} [{dur:.1f}s]")

    # Restore gammas
    for (name, mod), g_orig in zip(asdw_modules, original_gammas):
        mod.gamma.data.copy_(g_orig)

    # Save summary
    df = pd.DataFrame(results_table)
    csv_path = os.path.join(args.output_dir, "routing_and_asdw_ablation_summary.csv")
    json_path = os.path.join(args.output_dir, "routing_and_asdw_ablation_summary.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_table, f, indent=2)

    logger.info(f"\nAll evaluations completed! Saved to:\n- {csv_path}\n- {json_path}")
    print("\nFINAL SUMMARY TABLE:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
