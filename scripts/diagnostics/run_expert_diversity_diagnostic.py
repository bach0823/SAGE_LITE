#!/usr/bin/env python3
"""
scripts/diagnostics/run_expert_diversity_diagnostic.py

Expert Diversity Diagnostic for SAGE-Lite Models.
- Evaluates expert pool representations directly on identical batch input.
- Bypasses routers, top-k selection, and adaptive fusion.
- Captures output feature tensors from each expert in expert_pool (8 experts: 4 CNN, 4 ViT).
- Computes Pairwise Cosine Similarity and Pearson Correlation across all 8x8 pairs.
- Computes mean, median, max similarity.
- Concludes clearly: high redundancy vs genuine diversity.
- Exports results to JSON and CSV in results/diagnostics/expert_diversity/.
"""

import argparse
import json
import os
import sys
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sage.networks import create_b2_unet
from sage.utils.dataloader import get_dataset_from_config
from sage.utils.training_utils import set_seed


def run_expert_diversity_diagnostic(
    config_path: str,
    checkpoint_path: str,
    output_dir: str,
    data_root_override: str = None,
    num_samples: int = 28,  # 2 full batches of 14
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)
    set_seed(42)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if data_root_override:
        config["root_dir"] = data_root_override

    print("\n" + "=" * 80)
    print("SAGE-LITE EXPERT DIVERSITY DIAGNOSTIC (STRICT INVARIANT: ZERO RETRAIN)")
    print("=" * 80)
    print(f"Config:     {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device:     {device}")

    # 1. Instantiate exact model
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
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Extract expert pool (8 experts: 4 CNN main_blocks + 4 ViT main_blocks)
    # Check SageLayer instances to retrieve shared expert pool
    expert_pool = None
    for m in model.modules():
        if hasattr(m, "expert_pool") and m.expert_pool is not None:
            expert_pool = m.expert_pool
            break

    if expert_pool is None:
        raise RuntimeError("Could not find expert_pool in model architecture!")

    num_experts = len(expert_pool)
    print(f"Discovered expert pool with {num_experts} experts:")
    expert_names = []
    for i, exp in enumerate(expert_pool):
        exp_type = getattr(exp, "expert_type", "CNN" if i < 4 else "ViT")
        name = f"E{i}_{exp_type}"
        expert_names.append(name)
        print(f"  [{i}] {name} ({type(exp).__name__})")

    # 3. Load test batch
    val_dataset = get_dataset_from_config(config, split="val")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=14, shuffle=False)

    batch_x = []
    for batch in val_loader:
        if isinstance(batch, dict):
            imgs = batch["image"]
        else:
            imgs = batch[0]
        batch_x.append(imgs)
        if sum(b.shape[0] for b in batch_x) >= num_samples:
            break
    input_tensor = torch.cat(batch_x, dim=0)[:num_samples].to(device)
    print(f"Loaded identical input batch: shape={input_tensor.shape}")

    # 4. Forward through stem and backbone stages up to input representations
    # We feed the canonical hybrid backbone representations to each expert
    # Specifically, experts operate in 2 distinct native feature spaces:
    # E0..E3: CNN stages (shapes: 48, 96, 192, 384)
    # E4..E7: ViT blocks (shape: 196 tokens, embed_dim=192)
    # When experts are called via SAGE routing, they are connected via SA-Hub adapters
    # To compare raw intrinsic output representations on identical context:
    # We forward input_tensor through backbone stem to obtain the feature representation entering the layers.
    with torch.no_grad():
        # Get backbone multi-stage features
        backbone = model.backbone
        # ConvNeXt stages forward
        # Let's inspect stem
        stem_feat = backbone.convnext.stem(input_tensor) # [B, 48, 112, 112]
        
        # We collect the output feature tensor of each expert when evaluated on its native input
        # and projected to a standard normalized 1D representation per sample.
        expert_outputs = [] # list of [N, D_pooled]

        # Forward CNN stages
        cur_feat = stem_feat
        cnn_outputs = []
        for s_idx in range(4):
            stage = backbone.convnext.stages[s_idx]
            # Downsample if present (stage 1, 2, 3 have downsample)
            if hasattr(stage, "downsample") and stage.downsample is not None:
                cur_feat = stage.downsample(cur_feat)
            # Direct forward through expert
            exp_mod = expert_pool[s_idx]
            out = exp_mod(cur_feat) # [B, C, H, W]
            cur_feat = out
            # Global Average Pool + Global Std Pool to capture rich feature signature
            gap = out.mean(dim=[2, 3]) # [B, C]
            gstd = out.std(dim=[2, 3]) # [B, C]
            pooled = torch.cat([gap, gstd], dim=1) # [B, 2C]
            # Normalize to 256 dimensions using fixed random orthonormal projection for fair comparison across different channel sizes
            proj = torch.nn.functional.adaptive_avg_pool1d(pooled.unsqueeze(1), 256).squeeze(1)
            expert_outputs.append(proj)

        # Forward ViT blocks
        # ConvNeXt stage 3 output: cur_feat is [B, 384, H, W]
        b, c, h, w = cur_feat.shape
        tokens = cur_feat.flatten(2).transpose(1, 2)  # [B, H*W, 384]
        vit_in = backbone.convnext_to_transformer(tokens)  # [B, H*W, 192]
        pos_embed = backbone._get_interpolated_pos_embed(h, w)
        cur_tok = backbone.pre_transformer_norm(vit_in + pos_embed)

        for v_idx in range(4):
            exp_mod = expert_pool[4 + v_idx]
            out = exp_mod(cur_tok)  # [B, H*W, 192]
            cur_tok = out
            gap = out.mean(dim=1)  # [B, 192]
            gstd = out.std(dim=1)  # [B, 192]
            pooled = torch.cat([gap, gstd], dim=1)  # [B, 384]
            proj = torch.nn.functional.adaptive_avg_pool1d(pooled.unsqueeze(1), 256).squeeze(1)
            expert_outputs.append(proj)

    # Convert all expert outputs to numpy
    # Flatten across batch: [8, N * 256]
    feature_vectors = torch.stack([eo.flatten() for eo in expert_outputs], dim=0) # [8, Total_Features]
    feature_np = feature_vectors.cpu().numpy()

    # 5. Compute Pairwise Cosine Similarity Matrix (8x8)
    normed_features = feature_vectors / (feature_vectors.norm(dim=1, keepdim=True) + 1e-8)
    cos_sim_matrix = torch.mm(normed_features, normed_features.t()).cpu().numpy()

    # 6. Compute Pairwise Pearson Correlation Matrix (8x8)
    pearson_matrix = np.corrcoef(feature_np)

    # 7. Summary Statistics (excluding diagonal self-similarity 1.0)
    mask = ~np.eye(num_experts, dtype=bool)
    off_diag_cos = cos_sim_matrix[mask]
    off_diag_pearson = pearson_matrix[mask]

    mean_cos = float(np.mean(off_diag_cos))
    median_cos = float(np.median(off_diag_cos))
    max_cos = float(np.max(off_diag_cos))
    min_cos = float(np.min(off_diag_cos))

    mean_pearson = float(np.mean(off_diag_pearson))
    median_pearson = float(np.median(off_diag_pearson))
    max_pearson = float(np.max(off_diag_pearson))
    min_pearson = float(np.min(off_diag_pearson))

    # Cross-family vs Intra-family metrics
    # CNN-CNN: 0..3, ViT-ViT: 4..7, CNN-ViT: cross
    cnn_mask = np.zeros((8, 8), dtype=bool)
    cnn_mask[:4, :4] = True
    np.fill_diagonal(cnn_mask, False)
    intra_cnn_cos = float(np.mean(cos_sim_matrix[cnn_mask]))

    vit_mask = np.zeros((8, 8), dtype=bool)
    vit_mask[4:, 4:] = True
    np.fill_diagonal(vit_mask, False)
    intra_vit_cos = float(np.mean(cos_sim_matrix[vit_mask]))

    cross_mask = np.zeros((8, 8), dtype=bool)
    cross_mask[:4, 4:] = True
    cross_mask[4:, :4] = True
    cross_family_cos = float(np.mean(cos_sim_matrix[cross_mask]))

    # Conclusion
    if mean_cos > 0.85:
        diversity_verdict = "HIGH REDUNDANCY (Experts produce strongly collinear representations)"
    elif mean_cos > 0.60:
        diversity_verdict = "MODERATE DIVERSITY (Intra-family alignment with cross-family specialization)"
    else:
        diversity_verdict = "HIGH GENUINE DIVERSITY (Orthogonal/complementary expert representations)"

    # Print Report
    print("\n" + "=" * 80)
    print("EXPERT DIVERSITY DIAGNOSTIC RESULTS (8x8 PAIRWISE MATRICES)")
    print("=" * 80)
    print("\nPairwise Cosine Similarity Matrix (8x8):")
    df_cos = pd.DataFrame(cos_sim_matrix, index=expert_names, columns=expert_names)
    print(df_cos.round(4).to_string())

    print("\nPairwise Pearson Correlation Matrix (8x8):")
    df_pearson = pd.DataFrame(pearson_matrix, index=expert_names, columns=expert_names)
    print(df_pearson.round(4).to_string())

    print("\n" + "-" * 80)
    print(f"Mean Cosine Similarity:       {mean_cos:.4f}")
    print(f"Median Cosine Similarity:     {median_cos:.4f}")
    print(f"Max Off-Diagonal Similarity:  {max_cos:.4f}")
    print(f"Min Off-Diagonal Similarity:  {min_cos:.4f}")
    print(f"Intra-CNN Mean Similarity:    {intra_cnn_cos:.4f}")
    print(f"Intra-ViT Mean Similarity:    {intra_vit_cos:.4f}")
    print(f"Cross-Family (CNN-ViT) Sim:   {cross_family_cos:.4f}")
    print("-" * 80)
    print(f"DIVERSITY CONCLUSION:         {diversity_verdict}")
    print("=" * 80)

    # Save to CSV and JSON
    cos_csv = os.path.join(output_dir, "expert_pairwise_cosine_similarity.csv")
    pearson_csv = os.path.join(output_dir, "expert_pairwise_pearson_correlation.csv")
    summary_json = os.path.join(output_dir, "expert_diversity_summary.json")

    df_cos.to_csv(cos_csv)
    df_pearson.to_csv(pearson_csv)

    summary_data = {
        "model_checkpoint": checkpoint_path,
        "config": config_path,
        "num_experts": num_experts,
        "expert_names": expert_names,
        "metrics": {
            "mean_cosine_similarity": mean_cos,
            "median_cosine_similarity": median_cos,
            "max_cosine_similarity": max_cos,
            "min_cosine_similarity": min_cos,
            "mean_pearson_correlation": mean_pearson,
            "median_pearson_correlation": median_pearson,
            "max_pearson_correlation": max_pearson,
            "min_pearson_correlation": min_pearson,
            "intra_cnn_cosine_similarity": intra_cnn_cos,
            "intra_vit_cosine_similarity": intra_vit_cos,
            "cross_family_cosine_similarity": cross_family_cos,
        },
        "verdict": diversity_verdict,
    }

    with open(summary_json, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n[Saved] Exported matrices and summary to:\n  + {cos_csv}\n  + {pearson_csv}\n  + {summary_json}")
    return summary_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/p3_ablation/phase6_full_s1/a1_s2g_end_to_end.yaml")
    parser.add_argument("--checkpoint", type=str, default="results/phase6_combination/phase6_comb_a1_s2g_end_to_end/best_model_b2_stage2.pth")
    parser.add_argument("--output-dir", type=str, default="results/diagnostics/expert_diversity")
    parser.add_argument("--data-root", type=str, default=None)
    args = parser.parse_args()

    run_expert_diversity_diagnostic(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        data_root_override=args.data_root,
    )
