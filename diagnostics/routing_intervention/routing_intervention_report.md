# SAGE-Lite Routing Intervention Diagnostic Report

*Date:* 2026-09-28 22:27:57  
*Model:* Canonical D4-P3-C Standalone Model (8 experts: 4 CNN + 4 ViT, $top\_k=4$)  
*Checkpoint:* `D:\truong\SpecialSubjectTTNT\results\checkpoints\P3_C_D4_best_model_b2_global.pth` (Epoch 14, Best Val Dice: 0.7639)  
*Dataset Split:* Crack500 Validation Set (348 samples)  
*Protocol:* Setting A (Non-overlapping 448x448 Tiling)  
*Intervention Principle:* Non-invasive replacement of selected indices while preserving exact router-derived gating weights:
$$\text{selected\_logits} = \operatorname{gather}(\text{modulated\_logits}, \text{selected\_indices}), \quad g = \sigma(\text{selected\_logits})$$

---

## 1. Executive Summary & Overall Performance

| Mode | Overall Dice (Mean ± Std) | Median Dice | Overall IoU | Precision | Recall | Description |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Adaptive Baseline** | **0.7639** ± 0.1640 | 0.8049 | 0.6412 | 0.7298 | 0.8498 | Deterministic evaluation pass (exploration noise OFF) |
| **Static Top-4** | **0.7636** ± 0.1648 | 0.8074 | 0.6409 | 0.7291 | 0.8506 | Fixed Top-4 experts per router from training usage count |
| **Random Top-4 (10 seeds)** | **0.7633** ± 0.0004 | 0.8057 | 0.6404 | 0.7277 | 0.8517 | Uniform random sampling without replacement ($K=4, M=8$) |

---

## 2. Paired Statistical Comparisons (N = 348 Samples)

All comparisons are evaluated per-sample on identical validation images:

| Comparison | Mean $\Delta$ | Median $\Delta$ | 95% Confidence Interval | Paired $t$-stat ($p$-value) | Wilcoxon $W$ ($p$-value) | Win / Tie / Loss (Win Rate) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Adaptive vs Static** | +0.00037 | +0.00021 | [-0.00051, +0.00124] | $t = +0.82$ ($p = 4.1326e-01$) | $W = 27942.0$ ($p = 3.0529e-01$) | 189 / 4 / 155 (**54.3%**) |
| **Adaptive vs Random** | +0.00063 | +0.00036 | [+0.00005, +0.00121] | $t = +2.13$ ($p = 3.3535e-02$) | $W = 22814.0$ ($p = 1.0984e-04$) | 208 / 2 / 138 (**59.8%**) |
| **Static vs Random** | +0.00026 | +0.00017 | [-0.00057, +0.00110] | $t = +0.62$ ($p = 5.3702e-01$) | $W = 25539.0$ ($p = 1.6207e-02$) | 190 / 2 / 156 (**54.6%**) |

---

## 3. Thinness Quartile Breakdown

Quartiles were partitioned **once** from ground truth crack masks ($Q1 \le 0.083 < Q2 \le 0.124 < Q3 \le 0.176 < Q4$):

| Mode | Q1 (Lowest Thinness) | Q2 | Q3 | Q4 (Highest Thinness) |
| :--- | :---: | :---: | :---: | :---: |
| **Adaptive Baseline** | 0.8473 | 0.8005 | 0.7443 | 0.6637 |
| **Static Top-4** | 0.8474 | 0.8006 | 0.7427 | 0.6636 |
| **Random Top-4 (10 seeds)** | 0.8472 ± 0.0004 | 0.8000 ± 0.0006 | 0.7437 ± 0.0010 | 0.6623 ± 0.0005 |

---

## 4. Static Expert Policy (Option A: Checkpoint Training Accumulation)

Extracted from `expert_usage_count` accumulated over 13,552 training calls:

| Router | Alias | Selected Static Top-4 Experts | Selection Share in Training | Self-Expert (`my_index`) Included? |
| :--- | :---: | :---: | :---: | :---: |
| ConvNeXt Stage 0 | **S0** | `[E4, E5, E6, E7]` (All 4 ViT Experts) | 53.6% | No (Self: E0) |
| ConvNeXt Stage 1 | **S1** | `[E6, E1, E5, E3]` | 58.7% | **Yes (Self: E1)** |
| ConvNeXt Stage 2 | **S2** | `[E4, E6, E3, E5]` | 57.5% | No (Self: E2) |
| ConvNeXt Stage 3 | **S3** | `[E3, E5, E0, E2]` | 56.6% | **Yes (Self: E3)** |
| ViT Block 0 | **B0** | `[E0, E2, E6, E1]` | 51.5% | No (Self: E4) |
| ViT Block 1 | **B1** | `[E3, E4, E1, E2]` | 52.4% | No (Self: E5) |
| ViT Block 2 | **B2** | `[E0, E2, E4, E5]` | 52.2% | No (Self: E6) |
| ViT Block 3 | **B3** | `[E0, E6, E3, E4]` | 51.8% | No (Self: E7) |

---

## 5. Router-wise Subgroup Intervention (Depth Contribution)

| Subgroup Configuration | Intervened Routers | Kept Adaptive | Mean Dice | Std | $\Delta$ vs Adaptive | Mean IoU |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **All_Random** | `['All (S0-B3)']` | `['S0', 'S1', 'S2', 'S3', 'B0', 'B1', 'B2', 'B3']` | **0.7634** | ± 0.0004 | -0.00051 | 0.6405 |
| **Shallow_CNN_Only (S0-S2)** | `['S0', 'S2', 'S1']` | `['S3', 'B0', 'B1', 'B2', 'B3']` | **0.7632** | ± 0.0004 | -0.00070 | 0.6403 |
| **Deep_ViT_Only (B0-B3)** | `['B3', 'B0', 'B2', 'B1']` | `['S0', 'S1', 'S2', 'S3']` | **0.7643** | ± 0.0001 | +0.00036 | 0.6417 |
| **All_CNN_Only (S0-S3)** | `['S0', 'S2', 'S3', 'S1']` | `['B0', 'B1', 'B2', 'B3']` | **0.7632** | ± 0.0005 | -0.00072 | 0.6402 |

---

## 6. Routing Diagnostics & Sanity Verification

1. **Static Policy Exactness**: Verified 100% of selections on Static mode adhered strictly to the 4 allocated experts with 0 selections for other experts.
2. **Random Uniformity**: Verified random sampling achieved approximately 50% selection share across all 8 experts ($K=4 / M=8$).
3. **Weighting Fidelity**: In all modes, gating weights were computed identically via $\sigma(\text{modulated\_logits}[e])$, ensuring that differences reflect solely the **expert selection policy**.
4. **State Mutation**: Verified 0 parameters modified, model weights identical before and after hooks.

---
*Report auto-generated by `scripts/diagnostics/evaluate_routing_intervention.py`.*
