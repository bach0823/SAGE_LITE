# SAGE-Lite Empirical Diagnostics: Global Average Pooling (GAP) Routing-Signal Bottleneck Study

*Date: September 2026*
*Architecture Target: Canonical D4 P3-C (ASDW) Standalone Model*
*Dataset Split: Crack500 Validation Split (N = 348 samples)*
*Execution Protocol: Deterministic Evaluation Pass (model.eval(), torch.no_grad(), exploration noise OFF)*

---

## 1. Test 1: Validation Set Foreground Area Ratio Distribution

$$r_i = \frac{\#\text{positive pixels}}{H \times W} = \frac{\text{gt\_area}}{448 \times 448}$$

- **Total Samples (N)**: `348`
- **Mean Foreground Ratio**: `5.6114%`
- **Standard Deviation**: `5.0250%`
- **Median**: `4.1780%`
- **Range [Min, Max]**: `[0.0000%, 25.7289%]`
- **Quartiles [Q1, Q2, Q3, Q4]**: `[1.7388%, 4.1780%, 8.5576%, 25.7289%]`

### Extreme Sparsity Tiers
- **Samples with Foreground < 1%**: `17.53%`
- **Samples with Foreground < 2%**: `28.45%`
- **Samples with Foreground < 5%**: `56.61%`
- **Samples with Foreground < 10%**: `80.46%`

| Bin Interval | Pixel Ratio Range | Sample Count | Percentage |
|:---:|:---:|:---:|:---:|
| Bin | `[0.0000%, 2.5729%]` | 119 | 34.20% |
| Bin | `[2.5729%, 5.1458%]` | 79 | 22.70% |
| Bin | `[5.1458%, 7.7187%]` | 55 | 15.80% |
| Bin | `[7.7187%, 10.2916%]` | 31 | 8.91% |
| Bin | `[10.2916%, 12.8645%]` | 24 | 6.90% |
| Bin | `[12.8645%, 15.4374%]` | 24 | 6.90% |
| Bin | `[15.4374%, 18.0103%]` | 8 | 2.30% |
| Bin | `[18.0103%, 20.5831%]` | 4 | 1.15% |
| Bin | `[20.5831%, 23.1560%]` | 2 | 0.57% |
| Bin | `[23.1560%, 25.7289%]` | 2 | 0.57% |

---

## 2. Test 2: Captured Router Pooled Features Metadata

- **Stored Array Archive**: `router_pooled_features.npz` (348 samples)
- **Metadata File**: `router_pooled_features_meta.json`

| Router | Alias | Layer Type | Feature Dim (Pooled) | Experts Pool | Top-k Selection |
|:---|:---:|:---:|:---:|:---:|:---:|
| `convnext.stage_0` | **S0** | CNN | `48` | `8` | `4` |
| `convnext.stage_1` | **S1** | CNN | `48` | `8` | `4` |
| `convnext.stage_2` | **S2** | CNN | `96` | `8` | `4` |
| `convnext.stage_3` | **S3** | CNN | `192` | `8` | `4` |
| `transformer.block_0` | **B0** | ViT | `192` | `8` | `4` |
| `transformer.block_1` | **B1** | ViT | `192` | `8` | `4` |
| `transformer.block_2` | **B2** | ViT | `192` | `8` | `4` |
| `transformer.block_3` | **B3** | ViT | `192` | `8` | `4` |

---

## 3. Test 3: Linear Probe (Ridge Regression 5-fold CV)

Evaluation of morphology information retained in the pooled features:
$$\text{pooled feature } (\mathbf{h}) \xrightarrow{\text{Ridge Regression (5-fold CV)}} \text{target morphology}$$

| Router | Type | Feature Dim | Target: `gt_area` ($r_i$) $R^2$ | `gt_area` MAE | Target: `thinness` $R^2$ | `thinness` MAE |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **S0** (`convnext.stage_0`) | CNN | 48 | `+0.0489 ± 0.1331` | `0.038366` | `-0.1452 ± 0.0802` | `0.0862` |
| **S1** (`convnext.stage_1`) | CNN | 48 | `+0.4561 ± 0.0892` | `0.027792` | `+0.0880 ± 0.0763` | `0.0736` |
| **S2** (`convnext.stage_2`) | CNN | 96 | `+0.5150 ± 0.0735` | `0.025365` | `+0.1223 ± 0.1042` | `0.0722` |
| **S3** (`convnext.stage_3`) | CNN | 192 | `+0.7934 ± 0.0580` | `0.016954` | `+0.0494 ± 0.2497` | `0.0761` |
| **B0** (`transformer.block_0`) | ViT | 192 | `+0.8186 ± 0.0350` | `0.016346` | `+0.1998 ± 0.0854` | `0.0723` |
| **B1** (`transformer.block_1`) | ViT | 192 | `+0.8363 ± 0.0471` | `0.015268` | `+0.1169 ± 0.1427` | `0.0774` |
| **B2** (`transformer.block_2`) | ViT | 192 | `+0.8477 ± 0.0370` | `0.014908` | `-0.1787 ± 0.2062` | `0.0875` |
| **B3** (`transformer.block_3`) | ViT | 192 | `+0.8292 ± 0.0484` | `0.015509` | `-0.1646 ± 0.1404` | `0.0852` |

---

## 4. Test 4: Routing Sensitivity by Morphology Quartiles

Distribution shift comparison between Q1 (least pronounced) vs Q4 (most pronounced):

### 4.1. Sensitivity Across Thinness Quartiles

| Router | Q1 vs Q4 JSD (bits) | Q1 vs Q4 Total Abs Diff | Q1 CNN / ViT Fraction | Q4 CNN / ViT Fraction | Q1 Mean $g_s$ | Q4 Mean $g_s$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **S0** | `0.0058` | `0.0000` | `25.0% / 75.0%` | `24.7% / 75.3%` | `0.4873` | `0.4873` |
| **S1** | `0.0002` | `0.0000` | `40.5% / 59.5%` | `40.2% / 59.8%` | `0.4624` | `0.4622` |
| **S2** | `0.0088` | `0.0000` | `37.1% / 62.9%` | `43.7% / 56.3%` | `0.5258` | `0.5258` |
| **S3** | `0.0248` | `0.0000` | `58.0% / 42.0%` | `53.2% / 46.8%` | `0.5062` | `0.5037` |
| **B0** | `0.0167` | `0.0000` | `31.3% / 68.7%` | `33.0% / 67.0%` | `0.5192` | `0.5193` |
| **B1** | `0.0440` | `0.0000` | `49.4% / 50.6%` | `36.8% / 63.2%` | `0.5020` | `0.5013` |
| **B2** | `0.0164` | `0.0000` | `22.7% / 77.3%` | `21.3% / 78.7%` | `0.5029` | `0.5044` |
| **B3** | `0.0124` | `0.0000` | `27.3% / 72.7%` | `30.2% / 69.8%` | `0.5040` | `0.5059` |

### 4.2. Sensitivity Across Crack Area Quartiles (`gt_area`)

| Router | Q1 vs Q4 JSD (bits) | Q1 vs Q4 Total Abs Diff | Q1 CNN / ViT Fraction | Q4 CNN / ViT Fraction | Q1 Mean $g_s$ | Q4 Mean $g_s$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **S0** | `0.0029` | `0.0000` | `25.0% / 75.0%` | `25.0% / 75.0%` | `0.4872` | `0.4876` |
| **S1** | `0.0031` | `0.0000` | `39.7% / 60.3%` | `40.5% / 59.5%` | `0.4621` | `0.4626` |
| **S2** | `0.0150` | `0.0000` | `42.8% / 57.2%` | `35.3% / 64.7%` | `0.5260` | `0.5258` |
| **S3** | `0.0576` | `0.0000` | `56.3% / 43.7%` | `57.2% / 42.8%` | `0.5051` | `0.5060` |
| **B0** | `0.0905` | `0.0000` | `32.5% / 67.5%` | `32.8% / 67.2%` | `0.5199` | `0.5185` |
| **B1** | `0.2183` | `0.0000` | `31.0% / 69.0%` | `59.5% / 40.5%` | `0.5008` | `0.5029` |
| **B2** | `0.1147` | `0.0000` | `25.0% / 75.0%` | `18.1% / 81.9%` | `0.5054` | `0.5014` |
| **B3** | `0.3789` | `0.0000` | `57.2% / 42.8%` | `7.2% / 92.8%` | `0.5082` | `0.5011` |

---

## 5. Test 5: Empirical Magnitude of $g_s$ Logit Modulation

$$\Delta_{\text{family}} = \log\frac{g_s}{1 - g_s}$$

| Router | Alias | Mean $\Delta_{\text{family}}$ | Std $\Delta_{\text{family}}$ | $\Delta$ Range [Min, Max] | $\Delta$ [P5, P50, P95] | Mean $g_s$ | Base Logits Std | Base Logits Range |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `convnext.stage_0` | **S0** | `-0.0508` | `0.0067` | `[-0.0642, -0.0310]` | `[-0.0596, -0.0518, -0.0388]` | `0.4873` | `0.0246` | `0.1242` |
| `convnext.stage_1` | **S1** | `-0.1511` | `0.0058` | `[-0.1594, -0.1300]` | `[-0.1581, -0.1523, -0.1395]` | `0.4623` | `0.0869` | `0.3467` |
| `convnext.stage_2` | **S2** | `+0.1035` | `0.0039` | `[+0.0953, +0.1152]` | `[+0.0992, +0.1019, +0.1115]` | `0.5258` | `0.0663` | `0.2071` |
| `convnext.stage_3` | **S3** | `+0.0201` | `0.0115` | `[-0.0043, +0.0449]` | `[+0.0020, +0.0201, +0.0398]` | `0.5050` | `0.0158` | `0.1086` |
| `transformer.block_0` | **B0** | `+0.0770` | `0.0048` | `[+0.0540, +0.0880]` | `[+0.0699, +0.0773, +0.0833]` | `0.5192` | `0.0466` | `0.1897` |
| `transformer.block_1` | **B1** | `+0.0073` | `0.0047` | `[-0.0058, +0.0200]` | `[-0.0001, +0.0071, +0.0155]` | `0.5018` | `0.0141` | `0.1068` |
| `transformer.block_2` | **B2** | `+0.0138` | `0.0078` | `[-0.0089, +0.0350]` | `[+0.0008, +0.0134, +0.0265]` | `0.5034` | `0.0305` | `0.1882` |
| `transformer.block_3` | **B3** | `+0.0188` | `0.0124` | `[-0.0203, +0.0438]` | `[-0.0009, +0.0184, +0.0378]` | `0.5047` | `0.0202` | `0.1130` |

---

## 6. Test 6: Original SAGE Codebase Audit

- **Source File**: `d:\truong\SpecialSubjectTTNT\SAGE\sage\components\router.py` (Exists: True, Total Lines: 406)
- **Audit Status**: `verified_present`
- **Has GAP or Mean Pooling**: `True`

### Verified Evidence from Source Code
1. **CNN Pooling Definition (Line 106)**:
   ```python
   self.feature_aggregator = nn.AdaptiveAvgPool2d((1, 1))
   ```
   Execution (Line 189):
   ```python
   aggregated = self.feature_aggregator(x).squeeze(-1).squeeze(-1)
   ```
2. **Transformer Mean Pooling Execution (Line 191)**:
   ```python
   aggregated = x.mean(dim=1)
   ```
3. **Shared Expert Gate $g_s$ Coupling (Line 238)**:
   ```python
   g_s = torch.sigmoid(self.shared_expert_gate(aggregated_features))
   ```
4. **Query Projection SAR Coupling (Line 241)**:
   ```python
   query = self.query_projection(aggregated_features)
   ```

---

## 7. Artifact Manifest
- `foreground_ratio.csv`: Individual foreground ratio per sample
- `foreground_ratio_summary.json`: Detailed distribution summary
- `router_pooled_features.npz`: Compressed arrays of all captured pooled features
- `router_pooled_features_meta.json`: Metadata for pooled feature arrays
- `linear_probe_results.csv`: 5-fold CV R², MAE, RMSE per router and target
- `linear_probe_summary.json`: Complete fold-level linear probe data
- `routing_sensitivity_by_quartile.csv`: Quartile distributions and frequencies
- `routing_sensitivity_summary.json`: Quartile data and JSD metrics
- `gs_modulation_magnitude.csv`: Modulation magnitude metrics
- `gs_modulation_summary.json`: Statistical bounds on g_s and Delta_family
- `test6_sage_original_audit.json`: Source code analysis of original SAGE
- `GAP_ROUTING_STUDY_REPORT.md`: This consolidated technical report
