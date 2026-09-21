# Phase 2: B0 Baseline (Pure ConvNeXt) - Comprehensive Log

## 1. Architecture Audit (Shape & Parameter Count)
- **B0 (Pure ConvNeXtV2-Femto + U-Net):**
  - Params: 7,376,593 (7.37 M)
  - ViT Blocks: 0
  - Forward 448x448 -> 448x448: PASS
- **B1 (Hybrid ConvNeXtV2 + ViT-Tiny + U-Net):**
  - Params: 10,232,593 (10.23 M)
  - ViT Blocks: 6
  - Forward 448x448 -> 448x448: PASS

## 2. Dataset Sanity Check (CRACK500)
- **Splits:** Train (1896 pairs), Val (348 pairs), Test (1124 pairs).
- **Split Sanity:** 4 overlaps identified between Train/Test (kept as natural dataset characteristic).
- **Geometry & Mask:** Foreground ratios generally ~4.5% to 6.3% median. No pure background (0 fg) masks found.
- **Pipeline:** Albumentations transforms & Tensor generation verified via visual check (Raw Image -> Processed Overlay alignment).

## 3. T4 Throughput Benchmark (B0)
- **GPU:** Tesla T4 (Colab, 16GB VRAM)
- **Image Size:** 448x448
- **Results:**
  - BS 16: 66.7 img/s, Peak VRAM 2.68 GB (Chosen as safe working batch size)
  - BS 28: 63.4 img/s, Peak VRAM 4.60 GB (Max tested)

## 4. Full Training Log (B0)
- **Config:** BS=16, LR=1e-4 (Backbone 1e-5), Max Budget = 30 epochs (Stage 1: 15, Stage 2: 15).
- **Stage 2 Note:** For B0, Stage 2 is purely a continuation training from best Stage-1 checkpoint.
- **Results:** 
  - Ran fully up to Epoch 30 (Budget cap, Early Stopping not triggered).
  - **Best Global Epoch:** 28 (Stage 2 Epoch 13).
  - **Best Val Dice (Per-sample mean via Training Loop):** 0.8082.
  - **Best Val Loss:** 0.8421.
- **Overfit Analysis:** No signs of overfitting (Train Loss 0.8667 vs Val Loss 0.8316).

## 5. Quick Diagnostic Evaluation (Pooled Metrics - Naive Resize)
- **Goal:** Determine over-predict vs under-predict trend to assess Candidate 2 (Weighted BCE).
- **Val Results:** Precision 0.7649 | Recall 0.8643 | Dice 0.8116
- **Test Results:** Precision 0.6716 | Recall 0.8160 | Dice 0.7368
- **Conclusion:** Recall >> Precision. Model over-predicts (high false positives). Candidate 2 (pos_weight) is put on STANDBY as it would worsen Precision.

## 6. Official Tiling Evaluation (Crack500 Protocol)
*Status: Pending execution on Colab.*
- **Protocol:** Original resolution -> Non-overlapping 448x448 tiling -> Reconstruct -> Per-sample Mean calculation.
- **Val Official Score:** [To be updated]
- **Test Official Score:** [To be updated]
