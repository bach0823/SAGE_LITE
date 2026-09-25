# B1 Baseline Experiment Log (Crack500 Depth 4 — Screening Result)

*Date: 2026-09-22*  
*Hardware: Tesla T4 (Google Colab)*  
*Architecture: B1 (ConvNeXtV2-Femto + 4 ViT-Tiny blocks + U-Net decoder, Late Fusion, không SAGE)*  
*Parameters: 9,342,865 params*  
*Loss: CrackBinaryLoss (BCE=1.0, Dice=1.5, smooth=1e-5)*  
*Checkpoints: `/content/drive/MyDrive/crack_seg/B1_Crack500_Depth4/` (`best_model_b1.pth`)*

---

## 1. Setup & Hyperparameters

- **Dataset:** Crack500 (1896 train, 348 val, 1124 test)
- **Branch / Commit:** `crack500-audit` (`48af6ee`)
- **Strategy:** Single-Stage training, 1 Optimizer (AdamW), 1 Cosine Annealing Scheduler (3 warmup epochs)
- **Batch Size:** 20 (95 steps/epoch)
- **Epoch Budget / Patience:** 30 epochs (đã chạy trọn vẹn 30 epochs)
- **Learning Rate:** Base 1e-4 (Backbone 1e-5, Decoder 1e-4)
- **Weight Decay:** 0.05 (LayerNorm & bias: 0.0)
- **Backbone Weights:** Pretrained `convnextv2_femto.fcmae` (classification head removed)
- **ViT Weights:** Pretrained `vit_tiny_patch16_224.augreg_in21k_ft_in1k` (4 blocks, embed_dim=192, CLS token discarded, 196 spatial patch tokens retained)
- **Augmentation (Frozen Canonical):** `HorizontalFlip`, `VerticalFlip`, `RandomRotate90`, `RandomBrightnessContrast`, `GaussianBlur`
- **Smart Filter:** Reflect Pad -> RandomCrop 448x448 -> Reject if `fg_pixels < 20`

---

## 2. Training Convergence & Validation

- **Total Epochs run:** 30 / 30 (chạy đủ budget, học ổn định không bị dừng sớm)
- **Best Epoch:** Epoch 28 (Val Dice: **0.7428**, Val Loss: **1.1364**)
  - Epoch 1: Val Dice 0.1492
  - Epoch 4: Val Dice 0.6482
  - Epoch 8: Val Dice 0.7121
  - Epoch 13: Val Dice 0.7326
  - Epoch 24: Val Dice 0.7428 (Val Loss 1.1445)
  - Epoch 28: Val Dice **0.7428** (Val Loss **1.1364** — Best checkpoint nhờ tie-break loss thấp hơn)
- **Thứ hạng trong toàn bộ ViT-depth sweep:** **TOP 1 Validation** (vượt Depth 6, Depth 8, Depth 12 và B0).

---

## 3. Official Evaluation Metrics (`best_model_b1.pth`)

### Setting A (Non-overlapping Tiling 448×448)

| Metric | VAL (348 images) | TEST (1124 images) | Delta vs B0 (Test) |
|:---|---:|---:|:---:|
| **Loss** | 1.1364 | 1.1758 | **-0.2446** |
| **Precision** | 0.6738 | 0.6005 | -0.0026 |
| **Recall** | 0.8820 | **0.8548** | **+0.0219** |
| **Dice / F1** | **0.7428** | **0.6829** | **+0.0058** |
| **Global Pixel IoU** | 0.6696 | **0.5700** | **+0.0055** |
| **Crack-Present IoU** | 0.6149 | 0.5381 | +0.0071 |
| **Macro IoU** | 0.6149 | 0.5381 | +0.0071 |
| **Boundary IoU** *(Diag)* | 0.0891 | 0.0724 | +0.0048 |
| **HD95 (px)** *(Diag)* | 80.3944 | 96.7220 | +1.7395 px |

### Setting B (50% Overlap Tiling 448×448, stride 224, blend_mode='probs', threshold=0.5)

| Metric | VAL (348 images) | TEST (1124 images) | Delta vs Setting A (Test) | Delta vs B0 Setting B (Test) |
|:---|---:|---:|:---:|:---:|
| **Loss** | 2.0571 | 2.0627 | - | **-0.0565** |
| **Precision** | 0.6800 | 0.6031 | +0.0026 | -0.0002 |
| **Recall** | 0.8880 | **0.8589** | +0.0041 | **+0.0203** |
| **Dice / F1** | **0.7484** | **0.6867** | **+0.0038** | **+0.0066 (+0.66%)** |
| **Global Pixel IoU** | 0.6727 | **0.5741** | **+0.0041** | **+0.0059** |
| **Crack-Present IoU** | 0.6213 | 0.5424 | +0.0043 | +0.0079 |
| **Macro IoU** | 0.6213 | 0.5424 | +0.0043 | +0.0079 |
| **Boundary IoU** *(Diag)* | 0.0897 | 0.0723 | -0.0001 | +0.0047 |
| **HD95 (px)** *(Diag)* | 68.5749 | 90.6362 | -6.0858 px | -3.7326 px |

---

## 4. Bảng So Sánh Đối Chứng: B0 vs B1 Depth 4 (Crack500)

| Metric (Test Set) | B0 Baseline (0 blocks) | B1 Depth 4 (Locked) | Delta (B1 vs B0) | Đánh giá |
|:---|:---:|:---:|:---:|:---|
| **Số tham số** | 10.23M | **9.34M** | **-0.89M (-8.7%)** | Mô hình gọn nhẹ hơn |
| **Best Val Dice** | 0.7318 (@ ep 8) | **0.7428** (@ ep 28) | **+0.0110** | Học bền, hội tụ sâu hơn |
| **Best Val Loss** | 1.3962 | **1.1364** | **-0.2598** | Giảm mạnh sai lệch dự đoán |
| **Test Setting A Dice** | 0.6771 | **0.6829** | **+0.0058** | Tăng trên non-overlap tiling |
| **Test Setting B Dice** | 0.6801 | **0.6867** | **+0.0066** | Cải thiện nhất quán |
| **Test Setting B Recall** | 0.8386 | **0.8589** | **+0.0203** | Tăng mạnh khả năng bắt vết nứt |
| **Test Setting B Global IoU** | 0.5682 | **0.5741** | **+0.0059** | Độ phủ diện tích tăng |

---

## 5. Nhận Định & Ý Nghĩa Thực Nghiệm
- B1 Depth 4 đạt **Val Dice cao nhất (0.7428)** và **Val Loss thấp nhất (1.1364)** trong toàn bộ dải thực nghiệm `{4, 6, 8, 12}` blocks.
- Tuy nhiên, Val Dice giữa Depth 4/6/12 gần như tương đương (~0.742), và B1 chỉ phản ánh năng lực representation khi chưa có cơ chế điều hướng MoE.
- Do đó, **chưa freeze ViT depth từ B1**. Kết quả B1 Depth 4 được lưu giữ như một mốc screening quan trọng, và depth tối ưu cho SAGE-Lite sẽ được quyết định qua **B2 Depth Ablation** (tập trung vào `{4, 6, 12}`).
