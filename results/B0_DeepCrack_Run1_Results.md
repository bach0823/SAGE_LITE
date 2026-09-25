# B0 Baseline Experiment Log (DeepCrack Run-1)

*Date: 2026-09-22*  
*Hardware: Tesla T4 (Google Colab)*  
*Architecture: B0 (Pure ConvNeXtV2-Femto + U-Net, 10.2M params)*  
*Loss: CrackBinaryLoss (BCE=1.0, Dice=1.5, smooth=1e-5)*  
*Checkpoints: `/content/drive/MyDrive/crack_seg/B0_DeepCrack_Run1/` (`best_model_b0.pth`)*

---

## 1. Setup & Hyperparameters

- **Dataset:** DeepCrack (240 train, 60 val, 237 test)
- **Branch:** `crack500-audit`
- **Protocol:** Direct full-image prediction (Dynamic Pad-to-Square: `target = max(H, W, 448)` -> `Resize 448x448`, `cv2.INTER_NEAREST` cho mask).
- **Training Strategy:** Single-Stage training, 1 Optimizer (AdamW), 1 Cosine Annealing Scheduler (3 warmup epochs)
- **Batch Size:** 20 (12 steps/epoch)
- **Epochs / Patience:** 30 epochs max, EarlyStopping patience = 6 (Val Dice)
- **Learning Rate:** Base 1e-4 (Backbone 1e-5, Decoder 1e-4)
- **Augmentation:** `HorizontalFlip`, `VerticalFlip`, `RandomRotate90`, `RandomBrightnessContrast`, `GaussianBlur` (CLAHE, ElasticTransform, GridDistortion removed)
- **Smart Filter:** `false` (Direct pad-to-square & resize, no crop filter)

---

## 2. Training Convergence & Early Stopping Log

- **Total Epochs run:** 17 (EarlyStopping triggered at epoch 17, Best at **Epoch 11**)
- **Best Val Dice:** **0.6240** (Val Loss: **1.7366**)
- **Runtime:** ~3 phút trên Tesla T4 (~6–7s/epoch cho 12 steps).

| Epoch | Train Loss | Train Dice | Val Loss | Val Dice | LR Backbone | LR Decoder | Note |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | 2.0126 | 0.0878 | 2.1056 | 0.0136 | 4.00e-06 | 3.40e-05 | Warmup |
| 2 | 1.9900 | 0.1876 | 2.0636 | 0.2488 | 7.00e-06 | 6.70e-05 | |
| 3 | 1.9358 | 0.2612 | 1.9126 | 0.3832 | 9.78e-06 | 9.76e-05 | Warmup Peak |
| 4 | 1.8632 | 0.3808 | 1.7981 | 0.5071 | 9.61e-06 | 9.57e-05 | |
| 5 | 1.8187 | 0.4492 | 1.7688 | 0.5656 | 9.40e-06 | 9.34e-05 | |
| 6 | 1.7939 | 0.4862 | 1.7468 | 0.5693 | 9.14e-06 | 9.05e-05 | |
| 7 | 1.7795 | 0.5075 | 1.7395 | 0.5747 | 8.84e-06 | 8.73e-05 | |
| 8 | 1.7700 | 0.5294 | 1.7393 | 0.5749 | 8.51e-06 | 8.36e-05 | |
| 9 | 1.7609 | 0.5365 | 1.7477 | 0.5785 | 8.15e-06 | 7.96e-05 | |
| 10 | 1.7542 | 0.5369 | 1.7333 | 0.6056 | 7.75e-06 | 7.52e-05 | |
| **11** | **1.7482** | **0.5532** | **1.7366** | **0.6240** | **7.33e-06** | **7.06e-05** | **BEST CHECKPOINT** |
| 12 | 1.7415 | 0.5530 | 1.7313 | 0.6005 | 6.89e-06 | 6.58e-05 | Patience 1/6 |
| 13 | 1.7380 | 0.5618 | 1.7294 | 0.6206 | 6.44e-06 | 6.08e-05 | Patience 2/6 |
| 14 | 1.7344 | 0.5655 | 1.7380 | 0.6005 | 5.97e-06 | 5.57e-05 | Patience 3/6 |
| 15 | 1.7302 | 0.5752 | 1.7459 | 0.5532 | 5.50e-06 | 5.05e-05 | Patience 4/6 |
| 16 | 1.7272 | 0.5665 | 1.7180 | 0.6025 | 5.03e-06 | 4.53e-05 | Patience 5/6 |
| 17 | 1.7259 | 0.5780 | 1.7252 | 0.5917 | 4.56e-06 | 4.02e-05 | Patience 6/6 (EarlyStop) |

---

## 3. Official Evaluation Metrics (Direct 1-Pass Full-Image)

Checkpoint: `best_model_b0.pth` (Epoch 11)

| Metric | VAL Set (60 images) | TEST Set (237 images) |
|:---|---:|---:|
| **Loss** | 1.7366 | 1.6786 |
| **Precision** | 0.5075 | 0.6156 |
| **Recall** | 0.8741 | 0.8586 |
| **Dice / F1** | **0.6240** | **0.6953** |
| **Global Pixel IoU** | 0.5162 | 0.6041 |
| **Crack-Present IoU** | 0.4778 | 0.5522 |
| **Macro IoU** | 0.4778 | 0.5522 |
| **Boundary IoU** *(Diagnostic)* | 0.1103 | 0.1365 |
| **HD95 (px)** *(Diagnostic)* | 36.2463 | 30.8855 |

---

## 4. Key Observations & Comparison with Crack500

1. **Test Performance cao hơn Val (Inverted Gap):**
   - Val Dice: **0.6240** → Test Dice: **0.6953** (+0.0713).
   - Global Pixel IoU đạt **0.6041** trên Test. Do tập Val DeepCrack chỉ có 60 ảnh (ngẫu nhiên nhỏ), Test set (237 ảnh) thể hiện rõ ràng khả năng tổng quát hóa tốt hơn.
2. **Localization & Biên giới:**
   - **HD95 trên DeepCrack cực kỳ thấp:** chỉ **30.89 px** (so với 94.37–94.98 px trên Crack500).
   - **Boundary IoU:** đạt **0.1365** (gấp đôi so với 0.0676 trên Crack500).
   - Điều này thể hiện bề mặt nền của DeepCrack ít đa dạng nhiễu hạt nặng như mặt đường nhựa của Crack500, giúp mô hình bám viền crack tốt hơn rất nhiều.
3. **Đặc trưng Precision vs Recall:**
   - Tương đồng với Crack500: **Recall (~0.86)** cao hơn **Precision (~0.62)**. Mô hình nhạy bắt nứt, ít bỏ sót.

---

## 5. Conclusion & Baseline Status

> **B0 DeepCrack: Train PASS. Best epoch 11, Val Dice = 0.6240. Official Test Dice = 0.6953. Precision = 0.6156, Recall = 0.8586, Global Pixel IoU = 0.6041. Boundary IoU = 0.1365, HD95 = 30.89.**

- **Baseline B0 Complete:** Đã hoàn thành baseline B0 chuẩn mực trên cả 2 dataset Crack500 và DeepCrack.
- **Protocol Integrity:** Giữ nguyên checkpoint theo Val Dice (0.6240 @ epoch 11), đảm bảo tính khách quan khoa học, không chọn checkpoint theo Test set.
- **Next Steps:** Sẵn sàng chuyển sang B1 hoặc phát triển kiến trúc nhánh SAGE-Lite.
