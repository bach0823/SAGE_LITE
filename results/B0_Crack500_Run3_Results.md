# B0 Baseline Experiment Log (Crack500 Run-3)

*Date: 2026-09-22*  
*Hardware: Tesla T4 (Google Colab)*  
*Architecture: B0 (Pure ConvNeXtV2-Femto + U-Net, 10.2M params)*  
*Loss: CrackBinaryLoss (BCE=1.0, Dice=1.5, smooth=1e-5)*  
*Checkpoints: `/content/drive/MyDrive/crack_seg/B0_Crack500_Run3/` (`best_model_b0.pth`)*

---

## 1. Setup & Hyperparameters

- **Dataset:** Crack500 (1896 train, 348 val, 1124 test)
- **Branch:** `crack500-audit` (`98183d4`)
- **Strategy:** Single-Stage training, 1 Optimizer (AdamW), 1 Cosine Annealing Scheduler (3 warmup epochs)
- **Batch Size:** 20 (95 steps/epoch)
- **Epochs / Patience:** 30 epochs, early stopping patience = 6 (Val Dice)
- **Learning Rate:** Base 1e-4 (Backbone 1e-5, Decoder 1e-4)
- **Augmentation:** `HorizontalFlip`, `VerticalFlip`, `RandomRotate90`, `RandomBrightnessContrast`, `GaussianBlur` (CLAHE, ElasticTransform, GridDistortion removed)
- **Smart Filter:** Reflect Pad -> RandomCrop 448x448 -> Reject if `fg_pixels < 20`

---

## 2. Training Convergence & Early Stopping

- **Total Epochs run:** 14 (EarlyStopping triggered at epoch 14)
- **Best Epoch:** Epoch 8 (Val Dice: **0.7318**, Val Loss: **1.3962**)
- **Training Time:** ~12 minutes on Tesla T4 (~38-40s / epoch)

| Epoch | Train Loss | Train Dice | Val Loss | Val Dice | LR Backbone | LR Decoder | Note |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | 1.9186 | 0.1530 | 1.9141 | 0.2398 | 4.00e-06 | 3.40e-05 | Warmup |
| 2 | 1.7078 | 0.4921 | 1.6043 | 0.6090 | 7.00e-06 | 6.70e-05 | |
| 3 | 1.5723 | 0.6045 | 1.5012 | 0.6403 | 9.78e-06 | 9.76e-05 | Peak Warmup |
| 4 | 1.5137 | 0.6234 | 1.4469 | 0.6945 | 9.61e-06 | 9.57e-05 | |
| 5 | 1.4633 | 0.6383 | 1.4687 | 0.6735 | 9.40e-06 | 9.34e-05 | |
| 6 | 1.4155 | 0.6520 | 1.4440 | 0.6842 | 9.14e-06 | 9.05e-05 | |
| 7 | 1.3744 | 0.6589 | 1.3522 | 0.7026 | 8.84e-06 | 8.73e-05 | |
| **8** | **1.3331** | **0.6690** | **1.3962** | **0.7318** | **8.51e-06** | **8.36e-05** | **BEST CHECKPOINT** |
| 9 | 1.2996 | 0.6752 | 1.2949 | 0.7288 | 8.15e-06 | 7.96e-05 | Patience 1/6 |
| 10 | 1.2614 | 0.6766 | 1.3069 | 0.7221 | 7.75e-06 | 7.52e-05 | Patience 2/6 |
| 11 | 1.2337 | 0.6789 | 1.2737 | 0.7170 | 7.33e-06 | 7.06e-05 | Patience 3/6 |
| 12 | 1.2034 | 0.6853 | 1.2263 | 0.7220 | 6.89e-06 | 6.58e-05 | Patience 4/6 |
| 13 | 1.1723 | 0.6937 | 1.1741 | 0.7293 | 6.44e-06 | 6.08e-05 | Patience 5/6 |
| 14 | 1.1495 | 0.6945 | 1.1993 | 0.7134 | 5.97e-06 | 5.57e-05 | Patience 6/6 (EarlyStop) |

---

## 3. Official Evaluation Metrics (best_model_b0.pth)

### Setting A (Non-overlapping Tiling 448x448)

| Metric | VAL (348) | TEST (1124) |
|:---|---:|---:|
| **Loss** | 1.3962 | 1.4204 |
| **Precision** | 0.6755 | 0.6031 |
| **Recall** | 0.8619 | 0.8329 |
| **Dice / F1** | **0.7318** | **0.6771** |
| **Global Pixel IoU** | 0.6615 | 0.5645 |
| **Crack-Present IoU** | 0.6023 | 0.5310 |
| **Macro IoU** | 0.6023 | 0.5310 |
| **Boundary IoU** *(Diag)* | 0.0816 | 0.0676 |
| **HD95 (px)** *(Diag)* | 71.2971 | 94.9825 |

### Setting B (50% Overlap Tiling 448x448, stride 224, blend_mode='probs', threshold=0.5)

| Metric | VAL (348) | TEST (1124) | Delta vs Setting A (Test) |
|:---|---:|---:|:---:|
| **Loss** | 2.1153 | 2.1192 | - |
| **Precision** | 0.6782 | 0.6033 | +0.0002 |
| **Recall** | 0.8675 | 0.8386 | +0.0057 |
| **Dice / F1** | **0.7360** | **0.6801** | **+0.0030** |
| **Global Pixel IoU** | 0.6636 | 0.5682 | +0.0037 |
| **Crack-Present IoU** | 0.6066 | 0.5345 | +0.0035 |
| **Macro IoU** | 0.6066 | 0.5345 | +0.0035 |
| **Boundary IoU** *(Diag)* | 0.0812 | 0.0676 | 0.0000 |
| **HD95 (px)** *(Diag)* | 63.8870 | 94.3688 | **-0.6137 px** |

---

## 4. Conclusion & Observations

> **B0 Crack500: train PASS. Best Val Dice = 0.7318 @ epoch 8. Official Test Dice = 0.6771 (Setting A), 0.6801 (Setting B). Setting B chỉ cải thiện ~0.003 Dice. Recall cao (~0.83), nhưng Boundary IoU thấp và HD95 cao, cho thấy localization của crack còn chưa tốt.**

- **Training Pipeline:** Single-stage training converged cleanly in ~10 mins (14 epochs, EarlyStopping triggered as best was at epoch 8). LR warmup + cosine schedule operated as expected.
- **Setting A vs Setting B:** Overlap 50% chỉ tăng nhẹ Dice (+0.003 test), không tạo ra bước nhảy lớn nhưng hỗ trợ làm mịn mép ghép tile (HD95 val giảm từ 71.30 xuống 63.89 px).
- **Localization Profile:** Model bắt crack nhạy (Recall ~83-84%), nhưng Precision thấp hơn (~60%) do còn false positive. Boundary IoU (0.068–0.082) và HD95 (71–95 px) phản ánh đúng tính chất crack mảnh, ranh giới dự đoán chưa thật sự khít.
- **Ready for Next Phase:** B0 Crack500 baseline completed. Next: Train B0 on DeepCrack.
