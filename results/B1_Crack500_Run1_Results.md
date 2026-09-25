# B1 Baseline Experiment Log (Crack500 Run-1)

*Date: 2026-09-22*  
*Hardware: Tesla T4 (Google Colab)*  
*Architecture: B1 (ConvNeXtV2-Femto + 6 ViT-Tiny + U-Net, Late Fusion, không SAGE)*  
*Loss: CrackBinaryLoss (BCE=1.0, Dice=1.5, smooth=1e-5)*  
*Checkpoints: `/content/drive/MyDrive/crack_seg/B1_Crack500_Run1/` (`best_model_b1.pth`)*

---

## 1. Setup & Hyperparameters

- **Dataset:** Crack500 (1896 train, 348 val, 1124 test)
- **Branch / Commit:** `crack500-audit` (`ee7d3ea`)
- **Strategy:** Single-Stage training, 1 Optimizer (AdamW), 1 Cosine Annealing Scheduler (3 warmup epochs)
- **Batch Size:** 16 (95 steps/epoch)
- **Epoch Budget / Patience:** 30 epochs, early stopping patience = 6 (Val Dice)
- **Learning Rate:** Base 1e-4 (Backbone 1e-5, Decoder 1e-4)
- **Weight Decay:** 0.05 (LayerNorm & bias: 0.0)
- **Backbone Weights:** Pretrained `convnextv2_femto.fcmae` (classification head removed)
- **ViT Weights:** Pretrained `vit_tiny_patch16_224.augreg_in21k_ft_in1k` (6 blocks hard-locked, embed_dim=192, CLS token discarded, 196 spatial patch tokens retained)
- **Augmentation (Frozen Canonical):** `HorizontalFlip`, `VerticalFlip`, `RandomRotate90`, `RandomBrightnessContrast`, `GaussianBlur` (CLAHE, ElasticTransform, GridDistortion removed)
- **Smart Filter:** Reflect Pad -> RandomCrop 448x448 -> Reject if `fg_pixels < 20`

---

## 2. Training Convergence & Early Stopping

- **Total Epochs run:** 27 (EarlyStopping triggered at epoch 27, patience 6)
- **Best Epoch:** Epoch 21 (Val Dice: **0.7420**, Val Loss: **1.1453**)
- **Training Time:** ~20 minutes on Tesla T4 (~43-45s / epoch)

| Epoch | Train Loss | Train Dice | Val Loss | Val Dice | LR Backbone | LR Decoder | Note |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | 2.0179 | 0.1302 | 2.0223 | 0.1531 | 4.00e-06 | 3.40e-05 | Warmup 1/3 |
| 2 | 1.7685 | 0.4572 | 1.6660 | 0.5529 | 7.00e-06 | 6.70e-05 | Warmup 2/3 |
| 3 | 1.6243 | 0.5903 | 1.6001 | 0.6215 | 9.78e-06 | 9.76e-05 | Peak Warmup |
| 4 | 1.5680 | 0.6098 | 1.5611 | 0.6486 | 9.61e-06 | 9.57e-05 | |
| 5 | 1.5187 | 0.6260 | 1.5180 | 0.6532 | 9.40e-06 | 9.34e-05 | |
| 6 | 1.4782 | 0.6376 | 1.4902 | 0.6570 | 9.14e-06 | 9.05e-05 | |
| 7 | 1.4419 | 0.6480 | 1.4576 | 0.6924 | 8.84e-06 | 8.73e-05 | |
| 8 | 1.4021 | 0.6621 | 1.4406 | 0.7013 | 8.51e-06 | 8.36e-05 | |
| 9 | 1.3655 | 0.6692 | 1.3857 | 0.7269 | 8.15e-06 | 7.96e-05 | |
| 10 | 1.3329 | 0.6722 | 1.3285 | 0.7321 | 7.75e-06 | 7.52e-05 | |
| 11 | 1.3035 | 0.6801 | 1.3533 | 0.7175 | 7.33e-06 | 7.06e-05 | |
| 12 | 1.2763 | 0.6806 | 1.3055 | 0.7154 | 6.89e-06 | 6.58e-05 | |
| 13 | 1.2471 | 0.6839 | 1.2957 | 0.7359 | 6.44e-06 | 6.08e-05 | |
| 14 | 1.2216 | 0.6923 | 1.2678 | 0.7037 | 5.97e-06 | 5.57e-05 | |
| 15 | 1.2027 | 0.6946 | 1.2074 | 0.7338 | 5.50e-06 | 5.05e-05 | |
| 16 | 1.1803 | 0.6946 | 1.2483 | 0.7361 | 5.03e-06 | 4.53e-05 | |
| 17 | 1.1592 | 0.7009 | 1.2342 | 0.7336 | 4.56e-06 | 4.02e-05 | |
| 18 | 1.1488 | 0.7003 | 1.1888 | 0.7318 | 4.11e-06 | 3.52e-05 | |
| 19 | 1.1337 | 0.7043 | 1.1905 | 0.7385 | 3.67e-06 | 3.04e-05 | |
| 20 | 1.1216 | 0.7041 | 1.1696 | 0.7406 | 3.25e-06 | 2.58e-05 | |
| **21** | **1.1157** | **0.7077** | **1.1453** | **0.7420** | **2.85e-06** | **2.14e-05** | **BEST CHECKPOINT** |
| 22 | 1.1013 | 0.7096 | 1.1615 | 0.7390 | 2.49e-06 | 1.74e-05 | Patience 1/6 |
| 23 | 1.0945 | 0.7076 | 1.1698 | 0.7298 | 2.16e-06 | 1.37e-05 | Patience 2/6 |
| 24 | 1.0910 | 0.7077 | 1.1572 | 0.7377 | 1.86e-06 | 1.05e-05 | Patience 3/6 |
| 25 | 1.0816 | 0.7105 | 1.1464 | 0.7377 | 1.60e-06 | 7.63e-06 | Patience 4/6 |
| 26 | 1.0817 | 0.7146 | 1.1466 | 0.7352 | 1.39e-06 | 5.28e-06 | Patience 5/6 |
| 27 | 1.0746 | 0.7158 | 1.1584 | 0.7350 | 1.22e-06 | 3.42e-06 | Patience 6/6 (EarlyStop) |

---

## 3. Official Evaluation Metrics (`best_model_b1.pth`)

### Setting A (Non-overlapping Tiling 448x448)

| Metric | VAL (348) | TEST (1124) | Delta vs B0 (Test) |
|:---|---:|---:|:---:|
| **Loss** | 1.1453 | 1.1814 | **-0.2390** |
| **Precision** | 0.6859 | 0.6081 | +0.0050 |
| **Recall** | 0.8690 | 0.8466 | +0.0137 |
| **Dice / F1** | **0.7420** | **0.6857** | **+0.0086 (+0.86%)** |
| **Global Pixel IoU** | 0.6718 | **0.5730** | **+0.0085** |
| **Crack-Present IoU** | 0.6148 | 0.5414 | +0.0104 |
| **Macro IoU** | 0.6148 | 0.5414 | +0.0104 |
| **Boundary IoU** *(Diag)* | 0.0890 | 0.0729 | +0.0053 |
| **HD95 (px)** *(Diag)* | 62.2078 | **83.1099** | **-11.8726 px** |

### Setting B (50% Overlap Tiling 448x448, stride 224, blend_mode='probs', threshold=0.5)

| Metric | VAL (348) | TEST (1124) | Delta vs Setting A (Test) | Delta vs B0 Setting B (Test) |
|:---|---:|---:|:---:|:---:|
| **Loss** | 2.0586 | 2.0641 | - | -0.0551 |
| **Precision** | 0.6898 | 0.6113 | +0.0032 | +0.0080 |
| **Recall** | 0.8755 | 0.8501 | +0.0035 | +0.0115 |
| **Dice / F1** | **0.7481** | **0.6895** | **+0.0038** | **+0.0094 (+0.94%)** |
| **Global Pixel IoU** | 0.6756 | **0.5770** | **+0.0040** | **+0.0088** |
| **Crack-Present IoU** | 0.6213 | 0.5462 | +0.0048 | +0.0117 |
| **Macro IoU** | 0.6213 | 0.5462 | +0.0048 | +0.0117 |
| **Boundary IoU** *(Diag)* | 0.0893 | 0.0731 | +0.0002 | +0.0055 |
| **HD95 (px)** *(Diag)* | 57.4195 | **77.4342** | **-5.6757 px** | **-16.9346 px** |

---

## 4. Comprehensive Comparison: B0 vs B1 (Crack500)

| Metric | B0 (Run-3) | B1 (Run-1) | Delta (B1 vs B0) | Nhận định |
|:---|:---:|:---:|:---:|:---|
| **Architecture** | ConvNeXtV2-Femto + UNet | B0 + 6 ViT-Tiny | +6 Transformer Blocks | Cô lập đóng góp Global Context |
| **Best Val Dice** | 0.7318 (@ ep 8) | **0.7420** (@ ep 21) | **+0.0102** | Học bền vững, hội tụ sâu hơn |
| **Val Loss** | 1.3962 | **1.1453** | **-0.2509** | Sai số dự đoán giảm rõ rệt |
| **Test Dice (Setting A)** | 0.6771 | **0.6857** | **+0.0086** | Cải thiện trên test non-overlap |
| **Test Dice (Setting B)** | 0.6801 | **0.6895** | **+0.0094** | Tăng tiệm cận mốc 0.69 |
| **Test Global IoU (B)** | 0.5682 | **0.5770** | **+0.0088** | IoU toàn cục tăng đồng đều |
| **Test Boundary IoU (B)** | 0.0676 | **0.0731** | **+0.0055** | Ranh giới vết nứt sắc nét hơn |
| **Test HD95 (B)** | 94.3688 px | **77.4342 px** | **-16.9346 px** | **Khoảng cách sai lệch biên giảm mạnh** |

---

## 5. Key Findings & Conclusions (Nhận định & Đánh giá Học thuật)

### 1. B1 không underperform — Cải thiện nhất quán nhưng ở mức vừa phải
- B1 **không underperform**. Ngược lại, B1 **cải thiện nhất quán so với B0 trên toàn bộ các metric**, nhưng mức cải thiện là **vừa phải (moderate)**, chưa phải bước nhảy lớn:
  - Setting A: Test Dice **0.6771 → 0.6857** (**+0.0086**)
  - Setting B: Test Dice **0.6801 → 0.6895** (**+0.0094**)
  - Global Pixel IoU: **0.5682 → 0.5770** (**+0.0088**)
  - Boundary IoU: **0.0676 → 0.0731** (**+0.0055**)
  - HD95: **94.37 px → 77.43 px** (**-16.93 px**)
- B1 thêm 6 ViT-Tiny **không làm bất kỳ metric nào trong nhóm chính bị tụt đáng kể**, đồng thời Dice/IoU tăng đều và HD95 giảm khá rõ.

> **Kết luận chính thức:**  
> **B1 outperforms B0 trên Crack500 test ở cả Setting A và B, với Test Dice tăng ~0.009 và HD95 giảm ~12–17 px. Tuy nhiên mức cải thiện Dice là vừa phải, chưa phải cải thiện lớn.**

### 2. Lợi ích từ Overlap Blending (Setting B)
- Một điểm rất đẹp là **Setting B vẫn chỉ tăng khoảng 0.004 Dice so với Setting A ở cả B0 và B1**:
  - B0: `0.6771 → 0.6801` (+0.0030)
  - B1: `0.6857 → 0.6895` (+0.0038)
- Điều này khẳng định 50% overlap blending mang lại lợi ích nhỏ nhưng ổn định, khử hiệu quả sai lệch tại ranh giới cắt ghép tile.

### 3. Về câu hỏi "underperform" & Thang đo Baseline Ladder
- Với kết quả này, câu hỏi *"Liệu 6 ViT blocks có làm underperform hay overfit trên bài toán vết nứt không?"* đã có câu trả lời thực nghiệm rõ ràng: **Hoàn toàn không underperform**.
- Kết quả tạo ra một Baseline Ladder đối chứng rất sạch và chặt chẽ:
  ```text
  B0 Test Dice A = 0.6771
  B1 Test Dice A = 0.6857  (↑ +0.0086)

  B0 Test Dice B = 0.6801
  B1 Test Dice B = 0.6895  (↑ +0.0094, HD95: 94.37 → 77.43 px)
  ```
- **B0 + B1 đã hoàn tất baseline ladder trên Crack500**. Bước tiếp theo là **B2 = full SAGE-Lite** (B1 + SAGE Router + Expert Pool + SA-Hub + Load-Balance Loss), để kiểm chứng xem cơ chế định tuyến chuyên gia thích ứng hình thái (SAGE) có tạo thêm cải thiện đột phá so với B1 hay không.


