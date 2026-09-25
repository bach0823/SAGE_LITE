# B1 Baseline ViT-Depth Ablation Report (Crack500)

*Date: 2026-09-22*  
*Hardware: Tesla T4 (Google Colab)*  
*Dataset: Crack500 (1896 train, 348 val, 1124 test)*  
*Protocol: Frozen Canonical Preprocessing, Single-Stage AdamW, cosine warmup 3 epochs, BS=20, lr=1e-4*  
*Validation Protocol: Setting A (deterministic 448x448 non-overlapping tiling)*  
*Official Test Protocol: Setting A (non-overlap) & Setting B (50% overlap, stride 224, blend_mode='probs', threshold=0.5)*  
*Decision Criterion: Chốt depth dựa hoàn toàn trên **VALIDATION SET** (chống Data Snooping)*

---

## 1. Bảng Tổng Hợp Huấn Luyện & Validation

| Cấu hình | Số ViT Blocks | Tham số | Best Epoch | Stop Epoch | Best Val Loss | Best Val Dice | Thứ hạng Val |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **B0 (CNN thuần)** | 0 | 10.23M | 8 | 14 | 1.3962 | 0.7318 | 5 |
| **B1 - Depth 4** | **4** | **9.34M** | **28** | **30 (Full)** | **1.1364** | **0.7428** | **Top 1 (Best)** |
| **B1 - Depth 6** | 6 | 10.23M | 21 | 27 | 1.1453 | 0.7420 | Top 2 |
| **B1 - Depth 8** | 8 | 11.12M | 19 | 25 | 1.2006 | 0.7379 | 4 |
| **B1 - Depth 12** | 12 | 12.90M | 16 | 22 | 1.2332 | 0.7419 | Top 3 |

---

## 2. Bảng Đánh Giá Chính Thức (Official Evaluation: Setting A & Setting B)

### A. Setting A (Non-overlapping Tiling 448×448)

| Cấu hình | Val Loss | Val Dice | Val Global IoU | Test Loss | Test Prec | Test Rec | Test Dice | Test Global IoU | Test Boundary IoU | Test HD95 (px) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **B0 (0 blocks)** | 1.3962 | 0.7318 | 0.6555 | 1.4204 | 0.6031 | 0.8329 | 0.6771 | 0.5645 | 0.0676 | 94.9825 |
| **Depth 4** | **1.1364** | **0.7428** | 0.6696 | **1.1758** | 0.6005 | **0.8548** | 0.6829 | 0.5700 | 0.0724 | 96.7220 |
| **Depth 6** | 1.1453 | 0.7420 | **0.6718** | 1.1814 | 0.6081 | 0.8466 | 0.6857 | 0.5730 | 0.0729 | 83.1099 |
| **Depth 8** | 1.2006 | 0.7379 | 0.6670 | 1.2320 | 0.6003 | 0.8513 | 0.6819 | 0.5688 | 0.0673 | 81.9521 |
| **Depth 12** | 1.2332 | 0.7419 | 0.6706 | 1.2683 | **0.6172** | 0.8389 | **0.6902** | **0.5774** | **0.0746** | **80.7891** |

### B. Setting B (50% Overlapping Tiling 448×448, stride 224, blend_mode='probs', threshold=0.5)

| Cấu hình | Val Loss | Val Dice | Val Global IoU | Test Loss | Test Prec | Test Rec | Test Dice | Test Global IoU | Test Boundary IoU | Test HD95 (px) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **B0 (0 blocks)** | 2.1137 | 0.7381 | 0.6599 | 2.1192 | 0.6033 | 0.8386 | 0.6801 | 0.5682 | 0.0676 | 94.3688 |
| **Depth 4** | **2.0571** | 0.7484 | 0.6727 | **2.0627** | 0.6031 | **0.8589** | 0.6867 | 0.5741 | 0.0723 | 90.6362 |
| **Depth 6** | 2.0586 | 0.7481 | **0.6756** | 2.0641 | 0.6113 | 0.8501 | 0.6895 | 0.5770 | 0.0731 | 77.4342 |
| **Depth 8** | 2.0695 | 0.7447 | 0.6708 | 2.0740 | 0.6028 | 0.8555 | 0.6854 | 0.5730 | 0.0673 | **74.9342** |
| **Depth 12** | 2.0750 | **0.7487** | 0.6744 | 2.0810 | **0.6188** | 0.8425 | **0.6926** | **0.5812** | **0.0746** | 77.5123 |

---

## 3. Chi Tiết Tiến Trình Hội Tụ Từng Cấu Hình

### A. B1 Depth 4
- **Checkpoint:** `/content/drive/MyDrive/crack_seg/B1_Crack500_Depth4/best_model_b1.pth`
- **Số tham số:** 9,342,865 params (nhẹ nhất nhóm B1).
- **Đặc điểm hội tụ:** Chạy trọn vẹn 30/30 epochs, không bị dừng sớm, loss giảm đều đặn.
  - Best checkpoint đạt tại **Epoch 28**: Val Dice **0.7428**, Val Loss **1.1364**.
  - Test Setting A Dice: **0.6829**, Test Setting B Dice: **0.6867**.
  - Đạt Recall cao nhất trong toàn bộ các cấu hình (**0.8589** ở Setting B), rất nhạy với các vết nứt mảnh.

### B. B1 Depth 6
- **Checkpoint:** `/content/drive/MyDrive/crack_seg/B1_Crack500_Run1/best_model_b1.pth`
- **Số tham số:** 10,232,593 params.
- **Đặc điểm hội tụ:** EarlyStopping tại epoch 27 (patience 6).
  - Best checkpoint tại **Epoch 21**: Val Dice **0.7420**, Val Loss **1.1453**.
  - Test Setting A Dice: **0.6857**, Test Setting B Dice: **0.6895**, Test HD95: **77.43 px**.

### C. B1 Depth 8
- **Checkpoint:** `/content/drive/MyDrive/crack_seg/B1_Crack500_Depth8/best_model_b1.pth`
- **Số tham số:** 11,122,321 params.
- **Đặc điểm hội tụ:** EarlyStopping sớm tại epoch 25 (patience 6).
  - Best checkpoint tại **Epoch 19**: Val Dice **0.7379**, Val Loss **1.2006**.
  - Hiệu năng chững lại sớm nhất và thấp nhất trong các cấu hình ViT (Val Dice 0.7379, Test Dice B 0.6854).

### D. B1 Depth 12
- **Checkpoint:** `/content/drive/MyDrive/crack_seg/B1_Crack500_Depth12/best_model_b1.pth`
- **Số tham số:** 12,901,777 params (+38% params so với Depth 4).
- **Đặc điểm hội tụ:** EarlyStopping rất sớm tại epoch 22 (patience 6).
  - Best checkpoint tại **Epoch 16**: Val Dice **0.7419**, Val Loss **1.2332**.
  - Do số tham số lớn, mô hình có xu hướng overfit nhanh trên tập train nhỏ, thể hiện qua việc Val Loss cao nhất (1.2332) và dừng học chỉ sau 16 epoch hiệu quả.
  - Mặc dù Test Dice đạt 0.6926 trên Test set, nhưng khoảng cách Val Loss cho thấy mô hình kém ổn định hơn Depth 4.

---

---

## 4. Phân Tích Thực Nghiệm & Quyết Định Chiến Lược (Interpretation & Decision Note)

1. **Hiệu năng Validation:**
   - B1 Validation Dice của các cấu hình Depth 4, Depth 6 và Depth 12 gần như ngang nhau (~0.742):
     - Depth 4: Best Val Dice = **0.7428** (epoch 24, đạt lại 0.7428 ở epoch 28), Best Val Loss = **1.1364**
     - Depth 6: Best Val Dice = **0.7420** (epoch 21), Best Val Loss = **1.1453**
     - Depth 8: Best Val Dice = **0.7379** (epoch 19), Best Val Loss = **1.2006**
     - Depth 12: Best Val Dice = **0.7419** (epoch 16), Best Val Loss = **1.2332**
   - Sự chênh lệch Val Dice giữa Depth 4, 6, 12 là rất nhỏ (< 0.001), do đó **Val Dice đơn thuần không đủ để phân biệt rõ ràng sự vượt trội giữa 3 depth này**.

2. **Official Test Evaluation:**
   - Trên tập Test, Depth 12 đạt Test Dice cao hơn Depth 4, 6, 8 ở cả Setting A (0.6902) và Setting B (0.6926).
   - Tuy nhiên, **B1 vẫn chỉ là mô hình U-Net kết hợp ViT thuần túy (Late Fusion) và CHƯA có cơ chế SAGE routing hay expert specialization**. Test Dice cao hơn ở B1 chủ yếu phản ánh năng lực biểu diễn của một backbone sâu hơn, chưa phản ánh tính hiệu quả của cơ chế điều hướng đa chuyên gia.
   - Do đó, **tuyệt đối KHÔNG dùng kết quả B1 Test để kết luận vội vàng rằng "Depth 12 là depth tối ưu của SAGE-Lite"**.

3. **Bản chất của B1 Depth Sweep:**
   - Đợt thực nghiệm B1 ViT-depth sweep này đóng vai trò là bước **Screening / Diagnostic** nhằm khảo sát ảnh hưởng của chiều sâu ViT trước khi có SAGE.
   - Vì số lượng ViT blocks ảnh hưởng trực tiếp đến số lượng điểm injection, số Router, và cơ hội chuyên môn hóa của các chuyên gia trong B2 ($\mathcal{N}_{\text{injection}} = 4 \text{ CNN} + \text{num\_transformer\_layers}$), nên hiệu quả thực sự của từng mức depth **bắt buộc phải được kiểm chứng lại trong B2**, khi Router + Expert Pool + SA-Hub + Load-Balance Loss thực sự hoạt động đồng bộ.

---

## 5. Kế Hoạch Tiếp Theo (Next Plan)

1. **Chưa freeze ViT depth từ B1:** Không vội chốt cứng một depth duy nhất tại thời điểm này.
2. **Không cần chạy lại B1:** Toàn bộ dữ liệu screening của B1 `{4, 6, 8, 12}` đã hoàn chỉnh và được lưu trữ đầy đủ.
3. **B2 Depth Ablation tập trung vào `{4, 6, 12}`:** 
   - Sau khi hoàn thành B1 và xây dựng xong core components của B2, sẽ tiến hành ablation trực tiếp trên B2 với 3 mức depth trọng tâm: **4, 6 và 12 blocks** để đo đạc chính xác tác động khi có SAGE routing.
4. **Bảo toàn tính công bằng thực nghiệm:**
   - Giữ nguyên toàn bộ các thiết lập còn lại giữa các depth (single-stage, AdamW, cosine warmup 3 epochs, differential LR, seed 42, frozen canonical preprocessing).
   - Quyết định depth chính thức cho SAGE-Lite cuối cùng sẽ được đưa ra **sau khi có kết quả B2 Depth Ablation**.

