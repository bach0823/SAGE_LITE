# SAGE-Lite B2 Proposal 3 (P3) Architecture Comparison Report (Crack500)
## Nghiệm Thu Thực Nghiệm So Sánh Toàn Diện: P3-A vs P3-B vs P3-C (D12, BS14, 30 Epochs)

*Date: 2026-09-26*  
*Hardware: Tesla T4 (Google Colab, 14.56 GB usable VRAM, AMP FP16)*  
*Git Branch: `crack500-audit`*  
*Dataset: Crack500 (1896 train samples, 348 val pairs, 448x448 canonical)*  
*Protocol: Two-Stage Ladder (Stage 1: 15 epochs, Stage 2: 15 epochs, patience 6, warmup 3 epochs)*  
*Evaluation Setting: Setting A (deterministic 448x448 tiling on Crack500 Val)*  

---

## 1. Bảng Tổng Hợp Kết Quả Thực Nghiệm (Consolidated Comparison Table)

Tất cả 3 cấu hình P3 đều được huấn luyện đầy đủ (Full Confirmation - 30 epochs) ở chế độ **Standalone ImageNet-pretrained** trên cùng một môi trường phần cứng, cùng bộ tiền xử lý, cùng seed 42, cùng số lượng ViT blocks (D12), cùng batch size (14) và cùng bộ siêu tham số optimizer:

| Cấu hình (Mode) | Refinement Module | Tham số P3 | Best Val Dice | Best Val Loss | Best Epoch | Stage 1 Best Dice | Throughput (Colab T4) | Final $\gamma$ (S0 / S1) | Trạng thái Nghiệm thu |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **P3-A (Identity Control)** | `nn.Identity()` | **0** | **0.7543** | **1.0140** | Stage 2, Ep 13 | 0.7158 (Ep 14) | **~3.9 min / epoch** (~1.72s/it) | N/A (Identity) | **CONFIRMED CANDIDATE** |
| **P3-B (Generic DW Control)** | $3 \times \text{DWConv}(3\times3)$ | 38,450 | **0.7496** | **1.0495** | Stage 2, Ep 15 | 0.7179 (Ep 10) | **~4.1 min / epoch** (~1.80s/it) | 0.0105 / 0.0136 | **CONFIRMED** |
| **P3-C (ASDW Refinement)** | ASDW ($1\times7 + 7\times1 + 3\times3$) | 37,874 | **0.7557** | **1.0889** | Stage 2, Ep 9 | 0.7180 (Ep 10) | **~4.0 min / epoch** (~1.78s/it) | 0.0063 / 0.0071 | **TOP 1 DICE CANDIDATE** |
| *B2 Base D12 (No P3 / Pure)* | *None (Full $112\times112$)* | *0* | *0.7057 (S1 ep 4)* | *1.5203* | *In-Progress* | *0.7057 (S1)* | *~13.0 min / epoch* (~4.90s/it) | *N/A* | *Bottleneck Baseline* |

---

## 2. Chi Tiết Tiến Trình Huấn Luyện Từng Cấu Hình

### 2.1. Run A — P3-A (Identity / Spatial Compression Control)
- **Đặc tả kiến trúc:** Không có module tinh chỉnh lọc đặc trưng. Feature map của CNN Stage 0 ($112 \times 112$) và Stage 1 ($56 \times 56$) được đưa trực tiếp qua `AdaptiveAvgPool2d(28, 28)` + bộ đệm `pe28_fixed` (nội suy bicubic 2D từ PE14) trước khi nạp vào ViT expert.
- **Tiến trình Stage 1 (15 epochs):**
  - Khởi điểm: Epoch 1 đạt Val Dice `0.1461`, Val Loss `2.0335`.
  - Tăng tốc ổn định: Vượt ngưỡng 0.60 tại Epoch 4 (`0.6821`), vượt 0.70 tại Epoch 8 (`0.7060`).
  - Kỷ lục Stage 1: Đạt tại **Epoch 14** với Val Dice = **0.7158**, Val Loss = **1.2967** (Train Dice `0.6864`).
- **Tiến trình Stage 2 (Fine-tuning toàn diện 15 epochs):**
  - Nạp best Stage 1 checkpoint tại Epoch 14. Mở khóa toàn bộ backbone và CNN shared experts với `shared_lr = 1e-4`, `base_lr = 1e-4`.
  - Epoch 5: Val Dice tăng lên `0.7310`, Val Loss giảm xuống `1.1985`.
  - Epoch 10: Val Dice bứt phá lên `0.7492`, Val Loss giảm sâu xuống `1.0308`.
  - **Điểm tối ưu toàn cục (Global Best):** Đạt tại **Stage 2, Epoch 13**:
    - **Val Dice: 0.7543** (75.43%)
    - **Val Loss: 1.0140** (mức loss thấp nhất trong toàn bộ các cấu hình đã thử nghiệm)
    - Train Loss: `1.1365` (LB: `0.1607`), Train Dice: `0.7178`.
  - Hai epoch cuối (14 & 15) duy trì trạng thái ổn định cao: Epoch 14 đạt `0.7495` (Loss `1.0311`), Epoch 15 đạt `0.7485` (Loss `1.0368`).

### 2.2. Run B — P3-B (Generic Isotropic Depthwise Refinement)
- **Đặc tả kiến trúc:** 3 nhánh $3 \times 3$ Depthwise Conv đối xứng năng lực ($38,450$ tham số) $\to$ Concat $\to$ GELU $\to$ PWConv $1 \times 1 \to$ Residual ($\gamma$) $\to$ `AdaptiveAvgPool2d(28, 28)`.
- **Tiến trình Stage 1 (15 epochs):**
  - Khởi điểm: Epoch 1 đạt Val Dice `0.1535`, Val Loss `2.0209`.
  - Epoch 4: Val Dice `0.6635`, Val Loss `1.5530`.
  - Kỷ lục Stage 1: Đạt tại **Epoch 10** với Val Dice = **0.7179**, Val Loss = **1.3120** (Train Dice `0.6626`).
- **Tiến trình Stage 2 (15 epochs):**
  - Nạp best Stage 1 checkpoint tại Epoch 10.
  - Epoch 6: Val Dice `0.7294`, Val Loss `1.2001`.
  - Epoch 10: Val Dice `0.7464`, Val Loss `1.0665`.
  - **Điểm tối ưu toàn cục (Global Best):** Đạt tại **Stage 2, Epoch 15** (chạy trọn vẹn 30/30 epochs):
    - **Val Dice: 0.7496** (74.96%)
    - **Val Loss: 1.0495**
    - Train Loss: `1.1466` (LB: `0.1607`), Train Dice: `0.7199`.
    - Hệ số $\gamma$ thích nghi: $\gamma_{\text{S0}} = 0.0105$, $\gamma_{\text{S1}} = 0.0136$ (tăng trưởng đều đặn từ giá trị khởi tạo $0.0100$).

### 2.3. Run C — P3-C (Anisotropic Strip Depthwise Refinement - ASDW)
- **Đặc tả kiến trúc:** 3 nhánh bất đẳng hướng $1 \times 7 + 7 \times 1 + 3 \times 3$ ($37,874$ tham số) bám sát hình thái vết nứt mảnh $\to$ Concat $\to$ GELU $\to$ PWConv $1 \times 1 \to$ Residual ($\gamma$) $\to$ `AdaptiveAvgPool2d(28, 28)`.
- **Tiến trình Stage 1 (15 epochs):**
  - Kỷ lục Stage 1: Đạt tại **Epoch 10** với Val Dice = **0.7180**, Val Loss = **1.3145**.
- **Tiến trình Stage 2:**
  - **Điểm tối ưu toàn cục (Global Best):** Đạt tại **Stage 2, Epoch 9**:
    - **Val Dice: 0.7557** (75.57% — Cao nhất toàn bảng)
    - **Val Loss: 1.0889**
    - Train Loss: `1.2095` (LB: `0.1607`), Train Dice: `0.7030`.
    - Hệ số $\gamma$ hội tụ: $\gamma_{\text{S0}} = 0.0063$, $\gamma_{\text{S1}} = 0.0071$.
  - Early stopping kích hoạt tại Stage 2 Epoch 15 (sau 6 epoch không vượt qua kỷ lục Epoch 9).

---

## 3. Phân Tích Khoa Học & Đánh Giá Thực Nghiệm (Scientific Analysis)

### 3.1. Phân Tích Khoảng Cách P3-C vs P3-A ($\Delta = +0.14\%$)
- **Mức chênh lệch Val Dice:**
  $$\Delta \text{Val Dice} = \text{Dice}_{\text{P3-C}} - \text{Dice}_{\text{P3-A}} = 0.7557 - 0.7543 = +0.0014 \ (+0.14\%)$$
- **Mức chênh lệch Val Loss:** P3-A đạt Val Loss thấp hơn rõ rệt (**1.0140** vs **1.0889**, giảm $0.0749$).
- **Chi phí phần cứng & Độ phức tạp:**
  - P3-A: Đúng **0 tham số bổ sung**, không có thêm phép nhân chập nào, độ phức tạp kiến trúc tối thiểu.
  - P3-C: Bổ sung 37,874 tham số dạng dải ASDW ($1\times7, 7\times1$), mang lại mức cải thiện nhẹ $+0.14\%$ Dice.
- **Ý nghĩa cơ chế:**
  - Việc nén không gian về $28 \times 28$ kết hợp với bộ đệm vị trí bicubic PE28 (`AdaptiveAvgPool2d(28, 28) + pe28_fixed`) chính là **yếu tố nền tảng cốt lõi** quyết định thành công của cả nhánh P3. Bản thân cơ chế nén này (Run A) đã đủ để đưa mô hình đạt Val Dice `0.7543` (vượt xa baseline B1 D12 `0.7419`).
  - Nhánh ASDW (Run C) tinh chỉnh thêm chi tiết vi mô, giúp mô hình nhỉnh hơn ở ngưỡng cực đại (`0.7557`).

### 3.2. Vị Trí Của Run B (Generic DW Refinement)
- Run B đạt Val Dice `0.7496`, thấp hơn P3-C $0.61\%$ và thấp hơn P3-A $0.47\%$.
- Kết quả này chứng minh rằng việc áp dụng các bộ lọc tích chập đẳng hướng thông thường ($3 \times 3\text{ DWConv}$) **không mang lại lợi ích** cho việc trích xuất đặc trưng vết nứt trước khi nén, thậm chí còn gây nhiễu nhẹ so với việc nén trung bình trực tiếp (Run A). Chỉ có thiết kế dải bất đẳng hướng ASDW (Run C) mới phát huy được hiệu quả bổ trợ.

### 3.3. Hiệu Quả Tăng Tốc Throughput Đột Phá So Với B2 Base Thuần
- **B2 Base D12 (Chưa nén - Gọi ViT Expert ở $112\times112$ và $56\times56$):**
  - Tốc độ huấn luyện: **~13.0 phút / epoch** (tương đương $\sim 4.90 - 6.64$ giây/iteration ở BS12).
- **Các biến thể P3 (P3-A, P3-B, P3-C - Nén về $28\times28$):**
  - Tốc độ huấn luyện: **~3.9 - 4.1 phút / epoch** (tương đương $\sim 1.72 - 1.80$ giây/iteration ở BS14).
- $\implies$ **Tăng tốc thực tế:** P3 giúp tăng tốc độ huấn luyện end-to-end gấp **$\sim 3.2$ lần** trên GPU Tesla T4, đồng thời giải phóng VRAM cho phép nâng batch size từ 12 lên 14 an toàn.

---

## 4. Quyết Định Thực Nghiệm Tại Decision Gate 1 (Phase 1 Verdict)

Căn cứ theo quy tắc phân nhánh đã đăng ký trong [`docs/Crack500_Experimental_Protocol_Roadmap.md`](file:///d:/truong/SpecialSubjectTTNT/docs/Crack500_Experimental_Protocol_Roadmap.md#L230-L234):
- Khoảng cách giữa P3-C (`0.7557`) và P3-A (`0.7543`) là $\Delta = 0.0014 < 0.003$ ($0.3\%$).
- Tình huống thực nghiệm thuộc về **Case B (Equivalence / Extremely Close Candidates)**.

### Kết Luận & Định Hướng Tuyển Chọn:
1. **Loại bỏ P3-B (Generic DW):** P3-B có hiệu năng kém nhất trong cả 3 biến thể (`0.7496`), chính thức bị loại khỏi các vòng khảo sát tiếp theo.
2. **Duy trì Cặp Ứng Viên Hàng Đầu: P3-C và P3-A:**
   - **P3-C**: Ứng viên số 1 về độ chính xác phân đoạn cực đại (Max Val Dice = **0.7557**).
   - **P3-A**: Ứng viên số 1 về tính tinh gọn kiến trúc (Zero Extra Params, Val Loss thấp nhất = **1.0140**, Val Dice = **0.7543**).
3. **Chuyển tiếp sang Phase 2 (Depth Screening):** Tiến hành khảo sát chiều sâu ViT (D4, D6) trên P3 để đánh giá xem dung lượng mô hình có thể tinh giản hơn nữa mà vẫn duy trì mức Dice $\ge 0.75$ hay không.
