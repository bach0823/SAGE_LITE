# Phase 7 End-to-End Evaluation Report: Dual S2-Gate (Block 1 + Block 2)

> **Mô hình**: B2 UNet Hybrid (ConvNeXt-V2-Femto + ViT-Tiny, $D=4, K=2$)  
> **Kiến trúc mới**: Dual Spatial Modulation Gates (S2-Gate Block 1 Conv $3\times 3$ + S2-Gate Block 2 Conv $3\times 3, d=2$)  
> **Loss**: BCE + Dice + SoftBIoU ($\lambda=0.50, d=2$)  
> **Batch Size**: 16  
> **Nguồn dữ liệu**: Trích xuất từ gói `phase7_s2g_checkpoints_and_logs.zip`

---

## 1. Kết quả Huấn luyện (Validation Split: 348 mẫu)

### Stage 1 (Backbone, Decoder, SAGE, S2-Gate khởi động cùng lúc)
- **Số epoch**: 17 epochs
- **Peak Val Dice**: **`0.7540`** (Epoch 13)
- **Val Loss**: `1.6149`
- **Val IoU**: `0.6276`
- **Precision / Recall**: `0.7161` / `0.8412`

### Stage 2 (Fine-tuning theo giao thức SAGE gốc: `base_lr = 1e-4`, `shared_lr = 1e-4`)
- **Số epoch**: 18 epochs (Đạt đỉnh tại Epoch 15/18)
- **Peak Val Dice**: **`0.7623`** (**76.23%**, tăng **+0.83%** so với đỉnh Stage 1!)
- **Val Loss**: **`1.3258`** (Giảm mạnh từ `1.6149` xuống `1.3258`)
- **Val IoU**: **`0.6400`** (64.00%)
- **Precision / Recall**: `0.7344` / `0.8372`

---

## 2. Kết quả Đánh giá Tập Test Chính thức (Crack500 Test Split: 1,124 mẫu)

Đánh giá đơn lẻ chính thức (Single-Shot Preregistered Test Evaluation) tại `best_model_b2_stage2.pth`:

| Giao thức Tiling | Mean Dice | Median Dice | Macro IoU | Global Pixel IoU | Crack-Present IoU | Precision | Recall | Boundary IoU | HD95 (px) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Setting A** (Non-overlapping) | **0.6996** | **0.7363** | **0.5573** | **0.5876** | **0.5573** | **0.6530** | **0.8061** | **0.0999** | **85.11** |
| **Setting B** (50% Overlapping) | **0.7038** | **0.7398** | **0.5624** | **0.5923** | **0.5624** | **0.6584** | **0.8073** | **0.1010** | **76.19** |

---

## 3. Danh mục Checkpoints và Tài nguyên Đã Lưu Trữ Cục Bộ

Thư mục lưu trữ: [`results/phase7_combination/phase7_s2g_b1_b2_end_to_end/`](file:///d:/truong/SpecialSubjectTTNT/SAGE_LITE/results/phase7_combination/phase7_s2g_b1_b2_end_to_end)

1. **`best_model_b2_global.pth`** (42.12 MB): Mô hình tốt nhất toàn cục (Epoch 15 Stage 2, Val Dice `0.7623`).
2. **`best_model_b2_stage2.pth`** (42.12 MB): Trọng số tốt nhất của Stage 2.
3. **`best_model_b2_stage1.pth`** (42.12 MB): Trọng số tốt nhất của Stage 1 (Val Dice `0.7540`).
4. **`last_model_b2_stage2.pth`** (123.12 MB): Checkpoint đầy đủ kết thúc Stage 2 (gồm model, optimizer, scheduler, RNG state).
5. **`last_model_b2_stage1.pth`** (123.12 MB): Checkpoint kết thúc Stage 1.
6. **`stage1_completion.json` & `stage2_completion.json`**: File JSON ghi nhận toàn bộ thông số và siêu tham số huấn luyện.
7. **`train.log` & `train_stage1.log`**: Nhật ký huấn luyện chi tiết từng epoch của cả 2 stages.
8. **Thư mục `diagnostics/`**:
   - `final_test_summary.md`: Báo cáo markdown tổng kết tập Test 1,124 ảnh.
   - `final_test_metrics.json`: JSON chi tiết điểm số Setting A & B.
   - `final_test_per_sample.csv`: Dữ liệu phân tích từng bức ảnh một của tập Test.
   - `test_evaluation_audit_trail.log`: Dấu vết kiểm toán của lượt chạy đánh giá.
