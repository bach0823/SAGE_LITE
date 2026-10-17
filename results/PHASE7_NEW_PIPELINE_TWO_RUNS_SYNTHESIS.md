# Báo Cáo Đối Soát Huấn Luyện Pipeline Mới (Crack500): Run 1 vs Run 2

> **Giao thức Chuẩn hóa**: Direct Resize ($640 \times 360 \to 448 \times 448$ online), Không Crop ngẫu nhiên, Purged `smart_filter` & `crop_mode`  
> **Kiến trúc**: B2 UNet Hybrid (ConvNeXt-V2-Femto + ViT-Tiny, $D=4, K=2$, Dual S2-Gate: Block 1 Conv $3\times 3$ + Block 2 Conv $3\times 3, d=2$)  
> **Loss**: BCE + Dice + SoftBIoU ($\lambda=0.50, d=2$)  
> **Nguồn lưu trữ Checkpoint**: 
> - Run 1: [`results/phase7_combination/phase7_s2g_new_pipeline_run1/`](file:///d:/truong/SpecialSubjectTTNT/SAGE_LITE/results/phase7_combination/phase7_s2g_new_pipeline_run1)
> - Run 2 (Best Checkpoint): [`results/phase7_combination/phase7_s2g_new_pipeline_run2/`](file:///d:/truong/SpecialSubjectTTNT/SAGE_LITE/results/phase7_combination/phase7_s2g_new_pipeline_run2)

---

## 1. Kết quả Huấn luyện trên Tập Validation (348 ảnh)

| Chỉ số | Pipeline Cũ (Phase 7 gốc) | **Pipeline Mới - Run 1** | **Pipeline Mới - Run 2** | Phương sai ($\Delta$ R2 - R1) | Nhận xét |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Stage 1 Best Epoch** | Epoch 13 | Epoch 10 | Epoch 10 | 0 | Đồng nhất tuyệt đối |
| **Stage 1 Best Val Dice** | 0.7540 | **0.7649** | **0.7629** | -0.0020 | Tăng **+1.0%** ngay từ Stage 1 |
| **Stage 1 Best Val Loss** | 1.6149 | 1.6118 | 1.6060 | -0.0058 | Ổn định |
| **Stage 2 Best Epoch** | Epoch 15 | Epoch 18 | Epoch 16 | -2 | Đều đạt đỉnh ở các epoch cuối |
| **Stage 2 Best Val Dice** | 0.7623 | **0.7783** | **🏆 0.7793** | `+0.0010` | **Kỷ lục mới**: Tăng **+1.70%** Val Dice |
| **Stage 2 Val Loss** | 1.3258 | 1.2315 | 1.2277 | -0.0038 | Loss thấp hơn rõ rệt |
| **Stage 2 Precision** | 0.7344 | 0.7424 | **0.7528** | `+0.0104` | Precision tăng mạnh |
| **Stage 2 Recall** | 0.8372 | **0.8538** | 0.8444 | -0.0094 | Duy trì recall cao |
| **Stage 2 IoU** | 0.6400 | 0.6560 | **0.6578** | `+0.0018` | Vượt mốc 65.7% |

---

## 2. Kết quả Đánh giá Chuẩn Chính thức trên Tập Test (1,124 ảnh)

> [!IMPORTANT]
> Toàn bộ quá trình đánh giá dưới đây được thực hiện bằng script chuẩn [`scripts/evaluate_crack_official.py`](file:///d:/truong/SpecialSubjectTTNT/SAGE_LITE/scripts/evaluate_crack_official.py) thông qua phương thức suy luận trực tiếp 1-pass (`predict_full_image_direct`: Resize $640 \times 360 \to 448 \times 448 \to$ Model $\to$ Logits $640 \times 360$), **hoàn toàn loại bỏ hiện tượng méo mó biên do cắt lát (tiling) gây ra**.

### Bảng Chỉ số Đánh giá Chính thức (Checkpoint Run 2: `best_model_b2_stage2.pth`)
*(Nguồn file lưu trữ: [`official_direct_evaluation.json`](file:///d:/truong/SpecialSubjectTTNT/SAGE_LITE/results/phase7_combination/phase7_s2g_new_pipeline_run2/diagnostics/official_direct_evaluation.json))*

| Tập Dữ liệu | Số mẫu | Dice / F1 | Macro IoU | Global Pixel IoU | Precision | Recall | Boundary IoU | HD95 (px) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **VAL Set** | 348 | **0.7793** | 0.6578 | 0.7013 | 0.7525 | 0.8446 | 0.1318 | **38.69 px** |
| **TEST Set** | 1,124 | **🏆 0.7149** | **0.5746** | **0.5996** | **0.6778** | **0.8011** | **0.1024** | **48.26 px** |

### So sánh với các phương thức suy luận trước đây:
- **Test Dice**: Tăng vọt từ **0.6950** (phép cắt lát tiling chồng lấp Setting B cũ) lên **0.7149** (**+1.99%** Dice tuyệt đối).
- **Test IoU**: Đạt **0.5746** (57.46%), cao nhất từ trước đến nay.
- **Khoảng cách sai lệch biên (HD95)**: Giảm ngoạn mục từ **99.43 px** xuống còn **48.26 px** (giảm hơn một nửa sai lệch biên độ nứt nhờ không bị chia cắt hình học ở mép tile).

---

## 3. Trạng thái Dọn dẹp & Giải quyết Xung đột Mã nguồn
1. **Đã xóa bỏ hoàn toàn `tools/evaluate_final_test.py`**: Triệt tiêu nguồn gốc sinh ra các bảng Setting A/B cũ.
2. **Đã cập nhật `tools/compare_topology_phase6_vs_phase7.py`**: Chuyển toàn bộ luồng kiểm toán tô-pô sang `predict_full_image_direct`.
3. **Đã vá lỗi phần cứng FP16 (GTX 1650/1660)**: Trong `predict_full_image_direct`, bổ sung cơ chế bypass autocast trên các dòng card Turing 16xx để triệt tiêu lỗi sinh giá trị `NaN`.
