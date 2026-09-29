# Residual / Fusion Diagnostic Report (Canonical D4-P3-C K4)

**Date**: 2026-09-29 01:01:26  
**Checkpoint**: `P3_C_D4_best_model_b2_global.pth` (Epoch 14, Best Val Dice: 0.763907)  
**SHA256**: `33b0299dde38a4f29cfe0c3b0c3b6a27f14d4381a7efffdfd4d009843fa058ef`  
**Dataset**: Crack500 Validation Split (Setting A, N=348 samples)  
**Protocol**: Strictly Frozen-Routing Residual Sweep with Natural Downstream Cascade  

---

## 1. Phương Pháp & Giao Thức Nghiên Cứu (Frozen-Routing Protocol)

Nghiên cứu chẩn đoán này kiểm chứng trực tiếp giả thuyết: **Liệu hệ số tỷ lệ residual scale (mặc định $\alpha=0.1$) có đang làm suy hao quá mức hoặc gây mất cân đối trong đóng góp của nhánh expert giữa các nhóm router hay không.**

Quy trình được thực hiện qua 2 giai đoạn bất biến:
1. **Phase 1 (Canonical routing cache)**:
   - Chạy mô hình ở trạng thái chuẩn tắc (canonical baseline) với $\alpha=0.1$, exploration noise tắt (`exploration_noise=False`), chế độ deterministic evaluation.
   - Cache toàn bộ quyết định routing `(top_k_indices, gating_weights)` cho **từng tile patch, từng router** ($S0..S3, B0..B3$) trên toàn bộ 348 mẫu validation vào CPU RAM.
   - Tuyệt đối không cache $M_l$ hay $E_l$ để tính toán hậu nghiệm.
2. **Phase 2 (Residual sweep with natural downstream cascade)**:
   - Chạy fresh full forward hoàn chỉnh từ input gốc cho từng giá trị $\alpha$.
   - Tại mỗi `SageLayer`, bypass hoàn toàn việc tính lại routing (query projection, gating modulation, top-k selection); inject trực tiếp cặp `(top_k_indices, gating_weights)` tương ứng đã cache.
   - Nhánh expert tính toán lại $E_l$ trên input hiện tại của tầng; nhánh chính tính $M_l$; hợp nhất $y_l = M_l + \alpha_l E_l$.
   - Tín hiệu hợp nhất $y_l$ tiếp tục lan truyền tự nhiên (downstream cascade) sang các tầng tiếp theo. Khi tầng upstream thay đổi $\alpha$, các tầng downstream nhận biểu diễn mới và tự động tính toán lại $M$ và $E$.

---

## 2. Baseline Verification

- **Canonical Checkpoint Expected Dice**: `0.763907`
- **Evaluated Baseline Mean Dice**: `0.763936` ($\Delta = 0.000029$)
- **Evaluated Baseline Mean IoU**: `0.641186` (Expected: `0.641199`)
- **Evaluated Baseline Median Dice**: `0.804921`
- **Baseline Dice Standard Deviation**: `0.163962`
- **Status**: **VERIFIED MATCH** (Độ lệch $< 0.0001$, tái lập chuẩn tắc hoàn toàn).

---

## 3. Kết Quả Chi Tiết 3 Đợt Quét (Sweeps)

### Sweep 1 — ViT Group {B0, B1, B2, B3} (S0..S3 giữ cố định = 0.1)

| $\alpha$ | Mean Dice $\pm$ Std | Median Dice | Mean IoU | Precision | Recall | $\Delta$ Dice (Mean) | 95% CI | $p$ (paired t) | $p$ (Wilcoxon) | Win/Tie/Loss |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.000** | 0.7642 $\pm$ 0.1653 | 0.8070 | 0.6419 | 0.7346 | 0.8448 | **+0.000312** | [-0.000638, +0.001262] | 0.5185 | 1.5153e-04 | 204.0/3.0/141.0 |
| **0.025** | 0.7643 $\pm$ 0.1648 | 0.8065 | 0.6418 | 0.7337 | 0.8460 | **+0.000357** | [-0.000364, +0.001078] | 0.3312 | 5.8077e-05 | 204.0/3.0/141.0 |
| **0.050** | 0.7643 $\pm$ 0.1644 | 0.8065 | 0.6417 | 0.7326 | 0.8472 | **+0.000337** | [-0.000136, +0.000810] | 0.1621 | 1.0035e-05 | 210.0/3.0/135.0 |
| **0.075** | 0.7642 $\pm$ 0.1642 | 0.8055 | 0.6415 | 0.7313 | 0.8484 | **+0.000221** | [-0.000017, +0.000459] | 0.0683 | 2.0209e-05 | 202.0/5.0/141.0 |
| **0.100** | 0.7639 $\pm$ 0.1640 | 0.8049 | 0.6412 | 0.7298 | 0.8498 | 0.000000 | [+0.000000, +0.000000] | nan | 1.0000 | 0.0/348.0/0.0 |
| **0.150** | 0.7634 $\pm$ 0.1635 | 0.8045 | 0.6405 | 0.7267 | 0.8528 | **-0.000494** | [-0.000924, -0.000063] | 2.4763e-02 | 1.6331e-06 | 131.0/2.0/215.0 |
| **0.200** | 0.7628 $\pm$ 0.1629 | 0.8048 | 0.6395 | 0.7234 | 0.8560 | **-0.001149** | [-0.002028, -0.000271] | 1.0456e-02 | 1.5643e-07 | 132.0/2.0/214.0 |


### Sweep 2 — S3 Isolated {S3} (Tất cả router khác giữ cố định = 0.1)

| $\alpha$ | Mean Dice $\pm$ Std | Median Dice | Mean IoU | Precision | Recall | $\Delta$ Dice (Mean) | 95% CI | $p$ (paired t) | $p$ (Wilcoxon) | Win/Tie/Loss |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.000** | 0.7635 $\pm$ 0.1642 | 0.8053 | 0.6407 | 0.7284 | 0.8508 | **-0.000450** | [-0.000721, -0.000180] | 1.1727e-03 | 8.9621e-06 | 128.0/4.0/216.0 |
| **0.050** | 0.7637 $\pm$ 0.1641 | 0.8051 | 0.6410 | 0.7292 | 0.8502 | **-0.000203** | [-0.000344, -0.000061] | 5.0830e-03 | 3.8866e-05 | 135.0/15.0/198.0 |
| **0.100** | 0.7639 $\pm$ 0.1640 | 0.8049 | 0.6412 | 0.7298 | 0.8498 | 0.000000 | [+0.000000, +0.000000] | nan | 1.0000 | 0.0/348.0/0.0 |
| **0.200** | 0.7643 $\pm$ 0.1637 | 0.8055 | 0.6416 | 0.7310 | 0.8492 | **+0.000402** | [+0.000143, +0.000661] | 2.3997e-03 | 1.8318e-04 | 208.0/6.0/134.0 |
| **0.300** | 0.7645 $\pm$ 0.1636 | 0.8054 | 0.6418 | 0.7317 | 0.8488 | **+0.000578** | [+0.000135, +0.001022] | 1.0695e-02 | 6.0876e-04 | 201.0/4.0/143.0 |


### Sweep 3 — Shallow CNN Group {S0, S1, S2} (S3, B0..B3 giữ cố định = 0.1)

| $\alpha$ | Mean Dice $\pm$ Std | Median Dice | Mean IoU | Precision | Recall | $\Delta$ Dice (Mean) | 95% CI | $p$ (paired t) | $p$ (Wilcoxon) | Win/Tie/Loss |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.000** | 0.7638 $\pm$ 0.1634 | 0.8066 | 0.6409 | 0.7294 | 0.8496 | **-0.000112** | [-0.000939, +0.000714] | 0.7893 | 9.7649e-03 | 152.0/3.0/193.0 |
| **0.050** | 0.7640 $\pm$ 0.1637 | 0.8069 | 0.6412 | 0.7295 | 0.8501 | **+0.000065** | [-0.000313, +0.000442] | 0.7364 | 0.1041 | 164.0/4.0/180.0 |
| **0.100** | 0.7639 $\pm$ 0.1640 | 0.8049 | 0.6412 | 0.7298 | 0.8498 | 0.000000 | [+0.000000, +0.000000] | nan | 1.0000 | 0.0/348.0/0.0 |
| **0.200** | 0.7634 $\pm$ 0.1647 | 0.8049 | 0.6406 | 0.7306 | 0.8478 | **-0.000550** | [-0.001431, +0.000330] | 0.2200 | 0.8602 | 174.0/3.0/171.0 |
| **0.300** | 0.7623 $\pm$ 0.1652 | 0.8043 | 0.6393 | 0.7312 | 0.8436 | **-0.001622** | [-0.003302, +0.000058] | 0.0585 | 0.1297 | 159.0/4.0/185.0 |



---

## 4. Phân Tích Hình Thái Vết Nứt (Morphology Quartiles Q1 - Q4)

> Ngưỡng thinness score ($P / 2A$) được khóa cố định từ toàn bộ Ground Truth của 348 mẫu:  
> **Q1** ($t \le 0.0830$), **Q2** ($0.0830 < t \le 0.1238$), **Q3** ($0.1238 < t \le 0.1764$), **Q4** ($t > 0.1764$).

| Sweep | $\alpha$ | Overall Dice | Q1 Dice (Thick cracks) | Q2 Dice | Q3 Dice | Q4 Dice (Ultra-thin cracks) | $\Delta$ Dice vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ViT Group** | 0.000 | 0.7642 | 0.8464 | 0.7989 | 0.7435 | 0.6682 | **+0.000312** |
| **ViT Group** | 0.025 | 0.7643 | 0.8467 | 0.7994 | 0.7437 | 0.6675 | **+0.000357** |
| **ViT Group** | 0.050 | 0.7643 | 0.8469 | 0.7998 | 0.7440 | 0.6664 | **+0.000337** |
| **ViT Group** | 0.075 | 0.7642 | 0.8470 | 0.8002 | 0.7443 | 0.6651 | **+0.000221** |
| **ViT Group** | 0.100 | 0.7639 | 0.8473 | 0.8005 | 0.7443 | 0.6637 | 0.000000 |
| **ViT Group** | 0.150 | 0.7634 | 0.8477 | 0.8011 | 0.7442 | 0.6607 | **-0.000494** |
| **ViT Group** | 0.200 | 0.7628 | 0.8482 | 0.8018 | 0.7436 | 0.6575 | **-0.001149** |
| **S3 Isolated** | 0.000 | 0.7635 | 0.8471 | 0.8004 | 0.7440 | 0.6624 | **-0.000450** |
| **S3 Isolated** | 0.050 | 0.7637 | 0.8472 | 0.8004 | 0.7442 | 0.6631 | **-0.000203** |
| **S3 Isolated** | 0.100 | 0.7639 | 0.8473 | 0.8005 | 0.7443 | 0.6637 | 0.000000 |
| **S3 Isolated** | 0.200 | 0.7643 | 0.8473 | 0.8006 | 0.7446 | 0.6648 | **+0.000402** |
| **S3 Isolated** | 0.300 | 0.7645 | 0.8472 | 0.8006 | 0.7448 | 0.6654 | **+0.000578** |
| **Shallow CNN** | 0.000 | 0.7638 | 0.8478 | 0.8008 | 0.7448 | 0.6619 | **-0.000112** |
| **Shallow CNN** | 0.050 | 0.7640 | 0.8475 | 0.8010 | 0.7447 | 0.6628 | **+0.000065** |
| **Shallow CNN** | 0.100 | 0.7639 | 0.8473 | 0.8005 | 0.7443 | 0.6637 | 0.000000 |
| **Shallow CNN** | 0.200 | 0.7634 | 0.8469 | 0.7983 | 0.7431 | 0.6652 | **-0.000550** |
| **Shallow CNN** | 0.300 | 0.7623 | 0.8462 | 0.7954 | 0.7415 | 0.6662 | **-0.001622** |

---

## 5. Diễn Giải Khoa Học Theo Framework

$$\text{Representation evidence} \neq \text{Routing utility} \neq \text{Expert utility} \neq \text{Segmentation utility}$$

1. **Fusion-scale Sensitivity**:
   - Khảo sát độ nhạy của Dice đối với biến thiên $\alpha$ từ $0.0$ đến $0.3$ trên từng nhóm tầng.
2. **Expert Utility Evidence**:
   - Đánh giá xem expert branch có thực sự cung cấp utility cải thiện segmentation hay chỉ hoạt động như một bộ đệm nhiễu tuyến tính (additive perturbation).
3. **Segmentation Relevance**:
   - Định lượng mức độ thay đổi thực tế trên các hình thái vết nứt phức tạp (Q4 ultra-thin vs Q1 thick).

---
*Báo cáo tự động sinh bởi `scripts/diagnostics/evaluate_residual_fusion.py`.*
