# Phase 7 — S2-Gate Block 2 Extension Roadmap

## Status
- **Current step:** Step 2 Completed (Critical Verification 100% PASS with Real Validation Batch) $\to$ Ready for Step 3 (Phase 7A Driver Execution on Tesla T4)
- **Overall status:** READY FOR COLAB T4 EXECUTION (Local Code Audit & Unit Tests Complete)
- **Last updated:** 2026-10-07 03:10:00 (Local Time)

---

## Prerequisites (Phase 0 Audit Findings)

1. **Best Checkpoint có S2-Gate Block 1**:
   - `results/phase6_combination/phase6_comb_a1_s2g_end_to_end/best_model_b2_global.pth`
     * Val Dice: **0.7676** (Epoch 16), Loss: 1.6372, Precision: 0.7408, Recall: 0.8384.
     * Contains native `decoder.s2_gate` (Conv3x3) integrated directly into `model_state_dict`.
   - `results/phase6_combination/phase6_comb_a1_s2g_stage2/`:
     * Global peak Val Dice: 🏆 **0.7702** (Epoch 14), Loss đáy: **1.3471**, Precision: 0.7510, Recall: 0.8301, Thin Crack Dice: 0.4260.
     * Inherited from Stage 1 checkpoint with strict RNG restoration.

2. **Implementation chính xác của S2-Gate Block 1**:
   - File: [`sage/networks/s2_gate.py`](file:///d:/truong/SpecialSubjectTTNT/SAGE_LITE/sage/networks/s2_gate.py)
   - Kiến trúc: `S2GateModule(s2_channels=192, skip_channels=96, mid_channels=32, kernel_size=3)`
   - Cấu trúc mạng:
     ```python
     Conv2d(288, 32, kernel_size=3, padding=1, bias=True)
     -> BatchNorm2d(32)
     -> ReLU(inplace=True)
     -> Conv2d(32, 1, kernel_size=1, bias=True)
     ```
   - Khởi tạo cận Identity ($t=0$): Conv cuối cùng có trọng số zero, bias $= +5.0 \implies \sigma(5.0) \approx 0.9933$.
   - Vị trí chèn: [`sage/networks/decoder_block.py`](file:///d:/truong/SpecialSubjectTTNT/SAGE_LITE/sage/networks/decoder_block.py#L398) tại `i=1` (Decoder Block 1), lấy ngữ cảnh $S2$ (`x_dec`, 192 ch, 28x28) nội suy lên 56x56 để điều tiết Skip $S1$ (96 ch, 56x56).

3. **Vị trí và độ phân giải của Block 2**:
   - Vị trí: `i=2` trong `self.decoder_blocks` của `UNetDecoder`.
   - Ngữ cảnh đầu vào (`x_dec` từ Block 1): **96 channels, độ phân giải 56x56**.
   - Skip connection tương ứng (`reversed_skips[2]` = Stage 0 skip): **48 channels, độ phân giải 112x112**.
   - Cấu hình cổng Block 2:
     * Nội suy ngữ cảnh $S1$ từ 56x56 lên 112x112 (`mode="bilinear", align_corners=False`).
     * Ghép kênh: $96 + 48 = 144$ channels đầu vào.
     * Mạng nén: $144 \to 32 \to 1$.
     * Kích thước không gian của cổng $\alpha$: **112x112**.
     * Skip sau điều biến: `skip_gated = gate_alpha * skip_s0` (48 channels, 112x112).
     * Tham số: 4,737 ($k=1$) hoặc 41,601 ($k=3$).

4. **Training/Evaluation Scripts hiện tại**:
   - Script huấn luyện/đánh giá chuyên biệt Phase 7: [`scripts/diagnostics/train_eval_phase7_s2_gate_block2.py`](file:///d:/truong/SpecialSubjectTTNT/SAGE_LITE/scripts/diagnostics/train_eval_phase7_s2_gate_block2.py)
   - Driver tự động hóa toàn chuỗi Phase 7: [`scripts/diagnostics/run_phase7_block2_pipeline.py`](file:///d:/truong/SpecialSubjectTTNT/SAGE_LITE/scripts/diagnostics/run_phase7_block2_pipeline.py)
   - Bộ kiểm thử tiền trạm: [`scripts/tests/test_phase7_critical_verification.py`](file:///d:/truong/SpecialSubjectTTNT/SAGE_LITE/scripts/tests/test_phase7_critical_verification.py)

5. **Protocol, Metrics và Tiêu chí Đánh giá Block 1 / Phase 6**:
   - Giao thức: Setting A Canonical Validation ($N=348$ mẫu, tiling không chồng lấp $448 \times 448$, ngưỡng phân loại $\tau = 0.5$, FP32 strict).
   - Chỉ số chính: Validation Dice.
   - Chỉ số hình thái và biên: Precision, Recall, IoU, Boundary IoU ($d=2$), HD95, BF1, Thin Crack Dice ($t \le 3$ px), clDice (skeleton).
   - Chỉ số topology sự kiện:
     * Tổng số false bridges ($N=118$ trên 110 ảnh val).
     * Phân tầng 43 wider-gap bridge events ($D_{\text{gap}} > 5$ px).
     * Theo dõi 11 worsened cases từ v1.
     * Số lượng break events (fragmented GT components) và spurious islands.

6. **Seed Convention**:
   - Đơn run / diagnostic probe: `seed = 42`.
   - Kiểm định lặp lại (Repeatability): `seed = 42` và `seed = 43`.

7. **Checkpoint Naming / Output Convention**:
   - Checkpoint: `best_model_*.pth`, `last_model_*.pth`.
   - Gate weights riêng biệt: `s2g_block2_lr_{lr}_k{k}_weights.pth`.
   - Output directory: `results/diagnostics/phase7_s2_gate_block2/`.

---

## 1. Hai Diagnostic QC Đang Treo (Đã Thẩm Định Hoàn Tất)

### 1.1 Resolution-Ceiling Diagnostic
- **File bằng chứng:** [`results/diagnostics/phase6_bridge_localization/README.md`](file:///d:/truong/SpecialSubjectTTNT/results/diagnostics/phase6_bridge_localization/README.md)
- **Kết quả đo đạc trên lưới 112×112 của Decoder (Stride 4):**
  * Trong **$67.3\%$ ($74/110$)** các ca bị false bridge ở Baseline, các thành phần liên thông GT **vẫn được phân tách hoàn toàn** trên lưới 112×112.
  * Chỉ có **$25.5\%$ ($28/110$)** ca bị gộp (merge) do độ phân giải lưới.
- **Ý nghĩa & Rủi ro đối với Phase 7 (Block 2):**
  * Độ phân giải 112×112 **KHÔNG PHẢI** là trần nghẽn hình học chính gây ra hiện tượng false bridge. Lưới 112×112 đủ khả năng phân tách trong hơn 2/3 trường hợp, nhưng mô hình vẫn dự đoán pixel cầu nứt với xác suất rất cao ($P > 0.86$).
  * Điều này chứng minh việc can thiệp tại Decoder Block 2 (lưới 112×112) là **có cơ sở hình học hợp lệ** (không bị nghẽn lượng tử hóa), nhưng cần lưu ý Skip S0 không phải là nguyên nhân nhân quả chính như Skip S1 (S1 gây ra 62.8% ca bridge theo nghiên cứu provenance).

### 1.2 Annotation-Geometry QC
- **File bằng chứng:** [`results/diagnostics/phase6_annotation_qc/DIAGNOSTIC_D_ANNOTATION_QC_REPORT.md`](file:///d:/truong/SpecialSubjectTTNT/results/diagnostics/phase6_annotation_qc/DIAGNOSTIC_D_ANNOTATION_QC_REPORT.md)
- **Kết quả phân tầng khoảng cách $D_{\text{gap}}$ (N = 118 events):**
  * **$52.54\%$ (62 / 118 events)** có $D_{\text{gap}} \le 3.0$ px: Ảnh RGB thực tế vết nứt liên tục, nhưng nhãn người ngắt rời 1–2 px do nét cọ vẽ. Mô hình nhận diện vết nứt thật nhưng bị metric phạt là "false bridge".
  * **$11.02\%$ (13 / 118 events)** có $3.0 < D_{\text{gap}} \le 5.0$ px.
  * **$36.44\%$ (43 / 118 events)** có $D_{\text{gap}} > 5.0$ px: Đây mới là **false bridge nhân quả cấu trúc thật sự** của mạng.
- **Ý nghĩa đối với Phase 7:**
  * Không được đánh giá hiệu quả của S2-Gate Block 2 một cách cảm tính chỉ dựa trên việc giảm 118 ca bridge. Phải giám sát chặt chẽ cohort **43 wider-gap events** và **Val Dice / Thin Crack Recall** để tránh hiện tượng dập tắt nhầm vết nứt thật.

---

## 2. Critical Verification Trước Phase 7A (100% PASS Bằng Code Thật)

Đã thẩm định thông qua test suite độc lập [`scripts/tests/test_phase7_critical_verification.py`](file:///d:/truong/SpecialSubjectTTNT/SAGE_LITE/scripts/tests/test_phase7_critical_verification.py) trên **batch thật gồm 4 ảnh validation từ Crack500** (`torch.Size([4, 3, 448, 448])`):

### 2.1 Optimizer Parameter Group
- **Mã nguồn thẩm định:** `test_2_1_optimizer_parameter_group`
- **Cấu trúc Optimizer:**
  ```python
  optimizer = torch.optim.AdamW(s2_gate_b2.parameters(), lr=1e-3, weight_decay=1e-2)
  ```
- **Bằng chứng thực nghiệm:**
  * Số lượng tensors trong Optimizer: đúng **6 tensors** (trọng số Conv1, bias Conv1, gamma BN, beta BN, trọng số Conv2, bias Conv2).
  * Tổng số phần tử tham số: **41,601 elements** ($k=3$).
  * Xác nhận **0% tham số** từ backbone, router, decoder body, hay S2-Gate Block 1 lọt vào optimizer.

### 2.2 Freeze + BatchNorm / Train-Eval Mode Protocol
- **Mã nguồn thẩm định:** `test_2_2_freeze_and_eval_train_mode_protocol`
- **Trình tự bắt buộc:**
  ```python
  model.eval()               # 1. model.eval() đệ quy xuống toàn bộ submodules
  s2_gate_b2.train()          # 2. Chỉ kích hoạt train mode cho S2-Gate Block 2
  ```
- **Bằng chứng thực nghiệm:**
  * Trạng thái module: `s2_gate_b2.training == True`, `model.backbone.training == False`, `model.decoder.decoder_blocks[0].training == False`, `model.decoder.s2_gate.training == False`.
  * Trạng thái tham số: Toàn bộ mô hình có `requires_grad == False`, duy nhất 6 tensors của `s2_gate_b2` có `requires_grad == True`.
  * Tính bất biến của BatchNorm bị đóng băng: `running_mean` và `running_var` của toàn bộ các lớp BN trong backbone và decoder có chênh lệch $\Delta = 0.000000$ sau bước forward + backward + optimizer step.
  * Cách ly Gradient: Toàn bộ các lớp ngoài Block 2 gate có `p.grad is None`.

### 2.3 Numeric Identity Check (1×1 $\to$ 3×3)
- **Mã nguồn thẩm định:** `test_2_3_numeric_identity_check_1x1_vs_3x3`
- **Phương pháp:** Chuyển đổi trọng số 1x1 sang 3x3 bằng `s2_gate_b2_3x3.warm_start_from_v1(...)` (nhúng kernel 1x1 vào tâm (1, 1), điền 0 vào 8 vị trí xung quanh).
- **Kết quả đo đạc trên Batch Thật (Crack500 Val):**
  * **Chế độ Eval Mode (`model.eval()`):**
    - `max_abs_diff`: **$2.384 \times 10^{-6}$** (< 1e-4)
    - `mean_abs_diff`: **$8.257 \times 10^{-8}$**
    - `torch.allclose(atol=1e-5)`: **`True`**
  * **Chế độ Train Protocol (`model.eval()` + `gate.train()`):**
    - `max_abs_diff`: **$2.861 \times 10^{-6}$** (< 1e-4)
    - `mean_abs_diff`: **$1.208 \times 10^{-7}$**
    - `torch.allclose(atol=1e-5)`: **`True`**
  * **Giải thích toán học:** Do các trọng số xung quanh tâm bằng 0 tuyệt đối, phép tích chập $3 \times 3$ với padding=1 tại mọi tọa độ không gian (kể cả biên) có kết quả đồng nhất bitwise với tích chập $1 \times 1$.

---

## 7A — LR Probe (Kế Hoạch & Ma Trận Thực Nghiệm)

Checkpoint xuất phát: `results/phase6_combination/phase6_comb_a1_s2g_end_to_end/best_model_b2_global.pth`  
Thiết lập: 8 epochs, AdamW (wd=1e-2), CosineAnnealingLR, `seed=42`, Block 1 FROZEN, chỉ Block 2 Trainable.

| LR | Seed | Kernel | Val Dice | Precision | Recall | Thin Dice | Bridges (118) | Bridges (43) | Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `1e-4` | 42 | 3 | *Chờ chạy* | — | — | — | — | — | QUEUED (Colab T4) |
| `3e-4` | 42 | 3 | *Chờ chạy* | — | — | — | — | — | QUEUED (Colab T4) |
| `1e-3` | 42 | 3 | *Chờ chạy* | — | — | — | — | — | QUEUED (Colab T4) |
| `3e-3` | 42 | 3 | *Chờ chạy* | — | — | — | — | — | QUEUED (Colab T4) |

- **Selected LR:** *Sẽ khóa sau khi có kết quả chạy trên Colab T4*
- **Tiêu chí lựa chọn:** Tương tự Block 1 (tối đa hóa Val Dice, kiểm soát sụt giảm Thin Crack Recall, ưu tiên giảm false bridge trên cohort 43 wider-gap).

---

## 7A' — Kernel Probe (1×1 vs 3×3)

Tại LR=1e-3 (theo yêu cầu trực tiếp của người dùng triển khai sớm 7A'), huấn luyện độc lập từ đầu 8 epochs (`seed=42`, batch size 14, AdamW wd=1e-2, CosineAnnealingLR):

| Kernel | LR | Seed | Parameters | Val Dice | Precision | Recall | Thin Dice | Bridges (43) | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `1x1` | 1e-3 | 42 | 4,737 | *Đang chạy* | — | — | — | — | **IN PROGRESS (Background Task)** |
| `3x3` | 1e-3 | 42 | 41,601 | *Xếp hàng* | — | — | — | — | QUEUED |

- **Selected Kernel:** *Sẽ cập nhật ngay khi 2 run hoàn tất.*

---

## 7B — Full B1+B2 Repeatability Runs

Sau khi khóa LR* và Kernel* cho Block 2:

| Run | Seed | Block 1 State | Block 2 State | Val Dice | IoU | HD95 | Checkpoint |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Run 1 | 42 | Conv3x3 (Frozen/Trained) | Locked Kernel* | *Chờ chạy* | — | — | `best_model_p7b_s42.pth` |
| Run 2 | 43 | Conv3x3 (Frozen/Trained) | Locked Kernel* | *Chờ chạy* | — | — | `best_model_p7b_s43.pth` |
| **Mean** | — | — | — | — | — | — | — |

*Lưu ý validity:* Báo cáo trung bình của $N=2$ là **repeatability check**, tuyệt đối không diễn giải như một ước lượng phương sai thống kê hình thức (formal variance estimate).

---

## 7C — Fusion Mechanism Ablation

Tại cấu hình tốt nhất của 7B:
- **Residual Fusion** vs **Adaptive Fusion** (2 runs / config, `seed=42, 43`).

---

## 7D — Top-K Capacity Sweep

Tại cấu hình tốt nhất của 7C:
- **Top-K = 3** vs **Top-K = 4** (2 runs / config, `seed=42, 43`).

---

## Final Locked Configuration
- *Sẽ cập nhật sau khi hoàn tất 7A–7D.*

---

## Sealed Test Set Evaluation
- **Trạng thái:** NIÊM PHONG TUYỆT ĐỐI (Sealed, untouched).
- Chỉ thực hiện **DUY NHẤT 1 LẦN** sau khi khóa toàn bộ cấu hình cuối cùng của Phase 7.

---

## Audit Các Nguy Cơ Tính Hợp Lệ (Threats to Validity)

1. **Validation Selection Bias**:
   - Ghi nhận rõ: Phase 7 trải qua nhiều lần đánh giá trên tập Validation (LR probe, Kernel probe, Fusion, Top-K). Toàn bộ quá trình chọn lọc chịu ảnh hưởng tích lũy của validation exposure.
2. **Cỡ mẫu lặp lại $N=2$**:
   - Chỉ được coi là **repeatability check** để phát hiện seed instability, không phải kiểm định thống kê mẫu lớn.
3. **Cure-rate Definition Caveat**:
   - Định nghĩa trong code: `is_cured = (not is_merged) or (z_bridge < 0)`.
   - Lưu ý tính hợp lệ: Điều kiện `OR` có thể tính một ca là "cured" nếu độ nổi bật logit bị dập xuống âm (`z_bridge < 0`), dù trên nhị phân component vẫn có thể còn dính nhẹ (`is_merged=True`). Phải báo cáo song song cả hai metric để minh bạch.
4. **Cohort 43 Wider-Gap Events**:
   - Kích thước mẫu $N=43$ là mẫu chẩn đoán định tính/bán định lượng; không khái quát hóa quá mức.
5. **Cohort 11 Worsened Cases**:
   - Theo dõi sự phục hồi hoặc suy giảm riêng biệt để kiểm tra xem việc thêm Block 2 có tiếp tục chữa lành hay làm phát sinh điểm mù mới.

---

## Discrepancies & Quyết Định Kỹ Thuật

1. **Discrepancy #1: `load_model_from_checkpoint` thiếu tham số S2-Gate**:
   - *Phát hiện:* Hàm `load_model_from_checkpoint` trong [`tools/run_phase6_c_topology_diagnostic.py`](file:///d:/truong/SpecialSubjectTTNT/SAGE_LITE/tools/run_phase6_c_topology_diagnostic.py) được viết từ Phase 6-C (trước khi S2-Gate ra đời), dẫn đến lỗi `Unexpected key(s)` khi load checkpoint có `decoder.s2_gate`.
   - *Xử lý chuẩn tắc:* Đã bổ sung việc đọc và truyền `use_s2_gate`, `s2_gate_kernel_size`, `use_s2_gate_block2`, `s2_gate_block2_kernel_size` vào `create_b2_unet`.
2. **Quy tắc phần cứng (Hardware Invariant)**:
   - GPU máy local là GTX 1650 (4GB VRAM). Theo quy tắc bất biến trong skill `sage-lite-colab`, tác vụ đo đạc tải chính thức và huấn luyện OOM probe / official training **BẮT BUỘC thực hiện trên Google Colab Tesla T4 (16GB)**. Code đã được kiểm thử tính đúng đắn (unit test, numeric identity, parameter audit) 100% trên máy local.

---

## Files Đã Tạo / Cập Nhật

1. [`sage/networks/decoder_block.py`](file:///d:/truong/SpecialSubjectTTNT/SAGE_LITE/sage/networks/decoder_block.py): Bổ sung `use_s2_gate_block2`, `s2_gate_block2_kernel_size` và cơ chế điều biến tại `i=2`.
2. [`sage/networks/b2_unet.py`](file:///d:/truong/SpecialSubjectTTNT/SAGE_LITE/sage/networks/b2_unet.py): Cập nhật `B2ConvNeXtViTUNet` và `create_b2_unet` truyền tham số Block 2 gate.
3. [`tools/run_phase6_c_topology_diagnostic.py`](file:///d:/truong/SpecialSubjectTTNT/SAGE_LITE/tools/run_phase6_c_topology_diagnostic.py): Khắc phục lỗi nạp checkpoint có chứa S2-Gate.
4. [`scripts/tests/test_phase7_critical_verification.py`](file:///d:/truong/SpecialSubjectTTNT/SAGE_LITE/scripts/tests/test_phase7_critical_verification.py): Test suite kiểm tra 3 điều kiện tiên quyết (100% PASS).
5. [`scripts/diagnostics/train_eval_phase7_s2_gate_block2.py`](file:///d:/truong/SpecialSubjectTTNT/SAGE_LITE/scripts/diagnostics/train_eval_phase7_s2_gate_block2.py): Script thực thi huấn luyện và đánh giá Setting A cho Phase 7.
6. [`scripts/diagnostics/run_phase7_block2_pipeline.py`](file:///d:/truong/SpecialSubjectTTNT/SAGE_LITE/scripts/diagnostics/run_phase7_block2_pipeline.py): Master driver tự động hóa Phase 7 trên Colab T4.
7. [`docs/PHASE7_S2GATE_BLOCK2_ROADMAP.md`](file:///d:/truong/SpecialSubjectTTNT/docs/PHASE7_S2GATE_BLOCK2_ROADMAP.md): File roadmap và nhật ký theo dõi tiến trình Phase 7.

---

## Commands Executed

```bash
# 1. Chạy test suite kiểm định điều kiện tiên quyết Phase 7 (PASS)
python scripts/tests/test_phase7_critical_verification.py

# 2. Kiểm tra tính hồi quy của test suites trước đó (PASS)
python scripts/tests/test_u1_s2g_v2_invariants.py
python scripts/tests/test_u1_s2g_invariants.py

# 3. Kiểm tra CLI help của runner Phase 7 (PASS)
python scripts/diagnostics/train_eval_phase7_s2_gate_block2.py --help
```
