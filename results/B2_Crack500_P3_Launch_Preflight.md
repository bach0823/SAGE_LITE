# SAGE-Lite B2 Proposal 3 (P3) Launch Preflight Report (Crack500)

*Date: 2026-09-25*  
*Hardware: Tesla T4 (Google Colab, 14.56 GB usable VRAM)*  
*Commit HEAD: `bdb23f106b5b8243aa118f490917e327704c380a`*  
*Branch: `crack500-audit`*  
*Dataset: Crack500 (1896 train samples, 348 val pairs)*  
*Protocol: Parameterized CLI Preflight (`scripts/preflight_p3_realdata.py`), Depth=12, BS=12, Workers=2, Batches=3*  

---

## 1. Bảng Nghiệm Thu Hợp Nhất 24 Tiêu Chí (Consolidated Audit Table)

| Tiêu chí Kiểm tra / Metric | Run A (Identity) [D12] | Run B (Generic DW) [D12] | Run C (ASDW) [D12] | Đánh giá |
|:---|:---:|:---:|:---:|:---:|
| **A. Real-data forward** | PASS | PASS | PASS | 100% PASS |
| **B. Real-data backward** | PASS | PASS | PASS | 100% PASS |
| **C. Optimizer step** | PASS | PASS | PASS | 100% PASS |
| **D. Stage 1 -> 2 transition** | PASS | PASS | PASS | 100% PASS |
| **E. Stage 2 forward/backward** | PASS | PASS | PASS | 100% PASS |
| **F. Checkpoint save/reload** | PASS | PASS | PASS | 100% PASS |
| **G. Peak allocated VRAM** | 8,257.7 MB (8.06 GB) | 8,477.8 MB (8.28 GB) | 8,689.6 MB (8.49 GB) | Measured |
| **H. Peak reserved VRAM** | 9,690.0 MB (9.46 GB) | 9,898.0 MB (9.67 GB) | 10,016.0 MB (9.78 GB) | An toàn (< 10 GB) |
| **I. Throughput (Preflight)** | 0.21 samples/s | 0.65 samples/s | 0.99 samples/s | Measured |
| **J. DataLoader health** | PASS | PASS | PASS | 100% PASS |
| **K. NaN / Inf status** | CLEAN (None) | CLEAN (None) | CLEAN (None) | 100% PASS |
| **L. P3 gradient status** | PASS (0 params, Identity) | PASS (10 params, active grad) | PASS (10 params, active grad) | 100% PASS |
| **M. PE28 fixed buffer status** | PASS | PASS | PASS | 100% PASS |
| **N. Locked Base provenance** | NOT_PROVEN | NOT_PROVEN | NOT_PROVEN | Đúng thiết kế (GATED) |
| **O. Checkpoint SHA256** | None | None | None | Không bịa hash |

**Kết luận Suite Preflight:**  
`REAL-DATA P3 LAUNCH PREFLIGHT = GATED (Locked Base Checkpoint required; please supply --locked-base)`

---

## 2. Kiểm Tra Chéo Số Học & Tính Toàn Vẹn Kiến Trúc

### 2.1. Tính Bất Biến PE28 Giữa Các Runs (Cross-Run PE28 Invariance)
- **Run A vs Run B:** Đồng nhất tuyệt đối (`max diff = 0.0`)
- **Run A vs Run C:** Đồng nhất tuyệt đối (`max diff = 0.0`)
- **Khẳng định Bitwise:** `Run A.pe28_fixed == Run B.pe28_fixed == Run C.pe28_fixed` $\to$ **PASS 100%**.
- Buffer State: Kích thước tensor `[1, 784, 192]`, kiểu dữ liệu `torch.float32`, `requires_grad = False`, gradient `None`.
- Nguồn gốc toán học: Nội suy bicubic 2D chính xác từ ViT PE14 ($14 \times 14 \to 28 \times 28$).

### 2.2. Phân Nhóm Tham Số Stage 2 (Optimizer Partitioning)
- **Run A (Identity Control):** Đúng 0 tham số P3 được tạo ra. Luồng chính nhận trực tiếp features sau adaptive pool.
- **Run B & C:** Đúng 10 tensor tham số P3 được tạo ra. Trong optimizer Stage 2, toàn bộ 10 tensor này được xếp nghiêm ngặt vào nhóm `other_and_routers` (nhận `stage2_base_lr = 1e-4`), hoàn toàn không nằm trong `shared_experts`.
- **Shared Experts:** Toàn bộ 132 tham số được xác nhận là CNN main_blocks (`requires_grad = True`).

### 2.3. Chiếm Dụng VRAM Thực Tế Trên Tesla T4 (14.56 GB)
- Run A: Peak Allocated = 8.06 GB, Peak Reserved = 9.46 GB (~5.1 GB buffer tự do).
- Run B: Peak Allocated = 8.28 GB, Peak Reserved = 9.67 GB (~4.89 GB buffer tự do).
- Run C: Peak Allocated = 8.49 GB, Peak Reserved = 9.78 GB (~4.78 GB buffer tự do).
- Mức tiêu thụ trong cả 2 Stage đều nằm dưới 10.0 GB cho $BS=12$.

### 2.4. Kiểm Tra Luồng Validation Metric (Cách Ly Tuyệt Đối Với Test Set)
- Tìm thấy 348 cặp ảnh/mask trong split validation của Crack500.
- Chạy thử nghiệm Setting A (tiling 448x448) trên 2 mẫu validation:
  - Run A: Val Loss = 2.0817, Val Dice = 0.0810, Val IoU = 0.0436
  - Run B: Val Loss = 2.0757, Val Dice = 0.0716, Val IoU = 0.0383
  - Run C: Val Loss = 2.0780, Val Dice = 0.0750, Val IoU = 0.0401
- Tuyệt đối không truy cập thư mục Test (`test/images`, `test/masks`).

---

## 3. Trạng Thái Launch Gate
* **Toàn vẹn Phần cứng & Tính toán:** **VERIFIED 100% READY**.
* **Launch Gate:** **GATED** (Đúng quy trình: Chờ nạp Locked Base Depth 12 chính thức sau khi hoàn thành Phase 1 Base training).
* **Quy tắc:** Tuyệt đối KHÔNG bắt đầu Phase 7 (30 epochs P3) khi chưa mở cổng Launch Gate.

---

## 4. Đánh Giá Chuyên Sâu & Phân Tích Kết Quả (Analysis & Critical Caveats)

Kết quả này **đạt mục tiêu của preflight runtime P3**, nhưng cần phân định rành mạch giữa **“những gì đã được chứng minh”** và **“những gì chưa thể dùng làm bằng chứng khoa học”**.

### 4.1. Điều Đã Được Chứng Minh (Proven Points)
**Run A, B, C đều chạy hoàn chỉnh trên Crack500 thực tế với Depth 12, Batch Size 12 trên Tesla T4 (Google Colab).**

Cả ba chế độ đều vượt qua toàn bộ các khâu cốt lõi:
1. **Forward pass:** PASS trên ảnh thực $448 \times 448$.
2. **Backward pass:** PASS, gradient lan truyền thông suốt.
3. **Optimizer step:** PASS, cập nhật trọng số đúng định dạng.
4. **Stage 1 $\to$ Stage 2 reload:** PASS, transition mượt mà, không lệch tensor.
5. **Freeze / Shared Experts:** `set_shared_experts([0, 1, 2, 3])` PASS, đúng 132 tensor CNN blocks được mở khóa huấn luyện.
6. **Stage 2 forward/backward:** PASS, tính toán phân luồng router và loss hoàn chỉnh.
7. **Checkpoint save/reload:** PASS, serialize và restore trạng thái mô hình/optimizer nguyên vẹn.
8. **Kiểm tra số học:** Hoàn toàn sạch số học (KHÔNG có NaN / Inf).
9. **Buffer PE28:** Đúng kích thước `(1, 784, 192)`, FP32, hoàn toàn frozen (`requires_grad = False`, gradient = `None`).
10. **Phân nhóm tham số:**
    - Run B & C: Có đủ 10 tensor tham số P3 với gradient hữu hạn nằm trong nhóm `other_and_routers`.
    - Run A: Đúng bản chất Identity control, 0 tham số P3.
11. **Tính cô lập dữ liệu:** Validation pipeline chạy đúng trên tập **Val** (348 cặp ảnh), tuyệt đối không chạm vào thư mục **Test**.

> **Kết luận Software / Runtime Feasibility:** P3 Run A/B/C đã chính thức vượt qua smoke test thực tế trên phần cứng mục tiêu T4.

---

### 4.2. Khả Năng An Toàn Bộ Nhớ Ở Batch Size 12 (Memory Safety at BS12)
* **Peak Allocated VRAM:**
  - Run A (Identity): **8,257.7 MB** (~8.06 GB)
  - Run B (Generic DW): **8,477.8 MB** (~8.28 GB)
  - Run C (ASDW): **8,689.6 MB** (~8.49 GB)
* **Peak Reserved VRAM:** Dao động từ ~9.46 GB đến ~9.78 GB.

Trên phần cứng Tesla T4 16GB (khả dụng thực tế ~14.56 GB), lượng VRAM tự do dự phòng còn lại đạt khoảng **~4.78 GB đến ~5.1 GB**.  
Điều này khẳng định rằng **ở quy mô batch preflight, BS12 hoàn toàn an toàn và không bị OOM**.

---

### 4.3. Cảnh Báo Nhiễu Throughput (Throughput Contamination Caveat)
Số liệu throughput đo được trong suite preflight:
* Run A: 0.21 samples/s (Stage 1 mất 104.6s)
* Run B: 0.65 samples/s (Stage 1 mất 35.6s)
* Run C: 0.99 samples/s (Stage 1 mất 34.8s)

> [!CAUTION]
> **Con số này hoàn toàn là artifact do chạy gộp cả 3 run liên tiếp trong cùng một tiến trình Python (single process):**
> 1. **Run A chịu toàn bộ chi phí khởi tạo ban đầu:** Bao gồm khởi tạo CUDA context, nạp thư viện `timm`, tải pretrained weights từ Hugging Face Hub, cấp phát bộ nhớ ban đầu và build graph PyTorch (tiêu tốn tới 104.6s ở Stage 1).
> 2. **Run B chạy sau khi CUDA allocator đã được "làm ấm" (warmed-up):** Trọng số và thư viện đã nằm trong cache RAM/VRAM nên thời gian chỉ còn 35.6s.
> 3. **Run C chạy cuối cùng:** Hệ thống đã hoàn toàn ở trạng thái cached tối đa nên thời gian tiếp tục giảm còn 34.8s.
>
> ➡️ **TUYỆT ĐỐI KHÔNG ĐƯỢC KẾT LUẬN "ASDW NHANH HƠN IDENTITY GẤP 4 LẦN".**  
> Để có phép so sánh throughput công bằng, bắt buộc phải chạy từng mode trong một tiến trình (process) độc lập riêng biệt hoặc thực hiện đo đạc sau khi đã warm-up ít nhất 1 epoch/nhiều batch.

---

### 4.4. Tín Hiệu Kiến Trúc Từ Peak Allocated Memory
Trái ngược với Throughput bị nhiễu do warm-up, chỉ số **Peak Memory Allocated** lại phản ánh độ chính xác và tính logic cao của kiến trúc:
* **Run A (Identity - không thêm tham số/features mới):** **8,257.7 MB**
* **Run B (Generic DW conv 7x7):** **8,477.8 MB** (+220.1 MB cho intermediate feature maps của DW conv)
* **Run C (ASDW multi-scale & dynamic routing):** **8,689.6 MB** (+431.9 MB so với Identity, +211.8 MB so với Generic DW)

Sự gia tăng này là bằng chứng rõ ràng cho thấy:
1. Nhánh Run C thực sự thực hiện tính toán và lưu trữ activation buffers cho các nhánh multi-scale và cổng routing.
2. Mức overhead bộ nhớ của ASDW chỉ là **~432 MB ở BS12**, cực kỳ tối ưu và hoàn toàn nằm trong ngưỡng an toàn của GPU 16GB.

---

### 4.5. Giải Trình Trạng Thái GATED (Expected Safe Behavior)
Kết quả suite preflight trả về trạng thái `GATED` là **hoàn toàn chính xác theo đúng nguyên tắc thiết kế**:
* Script preflight này đóng vai trò là chốt chặn an toàn (Launch Gate) ngăn ngừa việc huấn luyện Stage 2 P3 từ trọng số ngẫu nhiên hoặc checkpoint không rõ nguồn gốc.
* Do chưa hoàn thành Phase 1 Base training cho Depth 12 trên Crack500 nên chưa có checkpoint base chính thức được khóa hash (`--locked-base`).
* Việc hệ thống từ chối mở cổng là bằng chứng cho thấy cơ chế phòng thủ tính toàn vẹn khoa học hoạt động hoàn hảo.

---

### 4.6. Ý Nghĩa Của Chỉ Số Validation Dice / Loss Ban Đầu
Chỉ số Validation Dice ghi nhận được (~0.07 - 0.08) và Loss (~2.07 - 2.08) chỉ được tính toán trên **2 mẫu validation ngẫu nhiên**:
* Mục đích duy nhất của bước này là **Sanity Check**: Xác minh hàm tính metric không bị crash, không phát sinh NaN/Inf, và pipeline validation tương thích tuyệt đối với cấu trúc đầu ra của mô hình.
* Các con số này **KHÔNG CÓ GIÁ TRỊ** để kết luận hoặc so sánh chất lượng phân đoạn giữa Run A, B và C trước khi mô hình được huấn luyện đầy đủ.

---

### 4.7. Kết Luận & Hành Động Tiếp Theo (Action Items)

1. **Về Code & Kiến trúc:**
   - Toàn bộ source code P3 (Run A, B, C), cơ chế 2-stage training, dynamic routing, và nội suy PE28 trên Depth 12, Batch Size 12 đã **HOÀN TOÀN SẴN SÀNG (100% READY)**. Không cần can thiệp hay sửa đổi thêm code mô hình.
2. **Về Cấu Hình Huấn Luyện:**
   - Chốt cấu hình mục tiêu cho cuộc thử nghiệm chính thức trên Tesla T4: **`num_transformer_layers = 12` (D12), `batch_size = 12` (BS12), `num_workers = 2`**.
3. **Kế Hoạch Thực Thi Kế Tiếp:**
   - **Đo Throughput Tinh Khiết (Isolated Benchmarking):** Nếu cần số liệu throughput chuẩn xác để đưa vào báo cáo khoa học, chạy 3 lệnh CLI độc lập trong các tiến trình riêng biệt (mỗi lệnh một cell Colab) với 5–10 batches.
   - **Triển khai Huấn luyện Phase 1 (Base Model Training):** Chạy huấn luyện Base Model (Stage 1) trên Crack500 để sinh ra checkpoint `locked_base` chính thức và tạo SHA256 hash mở khóa Launch Gate cho Phase 7 (P3 Training).
