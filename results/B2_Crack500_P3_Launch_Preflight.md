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
