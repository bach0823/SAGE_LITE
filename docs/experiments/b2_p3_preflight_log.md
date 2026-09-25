# SAGE-Lite B2 Proposal 3 (P3) Launch Preflight Log

## 1. Executive Summary & Verification Context
- **Date:** September 25, 2026
- **Hardware Target:** Google Colab Tesla T4 (14.56 GB usable VRAM)
- **Commit HEAD:** `bdb23f106b5b8243aa118f490917e327704c380a`
- **Branch:** `crack500-audit`
- **Dataset:** Real Crack500 (`train`: 1,896 samples, `val`: 348 pairs) at `/content/dataset/Crack500`
- **Configuration & Overrides:**
  - Architecture: Full SAGE-Lite B2 UNet (Depth = 12, ViT-Tiny blocks = 12, Expert Pool = 16, Routers = 16)
  - Preprocessing: Canonical RandomCrop (448×448) with Smart Filter (`fg_pixels >= 20`)
  - Runtime Parameters: `batch_size = 12`, `num_workers = 2`, `num_batches = 3` (per stage)
  - Treatment: P3 Modes A (Identity), B (Generic Depthwise), C (Adaptive Spatial Depthwise - ASDW)
- **Command Executed:**
  ```python
  !python scripts/preflight_p3_realdata.py \
      --depth 12 \
      --p3-mode all \
      --batch-size 12 \
      --num-workers 2 \
      --num-batches 3 \
      --data-root /content/dataset/Crack500
  ```

---

## 2. Consolidated 24-Point Invariant Audit Table

| Check Item / Metric | Run A (Identity) [D12] | Run B (Generic DW) [D12] | Run C (ASDW) [D12] | Status |
|:---|:---:|:---:|:---:|:---:|
| **A. Real-data forward** | PASS | PASS | PASS | 100% PASS |
| **B. Real-data backward** | PASS | PASS | PASS | 100% PASS |
| **C. Optimizer step** | PASS | PASS | PASS | 100% PASS |
| **D. Stage 1 -> 2 transition** | PASS | PASS | PASS | 100% PASS |
| **E. Stage 2 forward/backward** | PASS | PASS | PASS | 100% PASS |
| **F. Checkpoint save/reload** | PASS | PASS | PASS | 100% PASS |
| **G. Peak allocated VRAM** | 8,257.7 MB (8.06 GB) | 8,477.8 MB (8.28 GB) | 8,689.6 MB (8.49 GB) | Measured |
| **H. Peak reserved VRAM** | 9,690.0 MB (9.46 GB) | 9,898.0 MB (9.67 GB) | 10,016.0 MB (9.78 GB) | < 10.0 GB (Safe) |
| **I. Throughput (Preflight)** | 0.21 samples/s | 0.65 samples/s | 0.99 samples/s | Measured |
| **J. DataLoader health** | PASS | PASS | PASS | 100% PASS |
| **K. NaN / Inf status** | CLEAN (None) | CLEAN (None) | CLEAN (None) | 100% PASS |
| **L. P3 gradient status** | PASS (0 params, Identity) | PASS (10 params, active grad) | PASS (10 params, active grad) | 100% PASS |
| **M. PE28 fixed buffer status** | PASS | PASS | PASS | 100% PASS |
| **N. Locked Base provenance** | NOT_PROVEN | NOT_PROVEN | NOT_PROVEN | Expected GATED |
| **O. Checkpoint SHA256** | None | None | None | No fake SHA |

**Final Suite Verdict:**
`REAL-DATA P3 LAUNCH PREFLIGHT = GATED (Locked Base Checkpoint required; please supply --locked-base)`

---

## 3. Detailed Cross-Run Numerical & Architectural Verification

### 3.1. PE28 Cross-Run Invariance Audit
- Run A vs Run B: Identical (max difference = 0.0)
- Run A vs Run C: Identical (max difference = 0.0)
- **Bitwise Equality:** `Run A.pe28_fixed == Run B.pe28_fixed == Run C.pe28_fixed` $\to$ **PASS**.
- Buffer State: Shape `[1, 784, 192]`, `dtype = torch.float32`, `requires_grad = False`, gradient is strictly `None`.
- Derivation: Exact mathematical bicubic 2D interpolation from ViT positional embeddings ($14 \times 14 \to 28 \times 28$).

### 3.2. Parameter Grouping & SAGE Injection Verification
- **Run A (Control):** Exactly 0 refinement parameters instantiated. Main stream directly takes pooled features.
- **Run B & C:** Exactly 10 refinement parameter tensors instantiated. In Stage 2 optimizer, all 10 tensors are strictly placed in `other_and_routers` (receiving `stage2_base_lr = 1e-4`), and strictly absent from `shared_experts`.
- **Shared Experts:** All 132 parameters confirmed as CNN main_blocks (requires_grad=True).

### 3.3. Memory & VRAM Headroom on T4 (14.56 GB)
- Run A: Peak Allocated = 8.06 GB, Peak Reserved = 9.46 GB (~5.1 GB headroom).
- Run B: Peak Allocated = 8.28 GB, Peak Reserved = 9.67 GB (~4.89 GB headroom).
- Run C: Peak Allocated = 8.49 GB, Peak Reserved = 9.78 GB (~4.78 GB headroom).
- Peak memory during Stage 1/2 remains under 10.0 GB across all three runs, confirming $BS=12$ operates stably within T4 memory boundaries during training.

### 3.4. Validation Metric Sanity Check
- 348 validation pairs discovered in Crack500 validation split.
- Executed strictly on 2 validation samples via non-overlapping tiling:
  - Run A: Val Loss = 2.0817, Val Dice = 0.0810, Val IoU = 0.0436
  - Run B: Val Loss = 2.0757, Val Dice = 0.0716, Val IoU = 0.0383
  - Run C: Val Loss = 2.0780, Val Dice = 0.0750, Val IoU = 0.0401
- Zero access to Test split (`test/images`, `test/masks`).

---

## 4. Launch Gate Status & Next Steps
- **Hardware & Numerical Invariants:** **READY & VERIFIED 100%**.
- **Launch Gate Status:** **GATED** (Chờ Locked Base checkpoint chính thức từ Phase 1 trước khi bắt đầu 30-epoch training).
- **Official P3 Phase-7 Training:** BLOCKED until Locked Base D12 is generated.
