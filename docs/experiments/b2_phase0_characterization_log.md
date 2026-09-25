# Phase 0: SAGE-Lite B2 Runtime & Hardware Characterization Log (Crack500)

## 1. Executive Summary
- **Date:** September 25, 2026
- **Hardware:** Google Colab Tesla T4 (14.56 GB usable VRAM, 2 vCPUs)
- **Repo & Branch:** `SAGE_LITE` @ `crack500-audit`, Commit `8ca7df7`
- **Dataset:** Real Crack500 (1,896 train samples, 348 val pairs) at `/content/dataset/Crack500`
- **Scope:** Complete hardware profiling for all 3 ViT depths: **D12, D6, D4** (Zero generalization across depths; empirical measurements only).
- **Execution Script:** `scripts/run_phase0_probe.py`

---

## 2. Unified Characterization Table (T4 14.56 GB VRAM)

| Depth | ViT Blocks | Expert Pool | Batch Size Tested | Max Safe BS | Peak VRAM (BS12) | Workers | Step Time | Throughput | Sec / Epoch |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **D12** | 12 | 16 (4 CNN + 12 ViT) | BS8: PASS, BS12: PASS, BS16: OOM, BS24: OOM | **12** | **14,734.0 MB** *(98.8%)* | 0<br>2<br>4 | 5,121.6 ms<br>5,280.9 ms<br>5,299.8 ms | 2.34 img/s<br>2.27 img/s<br>2.26 img/s | 809.2 s *(13.5 min)*<br>834.4 s *(13.9 min)*<br>837.4 s *(14.0 min)* |
| **D06** | 6 | 10 (4 CNN + 6 ViT) | BS8: PASS, BS12: PASS, BS16: OOM, BS24: OOM | **12** | **14,150.0 MB** *(94.9%)* | 0<br>2<br>4 | 5,130.2 ms<br>4,728.4 ms<br>4,433.5 ms | 2.34 img/s<br>2.54 img/s<br>2.71 img/s | 810.6 s *(13.5 min)*<br>747.1 s *(12.5 min)*<br>700.5 s *(11.7 min)* |
| **D04** | 4 | 8 (4 CNN + 4 ViT) | BS8: PASS, BS12: PASS, BS16: OOM, BS24: OOM | **12** | **14,168.0 MB** *(95.0%)* | 0<br>2<br>4 | 6,261.4 ms<br>4,950.1 ms<br>3,881.5 ms | 1.92 img/s<br>2.42 img/s<br>3.09 img/s | 989.3 s *(16.5 min)*<br>782.1 s *(13.0 min)*<br>613.3 s *(10.2 min)* |

---

## 3. Epoch Budget Projections (BS = 12, Workers = 4)

| Depth | 15 Epochs | 20 Epochs | 25 Epochs | 30 Epochs | Colab Session Feasibility |
|:---:|:---:|:---:|:---:|:---:|:---|
| **D12** | 209.3 min *(~3.5 h)* | 279.1 min *(~4.7 h)* | 348.9 min *(~5.8 h)* | **418.7 min *(~7.0 h)*** | High timeout risk at 30 epochs; 20-25 epochs recommended. |
| **D06** | 175.1 min *(~2.9 h)* | 233.5 min *(~3.9 h)* | 291.9 min *(~4.9 h)* | **350.2 min *(~5.8 h)*** | Safe for single continuous Colab session. |
| **D04** | 153.3 min *(~2.6 h)* | 204.4 min *(~3.4 h)* | 255.5 min *(~4.3 h)* | **306.6 min *(~5.1 h)*** | Lightest compute; zero session risk. |

---

## 4. Key Findings & Technical Decisions
1. **Physical VRAM Ceiling:** All 3 depths OOM at $BS=16$. $BS=12$ is the absolute upper limit on T4 ($14,734\text{ MB}$ reserved on D12, leaving only $178.7\text{ MB}$ buffer). Safe operational batch size is $BS=12$ with explicit `empty_cache()`, or $BS=8$ for zero-risk buffer.
2. **Compute Bottleneck:** Step time is 95-97% GPU-bound (Forward + Backward = 3.8s to 5.1s). DataLoader wait time is minimal ($0.3 - 0.8\text{ ms}$). Colab environment has 2 vCPUs; `num_workers = 2` is optimal.
3. **Compute Scaling:** D4 is ~26.8% faster than D12, D6 is ~16.4% faster than D12.
4. **Recommended Training Budget:** $N_{\text{total}} = 20\text{ epochs}$ (Stage 1 = 8 epochs, Stage 2 = 12 epochs) with early stopping patience 5.
