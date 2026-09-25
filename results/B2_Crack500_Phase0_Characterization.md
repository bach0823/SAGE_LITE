# SAGE-Lite B2 Phase 0 Runtime & Hardware Characterization Report (Crack500)

*Date: 2026-09-25*  
*Hardware: Tesla T4 (Google Colab, 14.56 GB usable VRAM, 2 vCPUs)*  
*Repo & Commit: `SAGE_LITE` @ `crack500-audit`, Commit `8ca7df7`*  
*Dataset: Real Crack500 (1896 train samples, 348 val pairs)*  
*Script: `scripts/run_phase0_probe.py`*  

---

## 1. Bảng Tổng Hợp Đo Đạc Phần Cứng Duy Nhất (T4 14.56 GB VRAM)

| Depth | ViT Blocks | Expert Pool | Batch Size Tested | Max Safe BS | Peak VRAM (BS12) | Workers | Step Time | Throughput | Sec / Epoch |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **D12** | 12 | 16 (4 CNN + 12 ViT) | BS8: PASS, BS12: PASS, BS16: OOM, BS24: OOM | **12** | **14,734.0 MB** *(98.8%)* | 0<br>2<br>4 | 5,121.6 ms<br>5,280.9 ms<br>5,299.8 ms | 2.34 img/s<br>2.27 img/s<br>2.26 img/s | 809.2 s *(13.5 min)*<br>834.4 s *(13.9 min)*<br>837.4 s *(14.0 min)* |
| **D06** | 6 | 10 (4 CNN + 6 ViT) | BS8: PASS, BS12: PASS, BS16: OOM, BS24: OOM | **12** | **14,150.0 MB** *(94.9%)* | 0<br>2<br>4 | 5,130.2 ms<br>4,728.4 ms<br>4,433.5 ms | 2.34 img/s<br>2.54 img/s<br>2.71 img/s | 810.6 s *(13.5 min)*<br>747.1 s *(12.5 min)*<br>700.5 s *(11.7 min)* |
| **D04** | 4 | 8 (4 CNN + 4 ViT) | BS8: PASS, BS12: PASS, BS16: OOM, BS24: OOM | **12** | **14,168.0 MB** *(95.0%)* | 0<br>2<br>4 | 6,261.4 ms<br>4,950.1 ms<br>3,881.5 ms | 1.92 img/s<br>2.42 img/s<br>3.09 img/s | 989.3 s *(16.5 min)*<br>782.1 s *(13.0 min)*<br>613.3 s *(10.2 min)* |

---

## 2. Dự Phóng Thời Gian Chạy Thực Tế (Epoch Budget Projections: BS=12, Workers=4)

| Depth | 15 Epochs | 20 Epochs | 25 Epochs | 30 Epochs | Khả năng duy trì Colab Session |
|:---:|:---:|:---:|:---:|:---:|:---|
| **D12** | 209.3 min *(~3.5 h)* | 279.1 min *(~4.7 h)* | 348.9 min *(~5.8 h)* | **418.7 min *(~7.0 h)*** | Rủi ro timeout cao ở 30 epochs; khuyến nghị chọn 20-25 epochs. |
| **D06** | 175.1 min *(~2.9 h)* | 233.5 min *(~3.9 h)* | 291.9 min *(~4.9 h)* | **350.2 min *(~5.8 h)*** | An toàn trong 1 session liên tục. |
| **D04** | 153.3 min *(~2.6 h)* | 204.4 min *(~3.4 h)* | 255.5 min *(~4.3 h)* | **306.6 min *(~5.1 h)*** | Nhẹ nhất; hoàn thành nhanh trong ~3-4h. |

---

## 3. Năm Kết Luận Kỹ Thuật Trọng Tâm
1. **Trần Vật Lý VRAM:** Toàn bộ 3 depth đều OOM tại $BS=16$. $BS=12$ là giới hạn kịch trần (D12 chiếm 14,734 MB, chỉ còn 178 MB buffer). Cần dọn cache sau mỗi epoch hoặc chọn $BS=8$ nếu muốn buffer dư dả (> 5 GB).
2. **Điểm Nghẽn Tính Toán:** Bước huấn luyện bị chi phối 95-97% bởi GPU (Forward + Backward mất 3.8s - 5.1s). DataLoader chỉ mất 0.3 - 0.8 ms. Cấu hình `num_workers = 2` là tối ưu nhất cho môi trường 2-vCPU của Colab.
3. **Tỷ Lệ Tăng Tốc ViT Depth:** D4 (613.3 s/epoch) nhanh hơn D12 (837.4 s/epoch) ~26.8%; D6 nhanh hơn D12 ~16.4%.
4. **Ngân Sách Khuyến Nghị:** $N_{\text{total}} = 20\text{ epochs}$ (Stage 1 = 8 epochs, Stage 2 = 12 epochs) kèm Early Stopping `patience = 5`.
