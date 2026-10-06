# SAGE-Lite Phase 6: S2-Gate Combination, Routing Mechanism & Capacity Synthesis

*Date: 2026-10-07*  
*Hardware: Tesla T4 (Google Colab, 14.56 GB usable VRAM)*  
*Evaluation Dataset: Crack500 Validation Split (348 image pairs, Setting A non-overlap 448x448, 81,594,144 total pixels)*  
*Branch: `crack500-audit`*  
*Canonical Base Configuration: ViT Depth D=4, Hidden Dim H=64, Load Balancing LB=0.010, SAGE LR=2e-4, Shared LR Ratio r=1.00*  

---

## 1. Executive Summary & Synthesis

Báo cáo này tổng hợp toàn diện kết quả thực nghiệm của **Phase 6: A1 + S2-Gate (End-to-End & Stage 2-Only)**, thẩm định độc lập đóng góp của **ASDW (Adaptive Spatial Dynamic Weighting)**, kiểm chứng hành vi **Routing (Dynamic Adaptive vs Static vs Random)**, khảo sát dung lượng **Static Top-$K$ ($K \in \{2, 3, 4, 6\}$)**, và phân tích phân rã biểu diễn chuyên gia (**Expert Diversity Decomposition**).

### Các kết luận khoa học then chốt:
1. **Hiệu năng A1 + S2-Gate bứt phá đỉnh cao mới**:
   - Run 1 (`a1_s2g_end_to_end`): Val Dice **0.7676** (Epoch 16).
   - Run 2 (`a1_s2g_stage2`): Val Dice **0.7702** (Epoch 14), Precision **0.7510**, Recall **0.8301**, Thin Crack Dice **0.4260**.
   - S2-Gate (Conv3x3 tại Decoder Block 1) và Soft Boundary IoU Loss ($\lambda=0.5, d=2$) tạo ra hiệp đồng mạnh mẽ mà không làm giảm recall biên.
2. **Loại bỏ chính thức ASDW (Zero-Contribution Confirmed)**:
   - Thử nghiệm bật/tắt ASDW trực tiếp trên cùng checkpoint `a1_s2g_end_to_end` khẳng định: $\Delta\text{Dice} = -0.000001$ ($0.767640 \to 0.767639$). Trọng số $\gamma$ học được quá nhỏ ($\gamma_{\text{S0}} = 0.0090, \gamma_{\text{S1}} = 0.0077$), chứng minh ASDW hoàn toàn bất hoạt trong thực tế.
3. **Phát hiện nghịch lý Routing: Static Top-2 vượt Dynamic Adaptive**:
   - Static Top-2 đạt Val Dice **0.7687** (+0.11% vs Dynamic 0.7676, +0.79% Precision).
   - Khẳng định mạng đã học được cấu trúc phân công lao động tối ưu cố định; gating động ở inference chỉ mang tính điều biến biên (marginal fine-tuning) và có xu hướng tăng false positives do noise gating.
4. **Dung lượng chuyên gia bão hòa ở $K=2$ ($25\%$ capacity)**:
   - Quét $K \in \{2, 3, 4, 6\}$ cho thấy hiệu năng cao nhất tại $K=2$ (Dice 0.7687). Khi tăng $K \ge 3$, việc kích hoạt nhiều hơn 2 chuyên gia không mang lại lợi ích mà còn pha loãng đặc trưng, tụt xuống **0.7666** ở $K=6$.
5. **Cấu trúc phân rã chuyên gia (Expert Diversity)**:
   - Các chuyên gia CNN (E0–E3) có tính trực giao cao (Intra-CNN Cosine Similarity = **0.0952**), đảm nhiệm trích xuất cấu trúc cục bộ đa tỷ lệ.
   - Các khối Transformer (E4–E7) đạt độ hội tụ ngữ cảnh cao (Intra-ViT Cosine Similarity = **0.6869**), khẳng định ViT Depth $D=4$ đã bão hòa trần biểu diễn; việc tăng thêm ViT block ($D > 4$) là dư thừa.

---

## 2. Chi Tiết Thực Nghiệm A1 + S2-Gate (Run 1 vs Run 2)

| Thông Số / Metric | Run 1: `a1_s2g_end_to_end` | Run 2: `a1_s2g_stage2` | Chênh Lệch ($\Delta$) |
|:---|:---:|:---:|:---:|
| **Phương thức chạy** | Full Stage 1 + Stage 2 | Stage 2-Only từ Best S1 Checkpoint | Tiết kiệm 17 epochs S1 |
| **Stage 1 Epochs / Best Dice** | 17 epochs / **0.7541** | Kế thừa Checkpoint S1 (Dice 0.7541) | Đồng nhất baseline xuất phát |
| **Stage 2 Best Epoch** | Epoch 16 | Epoch 14 | Hội tụ nhanh hơn 2 epochs |
| **Global Best Val Dice** | **0.7676** | 🏆 **0.7702** | **+0.0026** (+0.26 pp) |
| **Validation Loss** | 1.6372 | **1.3471** | **-0.2901** |
| **Precision** | 0.7408 | **0.7510** | **+0.0102** (+1.02 pp) |
| **Recall** | **0.8384** | 0.8301 | -0.0083 (-0.83 pp) |
| **Thin Crack Dice** | 0.4230 | **0.4260** | **+0.0030** |
| **Composite Objective Score** | 0.7728 | **0.7761** | **+0.0033** |

*Ghi chú Provenance*:
- Checkpoint Stage 1 dùng chung: `results/phase6_combination/phase6_comb_a1_s2g_end_to_end/best_model_b2_stage1.pth` (SHA256: `361c82b5...`).
- Run 2 được khôi phục toàn bộ RNG state (PyTorch CPU/CUDA, NumPy, Python RNG, DataLoader generator) từ `last_model_b2_stage1.pth` để đảm bảo tính tái lập chuẩn mực khoa học.

---

## 3. Thẩm Định Độc Lập ASDW (Ablation: ASDW ON vs ASDW OFF)

Thực hiện đánh giá trên toàn bộ 348 mẫu của tập Crack500 Val dưới 3 chế độ Routing khác nhau với cùng checkpoint `best_model_b2_stage2.pth`:

| Routing Mode | ASDW Status | Val Dice (Mean) | Val Dice (Median) | Mean IoU | Precision | Recall |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Adaptive Routing** | ASDW ON | **0.767640** | 0.807273 | 0.645171 | 0.740777 | 0.838409 |
| **Adaptive Routing** | ASDW OFF ($\gamma=0$) | **0.767639** | 0.807273 | 0.645170 | 0.740774 | 0.838412 |
| **Static Routing (Top-2)** | ASDW ON | **0.768655** | 0.805391 | 0.646700 | 0.748639 | 0.830686 |
| **Static Routing (Top-2)** | ASDW OFF ($\gamma=0$) | **0.768653** | 0.805391 | 0.646696 | 0.748634 | 0.830689 |
| **Random Routing** | ASDW ON | **0.767275** | 0.804895 | 0.645015 | 0.743223 | 0.835030 |
| **Random Routing** | ASDW OFF ($\gamma=0$) | **0.767275** | 0.804895 | 0.645016 | 0.743221 | 0.835035 |

### Nhận xét & Kết luận:
- Sai số giữa ASDW ON và ASDW OFF là $|\Delta\text{Dice}| < 10^{-6}$ trên tất cả các chế độ.
- Hệ số suy giảm học được trong Stage 2 là $\gamma_{\text{S0}} = 0.0090, \gamma_{\text{S1}} = 0.0077$. Kết hợp với SAGE `residual_scale = 0.10`, độ lớn logit truyền từ ASDW nhánh bypass bị triệt tiêu về mức vô nghĩa ($< 10^{-4}$).
- **Quyết định**: Khẳng định quyết định loại bỏ hoàn toàn P3-C ASDW khỏi kiến trúc canonical SAGE-Lite là hoàn toàn chính xác.

---

## 4. Khảo Sát Dung Lượng Chuyên Gia Static Top-$K$ ($K \in \{2, 3, 4, 6\}$)

Đánh giá độ nhạy của dung lượng định tuyến tĩnh trên 348 mẫu Crack500 Val:

| Cấu hình | Dung lượng Expert (%) | Val Dice | Median Dice | Mean IoU | Precision | Recall | Thời gian Eval (s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Dynamic Adaptive (K=2)** | $25.0\%$ (2/8) | 0.7676 | **0.8073** | 0.6452 | 0.7408 | **0.8384** | 54.1s |
| **Static Top-2** | $25.0\%$ (2/8) | 🏆 **0.7687** | 0.8054 | 🏆 **0.6467** | 🏆 **0.7486** | 0.8307 | 61.9s |
| **Static Top-3** | $37.5\%$ (3/8) | 0.7682 | 0.8055 | 0.6459 | 0.7448 | 0.8344 | 93.4s |
| **Static Top-4** | $50.0\%$ (4/8) | 0.7682 | 0.8055 | 0.6457 | 0.7459 | 0.8327 | 117.0s |
| **Static Top-6** | $75.0\%$ (6/8) | 0.7666 | 0.8053 | 0.6437 | 0.7406 | 0.8357 | 143.4s |

### Cơ chế suy thoái dung lượng (Capacity Dilution):
1. **Tại sao K=2 tối ưu?**: Mô hình được huấn luyện với $K=2$ và hàm phạt Load Balancing $LB=0.010$. Các chuyên gia được tối ưu hóa để bổ trợ cho nhau theo cặp.
2. **Hiện tượng pha loãng khi $K \ge 3$**: Khi ép mô hình nhận $K > 2$ chuyên gia ở inference, các chuyên gia xếp hạng thấp hơn (rank 3, 4, 6) có phân phối biểu diễn không được tối ưu để cộng gộp tuyến tính, gây nhiễu đặc trưng và làm giảm Precision từ $0.7486 \to 0.7406$.
3. **Hiệu suất tính toán**: Static Top-2 tốn ít chi phí tính toán hơn đáng kể ($61.9\text{s}$ vs $143.4\text{s}$ ở $K=6$), đạt điểm cân bằng Pareto hoàn hảo.

---

## 5. Phân Tích Độ Đa Dạng Biểu Diễn Chuyên Gia (Expert Diversity)

Đo lường độ tương đồng biểu diễn trên 8 chuyên gia (E0–E3: CNN stages, E4–E7: Transformer blocks) qua tập Validation:

| Chỉ số Đa dạng Biểu diễn | Giá trị Định lượng | Đánh giá & Diễn giải Cơ chế |
|:---|:---:|:---|
| **Mean Cosine Similarity** | **0.2694** | Rất thấp $\implies$ Các chuyên gia duy trì biểu diễn gần như trực giao. |
| **Median Cosine Similarity** | **0.1535** | Phân phối tương đồng dồn về lân cận 0. |
| **Intra-CNN Cosine Similarity** | **0.0952** | Gần như trực giao hoàn toàn. 4 stage CNN học các tầng phân giải và receptive field hoàn toàn tách biệt. |
| **Intra-ViT Cosine Similarity** | **0.6869** | Mức độ tương đồng cao. Các khối Transformer xử lý ở cùng độ phân giải không gian ($14 \times 14$), biểu diễn ngữ cảnh có xu hướng hội tụ mạnh. |
| **Cross-Family Cosine Similarity** | **0.1782** | Rất thấp. CNN và ViT hình thành 2 trường phái biểu diễn bù trừ lẫn nhau (Local Detail vs Global Context). |
| **Kết luận Đa dạng** | **HIGH DIVERSITY** | Xác nhận không có hiện tượng suy biến chuyên gia (no expert collapse). |

### Hàm ý kiến trúc về ViT Depth:
- Vì **Intra-ViT Cosine Similarity đạt tới 0.6869**, các khối Transformer có độ dư thừa ngữ cảnh tương đối cao.
- Điều này chứng minh rằng việc dừng ở **ViT Depth $D=4$** là hoàn toàn đúng đắn. Việc tăng thêm $D=6$ hay $D=8$ sẽ chỉ làm tăng độ trùng lặp đặc trưng mà không cung cấp thêm thông tin trực giao mới.

---

## 6. Phân Tích Cơ Chế S2-Gate Cho Decoder

### Rationale cho Decoder Block 1 (56x56):
- Tại điểm giao giữa Skip S1 ($56 \times 56, 96\text{ch}$) và Upsampled S2 ($56 \times 56, 192\text{ch}$), mức độ mơ hồ biểu diễn đạt đỉnh ($R_{\text{norm}} = 0.69$). S2-Gate với Conv3x3 cung cấp Receptive Field 3px để nhận diện hành lang nứt hẹp, dập tắt các cầu nứt giả (False Bridges).

### Đánh giá khả năng mở rộng cho Decoder Block 0 và Decoder Block 2:
1. **Decoder Block 0 ($28 \times 28$ - Bottleneck)**:
   - *Không cần thiết*: Tại $28 \times 28$, ngữ cảnh toàn cục sâu nhất đã được thiết lập bởi ViT và Stage 3 CNN. Không có luồng skip nông mang nhiễu tần số cao nào cần phải triệt tiêu tại đây.
2. **Decoder Block 2 ($112 \times 112$ - Shallow Skip)**:
   - *Rất tiềm năng về mặt lý thuyết*: Skip S0 ($112 \times 112, 48\text{ch}$) là luồng đặc trưng rất nông, chứa nhiều nhiễu vi cấu trúc bề mặt vật liệu (texture noise) và gờ mép giả.
   - Một cổng không gian tương tự S2-Gate (dùng context từ Decoder Block 1 để điều tiết Skip S0) có thể giúp làm sạch biên giới nứt ở bước tái tạo chi tiết cuối cùng.

---

## 7. Tổng Hợp Cấu Trúc Kết Quả Đã Sắp Xếp (Results Directory Manifest)

Thư mục `results/` đã được chuẩn hóa và tinh gọn:
- `results/phase6_combination/`:
  - Đã dọn dẹp các thư mục sao chép trùng lặp byte-for-byte (`_run1`, `_run2`, `_record_07700`).
  - Lưu trữ đầy đủ checkpoint gốc và log của các cấu hình đại diện: `a1_s2g_end_to_end`, `a1_s2g_stage2`, `a1_full_s1_l050`, `a1_full_s1_l075`, `b1_v1/v2/v3`, `a1_v1/v2/v3`, `stage2a_a1_b1`.
- `results/diagnostics/`:
  - `routing_and_asdw_ablation/`: Báo cáo và bảng dữ liệu JSON/CSV chi tiết về ASDW ON/OFF và Routing modes.
  - `static_topk_sweep/`: Dữ liệu quét Top-$K$ ($K \in \{2, 3, 4, 6\}$).
  - `expert_diversity/`: Ma trận cosine và pearson giữa 8 chuyên gia.
  - `routing_analysis/`: Bao gồm cả `a1_s2g_end_to_end` và `phase6_a2_pure_plu` (đã quy hoạch từ thư mục ngoài).
  - `error_analysis/`: Phân tích lỗi hình thái và trường hợp biên (best/worst/gallery).
  - `phase6_u1_s2g/`: Trọng số và bảng đánh giá cơ chế S2-Gate standalone.
