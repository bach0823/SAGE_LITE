# 📖 Running Guide: SAGE-Lite for Crack Segmentation

This document provides step-by-step instructions for reproducing and operating the **SAGE-Lite** framework, from environment setup and dataset preparation to two-stage model training, checkpoint resumption, and benchmark evaluation on crack segmentation benchmarks (**Crack500**, **DeepCrack**).

---

## 1. Environment Setup

SAGE-Lite is built on **PyTorch $\ge$ 2.0** and relies on `timm` for lightweight foundation backbones (**ConvNeXtV2-Femto** and **ViT-Tiny**).

```bash
# 1. Create and activate a Conda environment
conda create -n sage_lite python=3.10 -y
conda activate sage_lite

# 2. Install PyTorch with appropriate CUDA support (e.g., CUDA 12.1)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 3. Install project dependencies
pip install -r requirements.txt
```

---

## 2. Dataset Preparation

SAGE-Lite targets infrastructure crack segmentation benchmarks. We provide automated preprocessing scripts that handle downloading, verification, filtering, and directory restructuring.

### Preparing Crack500

The Crack500 dataset contains pavement crack images characterized by complex asphalt textures, low contrast, and non-uniform illumination.

Run the automated preparation script:
```bash
python prepare_data/prepare_crack500.py
```

This script performs:
1. Automated mirror download with fallback and integrity checks.
2. Extraction and directory layout matching the expected hierarchy:
```text
datasets/Crack500_ready/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

### Preparing DeepCrack (Optional Benchmark)

DeepCrack provides multi-scale crack masks across different materials. To prepare DeepCrack:
```bash
python prepare_data/prepare_deepcrack.py
```
This clones the upstream repository at the locked commit (`8202a60`), extracts the data, and structures train/val/test splits with verified masks.

---

## 3. Structural Model Verification

Before initiating long training runs, always execute the structural verification script to validate tensor shape contracts, SAGE expert wrapping, and forward-pass integrity:

```bash
python tools/verify_model.py
```

Expected output includes confirmation of 8 injected experts (4 ConvNeXt stages + 4 ViT blocks), correct bridge projections, and successful loss computation.

---

## 4. Model Training

The primary entry point is `scripts/train_crack.py`, configured via `configs/sage_b2_crack500.yaml`.

### Full Two-Stage Training (Standard Pipeline)

SAGE-Lite employs an automated two-stage training strategy:
- **Stage 1 (Epochs 1–17):** S2-Gate uses a higher learning rate ($\text{LR}_{s2} = 10^{-3}$) to rapidly learn spatial modulation, while base modules train at $\text{LR} = 10^{-4}$ and SAGE routers at $\text{LR}_{\text{sage}} = 2 \times 10^{-4}$.
- **Stage 2 (Epochs 18–35):** Refines expert routing and shared representations at $\text{LR} = 10^{-4}$, optimizing boundary details with Soft Boundary IoU ($\lambda_{\text{bound}} = 0.50$, $d = 2$).

To run full end-to-end training:
```bash
python scripts/train_crack.py --config configs/sage_b2_crack500.yaml
```

All checkpoints, event logs, and routing metrics are saved to `results/sage_b2_crack500/`.

---

## 5. Staged Execution & Checkpoint Resumption

### Running Stage 1 Only
To run only the initial phase (e.g., on a resource-constrained worker):
```bash
python scripts/train_crack.py \
    --config configs/sage_b2_crack500.yaml \
    --stage1-only
```

### Resuming to Stage 2 from Stage 1 Checkpoint
To resume training seamlessly into Stage 2 while restoring exact RNG states and optimizer configurations:
```bash
python scripts/train_crack.py \
    --config configs/sage_b2_crack500.yaml \
    --stage2-only \
    --checkpoint results/sage_b2_crack500/best_model_b2_stage1.pth \
    --rng-checkpoint results/sage_b2_crack500/last_model_b2_stage1.pth
```

### Extending Stage 2 at a Low LR Floor
To perform fine-tuning or post-training stabilization:
```bash
python scripts/train_crack.py \
    --config configs/sage_b2_crack500.yaml \
    --resume-stage2 \
    --resume-stage2-low-lr \
    --low-lr 1e-6 \
    --checkpoint results/sage_b2_crack500/best_model_b2_global.pth
```

---

## 6. Official Benchmark Evaluation

To evaluate a trained checkpoint on the Crack500 test set using standard benchmark metrics (Precision, Recall, F1-Score, IoU, and Boundary IoU):

```bash
python scripts/evaluate_crack_official.py \
    --config configs/sage_b2_crack500.yaml \
    --checkpoint results/sage_b2_crack500/best_model_b2_global.pth
```

---

## 7. Analysis & Diagnostics Utilities

SAGE-Lite includes diagnostic scripts in `tools/`:
- **Routing Entropy & Decision Audit:**
  ```bash
  python tools/analyze_routing.py \
      --config configs/sage_b2_crack500.yaml \
      --checkpoint results/sage_b2_crack500/best_model_b2_global.pth
  ```
- **Inspect Checkpoint Tensors:**
  ```bash
  python tools/inspect_checkpoint.py \
      --checkpoint results/sage_b2_crack500/best_model_b2_global.pth
  ```
