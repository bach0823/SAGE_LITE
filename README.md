# SAGE-Lite: Lightweight Shape-Adapting Gated Experts for Crack Segmentation

*Advancing Mixture of Local Experts (MoLEx) with ConvNeXtV2-Femto and Compact Vision Transformers for Industrial and Civil Infrastructure Crack Boundary Inspection*

---

Official PyTorch implementation of **SAGE-Lite**, an efficient, lightweight Mixture of Local Experts (MoLEx) architecture tailored for fine-grained crack detection and boundary segmentation on concrete, pavement, and structural surfaces. SAGE-Lite builds upon the foundational theoretical principles of **SAGE** and optimizes both model complexity and routing pathways using **ConvNeXtV2-Femto** coupled with a compact **Vision Transformer** bottleneck, Dual **$\text{S}^2\text{-Gate}$** spatial modulation, and boundary-aware optimization.

> [!NOTE]
> **Academic Origin & Lineage:**  
> This project builds upon the original work:  
> *“SAGE: Shape-Adapting Gated Experts for Adaptive Histopathology Image Segmentation”* (CVPR 2026 Findings) by Gia Huy Thai, Hoang-Nguyen Vu, Anh-Minh Phan, Quang-Thinh Ly, Thi-Ngoc-Truc Nguyen, and Nhat Ho.  
> **SAGE-Lite** transfers and refines the core MoLEx paradigm from multi-organ histopathology to industrial and civil engineering structural crack segmentation (evaluating on benchmarks such as **Crack500** and **DeepCrack**), adapting routing mechanisms to thin, elongated, high-curvature topological fissures under severe compute constraints.

---

## 🔬 Implementation Highlights — Key Code Architecture

Integration follows a strictly typed, modular, and non-destructive design pattern. Lightweight foundation backbones provide multi-scale feature hierarchies without incurring high latency or prohibitive parameter counts.

```
       Input (3, 448, 448)
              │
    ┌─────────▼─────────┐
    │  ConvNeXtV2-Femto  │ ── Stage 0 (48ch, 112x112)  ──► Skip 0
    │  4 Conv Stages    │ ── Stage 1 (96ch,  56x56)   ──► Dual S²-Gate ──► Skip 1
    │  (Stages 0 - 3)   │ ── Stage 2 (192ch, 28x28)   ──► Context S2 ────► Skip 2
    └─────────┬─────────┘ ── Stage 3 (384ch, 14x14)
              │
     ConvNeXt-to-ViT Bridge (1x1 Conv + LayerNorm)
              │
    ┌─────────▼─────────┐
    │    ViT-Tiny       │ 4 Transformer Blocks (D=192, 196 tokens)
    │  Bottleneck       │
    └─────────┬─────────┘
              │
    ┌───────────────────┐
    │  SAGE Layer Pool  │ 4 CNN Experts + 4 ViT Experts (8-Expert Pool)
    │  & SageRouters    │ Top-2 Routing with Shared Experts [0, 1, 2, 3] & Residual Fusion
    └─────────┬─────────┘
              │
    ┌─────────▼─────────┐
    │   UNet Decoder    │ Skip Connections + S²-Gate Modulation + SoftBIoU Optimization
    └─────────┬─────────┘
              │
       Mask (1, 448, 448)
```

### 1. ConvNeXtV2 + ViT Hybrid Backbone

`ConvNeXtV2ViTHybrid` (`sage/networks/convnextv2_vit_hybrid.py`) couples a hierarchical convolutional encoder with a sequence transformer bottleneck:
* **ConvNeXtV2-Femto** (`convnextv2_femto.fcmae`): 4 convolutional stages with channel dimensions $C \in [48, 96, 192, 384]$ extracting local high-frequency crack edges, micro-fissures, and surface texture gradients.
* **Compact Vision Transformer** (`vit_tiny_patch16_224`): Configured with `num_transformer_layers = 4` blocks ($D = 192$, $14 \times 14 = 196$ spatial tokens at 448 resolution), modeling long-range topological crack continuity across the entire scene.

```python
# sage/networks/convnextv2_vit_hybrid.py
self.convnext = timm.create_model("convnextv2_femto.fcmae", pretrained=pretrained)
vit_full = timm.create_model("vit_tiny_patch16_224", pretrained=pretrained)

# Extract first num_transformer_layers (default: 4 in B2 configuration)
self.transformer_blocks = nn.ModuleList([
    vit_full.blocks[i] for i in range(num_transformer_layers)
])
```

Spatial-to-sequence bridge projections align 2D feature maps from ConvNeXt Stage 3 ($384$ channels) into the 1D token space ($192$ channels) expected by the Transformer blocks:
```python
# sage/networks/convnextv2_vit_hybrid.py
self.proj_to_transformer = nn.Sequential(
    nn.Conv2d(384, 192, kernel_size=1, bias=False),
    nn.BatchNorm2d(192),
    nn.GELU()
)
```

### 2. Dual $\text{S}^2\text{-Gate}$ Spatial Skip Modulation

The **$\text{S}^2\text{-Gate}$** (Spatial Skip Gate) module (`sage/networks/s2_gate.py`) dynamically modulates multi-scale skip connections before merging into the UNet decoder blocks:
* **Block 1 $\text{S}^2\text{-Gate}$:** Modulates Stage 1 skip ($56 \times 56$, 96 ch) using Stage 2 context ($28 \times 28$, 192 ch) with $3 \times 3$ standard convolution.
* **Block 2 $\text{S}^2\text{-Gate}$:** Employs dilated convolution ($3 \times 3$, $\text{dilation} = 2$) expanding the receptive field to suppress background concrete speckles while maintaining sharp crack boundaries.

```python
# sage/networks/s2_gate.py
self.gate_net = nn.Sequential(
    nn.Conv2d(
        s2_channels + skip_channels,
        mid_channels,
        kernel_size=kernel_size,
        padding=padding,
        dilation=dilation,
        bias=True,
    ),
    nn.BatchNorm2d(mid_channels),
    nn.ReLU(inplace=True),
    nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True),
)
```

> [!TIP]
> **Identity Initialization at $t=0$:** The final projection layer in $\text{S}^2\text{-Gate}$ is initialized with zero weights and a bias of $+5.0$ ($\sigma(5.0) \approx 0.9933$), guaranteeing exact identity pass-through at epoch 0 without perturbing early training gradients.

### 3. SAGE Routing & Heterogeneous Expert Pool

In `B2ConvNeXtViTUNet` (`sage/networks/b2_unet.py`), all 4 ConvNeXt stages and 4 ViT blocks are injected into `SageLayer` modules, forming a heterogeneous pool of **8 experts**:
* **Shared Experts:** Defined by `shared_expert_indices: [0, 1, 2, 3]` (the 4 CNN stages), providing stable multi-scale spatial representations.
* **Top-2 Dynamic Gating:** `SageRouter` (`sage/components/router.py`) dynamically selects the Top-2 most specialized experts per sample via Semantic Affinity Routing (SAR) and Sigmoid logit modulation:

$$\mathbf{y}_{\text{fused}} = \mathbf{y}_{\text{main}} + \gamma \cdot \text{Dropout}(\mathbf{y}_{\text{expert}})$$

Where $\gamma = 0.1$ denotes the residual scaling factor, implementing stable residual fusion (`fusion_type: "residual"`).

---

## ⚖️ Parameter Status & Two-Stage Training Regimes

SAGE-Lite follows a calibrated two-stage training protocol designed to stabilize expert specialization and fine-tune routing dynamics on crack benchmarks:

| Component / Layer Group | Module Path | Stage 1 (Ep 1–17) Status | Stage 2 (Ep 18–35) Status | Base Learning Rate |
| :--- | :--- | :---: | :---: | :---: |
| **Shared CNN Experts** | `backbone.convnext.stages[0:4]` | **Trainable** | **Trainable** | $\text{LR} = 10^{-4}$ |
| **ViT Transformer Blocks** | `backbone.transformer_blocks[0:4]` | **Trainable** | **Trainable** | $\text{LR} = 10^{-4}$ |
| **SAGE Routers** | `SageRouter` instances | **Trainable** | **Trainable** | $\text{LR}_{\text{sage}} = 2 \times 10^{-4}$ |
| **SA-Hub Adapters** | `SAHub.adapters` | **Trainable** | **Trainable** | $\text{LR}_{\text{sage}} = 2 \times 10^{-4}$ |
| **Dual $\text{S}^2\text{-Gate}$ Modules** | `s2_gate`, `s2_gate_block2` | **Trainable** | **Trainable** | $\text{LR}_{s2} = 10^{-3}$ (S1) / $10^{-4}$ (S2) |
| **UNet Decoder & Heads** | `decoder.blocks`, `final_conv` | **Trainable** | **Trainable** | $\text{LR} = 10^{-4}$ |
| **Soft Boundary IoU** | `SoftBoundaryIoULoss` | *Objective Criterion* | *Objective Criterion* | *Loss weight $\lambda = 0.50$* |

---

## 📐 Loss Formulation & Optimization Objectives

The composite loss function $\mathcal{L}_{\text{total}}$ balances regional overlap, boundary fidelity, and expert distribution:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{seg}}(\mathbf{z}, \mathbf{y}) + \lambda_{\text{bound}} \cdot \mathcal{L}_{\text{bound}}(\mathbf{z}, \mathbf{y}) + \lambda_{\text{lb}} \cdot \mathcal{L}_{\text{lb}}(\mathbf{G})$$

### 1. Primary Segmentation Loss ($\mathcal{L}_{\text{seg}}$)
$$\mathcal{L}_{\text{seg}} = 1.0 \cdot \mathcal{L}_{\text{BCE}}(\sigma(\mathbf{z}), \mathbf{y}) + 1.5 \cdot \mathcal{L}_{\text{Dice}}(\sigma(\mathbf{z}), \mathbf{y})$$

### 2. Soft Boundary IoU Loss ($\mathcal{L}_{\text{bound}}$)
Tailored specifically for thin, irregular crack paths. Differentiable morphological dilation with radius $d=2$ extracts boundary bands $\mathcal{B}(\cdot)$:
$$\mathcal{L}_{\text{bound}} = 1 - \frac{|\mathcal{B}(\sigma(\mathbf{z})) \cap \mathcal{B}(\mathbf{y})| + \epsilon}{|\mathcal{B}(\sigma(\mathbf{z})) \cup \mathcal{B}(\mathbf{y})| + \epsilon}$$
Configured with $\lambda_{\text{bound}} = 0.50$ and dilation radius $d = 2$.

### 3. Load Balancing Loss ($\mathcal{L}_{\text{lb}}$)
$$\mathcal{L}_{\text{lb}} = E \sum_{i=1}^{E} f_i \cdot P_i$$
Prevents router collapse towards a single dominant expert, with load-balance weighting factor $\lambda_{\text{lb}} = 0.01$.

---

## 📁 Repository Structure

```text
SAGE_LITE/
├── configs/                  # Canonical experiment configuration
│   └── sage_b2_crack500.yaml # Active B2 Crack500 training configuration
├── sage/                     # Core library implementation
│   ├── components/           # SAGE Routers, SA-Hub adapters, SageLayer wrappers
│   │   ├── router.py         # SageRouter (SAR, g_s, top-k, load balancing)
│   │   ├── sage_layer.py     # SageLayer with residual fusion
│   │   └── sa_hub.py         # Shape-Adapting Hub (cross-space 2D/1D adapters)
│   ├── networks/             # Neural network architectures
│   │   ├── convnextv2_vit_hybrid.py # ConvNeXtV2-Femto + ViT-Tiny hybrid
│   │   ├── decoder_block.py  # UNet Decoder blocks with skip connections
│   │   ├── s2_gate.py        # Dual S2-Gate spatial skip modulation
│   │   └── sage_injection.py # In-place SAGE expert pool wrapping
│   └── utils/                # Dataloader, metrics, evaluation utilities
├── scripts/                  # Training and evaluation entry points
│   ├── train_crack.py        # Main two-stage crack training script
│   ├── stage2_utils.py       # Checkpoint and RNG management for Stage 2
│   └── evaluate_crack_official.py # Official benchmark evaluation
├── tools/                    # Verification and diagnostic utilities
│   ├── verify_model.py       # Structural verification & dummy forward check
│   ├── inspect_checkpoint.py # Checkpoint parameter inspection
│   └── analyze_routing.py    # Routing allocation & entropy analysis
├── prepare_data/             # Dataset preprocessing scripts
│   ├── prepare_crack500.py   # Automated Crack500 dataset preparation
│   └── prepare_deepcrack.py  # Automated DeepCrack dataset preparation
├── docs/                     # Guides and experiment logs
│   └── RUNNING_GUIDE.md      # Comprehensive execution guide
└── requirements.txt          # Python dependencies
```

---

## ⚡ Quick Start & Execution

### 1. Environment Setup

Clone the repository and install dependencies:
```bash
git clone git@github.com:bach0823/SAGE_LITE.git
cd SAGE_LITE
pip install -r requirements.txt
```

### 2. Verify Architecture & Forward Pass

Before running heavy training, verify tensor dimensions, in-place expert pool injection, and Dual $S^2$-Gate setup:
```bash
python tools/verify_model.py
```

### 3. Dataset Preparation

Download raw datasets and process them to standard format:
```bash
# Prepare Crack500 dataset (default target)
python prepare_data/prepare_crack500.py --raw-dir /path/to/raw/crack500 --output-dir data/crack500

# Prepare DeepCrack dataset (optional)
python prepare_data/prepare_deepcrack.py --raw-dir /path/to/raw/deepcrack --output-dir data/deepcrack
```

### 4. Training Pipeline

All hyperparameters are centrally defined in `configs/sage_b2_crack500.yaml`.

#### Option A: Full Two-Stage Training (Recommended)
Executes Stage 1 (Encoder/Decoder warmup + S2-Gate) followed immediately by Stage 2 (Global SAGE adaptation with Top-2 routing):
```bash
python scripts/train_crack.py --config configs/sage_b2_crack500.yaml
```

#### Option B: Stage-by-Stage Modular Training
Run Stage 1 independently:
```bash
python scripts/train_crack.py --config configs/sage_b2_crack500.yaml --stage1-only
```

Resume directly into Stage 2 with deterministic RNG state restoration:
```bash
python scripts/train_crack.py \
    --config configs/sage_b2_crack500.yaml \
    --stage2-only \
    --checkpoint results/sage_b2_crack500/best_model_b2_stage1.pth \
    --rng-checkpoint results/sage_b2_crack500/last_model_b2_stage1.pth
```

### 5. Benchmark Evaluation

Evaluate trained checkpoints under the official direct SOTA resize protocol:
```bash
python scripts/evaluate_crack_official.py \
    --config configs/sage_b2_crack500.yaml \
    --checkpoint results/sage_b2_crack500/best_model_b2_stage2.pth \
    --split test
```

### 6. Interactive Routing Diagnostics

Inspect expert utilization, gating values, and routing entropy on validation samples:
```bash
python tools/analyze_routing.py \
    --config configs/sage_b2_crack500.yaml \
    --checkpoint results/sage_b2_crack500/best_model_b2_stage2.pth \
    --split val
```
