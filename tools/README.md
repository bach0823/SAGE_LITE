# SAGE-Lite Diagnostic & Inspection Utilities

This directory contains standalone diagnostic, verification, and inspection utilities for validating model components, auditing routing entropy, and generating annotations.

---

## 🛠️ Category Breakdown & Usage

### 1. Structural Verification & Auditing

* **`verify_model.py`**  
  Instantiates the model architecture locally, verifies module injection (Mixture of Local Experts layer wrapping), checks shape contracts across all skip connections, and executes a dummy forward pass to confirm that no dimension mismatches occur.
  ```bash
  python tools/verify_model.py
  ```

* **`inspect_checkpoint.py`**  
  Inspects parameter keys, tensor shapes, and metadata from saved `.pth` checkpoint files to ensure compatibility across stages and architectures.
  ```bash
  python tools/inspect_checkpoint.py \
      --checkpoint results/sage_b2_crack500/best_model_b2_global.pth
  ```

---

### 2. Routing Diagnostics & Visualization

* **`analyze_routing.py`**  
  Audits sample-level dynamic expert routing decisions, measuring routing entropy and expert selection frequency across the heterogeneous expert pool.
  ```bash
  python tools/analyze_routing.py \
      --config configs/sage_b2_crack500.yaml \
      --checkpoint results/sage_b2_crack500/best_model_b2_global.pth
  ```

* **`visualize_expert_routing_decisions.py`**  
  Generates spatial heatmaps indicating which local image regions and crack fissures are routed to specific CNN vs. ViT experts.
  ```bash
  python tools/visualize_expert_routing_decisions.py \
      --config configs/sage_b2_crack500.yaml \
      --checkpoint results/sage_b2_crack500/best_model_b2_global.pth
  ```

* **`visualize_gs_evolution.py`** / **`extract_gs_metrics.py`**  
  Visualizes the evolution of shared expert gating values ($g_s$) and load-balancing distributions over training epochs.
