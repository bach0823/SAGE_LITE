# SAGE-Lite Utilities Module (`sage.utils`)

This package provides foundational utilities supporting dataset loading, metric computation, training routines, checkpoint serialization, and visualization across the SAGE-Lite framework.

---

## 📦 Submodule Overview

| Module | Core Functions / Classes | Description |
| :--- | :--- | :--- |
| **`dataloader.py`** | `ConfigurableCrackDataset`, `get_dataset_from_config` | Dataset reader supporting paired images and binary masks with custom augmentations. |
| **`metrics.py`** | `calculate_iou`, `calculate_dice_coefficient`, `calculate_pixel_accuracy` | Fast batch-wise semantic segmentation metrics. |
| **`advanced_metrics.py`** | `calculate_boundary_iou`, `calculate_cl_dice` | Advanced topological boundary and connectivity evaluation functions. |
| **`training_utils.py`** | `setup_logging`, `set_seed`, `seed_worker`, `EarlyStopping` | Deterministic seeding, file/stream logging handlers, and early stopping. |
| **`model_utils.py`** | `count_parameters`, `save_checkpoint`, `load_checkpoint` | Parameter auditing and safe state dictionary serialization. |
| **`evaluation.py`** | `evaluate_model`, `generate_evaluation_report` | Multi-threshold evaluation and aggregated metric reporting. |
| **`visualization.py`** | `visualize_predictions`, `save_training_plots` | Side-by-side visualization of input, ground truth, and predicted masks. |
| **`gs_tracker.py`** | `GsTracker` | Internal accumulator tracking shared gating distributions and routing metadata. |

---

## 💡 Usage Example

```python
import torch
from sage.utils.metrics import calculate_iou, calculate_dice_coefficient
from sage.utils.training_utils import setup_logging, set_seed, EarlyStopping
from sage.utils.model_utils import count_parameters, save_checkpoint

# 1. Reproducibility & Logging
set_seed(42)
logger = setup_logging(output_dir="results/run1", experiment_name="b2_crack500")

# 2. Parameter Audit
total_params, trainable_params = count_parameters(model)
logger.info(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

# 3. Validation Metrics
preds = (torch.sigmoid(logits) > 0.5).float()
iou = calculate_iou(preds, targets)
dice = calculate_dice_coefficient(preds, targets)

# 4. Checkpoint Management
early_stopping = EarlyStopping(patience=8, path="best_model.pth", metric_name="IoU", mode="max")
```
