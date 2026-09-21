import os
import yaml
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from sage.utils.dataloader import get_dataset_from_config
from sage.utils.training_utils import set_seed

def check_dataset():
    config_path = "configs/b0_crack500.yaml"
    print(f"Loading config from {config_path}")
    
    # 4. Read samples using the actual dataset class.
    print("\n--- 1. Testing Dataset Initialization ---")
    train_dataset = get_dataset_from_config(config_path, split='train')
    val_dataset = get_dataset_from_config(config_path, split='val')
    test_dataset = get_dataset_from_config(config_path, split='test')
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size:   {len(val_dataset)}")
    print(f"Test size:  {len(test_dataset)}")
    
    # 5. Verify original spatial size before preprocessing
    print("\n--- 2. Verifying Raw Spatial Sizes ---")
    samples_to_check = train_dataset.samples[:20]  # Take first 20 for quick check
    mismatch_count = 0
    for s in samples_to_check:
        img = cv2.imread(s['image'])
        mask = cv2.imread(s['label'], cv2.IMREAD_GRAYSCALE)
        if img.shape[:2] != mask.shape[:2]:
            print(f"[ERROR] Size mismatch: {s['case_name']} -> Img: {img.shape}, Mask: {mask.shape}")
            mismatch_count += 1
    
    if mismatch_count == 0:
        print(f"[OK] Checked {len(samples_to_check)} raw samples. All spatial dimensions match perfectly.")
    else:
        raise ValueError(f"Found {mismatch_count} spatial size mismatches!")
        
    # 6. Verify mask is binary/foreground
    print("\n--- 3. Verifying Binary Mask Properties ---")
    mask_vals = set()
    for s in samples_to_check:
        mask = cv2.imread(s['label'], cv2.IMREAD_GRAYSCALE)
        mask_vals.update(np.unique(mask).tolist())
    
    print(f"Unique pixel values in raw masks: {mask_vals}")
    
    # 8 & 9. DataLoader Check
    print("\n--- 4. Verifying DataLoader & Batches ---")
    loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    try:
        batch = next(iter(loader))
        images = batch['image']
        labels = batch['label']
        print(f"Batch images shape: {images.shape}, dtype: {images.dtype}")
        print(f"Batch labels shape: {labels.shape}, dtype: {labels.dtype}")
        
        # Check NaN/Inf
        assert not torch.isnan(images).any(), "NaN found in images!"
        assert not torch.isinf(images).any(), "Inf found in images!"
        assert not torch.isnan(labels).any(), "NaN found in labels!"
        
        # Check min/max
        print(f"Images Min: {images.min().item():.4f}, Max: {images.max().item():.4f}")
        print(f"Labels Min: {labels.min().item()}, Max: {labels.max().item()}")
        
        # Verify img_size matches config (448)
        assert images.shape[2:] == (448, 448), f"Expected 448x448, got {images.shape[2:]}"
        assert labels.shape[1:] == (448, 448), f"Expected 448x448 label, got {labels.shape[1:]}"
        
        print("[OK] DataLoader successfully created valid batches with correct shapes/dtypes.")
    except Exception as e:
        print(f"[ERROR] DataLoader failed: {e}")
        raise e
        
    # 7. Visualize 3-5 image/mask pairs
    print("\n--- 5. Generating Visualizations ---")
    num_vis = 4
    fig, axes = plt.subplots(num_vis, 3, figsize=(12, 4*num_vis))
    for i in range(num_vis):
        item = train_dataset[i]
        img_tensor = item['image']
        mask_tensor = item['label']
        name = item['case_name']
        
        # Denormalize image for viewing
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        
        mask_np = mask_tensor.numpy()
        
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"Image: {name}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask_np, cmap='gray')
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis('off')
        
        # Overlay
        overlay = img_np.copy()
        # Red tint on crack pixels
        overlay[mask_np > 0] = [1.0, 0.0, 0.0] 
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title("Overlay")
        axes[i, 2].axis('off')
        
    plt.tight_layout()
    out_path = "results/dataset_sanity_check.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"[OK] Saved visualization to {out_path}")
    print("\n[SUCCESS] Dataset Sanity Check PASSED! You may now proceed with the pilot benchmark.")

if __name__ == '__main__':
    check_dataset()
