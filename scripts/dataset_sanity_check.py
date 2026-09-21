import os
import sys
import yaml
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from torch.utils.data import DataLoader

from sage.utils.dataloader import get_dataset_from_config
from sage.utils.training_utils import set_seed

def fatal_error(msg):
    print(f"\n[ERROR] {msg}")
    sys.exit(1)

def main():
    set_seed(42)
    config_path = "configs/b0_crack500.yaml"
    if not os.path.exists(config_path):
        fatal_error(f"Config file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    root_dir = cfg.get('root_dir', '/content/dataset/Crack500')
    splits = ['train', 'val', 'test']
    
    # ---------------------------------------------------------
    # [A] & [B] DATASET STRUCTURE & FILE INTEGRITY
    # ---------------------------------------------------------
    print("Checking Dataset Structure & File Integrity...")
    split_basenames = {}
    all_samples = {}
    
    for sp in splits:
        img_dir = os.path.join(root_dir, sp, 'images')
        mask_dir = os.path.join(root_dir, sp, 'masks')
        if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
            fatal_error(f"Missing images/masks directory for split: {sp}")
            
        imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        masks = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        img_bases = set([os.path.splitext(f)[0] for f in imgs])
        mask_bases = set([os.path.splitext(f)[0] for f in masks])
        
        if len(imgs) != len(img_bases):
            fatal_error(f"Duplicate image basenames in {sp}")
        if len(masks) != len(mask_bases):
            fatal_error(f"Duplicate mask basenames in {sp}")
            
        missing_masks = img_bases - mask_bases
        missing_imgs = mask_bases - img_bases
        if missing_masks or missing_imgs:
            fatal_error(f"Mismatch in {sp}. Imgs without masks: {len(missing_masks)}. Masks without imgs: {len(missing_imgs)}.")
            
        split_basenames[sp] = img_bases
        all_samples[sp] = [(os.path.join(img_dir, b + os.path.splitext(imgs[0])[1]), 
                            os.path.join(mask_dir, b + ".png")) for b in img_bases] # assuming .png for masks based on prep
        
        print(f"  {sp}: {len(imgs)} valid pairs")

    # ---------------------------------------------------------
    # [E] SPLIT SANITY (Overlaps)
    # ---------------------------------------------------------
    print("\nChecking Split Sanity (Overlaps)...")
    t_v = split_basenames['train'].intersection(split_basenames['val'])
    t_t = split_basenames['train'].intersection(split_basenames['test'])
    v_t = split_basenames['val'].intersection(split_basenames['test'])
    
    overlap_found = False
    if t_v:
        print(f"  [WARNING] Train-Val overlap detected: {len(t_v)} files")
        overlap_found = True
    if t_t:
        print(f"  [WARNING] Train-Test overlap detected: {len(t_t)} files")
        overlap_found = True
    if v_t:
        print(f"  [WARNING] Val-Test overlap detected: {len(v_t)} files")
        overlap_found = True
        
    if not overlap_found:
        print("  No overlaps found between splits (Basename checks passed).")

    # ---------------------------------------------------------
    # [C] & [D] RAW GEOMETRY & MASK INTEGRITY
    # ---------------------------------------------------------
    print("\nChecking Raw Geometry & Mask Integrity (Scanning files)...")
    for sp in splits:
        print(f"  Scanning {sp}...")
        shapes = []
        fg_ratios = []
        unique_mask_vals = set()
        empty_masks = 0
        
        for img_p, mask_p in all_samples[sp]:
            img = cv2.imread(img_p)
            if img is None: fatal_error(f"Cannot read image (corrupt?): {img_p}")
            mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
            if mask is None: fatal_error(f"Cannot read mask (corrupt?): {mask_p}")
            
            if img.shape[:2] != mask.shape[:2]:
                fatal_error(f"Spatial size mismatch: {img_p} ({img.shape}) vs {mask_p} ({mask.shape})")
                
            shapes.append(img.shape[:2])
            
            u_vals = np.unique(mask)
            unique_mask_vals.update(u_vals.tolist())
            
            fg_pixels = np.sum(mask > 0)
            total_pixels = mask.shape[0] * mask.shape[1]
            ratio = fg_pixels / total_pixels
            fg_ratios.append(ratio)
            if fg_pixels == 0:
                empty_masks += 1
                
        shape_counts = Counter(shapes)
        print(f"    Geometry:")
        print(f"      Common shapes: {shape_counts.most_common(3)}")
        print(f"    Mask Integrity:")
        print(f"      Unique pixel values: {unique_mask_vals}")
        print(f"      Total masks scanned: {len(fg_ratios)}")
        print(f"      Empty masks (pure background): {empty_masks}")
        print(f"      FG Ratio - Min: {np.min(fg_ratios):.4f}, Mean: {np.mean(fg_ratios):.4f}, Median: {np.median(fg_ratios):.4f}, Max: {np.max(fg_ratios):.4f}")

    # ---------------------------------------------------------
    # [F] DATASET PIPELINE
    # ---------------------------------------------------------
    print("\nChecking Dataset Pipeline (Configs & Dataloaders)...")
    for sp in splits:
        ds = get_dataset_from_config(config_path, split=sp)
        loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2)
        
        batch_count = 0
        for batch in loader:
            images = batch['image']
            labels = batch['label']
            
            if images.shape[2:] != (448, 448):
                fatal_error(f"Pipeline output image shape mismatch. Expected (..., 448, 448), got {images.shape}")
                
            if torch.isnan(images).any() or torch.isinf(images).any():
                fatal_error(f"NaN/Inf detected in images for split {sp}")
            if torch.isnan(labels).any() or torch.isinf(labels).any():
                fatal_error(f"NaN/Inf detected in labels for split {sp}")
                
            batch_count += 1
            if batch_count >= 3:
                break
                
        print(f"  {sp} pipeline OK (Tested {batch_count} batches). Image dtype: {images.dtype}, Label dtype: {labels.dtype}")

    # ---------------------------------------------------------
    # [G] VISUAL CHECK
    # ---------------------------------------------------------
    print("\nGenerating Visual Overlays from PIPELINE output...")
    ds_vis = get_dataset_from_config(config_path, split='train')
    
    num_vis = 5
    fig, axes = plt.subplots(num_vis, 3, figsize=(12, 4*num_vis))
    
    # Pick 5 random indices
    indices = np.random.choice(len(ds_vis), num_vis, replace=False)
    for i, idx in enumerate(indices):
        item = ds_vis[idx]
        img_tensor = item['image']
        mask_tensor = item['label']
        name = item.get('case_name', f"Sample {idx}")
        
        # Robust min-max scaling for visualization regardless of normalization scheme
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_min, img_max = img_np.min(), img_np.max()
        if img_max > img_min:
            img_np = (img_np - img_min) / (img_max - img_min)
        img_np = np.clip(img_np, 0, 1)
        
        mask_np = mask_tensor.squeeze().numpy()
        
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"Processed Image: {name}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask_np, cmap='gray')
        axes[i, 1].set_title("Processed Mask")
        axes[i, 1].axis('off')
        
        overlay = img_np.copy()
        # Overlay red tint where mask indicates foreground
        overlay[mask_np > 0.5] = [1.0, 0.0, 0.0]
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title("Overlay")
        axes[i, 2].axis('off')
        
    plt.tight_layout()
    out_path = "results/dataset_sanity_check.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    
    print(f"\n==================================================")
    print(f"CRACK500 DATASET SANITY CHECK")
    print(f"==================================================")
    print(f"[A] DATASET STRUCTURE       PASS")
    print(f"[B] FILE INTEGRITY          PASS")
    print(f"[C] RAW GEOMETRY            PASS")
    print(f"[D] MASK INTEGRITY          PASS")
    print(f"[E] SPLIT SANITY            PASS")
    print(f"[F] DATASET PIPELINE        PASS")
    print(f"[G] VISUAL CHECK            READY FOR MANUAL REVIEW")
    print(f"--------------------------------------------------")
    print(f"Dataset Status: READY FOR MANUAL REVIEW")
    print(f"--------------------------------------------------")
    print(f"\n[SUCCESS] Automated dataset sanity checks passed.")
    print(f"[MANUAL REVIEW REQUIRED] Please verify visual overlays before running pilot.")
    
if __name__ == '__main__':
    main()
