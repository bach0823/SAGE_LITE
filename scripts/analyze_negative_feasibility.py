import os
import sys
import numpy as np
import cv2
from tqdm import tqdm

from sage.utils.dataloader import get_dataset_from_config

def run_feasibility_analysis():
    print("="*60)
    print("=== CRACK500 NEGATIVE-CROP FEASIBILITY ANALYSIS ===")
    print("="*60)
    
    config_path = "configs/b0_crack500.yaml"
    if not os.path.exists(config_path):
        print(f"ERROR: {config_path} not found. Run from repo root.")
        return
        
    ds = get_dataset_from_config(config_path, split='train', image_size=448)
    print(f"Loaded {len(ds.samples)} training samples.")
    
    crop_transform = ds.crop_transform
    if not crop_transform:
        print("ERROR: No crop_transform found in dataset.")
        return
        
    N_ATTEMPTS = 100
    p_neg_list = []
    
    print(f"\nScanning {len(ds.samples)} images. Generating {N_ATTEMPTS} crops per image...")
    
    for i in tqdm(range(len(ds.samples)), desc="Analyzing Negative Feasibility"):
        sample = ds.samples[i]
        image = cv2.imread(sample['image'])
        mask = cv2.imread(sample['label'], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8)
        
        neg_count = 0
        for _ in range(N_ATTEMPTS):
            cropped = crop_transform(image=image, mask=mask)
            fg = (cropped['mask'] > 0).sum()
            if fg < 20:
                neg_count += 1
                
        p_neg = neg_count / N_ATTEMPTS
        p_neg_list.append(p_neg)
        
    p_neg_array = np.array(p_neg_list)
    zero_prob_count = (p_neg_array == 0).sum()
    total_imgs = len(p_neg_array)
    
    print("\n" + "-"*40)
    print("--- FEASIBILITY DISTRIBUTION ---")
    print("-" * 40)
    print(f"Total source images: {total_imgs}")
    print(f"Images with NO negative crops (in 100) : {zero_prob_count} ({zero_prob_count/total_imgs*100:.1f}%)")
    print(f"Images with >= 1 negative crops      : {total_imgs - zero_prob_count} ({(total_imgs - zero_prob_count)/total_imgs*100:.1f}%)")
    
    print(f"\nProbability of getting a negative crop from a randomly chosen image:")
    print(f"  Mean   : {np.mean(p_neg_array):.4f}")
    print(f"  Median : {np.median(p_neg_array):.4f}")
    print(f"  Min    : {np.min(p_neg_array):.4f}")
    print(f"  Max    : {np.max(p_neg_array):.4f}")
    
    print("\n" + "-"*40)
    print("--- EXPECTED FAILURE PROBABILITIES ---")
    print("-" * 40)
    # The probability of failing to find a negative crop on image i after 20 attempts
    fail_1_source_probs = (1.0 - p_neg_array) ** 20
    
    # Expected probability of failing 1 random source is the mean across all images
    p_fail_1 = np.mean(fail_1_source_probs)
    
    print(f"P(fail 1 source | 20 crops)   : {p_fail_1 * 100:.4f}%")
    print(f"P(fail 2 sources | 40 crops)  : {(p_fail_1**2) * 100:.4f}%")
    print(f"P(fail 5 sources | 100 crops) : {(p_fail_1**5) * 100:.6f}%")
    print(f"P(fail 10 sources | 200 crops): {(p_fail_1**10) * 100:.8f}%")
    
    print("\nCONCLUSION:")
    if (p_fail_1**10) > 0.001:
        print(">> HIGH RISK: MAX_RESAMPLE=10 is statistically insufficient for NEGATIVE targets.")
    else:
        print(">> LOW RISK: Statistically, MAX_RESAMPLE=10 should safely cover NEGATIVE targets.")

if __name__ == '__main__':
    run_feasibility_analysis()
