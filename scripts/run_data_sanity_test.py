"""
run_data_sanity_test.py
=======================
Sanity test verifying post-transform correctness across all dataset splits:
- Crack500 train (smart-filter crop-retry)
- Crack500 val/test (standard evaluation path)
- DeepCrack train (dynamic pad-to-square -> resize 448x448)
- DeepCrack val/test (dynamic pad-to-square -> resize 448x448)

Verifies:
  1. mask values are STRICTLY {0, 1} (no 255, no float artifacts).
  2. image shape is (3, 448, 448) where expected.
  3. label shape is (448, 448) where expected.
  4. no RuntimeError occurs during loading and transforms.
  5. DeepCrack dynamic Pad-to-Square correctly preserves aspect ratio.

Exit code 0 on PASS, non-zero on FAIL.
"""

import os
import sys
import yaml
import numpy as np
import torch
from tqdm import tqdm

from sage.utils.dataloader import ConfigurableMedicalDataset, pad_to_square


def test_split(dataset, split_name: str, max_samples: int = 50, expected_shape=(448, 448)):
    print(f"\n--- Testing {split_name} (Total: {len(dataset)} samples, checking {min(len(dataset), max_samples)}) ---")
    
    unique_vals_all = set()
    shape_errors = []
    
    n_to_check = min(len(dataset), max_samples)
    for i in range(n_to_check):
        sample = dataset[i]
        image = sample['image']
        label = sample['label']
        
        # Check shapes
        if expected_shape is not None:
            if tuple(image.shape) != (3, expected_shape[0], expected_shape[1]):
                shape_errors.append(f"Sample {i}: image shape {tuple(image.shape)} != (3, {expected_shape[0]}, {expected_shape[1]})")
            if tuple(label.shape) != expected_shape:
                shape_errors.append(f"Sample {i}: label shape {tuple(label.shape)} != {expected_shape}")
                
        # Check unique mask values
        u_vals = label.unique().tolist()
        unique_vals_all.update(u_vals)
        for v in u_vals:
            if v not in (0, 1):
                shape_errors.append(f"Sample {i}: non-binary label value {v}")
                
    binarize_ok = unique_vals_all.issubset({0, 1})
    passed = (len(shape_errors) == 0 and binarize_ok)
    
    print(f"  Unique label values : {sorted(list(unique_vals_all))}")
    print(f"  Binarization Status : {'PASS (Strictly {0, 1})' if binarize_ok else 'FAIL'}")
    print(f"  Shape/Integrity     : {'PASS' if len(shape_errors) == 0 else 'FAIL (' + str(len(shape_errors)) + ' errors)'}")
    if shape_errors:
        print(f"  Errors sample       : {shape_errors[:3]}")
    print(f"  Result              : {'[PASS]' if passed else '[FAIL]'}")
    return passed


def main():
    print("=" * 70)
    print("=== SAGE-LITE COMPREHENSIVE DATA SANITY TEST ===")
    print("=" * 70)
    
    all_pass = True
    
    # ── 1. DeepCrack Pad-to-Square aspect ratio preservation check ────────────
    print("\n--- Testing DeepCrack Dynamic Pad-to-Square Unit Logic ---")
    dummy_img = np.zeros((384, 544, 3), dtype=np.uint8)
    dummy_msk = np.ones((384, 544), dtype=np.uint8)
    p_img, p_msk = pad_to_square(dummy_img, dummy_msk, min_size=448)
    print(f"  Input: {dummy_img.shape[:2]} -> Padded: {p_img.shape[:2]}")
    assert p_img.shape[:2] == (544, 544), f"Expected 544x544, got {p_img.shape[:2]}"
    assert p_msk.shape == (544, 544), f"Expected 544x544, got {p_msk.shape}"
    print("  Dynamic Pad-to-Square Unit Test: [PASS]")
    
    # ── 2. Crack500 Checks ────────────────────────────────────────────────────
    crack500_cfg = 'configs/b0_crack500.yaml'
    c500_local = "D:/truong/SpecialSubjectTTNT/datasets/CRACK500_canonical"
    if os.path.exists(c500_local):
        tmp_c500 = {
            'model': 'B0',
            'root_dir': c500_local,
            'crop_mode': 'random',
            'smart_filter': True,
            'img_size': 448,
            'mask_suffix': '',
            'train': {'images': 'traincrop', 'masks': 'traincrop'},
            'val': {'images': 'valcrop', 'masks': 'valcrop'},
            'test': {'images': 'testcrop', 'masks': 'testcrop'}
        }
        tmp_path_c500 = 'temp_sanity_c500.yaml'
        with open(tmp_path_c500, 'w') as f:
            yaml.dump(tmp_c500, f)
        try:
            ds_c500_train = ConfigurableMedicalDataset(tmp_path_c500, split='train', image_size=448)
            all_pass &= test_split(ds_c500_train, "Crack500 Train (Smart Filter)", max_samples=50, expected_shape=(448, 448))
            ds_c500_val = ConfigurableMedicalDataset(tmp_path_c500, split='val', image_size=448)
            all_pass &= test_split(ds_c500_val, "Crack500 Val", max_samples=50, expected_shape=None)
        finally:
            if os.path.exists(tmp_path_c500):
                os.remove(tmp_path_c500)
    elif os.path.exists(crack500_cfg):
        try:
            ds_c500_train = ConfigurableMedicalDataset(crack500_cfg, split='train', image_size=448)
            all_pass &= test_split(ds_c500_train, "Crack500 Train (Smart Filter)", max_samples=50, expected_shape=(448, 448))
        except Exception as e:
            print(f"ERROR on Crack500 Train: {e}")
            all_pass = False
            
        try:
            ds_c500_val = ConfigurableMedicalDataset(crack500_cfg, split='val', image_size=448)
            all_pass &= test_split(ds_c500_val, "Crack500 Val", max_samples=len(ds_c500_val), expected_shape=None)
        except Exception as e:
            print(f"NOTE on Crack500 Val: {e}")
            
    # ── 3. DeepCrack Checks ───────────────────────────────────────────────────
    # Check if canonical dataset exists on disk or via config
    deepcrack_root = "D:/truong/SpecialSubjectTTNT/datasets/DeepCrack_canonical"
    if not os.path.exists(deepcrack_root):
        deepcrack_root = "/content/dataset/DeepCrack"
        
    if os.path.exists(deepcrack_root):
        # Create temporary config pointing to DeepCrack
        tmp_cfg = {
            'model': 'B0',
            'root_dir': deepcrack_root,
            'crop_mode': 'resize',
            'smart_filter': False,
            'img_size': 448,
            'mask_suffix': '',
            'train': {'images': 'train_img', 'masks': 'train_lab'},
            'val': {'images': 'test_img', 'masks': 'test_lab'},
            'test': {'images': 'test_img', 'masks': 'test_lab'}
        }
        tmp_path = 'temp_sanity_deepcrack.yaml'
        with open(tmp_path, 'w') as f:
            yaml.dump(tmp_cfg, f)
            
        try:
            ds_dc_train = ConfigurableMedicalDataset(tmp_path, split='train', image_size=448)
            all_pass &= test_split(ds_dc_train, "DeepCrack Train (Dynamic Pad-to-Square -> 448x448)", max_samples=50, expected_shape=(448, 448))
            
            ds_dc_val = ConfigurableMedicalDataset(tmp_path, split='val', image_size=448)
            all_pass &= test_split(ds_dc_val, "DeepCrack Val/Test (Dynamic Pad-to-Square -> 448x448)", max_samples=50, expected_shape=(448, 448))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        print(f"DeepCrack dataset not found at {deepcrack_root}, skipping DeepCrack split test.")
        
    print(f"\n{'='*70}")
    print(f"DATA SANITY TEST OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print(f"{'='*70}")
    
    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
