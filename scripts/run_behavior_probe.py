import torch
import sys
import os
import random
import cv2
import numpy as np
from tqdm import tqdm

from sage.utils.dataloader import get_dataset_from_config

def run_real_probe():
    print("="*60)
    print("=== REAL DATALOADER BEHAVIOR PROBE (CRACK500) ===")
    print("="*60)
    
    config_path = "configs/b0_crack500.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"{config_path} not found. Must run from repo root.")
        
    print(f"1. Loading real dataset using {config_path}...")
    try:
        ds = get_dataset_from_config(config_path, split='train', image_size=448)
        print(f"   -> Successfully loaded {len(ds)} training samples.")
    except Exception as e:
        print(f"Error loading dataset: {e}. If dataset is missing, please ensure prepare_crack500.py was run.")
        return
    
    print("\n2. Injecting Telemetry into ConfigurableMedicalDataset...")
    
    metrics = {
        'accepted_positive': 0,
        'accepted_negative': 0,
        'retry_counts': [],
        'source_resamples': 0,
        'hit_max_resample': 0,
        'runtime_errors': 0,
        'accepted_fg_pixels': [],
        'shape_valid': True
    }
    
    # We redefine the method dynamically to capture the loops exactly as requested
    def probed_getitem(self, idx):
        MAX_RESAMPLE = 10
        import random
        
        # 1. Tạo target 1 lần duy nhất cho toàn bộ request này
        is_positive_target = random.random() < 0.85
        
        for source_try in range(MAX_RESAMPLE):
            sample = self.samples[idx]
            
            image = cv2.imread(sample['image'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(sample['label'], cv2.IMREAD_GRAYSCALE)
            mask = (mask > 0).astype(np.uint8)
            
            if self.split == 'train' and self.crop_transform and self.use_smart_filter:
                success = False
                for r in range(20):
                    cropped = self.crop_transform(image=image, mask=mask)
                    fg_pixels = (cropped['mask'] > 0).sum()
                    has_crack = fg_pixels >= 20
                    
                    if is_positive_target and has_crack:
                        success = True; break
                    elif not is_positive_target and not has_crack:
                        success = True; break
                        
                if not success:
                    # METRIC TRACKING
                    metrics['source_resamples'] += 1
                    
                    idx = random.randint(0, len(self.samples) - 1)
                    continue
                    
                # METRIC TRACKING
                if is_positive_target:
                    metrics['accepted_positive'] += 1
                else:
                    metrics['accepted_negative'] += 1
                    
                metrics['retry_counts'].append(r)
                metrics['accepted_fg_pixels'].append(fg_pixels)
                
                augmented = self.aug_transform(image=cropped['image'], mask=cropped['mask'])
                image_out = augmented['image']
                label_out = augmented['mask']
                
                if label_out.dtype != torch.long:
                    label_out = label_out.long()
                    
                # METRIC TRACKING
                if image_out.shape != torch.Size([3, 448, 448]) or label_out.shape != torch.Size([448, 448]):
                    metrics['shape_valid'] = False
                    
                return {'image': image_out, 'label': label_out, 'case_name': sample['case_name']}
                
            else:
                raise RuntimeError("Smart filter should be active for Crack500 probe!")
                
        metrics['hit_max_resample'] += 1
        metrics['runtime_errors'] += 1
        raise RuntimeError(f"FAIL-FAST: Exceeded {MAX_RESAMPLE} source resampling attempts in dataloader.")
        
    ds.__class__.__getitem__ = probed_getitem
    
    PROBE_SIZE = 3000
    print(f"\n3. Executing Real Behavior Probe ({PROBE_SIZE} requests)...")
    for i in tqdm(range(PROBE_SIZE)):
        try:
            sample_idx = random.randint(0, len(ds) - 1)
            ds[sample_idx]
        except RuntimeError as e:
            if "FAIL-FAST" not in str(e):
                raise e

    print("\n" + "="*60)
    print("=== REAL BEHAVIOR PROBE RESULTS ===")
    print("="*60)
    print(f"Total simulated batches requested: {PROBE_SIZE}")
    
    total_accepted = metrics['accepted_positive'] + metrics['accepted_negative']
    if total_accepted > 0:
        pos_pct = (metrics['accepted_positive'] / total_accepted) * 100
        neg_pct = (metrics['accepted_negative'] / total_accepted) * 100
    else:
        pos_pct = neg_pct = 0
        
    print(f"Final accepted positive : {metrics['accepted_positive']} ({pos_pct:.1f}%)")
    print(f"Final accepted negative : {metrics['accepted_negative']} ({neg_pct:.1f}%)")
    
    avg_retry = np.mean(metrics['retry_counts']) if metrics['retry_counts'] else 0
    print(f"Average retries per crop: {avg_retry:.2f}")
    print(f"Total source resamples  : {metrics['source_resamples']}")
    print(f"Times hitting MAX_RESAMPLE: {metrics['hit_max_resample']}")
    print(f"Total RuntimeErrors     : {metrics['runtime_errors']}")
    print(f"All output shapes exactly (3,448,448) & (448,448): {metrics['shape_valid']}")
    
    if metrics['accepted_fg_pixels']:
        fg = metrics['accepted_fg_pixels']
        print(f"\nForeground Distribution of ACCEPTED crops:")
        print(f"  Min: {np.min(fg)}")
        print(f"  Max: {np.max(fg)}")
        print(f"  Mean: {np.mean(fg):.1f}")
        print(f"  Median: {np.median(fg):.1f}")
        print(f"  < 20 px: {sum(1 for x in fg if x < 20)}")
        print(f"  >= 20 px: {sum(1 for x in fg if x >= 20)}")

if __name__ == '__main__':
    run_real_probe()
