import os
import sys
import torch
import random
import numpy as np
import cv2
import traceback
from torch.utils.data import DataLoader
from tqdm import tqdm

from sage.utils.dataloader import get_dataset_from_config

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def run_test(num_workers):
    print("\n" + "="*70)
    print(f"=== DATALOADER SMOKE TEST (num_workers={num_workers}) ===")
    print("="*70)
    
    config_path = "configs/b0_crack500.yaml"
    if not os.path.exists(config_path):
        print(f"ERROR: {config_path} not found. Run from repo root.")
        return
        
    ds = get_dataset_from_config(config_path, split='train', image_size=448)
    print(f"Loaded {len(ds)} samples.")
    
    # Monkey-patch __getitem__ to return telemetry
    def probed_getitem(self, idx):
        MAX_RESAMPLE = 10
        import random
        import torch
        
        is_positive_target = random.random() < 0.85
        
        metrics = {
            'pos': int(is_positive_target),
            'neg': int(not is_positive_target),
            'retries': 0,
            'resamples': 0,
            'fg': 0
        }
        
        original_idx = idx
        
        for source_try in range(MAX_RESAMPLE):
            sample = self.samples[idx]
            
            image = cv2.imread(sample['image'])
            if image is None:
                raise ValueError(f"Failed to read image: {sample['image']}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            mask = cv2.imread(sample['label'], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to read mask: {sample['label']}")
            mask = (mask > 0).astype(np.uint8)
            
            if self.split == 'train' and self.crop_transform and self.use_smart_filter:
                success = False
                for r in range(20):
                    cropped = self.crop_transform(image=image, mask=mask)
                    fg = (cropped['mask'] > 0).sum()
                    if is_positive_target and fg >= 20:
                        success = True; break
                    elif not is_positive_target and fg < 20:
                        success = True; break
                
                if not success:
                    metrics['resamples'] += 1
                    idx = random.randint(0, len(self.samples) - 1)
                    continue
                    
                metrics['retries'] = r
                metrics['fg'] = fg
                
                augmented = self.aug_transform(image=cropped['image'], mask=cropped['mask'])
                img_out = augmented['image']
                lbl_out = augmented['mask']
                if lbl_out.dtype != torch.long: lbl_out = lbl_out.long()
                
                return {'image': img_out, 'label': lbl_out, 'metrics': metrics}
                
        worker_info = torch.utils.data.get_worker_info()
        wid = worker_info.id if worker_info else 'main'
        target_str = 'POSITIVE' if is_positive_target else 'NEGATIVE'
        raise RuntimeError(f"FAIL-FAST: Exceeded {MAX_RESAMPLE} source resampling attempts. Worker={wid}, Target={target_str}, Orig_Idx={original_idx}, Stuck_Idx={idx}")
        
    ds.__class__.__getitem__ = probed_getitem
    
    g = torch.Generator()
    g.manual_seed(42)
    
    dl = DataLoader(
        ds,
        batch_size=16,
        num_workers=num_workers,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=False
    )
    
    total_batches = 0
    total_pos = 0
    total_neg = 0
    total_resamples = 0
    retries = []
    
    try:
        for batch_idx, batch in enumerate(tqdm(dl, desc=f"Testing num_workers={num_workers}")):
            metrics = batch['metrics']
            total_pos += metrics['pos'].sum().item()
            total_neg += metrics['neg'].sum().item()
            total_resamples += metrics['resamples'].sum().item()
            
            # handle flat lists from default_collate
            for r in metrics['retries']:
                retries.append(r.item())
                
            total_batches += 1
            if total_batches >= 200:  # ~3200 samples
                break
                
        print(f"\nSUCCESS: Completed {total_batches} batches without FAIL-FAST.")
        total_samples = total_pos + total_neg
        if total_samples > 0:
            print(f"Accepted Target Ratio: POS={total_pos} ({total_pos/total_samples*100:.1f}%), NEG={total_neg} ({total_neg/total_samples*100:.1f}%)")
        print(f"Total Source Resamples : {total_resamples}")
        print(f"Average Retries/Sample : {np.mean(retries):.2f}")
        
    except Exception as e:
        print(f"\n[!] FAILED at batch {total_batches}!")
        print(f"Exception Message: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    # Run num_workers=4 first (reproduce training failure)
    run_test(num_workers=4)
    # Run num_workers=0 next (isolate multiprocessing vs policy intrinsic)
    run_test(num_workers=0)
