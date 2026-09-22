import os
import sys
import shutil
import subprocess
import glob
import zipfile

def prepare_crack500():
    print("="*60)
    print("=== CRACK500 Dataset Preparation Script ===")
    print("="*60)
    
    target_dir = "/content/dataset/Crack500"
    raw_dir = "/content/raw_data"
    
    is_local = False
    if not os.path.exists("/content"):
        is_local = True
        target_dir = os.path.abspath("temp_data/dataset/Crack500")
        raw_dir = os.path.abspath("temp_data/raw_data")
        print(f"[NOTE] Not running on Colab. Redirecting to {target_dir}")
        
    os.makedirs(raw_dir, exist_ok=True)
    zip_path = os.path.join(raw_dir, "crack_datasets.zip")
    
    if not os.path.exists(zip_path):
        print("\n1. Downloading CRACK500 from author's Google Drive (~2GB)...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
        subprocess.run(f"gdown 13_vDYl54Mrd34dddX9w4ppAEiuWv4MlD -O {zip_path}", shell=True, check=True)
    else:
        print("\n1. ZIP file already exists. Skipping download.")
        
    print("\n2. Extracting Outer ZIP...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(raw_dir)
        
    print("\n3. Extracting Nested split ZIPs (traincrop.zip, valcrop.zip, testcrop.zip)...")
    nested_zips = ['traincrop.zip', 'valcrop.zip', 'testcrop.zip']
    for nested in nested_zips:
        found = False
        for root, _, files in os.walk(raw_dir):
            for f in files:
                if f.lower() == nested:
                    nested_path = os.path.join(root, f)
                    print(f"   -> Found and extracting {nested}...")
                    with zipfile.ZipFile(nested_path, 'r') as zf:
                        zf.extractall(root)
                    found = True
                    break
            if found:
                break
        if not found:
            raise FileNotFoundError(f"FAIL-FAST: Nested ZIP '{nested}' not found inside {zip_path}")
    
    print("\n4. Discovering and Pairing Dataset...")
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
        
    def process_split(split_kw, split_name):
        img_out = os.path.join(target_dir, split_name, "images")
        mask_out = os.path.join(target_dir, split_name, "masks")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(mask_out, exist_ok=True)
        
        img_dict, mask_dict = {}, {}
        
        for root, _, files in os.walk(raw_dir):
            path_lower = root.lower()
            if target_dir.lower() in path_lower:
                continue
            if split_kw in path_lower:
                for f in files:
                    ext = f.lower()
                    basename = os.path.splitext(f)[0]
                    filepath = os.path.join(root, f)
                    
                    if ext.endswith(('.jpg', '.jpeg')):
                        img_dict[basename] = filepath
                    elif ext.endswith(('.png', '.bmp', '.tif')):
                        clean_basename = basename
                        if clean_basename.endswith('_mask'):
                            clean_basename = clean_basename[:-5]
                        mask_dict[clean_basename] = filepath
                        
        paired = set(img_dict.keys()).intersection(set(mask_dict.keys()))
        if len(paired) == 0:
            raise ValueError(f"FAIL-FAST: 0 pairs found for {split_name}.")
            
        print(f"\n--- Processing Split: {split_name.upper()} ---")
        
        for b in sorted(list(paired)):
            src_img = img_dict[b]
            src_mask = mask_dict[b]
            img_ext = os.path.splitext(src_img)[1]
            
            dst_img = os.path.join(img_out, b + img_ext)
            dst_mask = os.path.join(mask_out, b + ".png")
            
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_mask, dst_mask)
            
        return paired

    train_pairs = process_split('train', 'train')
    val_pairs = process_split('val', 'val')
    test_pairs = process_split('test', 'test')
    
    print("\n5. Running Sanity Checks...")
    expected_counts = {'train': 1896, 'val': 348, 'test': 1124}
    actual_counts = {'train': len(train_pairs), 'val': len(val_pairs), 'test': len(test_pairs)}
    
    for split, exp in expected_counts.items():
        act = actual_counts[split]
        if act != exp:
            print(f"[FAIL] {split.upper()} count mismatch: Expected {exp}, Got {act}")
            sys.exit(1)
        print(f"[OK] {split.upper()} count: {act} (Matches expected)")
        
    overlap_train_test = train_pairs.intersection(test_pairs)
    if len(overlap_train_test) > 0:
        print(f"[WARNING] {len(overlap_train_test)} overlapping sample basenames detected between Train/Test.")
        print("[WARNING] The user has elected to leave these alone as per known Crack500 raw state.")

    print("\n6. Strict Mask Binary Values Check (Checking ALL masks)...")
    try:
        import cv2
        import numpy as np
    except ImportError as e:
        print(f"[FAIL] Dependency missing for validation: {e}. Please install opencv-python and numpy.")
        sys.exit(1)
        
    all_mask_files = glob.glob(os.path.join(target_dir, '*', 'masks', '*.png'))
    invalid_masks = []
    
    # Check all masks
    print(f"Scanning {len(all_mask_files)} masks for strict {{0, 255}} validation...")
    for mf in all_mask_files:
        mask = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            invalid_masks.append((mf, "Unreadable"))
            continue
        unique_vals = set(np.unique(mask).tolist())
        expected_vals = {0, 255}
        if not unique_vals.issubset(expected_vals):
            invalid_masks.append((mf, str(unique_vals)))
            
    if invalid_masks:
        print(f"[FAIL] {len(invalid_masks)} masks contain unexpected values!")
        for mf, err in invalid_masks[:10]:
            print(f"  - {mf}: {err}")
        if len(invalid_masks) > 10:
            print("  ... and more.")
        sys.exit(1)
    else:
        print(f"[OK] All {len(all_mask_files)} masks are strictly {{0, 255}} binary.")

    print("\n" + "="*60)
    print(f"=== [PASS] Dataset Preparation Completed at {target_dir} ===")
    print("="*60)

if __name__ == '__main__':
    prepare_crack500()
