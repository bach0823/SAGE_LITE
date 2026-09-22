import os
import sys
import shutil
import subprocess
import glob
import zipfile
import random

def prepare_deepcrack():
    print("="*60)
    print("=== DEEPCRACK Dataset Preparation Script ===")
    print("="*60)
    
    target_dir = "/content/dataset/DeepCrack"
    repo_dir = "/content/raw_data/DeepCrack_repo"
    raw_dir = "/content/raw_data/DeepCrack_extracted"
    
    is_local = False
    if not os.path.exists("/content"):
        is_local = True
        target_dir = os.path.abspath("temp_data/dataset/DeepCrack")
        repo_dir = os.path.abspath("temp_data/raw_data/DeepCrack_repo")
        raw_dir = os.path.abspath("temp_data/raw_data/DeepCrack_extracted")
        print(f"[NOTE] Not running on Colab. Redirecting to {target_dir}")
        
    os.makedirs(os.path.dirname(repo_dir), exist_ok=True)
    
    LOCKED_COMMIT = "8202a60"
    if not os.path.exists(repo_dir):
        print(f"\n1. Cloning upstream yhlleo/DeepCrack repo (Target commit: {LOCKED_COMMIT})...")
        subprocess.run(f"git clone https://github.com/yhlleo/DeepCrack.git {repo_dir}", shell=True, check=True)
        subprocess.run(f"git checkout {LOCKED_COMMIT}", cwd=repo_dir, shell=True, check=True)
    else:
        head_commit = subprocess.check_output("git rev-parse HEAD", cwd=repo_dir, shell=True, text=True).strip()
        if not head_commit.startswith(LOCKED_COMMIT):
            print(f"\n1. Repo exists but HEAD is {head_commit}. Hard resetting to {LOCKED_COMMIT}...")
            subprocess.run(f"git fetch --all && git reset --hard {LOCKED_COMMIT}", cwd=repo_dir, shell=True, check=True)
        else:
            print(f"\n1. Repo exists and HEAD verified at {LOCKED_COMMIT}.")
            
    zip_path = os.path.join(repo_dir, "dataset", "DeepCrack.zip")
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"FAIL-FAST: {zip_path} not found.")
        
    print("\n2. Extracting DeepCrack.zip...")
    if os.path.exists(raw_dir):
        shutil.rmtree(raw_dir)
    os.makedirs(raw_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(raw_dir)
        
    base_extracted = raw_dir
    if os.path.exists(os.path.join(raw_dir, "DeepCrack", "train_img")):
        base_extracted = os.path.join(raw_dir, "DeepCrack")
        
    print("\n3. Generating project-defined deterministic split (seed=42)...")
    print("   Upstream DeepCrack provides 300 train + 237 test images.")
    print("   Creating local split: Train=240, Val=60, Test=237.")
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
        
    train_img_dir_raw = os.path.join(base_extracted, "train_img")
    train_lab_dir_raw = os.path.join(base_extracted, "train_lab")
    test_img_dir_raw = os.path.join(base_extracted, "test_img")
    test_lab_dir_raw = os.path.join(base_extracted, "test_lab")
    
    disk_stems = sorted([os.path.splitext(f)[0] for f in os.listdir(train_img_dir_raw) if f.endswith(".jpg")])
    if len(disk_stems) != 300:
        raise ValueError(f"FAIL-FAST: Expected 300 train images, got {len(disk_stems)}")
        
    rng = random.Random(42)
    shuffled_indices = list(range(len(disk_stems)))
    rng.shuffle(shuffled_indices)
    
    train_stems = [disk_stems[i] for i in shuffled_indices[:240]]
    val_stems = [disk_stems[i] for i in shuffled_indices[240:]]
    
    test_stems = sorted([os.path.splitext(f)[0] for f in os.listdir(test_img_dir_raw) if f.endswith(".jpg")])
    if len(test_stems) != 237:
        raise ValueError(f"FAIL-FAST: Expected 237 test images, got {len(test_stems)}")
        
    def copy_split(stems, split_name, src_img_dir, src_lab_dir):
        img_out = os.path.join(target_dir, split_name, "images")
        mask_out = os.path.join(target_dir, split_name, "masks")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(mask_out, exist_ok=True)
        for s in stems:
            shutil.copy2(os.path.join(src_img_dir, s + ".jpg"), os.path.join(img_out, s + ".jpg"))
            shutil.copy2(os.path.join(src_lab_dir, s + ".png"), os.path.join(mask_out, s + ".png"))
            
    copy_split(train_stems, "train", train_img_dir_raw, train_lab_dir_raw)
    copy_split(val_stems, "val", train_img_dir_raw, train_lab_dir_raw)
    copy_split(test_stems, "test", test_img_dir_raw, test_lab_dir_raw)
    
    print("\n4. Running Sanity Checks...")
    expected_counts = {'train': 240, 'val': 60, 'test': 237}
    for split, exp in expected_counts.items():
        imgs = glob.glob(os.path.join(target_dir, split, 'images', '*.jpg'))
        if len(imgs) != exp:
            print(f"[FAIL] {split.upper()} count mismatch: Expected {exp}, Got {len(imgs)}")
            sys.exit(1)
        print(f"[OK] {split.upper()} count: {len(imgs)} (Matches expected)")

    print("\n5. Strict Mask Binary Values Check (Checking ALL masks)...")
    try:
        import cv2
        import numpy as np
    except ImportError as e:
        print(f"[FAIL] Dependency missing for validation: {e}. Please install opencv-python and numpy.")
        sys.exit(1)
        
    all_mask_files = glob.glob(os.path.join(target_dir, '*', 'masks', '*.png'))
    invalid_masks = []
    
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
        sys.exit(1)
    else:
        print(f"[OK] All {len(all_mask_files)} masks are strictly {{0, 255}} binary.")

    print("\n" + "="*60)
    print(f"=== [PASS] Dataset Preparation Completed at {target_dir} ===")
    print("="*60)

if __name__ == '__main__':
    prepare_deepcrack()
