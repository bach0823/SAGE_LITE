import argparse
import os
import sys
import yaml
import glob
import cv2
import numpy as np
import torch
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sage.networks import create_b0_unet
from sage.utils.training_utils import set_seed

def get_image_mask_pairs(config, split):
    root_dir = config.get('root_dir', '')
    split_config = config.get(split)
    if not split_config:
        return []
        
    img_dir = split_config['images']
    mask_dir = split_config['masks']
    
    if root_dir and not os.path.isabs(img_dir):
        img_dir = os.path.join(root_dir, img_dir)
    if root_dir and not os.path.isabs(mask_dir):
        mask_dir = os.path.join(root_dir, mask_dir)
        
    img_paths = sorted(glob.glob(os.path.join(img_dir, '*')))
    
    # Filter valid extensions
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    img_paths = [p for p in img_paths if os.path.splitext(p)[1].lower() in valid_exts]
    
    pairs = []
    mask_suffix = config.get('mask_suffix', '')
    for img_p in img_paths:
        stem = os.path.splitext(os.path.basename(img_p))[0]
        # In Crack500, masks are usually exactly the same name or have a suffix, and typically .png
        # Try to find corresponding mask
        found_mask = None
        for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            test_mask = os.path.join(mask_dir, stem + mask_suffix + ext)
            if os.path.exists(test_mask):
                found_mask = test_mask
                break
        if found_mask:
            pairs.append((img_p, found_mask))
            
    return pairs

def predict_full_image_tiling(model, image, device, tile_size=448, batch_size=16):
    H, W = image.shape[:2]
    
    # Calculate padding
    pad_h = (tile_size - (H % tile_size)) % tile_size
    pad_w = (tile_size - (W % tile_size)) % tile_size
    
    padded_img = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    pH, pW = padded_img.shape[:2]
    
    # Extract patches
    patches = []
    coords = []
    for y in range(0, pH, tile_size):
        for x in range(0, pW, tile_size):
            patch = padded_img[y:y+tile_size, x:x+tile_size]
            # Normalize exactly like training: ToTensor + div 255 (Albumentations default for ToTensorV2)
            patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
            patches.append(patch_tensor)
            coords.append((y, x))
            
    # Predict in batches
    pred_mask = np.zeros((pH, pW), dtype=np.float32)
    
    for i in range(0, len(patches), batch_size):
        batch = torch.stack(patches[i:i+batch_size]).to(device, non_blocking=True)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                logits = model(batch)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            
        for j, prob in enumerate(probs):
            y, x = coords[i+j]
            pred_mask[y:y+tile_size, x:x+tile_size] = prob
            
    # Crop to original size
    pred_mask = pred_mask[:H, :W]
    return (pred_mask > 0.5).astype(np.uint8)

def evaluate_split(model, pairs, device, tile_size=448):
    metrics = {'precision': [], 'recall': [], 'dice': [], 'iou': []}
    
    for img_p, mask_p in tqdm(pairs, desc="Evaluating"):
        img = cv2.imread(img_p)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        if mask is None: continue
        target = (mask > 0).astype(np.uint8)
        
        pred = predict_full_image_tiling(model, img, device, tile_size=tile_size)
        
        tp = np.sum((pred == 1) & (target == 1))
        fp = np.sum((pred == 1) & (target == 0))
        fn = np.sum((pred == 0) & (target == 1))
        
        precision = tp / (tp + fp + 1e-5)
        recall = tp / (tp + fn + 1e-5)
        dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-5)
        iou = tp / (tp + fp + fn + 1e-5)
        
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['dice'].append(dice)
        metrics['iou'].append(iou)
        
    return {k: np.mean(v) for k, v in metrics.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = create_b0_unet(pretrained=False).to(device)
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
        
    tile_size = config.get('img_size', 448)
    print(f"Using non-overlapping tiling with tile_size={tile_size}x{tile_size}")
    
    for split in ['val', 'test']:
        pairs = get_image_mask_pairs(config, split)
        if not pairs:
            continue
            
        print(f"\n--- Evaluating Official Protocol on {split.upper()} Set ({len(pairs)} images) ---")
        res = evaluate_split(model, pairs, device, tile_size=tile_size)
        
        print(f"{split.upper()} Precision: {res['precision']:.4f}")
        print(f"{split.upper()} Recall:    {res['recall']:.4f}")
        print(f"{split.upper()} Dice/F1:   {res['dice']:.4f}")
        print(f"{split.upper()} IoU:       {res['iou']:.4f}")

if __name__ == '__main__':
    main()
