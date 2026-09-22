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

from sage.networks import create_b0_unet, create_b1_unet, create_b2_unet
from sage.utils.training_utils import set_seed
from scripts.train_crack import CrackBinaryLoss

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
    
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    img_paths = [p for p in img_paths if os.path.splitext(p)[1].lower() in valid_exts]
    
    pairs = []
    mask_suffix = config.get('mask_suffix', '')
    for img_p in img_paths:
        stem = os.path.splitext(os.path.basename(img_p))[0]
        found_mask = None
        for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            test_mask = os.path.join(mask_dir, stem + mask_suffix + ext)
            if os.path.exists(test_mask):
                found_mask = test_mask
                break
        if found_mask:
            pairs.append((img_p, found_mask))
            
    return pairs

from sage.utils.advanced_metrics import calculate_hd95_bf1

def compute_boundary_iou(pred: np.ndarray, target: np.ndarray, d: int = 2) -> float:
    """
    Computes Boundary IoU (Cheng et al., 2021) with boundary dilation radius d.
    """
    if np.sum(target) == 0 and np.sum(pred) == 0:
        return 1.0
    if np.sum(target) == 0 or np.sum(pred) == 0:
        return 0.0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * d + 1, 2 * d + 1))
    gt_boundary = cv2.morphologyEx(target.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    pred_boundary = cv2.morphologyEx(pred.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0

    gt_region = gt_boundary & (target > 0)
    pred_region = pred_boundary & (pred > 0)

    intersection = np.sum(gt_region & pred_region)
    union = np.sum(gt_region | pred_region)
    if union == 0:
        return 1.0 if np.sum(pred) == 0 and np.sum(target) == 0 else 0.0
    return float(intersection / (union + 1e-5))

def predict_full_image_direct(model, image, device, target_size=448):
    """
    Direct full-image prediction without tiling (Protocol for DeepCrack baseline).
    Full-image pipeline: dynamic Pad-to-Square -> Resize 448x448 -> model 1 pass -> logits.
    """
    H, W = image.shape[:2]
    target_dim = max(H, W, target_size)
    pad_h = target_dim - H
    pad_w = target_dim - W
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    resized = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    tensor = (tensor - mean) / std
    
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            logits = model(tensor)
        logits_np = logits.squeeze().cpu().numpy()
        
    return logits_np

def predict_full_image_tiling_setting_a(model, image, device, tile_size=448, batch_size=16):
    """
    Crack500 Setting A: Non-overlapping tiling with tile_size x tile_size (stride = tile_size).
    """
    H, W = image.shape[:2]
    
    pad_h = (tile_size - (H % tile_size)) % tile_size
    pad_w = (tile_size - (W % tile_size)) % tile_size
    
    padded_img = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    pH, pW = padded_img.shape[:2]
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    patches = []
    coords = []
    for y in range(0, pH, tile_size):
        for x in range(0, pW, tile_size):
            patch = padded_img[y:y+tile_size, x:x+tile_size]
            patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
            patch_tensor = (patch_tensor - mean) / std
            patches.append(patch_tensor)
            coords.append((y, x))
            
    pred_logits = np.zeros((pH, pW), dtype=np.float32)
    
    for i in range(0, len(patches), batch_size):
        batch = torch.stack(patches[i:i+batch_size]).to(device, non_blocking=True)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                logits = model(batch)
            logits_np = logits.squeeze(1).cpu().numpy()
            
        for j, logit in enumerate(logits_np):
            y, x = coords[i+j]
            pred_logits[y:y+tile_size, x:x+tile_size] = logit
            
    pred_logits = pred_logits[:H, :W]
    return pred_logits

def predict_full_image_tiling_setting_b(model, image, device, tile_size=448, stride=224, batch_size=16, blend_mode='logits'):
    """
    Crack500 Setting B: Overlapping tiling (50% overlap, stride=tile_size//2),
    blending at overlap regions, and crop back to original (H, W).
    Supports blend_mode='logits' (Option 1) and blend_mode='probs' (Option 2).
    """
    H, W = image.shape[:2]
    
    if H < tile_size:
        pad_h = tile_size - H
    else:
        rem_h = (H - tile_size) % stride
        pad_h = (stride - rem_h) % stride if rem_h != 0 else 0
        
    if W < tile_size:
        pad_w = tile_size - W
    else:
        rem_w = (W - tile_size) % stride
        pad_w = (stride - rem_w) % stride if rem_w != 0 else 0
        
    padded_img = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    pH, pW = padded_img.shape[:2]
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    patches = []
    coords = []
    for y in range(0, pH - tile_size + 1, stride):
        for x in range(0, pW - tile_size + 1, stride):
            patch = padded_img[y:y+tile_size, x:x+tile_size]
            patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
            patch_tensor = (patch_tensor - mean) / std
            patches.append(patch_tensor)
            coords.append((y, x))
            
    accum_map = np.zeros((pH, pW), dtype=np.float32)
    count_map = np.zeros((pH, pW), dtype=np.float32)
    
    for i in range(0, len(patches), batch_size):
        batch = torch.stack(patches[i:i+batch_size]).to(device, non_blocking=True)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                logits = model(batch)
            logits_np = logits.squeeze(1).cpu().numpy()
            
        for j, logit in enumerate(logits_np):
            y, x = coords[i+j]
            if blend_mode == 'probs':
                prob = 1.0 / (1.0 + np.exp(-logit))
                accum_map[y:y+tile_size, x:x+tile_size] += prob
            else: # 'logits'
                accum_map[y:y+tile_size, x:x+tile_size] += logit
            count_map[y:y+tile_size, x:x+tile_size] += 1.0
            
    blended = accum_map / np.maximum(count_map, 1.0)
    blended = blended[:H, :W]
    return blended

# Alias for backward compatibility
predict_full_image_tiling = predict_full_image_tiling_setting_a

def resolve_protocol(config, cli_protocol=None):
    """
    Resolves evaluation protocol:
    - 'direct': DeepCrack (no tiling, full image)
    - 'setting_a': Crack500 Setting A (non-overlap 448x448)
    - 'setting_b': Crack500 Setting B (overlap 50%, stride=224, average blending)
    """
    if cli_protocol and cli_protocol not in ('auto', 'all'):
        return cli_protocol
        
    if 'eval_protocol' in config:
        return config['eval_protocol']

    dataset_name = str(config.get('dataset', '')).lower()
    root_dir = str(config.get('root_dir', '')).lower()
    crop_mode = str(config.get('crop_mode', '')).lower()
    
    if 'deepcrack' in dataset_name or 'deepcrack' in root_dir or crop_mode == 'resize':
        return 'direct'
    else:
        return 'setting_a'

def evaluate_split(model, pairs, device, protocol='setting_a', tile_size=448, blend_mode='probs', criterion=None, diagnostic=False, verbose=True):
    metrics = {
        'precision': [], 'recall': [], 'dice': [], 'iou': [],
        'crack_iou': []
    }
    if diagnostic:
        metrics['boundary_iou'] = []
        metrics['hd95'] = []
    if criterion is not None:
        metrics['loss'] = []
        
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    iterator = tqdm(pairs, desc=f"Evaluating [{protocol}]") if verbose else pairs
    
    for img_p, mask_p in iterator:
        img = cv2.imread(img_p)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        if mask is None: continue
        target = (mask > 0).astype(np.uint8)
        
        if protocol == 'direct':
            # Full-image DeepCrack: dynamic Pad-to-Square -> Resize 448x448 -> model
            logits_np = predict_full_image_direct(model, img, device, target_size=tile_size)
            pred = (logits_np > 0.0).astype(np.uint8)
            
            # Ground truth target: dynamic Pad-to-Square -> Resize 448x448
            H, W = target.shape[:2]
            target_dim = max(H, W, tile_size)
            pad_h = target_dim - H
            pad_w = target_dim - W
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left
            padded_target = cv2.copyMakeBorder(target, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
            target = cv2.resize(padded_target, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)
            
        elif protocol == 'setting_b':
            blended = predict_full_image_tiling_setting_b(model, img, device, tile_size=tile_size, stride=tile_size // 2, blend_mode=blend_mode)
            if blend_mode == 'probs':
                pred = (blended > 0.5).astype(np.uint8)
                logits_np = blended  # for loss computation approximation
            else:
                pred = (blended > 0.0).astype(np.uint8)
                logits_np = blended
        else: # setting_a
            logits_np = predict_full_image_tiling_setting_a(model, img, device, tile_size=tile_size)
            pred = (logits_np > 0.0).astype(np.uint8)
            
        if criterion is not None:
            log_tensor = torch.from_numpy(logits_np).unsqueeze(0).unsqueeze(0).to(device)
            tgt_tensor = torch.from_numpy(target).unsqueeze(0).unsqueeze(0).float().to(device)
            loss = criterion(log_tensor, tgt_tensor).item()
            metrics['loss'].append(loss)
        
        tp = int(np.sum((pred == 1) & (target == 1)))
        fp = int(np.sum((pred == 1) & (target == 0)))
        fn = int(np.sum((pred == 0) & (target == 1)))
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        precision = tp / (tp + fp + 1e-5)
        recall = tp / (tp + fn + 1e-5)
        dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-5)
        iou = tp / (tp + fp + fn + 1e-5)
        
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['dice'].append(dice)
        metrics['iou'].append(iou)
        
        if np.sum(target) > 0:
            metrics['crack_iou'].append(iou)
            
        if diagnostic:
            b_iou = compute_boundary_iou(pred, target, d=2)
            hd, _ = calculate_hd95_bf1(pred, target)
            metrics['boundary_iou'].append(b_iou)
            metrics['hd95'].append(hd)
            
    summary = {k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in metrics.items()}
    # Global Pixel IoU: total TP / (total TP + total FP + total FN)
    summary['global_pixel_iou'] = float(total_tp / (total_tp + total_fp + total_fn + 1e-5))
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--protocol', type=str, default='auto',
                        choices=['auto', 'direct', 'setting_a', 'setting_b', 'all'],
                        help="Eval protocol: 'direct' (DeepCrack full-image), 'setting_a' (Crack500 non-overlap), 'setting_b' (Crack500 50% overlap), 'all' (runs both Setting A & B)")
    parser.add_argument('--blend-mode', type=str, default='probs', choices=['logits', 'probs'],
                        help="Blend method for Setting B: 'probs' (average probabilities -> threshold 0.5) or 'logits' (average logits -> threshold 0)")
    parser.add_argument('--diagnostic', action='store_true',
                        help="Compute diagnostic metrics at best checkpoint (Boundary IoU, HD95)")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model_type = config.get('model', 'B0')
    if model_type == 'B0':
        model = create_b0_unet(pretrained=False).to(device)
    elif model_type == 'B1':
        vit_depth = int(config.get('num_transformer_layers', 6))
        model = create_b1_unet(num_transformer_layers=vit_depth, pretrained=False).to(device)
        print(f"Loaded B1 architecture with {vit_depth} ViT blocks")
    elif model_type == 'B2':
        vit_depth = int(config.get('num_transformer_layers', 12))
        sage_cfg = config.get('sage_config', {})
        model = create_b2_unet(num_transformer_layers=vit_depth, pretrained=False, sage_config=sage_cfg).to(device)
        print(f"Loaded B2 architecture with {vit_depth} ViT blocks and SAGE components")
    else:
        raise ValueError(f"Model {model_type} not implemented yet")
        
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
        
    tile_size = config.get('img_size', 448)
    
    if args.protocol == 'all':
        default_proto = resolve_protocol(config)
        if default_proto == 'direct':
            protocols_to_run = ['direct']
        else:
            protocols_to_run = ['setting_a', 'setting_b']
    elif args.protocol != 'auto':
        protocols_to_run = [args.protocol]
    else:
        protocols_to_run = [resolve_protocol(config)]

    criterion = CrackBinaryLoss()

    for protocol in protocols_to_run:
        print("\n" + "=" * 60)
        print(f"EVALUATION PROTOCOL: {protocol.upper()}")
        print("=" * 60)
        if protocol == 'direct':
            print("  Mode: Direct full-image prediction (DeepCrack baseline: dynamic Pad-to-Square -> Resize 448x448).")
        elif protocol == 'setting_a':
            print(f"  Mode: Crack500 Setting A (non-overlapping tiling, tile_size={tile_size}x{tile_size}).")
        elif protocol == 'setting_b':
            print(f"  Mode: Crack500 Setting B (50% overlapping tiling, tile_size={tile_size}x{tile_size}, stride={tile_size // 2}, blend_mode={args.blend_mode}).")
        
        for split in ['val', 'test']:
            pairs = get_image_mask_pairs(config, split)
            if not pairs:
                continue
                
            print(f"\n--- Evaluating [{protocol}] on {split.upper()} Set ({len(pairs)} images) ---")
            res = evaluate_split(
                model, pairs, device, protocol=protocol, tile_size=tile_size,
                blend_mode=args.blend_mode, criterion=criterion, diagnostic=args.diagnostic
            )
            
            print(f"{split.upper()} Loss:               {res.get('loss', 0.0):.4f}")
            print(f"{split.upper()} Precision:          {res['precision']:.4f}")
            print(f"{split.upper()} Recall:             {res['recall']:.4f}")
            print(f"{split.upper()} Dice/F1:            {res['dice']:.4f}")
            print(f"{split.upper()} Global Pixel IoU:   {res['global_pixel_iou']:.4f}")
            print(f"{split.upper()} Crack-Present IoU:  {res.get('crack_iou', res['iou']):.4f}")
            print(f"{split.upper()} Macro IoU:          {res['iou']:.4f}")
            if args.diagnostic:
                print(f"{split.upper()} Boundary IoU:       {res.get('boundary_iou', 0.0):.4f}")
                print(f"{split.upper()} HD95:               {res.get('hd95', 0.0):.4f}")

if __name__ == '__main__':
    main()

