import argparse
import json
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

from sage.networks import create_b2_unet
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
    Direct full-image prediction:
    Resize image (H, W) -> (target_size, target_size) -> model 1 pass -> logits -> resize logits back to original (H, W).
    Unified single-path pipeline for both Crack500 and DeepCrack.
    """
    H, W = image.shape[:2]
    resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    tensor = (tensor - mean) / std
    
    is_gtx16xx = False
    if device.type == 'cuda':
        dname = torch.cuda.get_device_name(device)
        if '1650' in dname or '1660' in dname:
            is_gtx16xx = True

    with torch.no_grad():
        if device.type == 'cuda' and not is_gtx16xx:
            with torch.amp.autocast('cuda'):
                logits = model(tensor)
        else:
            logits = model(tensor)
        logits_np = logits.squeeze().cpu().numpy()

    # Resize logits back to original image size (H, W) for exact ground-truth comparison
    logits_np = cv2.resize(logits_np, (W, H), interpolation=cv2.INTER_LINEAR)
        
    return logits_np


def resolve_protocol(config=None, cli_protocol=None):
    """
    Resolves evaluation protocol:
    Standardized to full-image direct evaluation (1-pass prediction).
    """
    return 'direct'

def evaluate_split(model, pairs, device, protocol='direct', tile_size=448, criterion=None, diagnostic=False, verbose=True):
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
        
        # Unified direct resize evaluation for both Crack500 and DeepCrack:
        # Resize original (H, W) -> 448x448 -> model -> resize logits back to (H, W)
        logits_np = predict_full_image_direct(model, img, device, target_size=tile_size)
        pred = (logits_np > 0.0).astype(np.uint8)
        # target remains in original resolution (H, W) - exact evaluation without padding!
            
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
    parser.add_argument('--protocol', type=str, default='direct',
                        choices=['direct'],
                        help="Eval protocol: 'direct' (1-pass full-image prediction)")
    parser.add_argument('--diagnostic', action='store_true',
                        help="Compute diagnostic metrics at best checkpoint (Boundary IoU, HD95)")
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test', 'all'],
                        help="Dataset split to evaluate (default: val to keep test set sealed)")
    parser.add_argument('--output_json', '--output-json', type=str, default=None,
                        help="Path to save evaluation metrics as JSON")
    parser.add_argument('--data-root', '--data_root', type=str, default=None,
                        help="Override dataset root directory")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.data_root:
        config['root_dir'] = args.data_root
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model_type = config.get('model', 'B2')
    if model_type in ['B2', 'SAGE_LITE', 'ConvNeXt_Transformer_UNet']:
        vit_depth = int(config.get('num_transformer_layers', 12))
        sage_cfg = config.get('sage_config', {})
        use_plu_head = config.get('use_plu_head', False)
        model = create_b2_unet(
            num_transformer_layers=vit_depth,
            pretrained=False,
            sage_config=sage_cfg,
            p3_mode=config.get('p3_mode'),
            use_plu_head=use_plu_head,
            use_s2_gate=config.get('use_s2_gate', False),
            s2_gate_kernel_size=config.get('s2_gate_kernel_size', 3),
            use_s2_gate_block2=config.get('use_s2_gate_block2', False),
            s2_gate_block2_kernel_size=config.get('s2_gate_block2_kernel_size', 3),
            s2_gate_block2_dilation=config.get('s2_gate_block2_dilation', 1),
        ).to(device)
        print(f"Loaded B2 architecture with {vit_depth} ViT blocks, SAGE components, use_plu_head={use_plu_head}, s2_gate={config.get('use_s2_gate')}, s2_gate_b2={config.get('use_s2_gate_block2')}")
        if use_plu_head:
            from sage.networks.decoder_block import ProgressiveLearnedUpsamplingHead
            assert model.use_plu_head is True, "model.use_plu_head is not True!"
            assert isinstance(model.decoder.segmentation_head, ProgressiveLearnedUpsamplingHead), \
                f"Expected ProgressiveLearnedUpsamplingHead, got {type(model.decoder.segmentation_head)}"
            print("[AUDIT PASS] Verified model is equipped with ProgressiveLearnedUpsamplingHead.")
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
    protocols_to_run = ['direct']

    criterion = CrackBinaryLoss()

    for protocol in protocols_to_run:
        print("\n" + "=" * 60)
        print(f"EVALUATION PROTOCOL: {protocol.upper()}")
        print("=" * 60)
        print("  Mode: Direct full-image prediction (Resize to 448x448 -> Model -> Resize logits back to original resolution for Crack500 & DeepCrack).")
        
        splits_to_run = ['val', 'test'] if args.split == 'all' else [args.split]
        for split in splits_to_run:
            pairs = get_image_mask_pairs(config, split)
            if not pairs:
                continue
                
            print(f"\n--- Evaluating [{protocol}] on {split.upper()} Set ({len(pairs)} images) ---")
            res = evaluate_split(
                model, pairs, device, protocol=protocol, tile_size=tile_size,
                criterion=criterion, diagnostic=args.diagnostic
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

            if args.output_json:
                out_p = args.output_json
                if os.path.dirname(out_p):
                    os.makedirs(os.path.dirname(os.path.abspath(out_p)), exist_ok=True)
                with open(out_p, 'w', encoding='utf-8') as f:
                    json.dump(res, f, indent=2)
                print(f"[AUDIT PASS] Saved canonical official evaluation metrics to: {out_p}")

if __name__ == '__main__':
    main()

