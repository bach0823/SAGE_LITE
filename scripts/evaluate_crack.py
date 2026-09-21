import argparse
import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sage.networks import create_b0_unet
from sage.utils.dataloader import get_dataset_from_config
from sage.utils.training_utils import set_seed

def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_intersection = 0
    total_union = 0
    total_pred = 0
    total_target = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            images = batch['image'].to(device, non_blocking=True)
            targets = batch['label'].to(device, non_blocking=True)
            
            if targets.dim() == 3:
                targets = targets.unsqueeze(1)
            targets = targets.float()
            
            with torch.amp.autocast('cuda'):
                logits = model(images)
            
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
            
            # Calculate TP, FP, FN for this batch
            tp = (preds * targets).sum().item()
            fp = (preds * (1 - targets)).sum().item()
            fn = ((1 - preds) * targets).sum().item()
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            total_intersection += tp
            total_union += tp + fp + fn
            total_pred += preds.sum().item()
            total_target += targets.sum().item()
            
    # Calculate dataset-wide metrics
    precision = total_tp / (total_tp + total_fp + 1e-5)
    recall = total_tp / (total_tp + total_fn + 1e-5)
    dice = (2.0 * total_intersection) / (total_pred + total_target + 1e-5)
    iou = total_intersection / (total_union + 1e-5)
    f1 = (2.0 * precision * recall) / (precision + recall + 1e-5) # Should match dice closely
    
    return {
        'precision': precision,
        'recall': recall,
        'dice': dice,
        'iou': iou,
        'f1': f1
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    img_size = config.get('img_size', 448)
    batch_size = config.get('batch_size', 16)
    num_workers = config.get('num_workers', 4)
    
    print("Loading Validation and Test datasets...")
    val_dataset = get_dataset_from_config(args.config, split='val', image_size=img_size)
    test_dataset = get_dataset_from_config(args.config, split='test', image_size=img_size)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    model = create_b0_unet(pretrained=False).to(device)
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    print("\n--- Evaluating on Validation Set ---")
    val_metrics = evaluate(model, val_loader, device)
    print(f"Val Precision: {val_metrics['precision']:.4f}")
    print(f"Val Recall:    {val_metrics['recall']:.4f}")
    print(f"Val Dice/F1:   {val_metrics['dice']:.4f}")
    print(f"Val IoU:       {val_metrics['iou']:.4f}")
    
    print("\n--- Evaluating on Test Set ---")
    test_metrics = evaluate(model, test_loader, device)
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall:    {test_metrics['recall']:.4f}")
    print(f"Test Dice/F1:   {test_metrics['dice']:.4f}")
    print(f"Test IoU:       {test_metrics['iou']:.4f}")

if __name__ == '__main__':
    main()
