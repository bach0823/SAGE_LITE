import json
import argparse
import os
import sys
import yaml
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from timm.scheduler.cosine_lr import CosineLRScheduler

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time

from sage.networks import create_b0_unet
from sage.utils.dataloader import get_dataset_from_config
from sage.utils.training_utils import setup_logging, set_seed, seed_worker

def get_optimizer_groups(model, lr_backbone, lr_decoder, weight_decay=0.05):
    optimizer_groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if 'LayerNorm' in name or name.endswith('.bias'):
            wd = 0.0
        else:
            wd = weight_decay
            
        if name.startswith('backbone'):
            lr = lr_backbone
        else:
            lr = lr_decoder
            
        optimizer_groups.append({
            'params': [param],
            'weight_decay': wd,
            'lr': lr
        })
    return optimizer_groups

def get_scheduler(optimizer, epochs, warmup_epochs=3):
    return CosineLRScheduler(
        optimizer,
        t_initial=epochs,
        lr_min=1e-6,
        warmup_t=warmup_epochs,
        warmup_lr_init=1e-6,
        t_in_epochs=True
    )

class CrackBinaryLoss(torch.nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        
        probs = torch.sigmoid(logits)
        smooth = 1e-5
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_loss = 1.0 - (2.0 * intersection + smooth) / (union + smooth)
        
        return bce_loss + dice_loss.mean()

def calculate_binary_metrics(probs, targets, threshold=0.5):
    preds = (probs > threshold).float()
    
    tp = (preds * targets).sum(dim=(2,3))
    fp = (preds * (1 - targets)).sum(dim=(2,3))
    fn = ((1 - preds) * targets).sum(dim=(2,3))
    tn = ((1 - preds) * (1 - targets)).sum(dim=(2,3))
    
    smooth = 1e-5
    acc = (tp + tn) / (tp + tn + fp + fn + smooth)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + smooth)
    iou = tp / (tp + fp + fn + smooth)
    
    return acc.mean().item(), dice.mean().item(), iou.mean().item()

def main(args):
    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    set_seed(config.get('seed', 42))
    
    output_dir = config.get('output_dir', 'results/runs')
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir, experiment_name='train')
    logger.info(f"Loaded config from {config_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    img_size = config.get('img_size', 448)
    batch_size = config.get('batch_size', 20)
    num_workers = config.get('num_workers', 4)
    
    train_dataset = get_dataset_from_config(config_path, split='train', image_size=img_size)
    from evaluate_crack_official import get_image_mask_pairs, evaluate_split
    val_pairs = get_image_mask_pairs(config, 'val')
    
    g = torch.Generator()
    g.manual_seed(config.get('seed', 42))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
        drop_last=False
    )
    
    model_type = config.get('model', 'B0')
    if model_type == 'B0':
        model = create_b0_unet(pretrained=True).to(device)
    else:
        raise ValueError(f"Model {model_type} not implemented yet")
        
    logger.info(f"Initialized {model_type} model")
    criterion = CrackBinaryLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    base_lr = float(config.get('lr', 1e-4))
    lr_backbone = base_lr * 0.1
    lr_decoder = base_lr
    
    total_budget = config.get('epochs', 30)
    stage1_max = min(config.get('stage1_epochs', total_budget // 2), total_budget)
    patience = config.get('patience', 6)
    
    global_best_dice = 0.0
    global_best_loss = float('inf')
    
    epochs_used_so_far = 0
    stages_to_run = [2] if args.stage2_only else [1, 2]
    
    if args.stage2_only:
        stage1_ckpt_path = os.path.join(output_dir, "best_model_b0_stage1.pth")
        if os.path.exists(stage1_ckpt_path):
            logger.info(f"Loading Stage 1 checkpoint for --stage2-only: {stage1_ckpt_path}")
            # Checkpoint loading happens later, here we just resolve epochs
            if args.stage1_epochs_used is not None:
                epochs_used_so_far = args.stage1_epochs_used
                logger.info(f"Using explicitly provided --stage1-epochs-used: {epochs_used_so_far}")
            else:
                completion_file = os.path.join(output_dir, "stage1_completion.json")
                if os.path.exists(completion_file):
                    with open(completion_file, 'r') as f:
                        meta = json.load(f)
                        epochs_used_so_far = meta.get('epochs_used', stage1_max)
                    logger.info(f"Loaded actual Stage 1 epochs from {completion_file}: {epochs_used_so_far}")
                else:
                    epochs_used_so_far = stage1_max
                    logger.warning(f"No explicit stage1 epochs provided, and completion file missing. Assuming max: {epochs_used_so_far}")
        else:
            logger.error(f"Cannot find Stage 1 checkpoint at {stage1_ckpt_path}")
            sys.exit(1)
    
    for stage in stages_to_run:
        logger.info(f"\n{'='*40}\nSTARTING STAGE {stage}\n{'='*40}")
        
        if stage == 1:
            max_stage_epochs = stage1_max
        else:
            max_stage_epochs = total_budget - epochs_used_so_far
            if max_stage_epochs <= 0:
                logger.info(f"Total epoch budget ({total_budget}) exhausted. Skipping Stage 2.")
                break
                
        if stage == 2:
            stage1_ckpt_path = os.path.join(output_dir, f"best_model_b0_stage1.pth")
            if os.path.exists(stage1_ckpt_path):
                logger.info(f"Loading best Stage 1 checkpoint from {stage1_ckpt_path}")
                checkpoint = torch.load(stage1_ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                logger.warning(f"Stage 1 checkpoint not found at {stage1_ckpt_path}. Proceeding anyway...")
        
        param_groups = get_optimizer_groups(model, lr_backbone, lr_decoder, weight_decay=0.05)
        optimizer = optim.AdamW(param_groups)
        scheduler = get_scheduler(optimizer, epochs=max_stage_epochs, warmup_epochs=3)
        
        best_stage_dice = 0.0
        best_stage_loss = float('inf')
        epochs_no_improve = 0
        actual_epochs_this_stage = 0
        
        for epoch in range(1, max_stage_epochs + 1):
            actual_epochs_this_stage = epoch
            model.train()
            train_loss = 0.0
            train_acc, train_dice, train_iou = 0.0, 0.0, 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{max_stage_epochs} [Train]")
            for batch in pbar:
                images = batch['image'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast('cuda'):
                    logits = model(images)
                    loss = criterion(logits, labels)
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    acc, dice, iou = calculate_binary_metrics(probs, labels)
                    train_acc += acc
                    train_dice += dice
                    train_iou += iou
                    
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'dice': f"{dice:.4f}"})
                
            scheduler.step()
            
            model.eval()
            with torch.no_grad():
                val_metrics = evaluate_split(model, val_pairs, device, tile_size=img_size, criterion=criterion, verbose=False)
                
            val_loss = val_metrics['loss']
            val_dice = val_metrics['dice']
            
            train_loss /= len(train_loader)
            train_dice /= len(train_loader)
            
            logger.info(f"Epoch {epoch}/{max_stage_epochs} - Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f} | Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
            
            is_best_stage = False
            if val_dice > best_stage_dice + 1e-4:
                is_best_stage = True
            elif abs(val_dice - best_stage_dice) <= 1e-4:
                if val_loss < best_stage_loss:
                    is_best_stage = True
                    
            if is_best_stage:
                best_stage_dice = val_dice
                best_stage_loss = val_loss
                epochs_no_improve = 0
                
                ckpt_path = os.path.join(output_dir, f"best_model_b0_stage{stage}.pth")
                torch.save({
                    'epoch': int(epoch),
                    'stage': int(stage),
                    'model_state_dict': model.state_dict(),
                    'best_dice': float(best_stage_dice),
                    'best_loss': float(best_stage_loss),
                }, ckpt_path)
                logger.info(f"New best Stage {stage} model saved with Val Dice: {best_stage_dice:.4f} and Val Loss: {best_stage_loss:.4f}")
                
                is_global_best = False
                if val_dice > global_best_dice + 1e-4:
                    is_global_best = True
                elif abs(val_dice - global_best_dice) <= 1e-4:
                    if val_loss < global_best_loss:
                        is_global_best = True
                        
                if is_global_best:
                    global_best_dice = val_dice
                    global_best_loss = val_loss
                    global_ckpt_path = os.path.join(output_dir, f"best_model_b0_global.pth")
                    torch.save({
                        'epoch': int(epoch),
                        'stage': int(stage),
                        'model_state_dict': model.state_dict(),
                        'best_dice': float(global_best_dice),
                        'best_loss': float(global_best_loss),
                    }, global_ckpt_path)
                    logger.info(f"*** New GLOBAL best model saved (Dice: {global_best_dice:.4f}) ***")
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= patience:
                logger.info(f"EarlyStopping triggered at epoch {epoch} (Patience: {patience})")
                break
                
        epochs_used_so_far += actual_epochs_this_stage
        
        # Save actual epochs used metadata for resuming logic
        completion_file = os.path.join(output_dir, f"stage{stage}_completion.json")
        with open(completion_file, 'w') as f:
            json.dump({'epochs_used': actual_epochs_this_stage}, f)
            
        logger.info(f"Stage {stage} finished. Total epochs used so far: {epochs_used_so_far}/{total_budget}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SAGE-Lite Models for Crack Segmentation')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--stage2-only', action='store_true', help='Skip Stage 1 and resume directly to Stage 2 using stage 1 checkpoint')
    parser.add_argument('--stage1-epochs-used', type=int, default=None, help='Explicitly specify how many epochs Stage 1 actually ran')
    args = parser.parse_args()
    main(args)

