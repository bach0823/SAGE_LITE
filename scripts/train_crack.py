import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from sage.networks import create_b0_unet
from sage.utils.dataloader import get_dataset_from_config
from sage.utils.training_utils import setup_logging, set_seed, seed_worker

def get_optimizer_groups(model, lr_backbone, lr_decoder, weight_decay=0.05):
    optimizer_groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Exclude LayerNorm and bias from weight decay
        if 'LayerNorm' in name or name.endswith('.bias'):
            wd = 0.0
        else:
            wd = weight_decay
            
        # Differential LR: backbone uses lower LR
        if 'encoder' in name or 'convnext' in name:
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
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
    
    if epochs <= warmup_epochs:
        warmup_epochs = max(1, epochs // 2)
        
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    return scheduler

class CrackBinaryLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.5, smooth=1e-5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        
        bce_loss = self.bce(logits, targets)
        
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score.mean()
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

def calculate_binary_metrics(probs, targets, threshold=0.5):
    preds = (probs > threshold).float()
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)
    targets = targets.float()
    
    intersection = (preds * targets).sum().item()
    union = (preds + targets).sum().item() - intersection
    
    correct = (preds == targets).sum().item()
    total = targets.numel()
    acc = correct / total if total > 0 else 0
    
    pred_sum = preds.sum().item()
    target_sum = targets.sum().item()
    dice = (2.0 * intersection) / (pred_sum + target_sum + 1e-5)
    
    iou = intersection / (union + 1e-5)
    
    return acc, dice, iou

def main(config_path):
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
    val_dataset = get_dataset_from_config(config_path, split='val', image_size=img_size)
    
    g = torch.Generator()
    g.manual_seed(config.get('seed', 42))
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
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
    
    max_epochs = config.get('epochs', 30)
    patience = config.get('patience', 6)
    
    global_best_dice = 0.0
    global_best_loss = float('inf')
    
    for stage in [1, 2]:
        logger.info(f"\n{'='*40}\nSTARTING STAGE {stage}\n{'='*40}")
        
        if stage == 2:
            stage1_ckpt_path = os.path.join(output_dir, f"best_model_b0_stage1.pth")
            if os.path.exists(stage1_ckpt_path):
                logger.info(f"Loading best Stage 1 checkpoint from {stage1_ckpt_path}")
                checkpoint = torch.load(stage1_ckpt_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                logger.warning(f"Stage 1 checkpoint not found at {stage1_ckpt_path}. Proceeding anyway...")
        
        param_groups = get_optimizer_groups(model, lr_backbone, lr_decoder, weight_decay=0.05)
        optimizer = optim.AdamW(param_groups)
        scheduler = get_scheduler(optimizer, epochs=max_epochs, warmup_epochs=3)
        
        best_stage_dice = 0.0
        best_stage_loss = float('inf')
        epochs_no_improve = 0
        
        for epoch in range(1, max_epochs + 1):
            model.train()
            train_loss = 0.0
            train_acc, train_dice, train_iou = 0.0, 0.0, 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs} [Train]")
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
            val_loss = 0.0
            val_acc, val_dice, val_iou = 0.0, 0.0, 0.0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{max_epochs} [Val]"):
                    images = batch['image'].to(device, non_blocking=True)
                    labels = batch['label'].to(device, non_blocking=True)
                    
                    with torch.amp.autocast('cuda'):
                        logits = model(images)
                        loss = criterion(logits, labels)
                        
                    val_loss += loss.item()
                    probs = torch.sigmoid(logits)
                    acc, dice, iou = calculate_binary_metrics(probs, labels)
                    val_acc += acc
                    val_dice += dice
                    val_iou += iou
                    
            train_loss /= len(train_loader)
            train_dice /= len(train_loader)
            val_loss /= len(val_loader)
            val_dice /= len(val_loader)
            
            logger.info(f"Epoch {epoch}/{max_epochs} - Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f} | Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
            
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
                    'epoch': epoch,
                    'stage': stage,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_dice': best_stage_dice,
                    'best_loss': best_stage_loss,
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
                        'epoch': epoch,
                        'stage': stage,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_dice': global_best_dice,
                        'best_loss': global_best_loss,
                    }, global_ckpt_path)
                    logger.info(f"*** New GLOBAL best model saved (Dice: {global_best_dice:.4f}) ***")
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= patience:
                logger.info(f"EarlyStopping triggered at epoch {epoch} (Patience: {patience})")
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SAGE-Lite Models for Crack Segmentation')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()
    main(args.config)
