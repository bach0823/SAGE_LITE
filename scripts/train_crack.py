import json
import argparse
import os
import sys
import yaml
from tqdm import tqdm
from typing import Optional, Set, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from timm.scheduler.cosine_lr import CosineLRScheduler

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
scripts_dir = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)


import time

from sage.networks import create_b0_unet, create_b1_unet, create_b2_unet
from sage.utils.dataloader import get_dataset_from_config
from sage.utils.training_utils import setup_logging, set_seed, seed_worker

DEFAULT_SHARED_PREFIXES = {
    "backbone.convnext.stages.0.main_block.",
    "backbone.convnext.stages.1.main_block.",
    "backbone.convnext.stages.2.main_block.",
    "backbone.convnext.stages.3.main_block.",
    "backbone.convnext.stages.0.expert_pool.0.",
    "backbone.convnext.stages.0.expert_pool.1.",
    "backbone.convnext.stages.0.expert_pool.2.",
    "backbone.convnext.stages.0.expert_pool.3.",
    "expert_pool.0.",
    "expert_pool.1.",
    "expert_pool.2.",
    "expert_pool.3.",
}


def set_p3_gamma_init(model: nn.Module, gamma_val: float):
    """Initializes gamma parameters in all P3 refinement stages to the specified value."""
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'convnext') and hasattr(model.backbone.convnext, 'stages'):
        for stage in model.backbone.convnext.stages:
            if hasattr(stage, 'p3_refinement') and stage.p3_refinement is not None:
                if hasattr(stage.p3_refinement, 'gamma'):
                    with torch.no_grad():
                        stage.p3_refinement.gamma.fill_(gamma_val)


def get_p3_gamma_values(model: nn.Module) -> Tuple[float, float]:
    """Retrieves current gamma values for Stage 0 and Stage 1."""
    g0, g1 = float('nan'), float('nan')
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'convnext') and hasattr(model.backbone.convnext, 'stages'):
        stages = model.backbone.convnext.stages
        if len(stages) > 0 and hasattr(stages[0], 'p3_refinement') and stages[0].p3_refinement is not None:
            if hasattr(stages[0].p3_refinement, 'gamma'):
                g0 = float(stages[0].p3_refinement.gamma.item())
        if len(stages) > 1 and hasattr(stages[1], 'p3_refinement') and stages[1].p3_refinement is not None:
            if hasattr(stages[1].p3_refinement, 'gamma'):
                g1 = float(stages[1].p3_refinement.gamma.item())
    return g0, g1


def create_stage2_optimizer(
    model: nn.Module,
    stage2_base_lr: float,
    stage2_shared_lr: float,
    stage2_p3_lr: Optional[float] = None,
    shared_prefixes: Optional[Set[str]] = None,
    weight_decay: float = 0.05,
) -> optim.Optimizer:
    """
    Construct Stage-2 optimizer parameter groups according to SAGE-Lite protocol:
    - Shared experts (CNN main_block stages): stage2_shared_lr
    - P3 refinement (ASDW / Generic DW): stage2_p3_lr (defaults to stage2_base_lr)
    - Other components (ViT blocks, routers, SA-Hub adapters, decoder, bridge layers): stage2_base_lr
    - Weight decay: 0.0 for LayerNorm/Norm, biases, and gamma; weight_decay (0.05) for weights.
    - All trainable parameters retain requires_grad=True (no freezing).
    """
    if stage2_p3_lr is None:
        stage2_p3_lr = stage2_base_lr
    if shared_prefixes is None:
        shared_prefixes = DEFAULT_SHARED_PREFIXES

    groups = {
        'shared_decay': {'params': [], 'weight_decay': weight_decay, 'lr': stage2_shared_lr, 'name': 'shared_experts'},
        'shared_no_decay': {'params': [], 'weight_decay': 0.0, 'lr': stage2_shared_lr, 'name': 'shared_experts'},
        'p3_decay': {'params': [], 'weight_decay': weight_decay, 'lr': stage2_p3_lr, 'name': 'p3_refinement'},
        'p3_no_decay': {'params': [], 'weight_decay': 0.0, 'lr': stage2_p3_lr, 'name': 'p3_refinement'},
        'others_decay': {'params': [], 'weight_decay': weight_decay, 'lr': stage2_base_lr, 'name': 'other_and_routers'},
        'others_no_decay': {'params': [], 'weight_decay': 0.0, 'lr': stage2_base_lr, 'name': 'other_and_routers'},
    }

    shared_param_ids = set()
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'convnext') and hasattr(model.backbone.convnext, 'stages'):
        for stage_idx in range(min(4, len(model.backbone.convnext.stages))):
            stage = model.backbone.convnext.stages[stage_idx]
            block = stage.main_block if hasattr(stage, 'main_block') else stage
            for p in block.parameters():
                shared_param_ids.add(id(p))

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_no_decay = 'layernorm' in name.lower() or 'norm' in name.lower() or name.endswith('.bias') or 'gamma' in name.lower()

        if 'p3_refinement' in name:
            tier = 'p3'
        elif (id(param) in shared_param_ids) or any(name.startswith(p) for p in shared_prefixes):
            tier = 'shared'
        else:
            tier = 'others'

        group_key = f"{tier}_no_decay" if is_no_decay else f"{tier}_decay"
        groups[group_key]['params'].append(param)

    param_groups = [g for g in groups.values() if len(g['params']) > 0]

    # Parameter partition integrity assertions
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    all_group_params = []
    for g in param_groups:
        all_group_params.extend(g['params'])

    assert len(trainable_params) == len(all_group_params), (
        f"Parameter count mismatch in Stage 2 optimizer: trainable={len(trainable_params)} vs groups={len(all_group_params)}"
    )
    assert len(set(trainable_params)) == len(set(all_group_params)), (
        "Duplicate parameters found across Stage 2 optimizer groups!"
    )

    return optim.AdamW(param_groups)

def get_optimizer_groups(model, lr_backbone, lr_decoder, lr_sage=None, lr_p3=None, weight_decay=0.05):
    """
    Construct optimizer parameter groups categorized by learning rate tiers and weight decay:
    - Backbone (pretrained ConvNeXt stages and ViT blocks): lr_backbone (1e-5)
    - Decoder & Interface (UNet decoder and newly initialized hybrid bridge layers): lr_decoder (1e-4)
    - SAGE components (routers, SA-Hub adapters, adaptive fusion alpha): lr_sage (1e-4)
    - P3 refinement (ASDW / Generic DW): lr_p3 (1e-4)
    - Weight decay: 0.0 for LayerNorm/Norm, biases, and gamma; weight_decay (0.05) for weights.
    """
    if lr_sage is None:
        lr_sage = lr_decoder
    if lr_p3 is None:
        lr_p3 = lr_decoder

    # Interface / bridge layers between ConvNeXt and ViT are newly initialized (NOT pretrained)
    interface_keys = (
        'convnext_to_transformer',
        'transformer_to_decoder',
        'pre_transformer_norm',
        'post_transformer_norm',
    )

    groups = {
        'backbone_decay': {'params': [], 'weight_decay': weight_decay, 'lr': lr_backbone, 'name': 'backbone'},
        'backbone_no_decay': {'params': [], 'weight_decay': 0.0, 'lr': lr_backbone, 'name': 'backbone'},
        'decoder_decay': {'params': [], 'weight_decay': weight_decay, 'lr': lr_decoder, 'name': 'decoder'},
        'decoder_no_decay': {'params': [], 'weight_decay': 0.0, 'lr': lr_decoder, 'name': 'decoder'},
        'sage_decay': {'params': [], 'weight_decay': weight_decay, 'lr': lr_sage, 'name': 'sage'},
        'sage_no_decay': {'params': [], 'weight_decay': 0.0, 'lr': lr_sage, 'name': 'sage'},
        'p3_decay': {'params': [], 'weight_decay': weight_decay, 'lr': lr_p3, 'name': 'p3_refinement'},
        'p3_no_decay': {'params': [], 'weight_decay': 0.0, 'lr': lr_p3, 'name': 'p3_refinement'},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        is_no_decay = 'layernorm' in name.lower() or 'norm' in name.lower() or name.endswith('.bias') or 'gamma' in name.lower()
        
        if 'p3_refinement' in name:
            tier = 'p3'
        elif 'router' in name or 'sa_hub' in name or 'alpha' in name:
            tier = 'sage'
        elif name.startswith('decoder') or any(k in name for k in interface_keys):
            tier = 'decoder'
        else:
            tier = 'backbone'
            
        group_key = f"{tier}_no_decay" if is_no_decay else f"{tier}_decay"
        groups[group_key]['params'].append(param)
        
    return [g for g in groups.values() if len(g['params']) > 0]

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
    def __init__(self, bce_weight=1.0, dice_weight=1.5, smooth=1e-5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = torch.nn.BCEWithLogitsLoss()

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
    
    tp = (preds * targets).sum(dim=(2, 3))
    fp = (preds * (1 - targets)).sum(dim=(2, 3))
    fn = ((1 - preds) * targets).sum(dim=(2, 3))
    tn = ((1 - preds) * (1 - targets)).sum(dim=(2, 3))
    
    smooth = 1e-5
    acc = (tp + tn) / (tp + tn + fp + fn + smooth)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + smooth)
    iou = tp / (tp + fp + fn + smooth)
    
    return acc.mean().item(), dice.mean().item(), iou.mean().item()

def main(args):
    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 1. Apply CLI Overrides to configuration
    if getattr(args, 'data_root', None):
        config['root_dir'] = args.data_root
    if getattr(args, 'depth', None) is not None:
        config['num_transformer_layers'] = args.depth
    if getattr(args, 'batch_size', None) is not None:
        config['batch_size'] = args.batch_size
    if getattr(args, 'num_workers', None) is not None:
        config['num_workers'] = args.num_workers
    if getattr(args, 'lr', None) is not None:
        config['lr'] = args.lr
    if getattr(args, 'p3_lr', None) is not None:
        config['p3_lr'] = args.p3_lr
    if getattr(args, 'gamma_init', None) is not None:
        config['gamma_init'] = args.gamma_init
    if getattr(args, 'epochs', None) is not None:
        config['epochs'] = args.epochs
    if getattr(args, 'warmup_epochs', None) is not None:
        config['warmup_epochs'] = args.warmup_epochs
    if getattr(args, 'output_dir', None) is not None:
        config['output_dir'] = args.output_dir
    if getattr(args, 'stage2_base_lr', None) is not None:
        config['stage2_base_lr'] = args.stage2_base_lr
    if getattr(args, 'stage2_shared_lr', None) is not None:
        config['stage2_shared_lr'] = args.stage2_shared_lr
    if getattr(args, 'stage2_p3_lr', None) is not None:
        config['stage2_p3_lr'] = args.stage2_p3_lr
    if getattr(args, 'patience', None) is not None:
        config['patience'] = args.patience

    set_seed(config.get('seed', 42))

    output_dir = config.get('output_dir', 'results/runs')
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir, experiment_name='train')
    logger.info(f"Loaded config from {config_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        dev_name = torch.cuda.get_device_name(0)
        if '1650' in dev_name or '1660' in dev_name:
            torch.backends.cudnn.enabled = False
            logger.info(f"Detected {dev_name}. Set torch.backends.cudnn.enabled = False for FP16 numerical stability.")
    logger.info(f"Using device: {device}")

    img_size = config.get('img_size', 448)
    batch_size = config.get('batch_size', 20)
    num_workers = config.get('num_workers', 4)

    train_dataset = get_dataset_from_config(config, split='train', image_size=img_size)
    from evaluate_crack_official import get_image_mask_pairs, evaluate_split, resolve_protocol
    val_pairs = get_image_mask_pairs(config, 'val')
    eval_protocol = resolve_protocol(config)
    logger.info(f"Evaluation protocol for validation: {eval_protocol}")

    g = torch.Generator()
    g.manual_seed(config.get('seed', 42))

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=(device.type == 'cuda'),
        drop_last=False
    )

    p3_mode = config.get('p3_mode', None)
    model_type = config.get('model', 'B0')
    if model_type == 'B0':
        model = create_b0_unet(pretrained=True).to(device)
    elif model_type == 'B1':
        vit_depth = int(config.get('num_transformer_layers', 6))
        model = create_b1_unet(num_transformer_layers=vit_depth, pretrained=True).to(device)
        logger.info(f"Loaded B1 with {vit_depth} ViT blocks")
    elif model_type == 'B2':
        vit_depth = int(config.get('num_transformer_layers', 12))
        sage_cfg = config.get('sage_config', {})
        model = create_b2_unet(
            num_classes=1,
            img_size=img_size,
            num_transformer_layers=vit_depth,
            pretrained=True,
            sage_config=sage_cfg,
            p3_mode=p3_mode,
        ).to(device)
        logger.info(f"Loaded B2 with {vit_depth} ViT blocks, full SAGE-Lite injection, and p3_mode='{p3_mode}'")
    else:
        raise ValueError(f"Model {model_type} not implemented yet")

    logger.info(f"Initialized {model_type} model")

    # Ingest locked-base checkpoint or pre-trained checkpoint if provided
    locked_base_path = getattr(args, 'locked_base', None) or config.get('locked_base_checkpoint')
    generic_ckpt_path = getattr(args, 'checkpoint', None) or config.get('checkpoint')
    is_stage2_resume = getattr(args, 'resume_stage2', False) or (getattr(args, 'stage2_only', False) and getattr(args, 'checkpoint', None) is not None)

    # Standalone vs Locked-Base Invariant Handling
    if locked_base_path:
        if generic_ckpt_path:
            raise ValueError(
                f"For P3 runs (p3_mode='{p3_mode}'), both --locked-base (locked_base_checkpoint) and "
                "--checkpoint cannot be supplied simultaneously. --locked-base is strictly for "
                "Locked Base provenance, and --checkpoint is strictly for resume/continue."
            )
        from sage.utils.model_utils import load_locked_base_into_p3
        from scripts.preflight_p3_realdata import compute_file_sha256, EXPECTED_LOCKED_BASE_SHA256_D4
        sha = compute_file_sha256(locked_base_path)
        logger.info(f"Ingesting locked base checkpoint from {locked_base_path} (SHA256: {sha[:16]}...) via load_locked_base_into_p3 (p3_mode='{p3_mode}')...")
        load_locked_base_into_p3(model, locked_base_path, p3_mode=p3_mode or "A")
    elif generic_ckpt_path:
        logger.info(f"Loading checkpoint from {generic_ckpt_path} (resume/continue mechanism)...")
        ckpt = torch.load(generic_ckpt_path, map_location=device, weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(sd, strict=False)
    else:
        if p3_mode in ("A", "B", "C"):
            logger.info(f"P3-{p3_mode} Standalone Initialization: ImageNet-pretrained, no parent checkpoint")

    # Initialize gamma parameters for P3
    gamma_init_val = float(config.get('gamma_init', 0.01))
    if p3_mode is not None:
        if is_stage2_resume and generic_ckpt_path:
            g0, g1 = get_p3_gamma_values(model)
            logger.info(f"P3 Refinement gamma preserved from checkpoint: S0={g0:.4f}, S1={g1:.4f}")
        else:
            set_p3_gamma_init(model, gamma_init_val)
            g0, g1 = get_p3_gamma_values(model)
            logger.info(f"P3 Refinement gamma initialized: S0={g0:.4f}, S1={g1:.4f}")

    criterion = CrackBinaryLoss()
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda'))

    base_lr = float(config.get('lr', 1e-4))
    p3_lr = float(config.get('p3_lr', base_lr))
    warmup_epochs = int(config.get('warmup_epochs', 3))
    lr_backbone = base_lr * 0.1
    lr_decoder = base_lr
    lr_sage = base_lr

    two_stage = getattr(args, 'two_stage', False) or config.get('two_stage', False)

    if not two_stage:
        logger.info(f"\n{'='*50}\nSTARTING SINGLE-STAGE TRAINING ({model_type})\n{'='*50}")
        total_epochs = int(config.get('epochs', 30))
        patience = int(config.get('patience', 6))
        logger.info(f"Total epochs: {total_epochs}, Patience: {patience}, Base LR: {base_lr}, P3 LR: {p3_lr}")

        param_groups = get_optimizer_groups(model, lr_backbone, lr_decoder, lr_sage=lr_sage, lr_p3=p3_lr, weight_decay=0.05)
        optimizer = optim.AdamW(param_groups)
        scheduler = get_scheduler(optimizer, epochs=total_epochs, warmup_epochs=warmup_epochs)

        best_dice = 0.0
        best_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(1, total_epochs + 1):
            model.train()
            train_loss = 0.0
            train_lb_loss = 0.0
            train_acc, train_dice, train_iou = 0.0, 0.0, 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs} [Train]")
            for batch in pbar:
                images = batch['image'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda')):
                    if hasattr(model, 'forward_with_routing_info'):
                        forward_out = model.forward_with_routing_info(images)
                        logits = forward_out['logits']
                        routing_infos = forward_out['routing_infos']
                        lb_loss = model.compute_total_load_balance_loss(routing_infos)
                    else:
                        logits = model(images)
                        lb_loss = torch.tensor(0.0, device=device)
                    seg_loss = criterion(logits, labels)
                    loss = seg_loss + 1.0 * lb_loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                train_lb_loss += lb_loss.item()
                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    acc, dice, iou = calculate_binary_metrics(probs, labels)
                    train_acc += acc
                    train_dice += dice
                    train_iou += iou

                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'dice': f"{dice:.4f}", 'lb': f"{lb_loss.item():.4f}"})

            scheduler.step(epoch)

            model.eval()
            with torch.no_grad():
                val_metrics = evaluate_split(model, val_pairs, device, protocol=eval_protocol, tile_size=img_size, criterion=criterion, verbose=False)

            val_loss = val_metrics['loss']
            val_dice = val_metrics['dice']

            train_loss /= len(train_loader)
            train_dice /= len(train_loader)
            train_lb_loss /= len(train_loader)

            bb_lr = next((g['lr'] for g in optimizer.param_groups if g.get('name') == 'backbone'), 0.0)
            dec_lr = next((g['lr'] for g in optimizer.param_groups if g.get('name') == 'decoder'), 0.0)
            sage_lr = next((g['lr'] for g in optimizer.param_groups if g.get('name') == 'sage'), None)
            p3_lr_cur = next((g['lr'] for g in optimizer.param_groups if g.get('name') == 'p3_refinement'), None)
            lr_str = f"BB={bb_lr:.2e}, Dec={dec_lr:.2e}"
            if sage_lr is not None:
                lr_str += f", SAGE={sage_lr:.2e}"
            if p3_lr_cur is not None:
                g0, g1 = get_p3_gamma_values(model)
                lr_str += f", P3={p3_lr_cur:.2e} (gamma: S0={g0:.4f}, S1={g1:.4f})"
            logger.info(f"Epoch {epoch}/{total_epochs} - Train Loss: {train_loss:.4f} (LB: {train_lb_loss:.4f}), Train Dice: {train_dice:.4f} | Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f} | LR: {lr_str}")
            
            is_best = False
            if val_dice > best_dice + 1e-4:
                is_best = True
            elif abs(val_dice - best_dice) <= 1e-4:
                if val_loss < best_loss:
                    is_best = True
                    
            if is_best:
                best_dice = val_dice
                best_loss = val_loss
                epochs_no_improve = 0
                
                ckpt_path = os.path.join(output_dir, f"best_model_{model_type.lower()}.pth")
                save_dict = {
                    'epoch': int(epoch),
                    'model_state_dict': model.state_dict(),
                    'best_dice': float(best_dice),
                    'best_loss': float(best_loss),
                    'model_type': model_type,
                }
                if model_type in ['B1', 'B2']:
                    save_dict['num_transformer_layers'] = vit_depth
                if model_type == 'B2':
                    save_dict['sage_config'] = sage_cfg
                    if p3_mode is not None:
                        save_dict['p3_mode'] = p3_mode
                torch.save(save_dict, ckpt_path)
                logger.info(f"*** New BEST model saved with Val Dice: {best_dice:.4f}, Val Loss: {best_loss:.4f} ***")
            else:
                epochs_no_improve += 1
                
            last_ckpt_path = os.path.join(output_dir, f"last_model_{model_type.lower()}.pth")
            last_save_dict = {
                'epoch': int(epoch),
                'model_state_dict': model.state_dict(),
                'val_dice': float(val_dice),
                'val_loss': float(val_loss),
                'model_type': model_type,
            }
            if model_type in ['B1', 'B2']:
                last_save_dict['num_transformer_layers'] = vit_depth
            if model_type == 'B2':
                last_save_dict['sage_config'] = sage_cfg
                if p3_mode is not None:
                    last_save_dict['p3_mode'] = p3_mode
            torch.save(last_save_dict, last_ckpt_path)
            
            if epochs_no_improve >= patience:
                logger.info(f"EarlyStopping triggered at epoch {epoch} (Patience: {patience})")
                break
                
        logger.info(f"Training completed. Best Val Dice: {best_dice:.4f}")
        return

    # ── Legacy Two-Stage Training (Only when --two-stage is requested) ────────
    total_budget = config.get('epochs', 30)
    stage1_max = min(config.get('stage1_epochs', total_budget // 2), total_budget)
    patience = config.get('patience', 6)
    
    global_best_dice = 0.0
    global_best_loss = float('inf')
    
    epochs_used_so_far = 0
    is_stage2_resume = getattr(args, 'resume_stage2', False) or (getattr(args, 'stage2_only', False) and getattr(args, 'checkpoint', None) is not None)
    stages_to_run = [2] if (args.stage2_only or is_stage2_resume) else [1, 2]
    
    if is_stage2_resume:
        resume_ckpt_path = args.checkpoint
        if not resume_ckpt_path or not os.path.exists(resume_ckpt_path):
            logger.error(f"Cannot find checkpoint for Stage 2 resumption/extension: {resume_ckpt_path}")
            sys.exit(1)
        logger.info(f"Loading Stage 2 checkpoint for continuation: {resume_ckpt_path}")
        resume_data = torch.load(resume_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(resume_data.get('model_state_dict', resume_data))
        global_best_dice = float(resume_data.get('best_dice', 0.0))
        global_best_loss = float(resume_data.get('best_loss', float('inf')))
        ckpt_epoch = resume_data.get('epoch', 'unknown')
        logger.info(
            f"Stage 2 Extension baseline initialized from {resume_ckpt_path}: "
            f"Best Val Dice={global_best_dice:.4f}, Best Val Loss={global_best_loss:.4f} (recorded at original Epoch {ckpt_epoch})"
        )
        if p3_mode is not None:
            g0, g1 = get_p3_gamma_values(model)
            logger.info(f"Stage 2 Extension learned gamma confirmed: S0={g0:.4f}, S1={g1:.4f}")
    elif args.stage2_only:
        stage1_ckpt_path = os.path.join(output_dir, f"best_model_{model_type.lower()}_stage1.pth")
        if os.path.exists(stage1_ckpt_path):
            logger.info(f"Loading Stage 1 checkpoint for --stage2-only: {stage1_ckpt_path}")
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
        elif is_stage2_resume and getattr(args, 'stage2_epochs', None) is not None:
            max_stage_epochs = args.stage2_epochs
        else:
            max_stage_epochs = total_budget - epochs_used_so_far
            if max_stage_epochs <= 0:
                logger.info(f"Total epoch budget ({total_budget}) exhausted. Skipping Stage 2.")
                break
                
        if stage == 2:
            if not is_stage2_resume:
                stage1_ckpt_path = os.path.join(output_dir, f"best_model_{model_type.lower()}_stage1.pth")
                if os.path.exists(stage1_ckpt_path):
                    logger.info(f"Loading best Stage 1 checkpoint from {stage1_ckpt_path}")
                    checkpoint = torch.load(stage1_ckpt_path, map_location=device, weights_only=False)
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    logger.warning(f"Stage 1 checkpoint not found at {stage1_ckpt_path}. Proceeding anyway...")

            shared_indices = config.get("sage_config", {}).get("shared_expert_indices") or config.get("sage", {}).get("shared_expert_indices", [0, 1, 2, 3])

            if hasattr(model, "set_shared_experts"):
                model.set_shared_experts(shared_indices)
                logger.info(f"Stage 2: Updated shared expert indices to {shared_indices}")

            stage2_base_lr = float(config.get("stage2_base_lr", base_lr))
            stage2_shared_lr = float(config.get("stage2_shared_lr", base_lr))
            stage2_p3_lr = float(config.get("stage2_p3_lr", p3_lr))
            logger.info(f"Stage 2 Optimizer: shared_lr={stage2_shared_lr:.2e}, base_lr={stage2_base_lr:.2e}, p3_lr={stage2_p3_lr:.2e}")
            optimizer = create_stage2_optimizer(
                model,
                stage2_base_lr=stage2_base_lr,
                stage2_shared_lr=stage2_shared_lr,
                stage2_p3_lr=stage2_p3_lr,
            )
        else:
            param_groups = get_optimizer_groups(model, lr_backbone, lr_decoder, lr_sage=lr_sage, lr_p3=p3_lr, weight_decay=0.05)
            optimizer = optim.AdamW(param_groups)

        scheduler = get_scheduler(optimizer, epochs=max_stage_epochs, warmup_epochs=warmup_epochs)
        
        if is_stage2_resume and stage == 2:
            best_stage_dice = global_best_dice
            best_stage_loss = global_best_loss
            epochs_no_improve = int(getattr(args, 'initial_epochs_no_improve', 0) or 0)
            logger.info(
                f"Stage 2 Extension starting with baseline Dice={best_stage_dice:.4f}, "
                f"Loss={best_stage_loss:.4f}, initial epochs_no_improve={epochs_no_improve}/{patience}"
            )
        else:
            best_stage_dice = 0.0
            best_stage_loss = float('inf')
            epochs_no_improve = 0
        actual_epochs_this_stage = 0
        
        for epoch in range(1, max_stage_epochs + 1):
            actual_epochs_this_stage = epoch
            model.train()
            train_loss = 0.0
            train_lb_loss = 0.0
            train_acc, train_dice, train_iou = 0.0, 0.0, 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{max_stage_epochs} [Train]")
            for batch in pbar:
                images = batch['image'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda')):
                    if hasattr(model, 'forward_with_routing_info'):
                        forward_out = model.forward_with_routing_info(images)
                        logits = forward_out['logits']
                        routing_infos = forward_out['routing_infos']
                        lb_loss = model.compute_total_load_balance_loss(routing_infos)
                    else:
                        logits = model(images)
                        lb_loss = torch.tensor(0.0, device=device)
                    seg_loss = criterion(logits, labels)
                    loss = seg_loss + 1.0 * lb_loss
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                train_lb_loss += lb_loss.item()
                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    acc, dice, iou = calculate_binary_metrics(probs, labels)
                    train_acc += acc
                    train_dice += dice
                    train_iou += iou
                    
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'dice': f"{dice:.4f}", 'lb': f"{lb_loss.item():.4f}"})
                
            scheduler.step(epoch)
            
            model.eval()
            with torch.no_grad():
                val_metrics = evaluate_split(model, val_pairs, device, protocol=eval_protocol, tile_size=img_size, criterion=criterion, verbose=False)
                
            val_loss = val_metrics['loss']
            val_dice = val_metrics['dice']
            
            train_loss /= len(train_loader)
            train_dice /= len(train_loader)
            train_lb_loss /= len(train_loader)
            
            if stage == 2:
                sh_lr = next((g['lr'] for g in optimizer.param_groups if g.get('name') == 'shared_experts'), 0.0)
                oth_lr = next((g['lr'] for g in optimizer.param_groups if g.get('name') == 'other_and_routers'), 0.0)
                p3_lr_cur = next((g['lr'] for g in optimizer.param_groups if g.get('name') == 'p3_refinement'), None)
                lr_str = f"Shared={sh_lr:.2e}, Others={oth_lr:.2e}"
                if p3_lr_cur is not None:
                    lr_str += f", P3={p3_lr_cur:.2e}"
            else:
                bb_lr = next((g['lr'] for g in optimizer.param_groups if g.get('name') == 'backbone'), 0.0)
                dec_lr = next((g['lr'] for g in optimizer.param_groups if g.get('name') == 'decoder'), 0.0)
                sage_lr = next((g['lr'] for g in optimizer.param_groups if g.get('name') == 'sage'), None)
                p3_lr_cur = next((g['lr'] for g in optimizer.param_groups if g.get('name') == 'p3_refinement'), None)
                lr_str = f"BB={bb_lr:.2e}, Dec={dec_lr:.2e}"
                if sage_lr is not None:
                    lr_str += f", SAGE={sage_lr:.2e}"
                if p3_lr_cur is not None:
                    lr_str += f", P3={p3_lr_cur:.2e}"
            g0, g1 = get_p3_gamma_values(model)
            if g0 is not None or g1 is not None:
                lr_str += f" (gamma: S0={g0:.4f}, S1={g1:.4f})"

            logger.info(f"Epoch {epoch}/{max_stage_epochs} - Train Loss: {train_loss:.4f} (LB: {train_lb_loss:.4f}), Train Dice: {train_dice:.4f} | Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f} | LR: {lr_str}")
            
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
                
                ckpt_path = os.path.join(output_dir, f"best_model_{model_type.lower()}_stage{stage}.pth")
                stage_ckpt_data = {
                    'epoch': int(epoch),
                    'stage': int(stage),
                    'model_state_dict': model.state_dict(),
                    'best_dice': float(best_stage_dice),
                    'best_loss': float(best_stage_loss),
                    'model_type': model_type,
                }
                if model_type in ['B1', 'B2']:
                    stage_ckpt_data['num_transformer_layers'] = vit_depth
                if model_type == 'B2':
                    stage_ckpt_data['sage_config'] = sage_cfg
                    if p3_mode is not None:
                        stage_ckpt_data['p3_mode'] = p3_mode
                torch.save(stage_ckpt_data, ckpt_path)
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
                    global_ckpt_path = os.path.join(output_dir, f"best_model_{model_type.lower()}_global.pth")
                    global_ckpt_data = {
                        'epoch': int(epoch),
                        'stage': int(stage),
                        'model_state_dict': model.state_dict(),
                        'best_dice': float(global_best_dice),
                        'best_loss': float(global_best_loss),
                        'model_type': model_type,
                    }
                    if model_type in ['B1', 'B2']:
                        global_ckpt_data['num_transformer_layers'] = vit_depth
                    if model_type == 'B2':
                        global_ckpt_data['sage_config'] = sage_cfg
                        if p3_mode is not None:
                            global_ckpt_data['p3_mode'] = p3_mode
                    torch.save(global_ckpt_data, global_ckpt_path)
                    logger.info(f"*** New GLOBAL best model saved (Dice: {global_best_dice:.4f}) ***")
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= patience:
                logger.info(f"EarlyStopping triggered at epoch {epoch} (Patience: {patience})")
                break
                
        epochs_used_so_far += actual_epochs_this_stage
        
        completion_file = os.path.join(output_dir, f"stage{stage}_completion.json")
        with open(completion_file, 'w') as f:
            json.dump({'epochs_used': actual_epochs_this_stage}, f)
            
        logger.info(f"Stage {stage} finished. Total epochs used so far: {epochs_used_so_far}/{total_budget}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SAGE-Lite Models for Crack Segmentation')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--stage2-only', action='store_true', help='Skip Stage 1 and resume directly to Stage 2 using stage 1 checkpoint')
    parser.add_argument('--stage1-epochs-used', type=int, default=None, help='Explicitly specify how many epochs Stage 1 actually ran')
    parser.add_argument('--two-stage', action='store_true', help='Enable legacy 2-stage ladder training (default is single-stage)')
    parser.add_argument('--locked-base', type=str, default=None, help='Path to locked base checkpoint to ingest via load_locked_base_into_p3')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file to resume or initialize from')
    parser.add_argument('--data-root', type=str, default=None, help='Override dataset root_dir')
    parser.add_argument('--depth', type=int, default=None, help='Override num_transformer_layers (ViT depth)')
    parser.add_argument('--batch-size', type=int, default=None, help='Override training batch_size')
    parser.add_argument('--num-workers', type=int, default=None, help='Override DataLoader num_workers')
    parser.add_argument('--lr', type=float, default=None, help='Override base learning rate')
    parser.add_argument('--p3-lr', type=float, default=None, help='Override P3 refinement learning rate')
    parser.add_argument('--gamma-init', type=float, default=None, help='Override initial gamma value for P3 refinement')
    parser.add_argument('--epochs', type=int, default=None, help='Override total training epochs')
    parser.add_argument('--warmup-epochs', type=int, default=None, help='Override scheduler warmup epochs')
    parser.add_argument('--output-dir', type=str, default=None, help='Override output directory')
    parser.add_argument('--resume-stage2', action='store_true', help='Resume/extend Stage 2 training from an existing Stage 2 or global checkpoint')
    parser.add_argument('--stage2-epochs', type=int, default=None, help='Number of epochs to run Stage 2 during this continuation session')
    parser.add_argument('--initial-epochs-no-improve', type=int, default=0, help='Initial epochs without improvement counter for early stopping (e.g. 2 if resuming after 2 non-improving epochs)')
    parser.add_argument('--stage2-base-lr', type=float, default=None, help='Override stage2_base_lr')
    parser.add_argument('--stage2-shared-lr', type=float, default=None, help='Override stage2_shared_lr')
    parser.add_argument('--stage2-p3-lr', type=float, default=None, help='Override stage2_p3_lr')
    parser.add_argument('--patience', type=int, default=None, help='Override early stopping patience')
    args = parser.parse_args()
    main(args)




