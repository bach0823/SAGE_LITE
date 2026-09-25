import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from sage.networks import create_b0_unet
from sage.networks.b1_unet import create_b1_unet
from sage.networks.b2_unet import create_b2_unet
from sage.utils.dataloader import get_dataset_from_config, seed_worker
from scripts.train_crack import CrackBinaryLoss, create_stage2_optimizer, get_optimizer_groups

def run_lr_range_test(
    config_path,
    min_lr=1e-6,
    max_lr=1e-1,
    num_iter=100,
    batch_size=None,
    output_csv=None,
    data_root=None,
):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if data_root:
        config['root_dir'] = data_root

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model_name = config.get('model', 'B0').upper()
    img_size = config.get('img_size', 448)
    bs = batch_size if batch_size is not None else config.get('batch_size', 16)
    
    if output_csv is None:
        output_csv = f"results/lr_range_test_{model_name.lower()}.csv"

    # Load dataset
    train_dataset = get_dataset_from_config(config, split='train', image_size=img_size)
    g = torch.Generator()
    g.manual_seed(config.get('seed', 42))

    loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=0, # keep deterministic and lightweight for LR test
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=True
    )

    print(f"Dataset length: {len(train_dataset)}, Batch size: {bs}, Steps per epoch: {len(loader)}")
    
    # Initialize model
    vit_depth = int(config.get('num_transformer_layers', 12))
    sage_cfg = config.get('sage_config', {})
    
    if model_name == 'B2':
        print(f"Instantiating B2 UNet (Depth={vit_depth})...")
        model = create_b2_unet(
            num_transformer_layers=vit_depth,
            pretrained=True,
            sage_config=sage_cfg,
            p3_mode=None,
        ).to(device)
        model.set_shared_experts([0, 1, 2, 3])
        # Use Stage 2 optimizer partitioning
        optimizer = create_stage2_optimizer(model, stage2_base_lr=min_lr, stage2_shared_lr=min_lr)
    elif model_name == 'B1':
        print(f"Instantiating B1 UNet (Depth={vit_depth})...")
        model = create_b1_unet(num_transformer_layers=vit_depth, pretrained=True).to(device)
        optimizer = torch.optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': min_lr * 0.1},
            {'params': model.decoder.parameters(), 'lr': min_lr * 1.0}
        ], weight_decay=1e-4)
    else:
        print("Instantiating B0 UNet...")
        model = create_b0_unet(pretrained=True).to(device)
        optimizer = torch.optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': min_lr * 0.1},
            {'params': model.decoder.parameters(), 'lr': min_lr * 1.0}
        ], weight_decay=1e-4)

    model.train()

    # Calculate initial group multipliers relative to min_lr
    base_lrs = [grp['lr'] for grp in optimizer.param_groups]
    group_ratios = [lr / min_lr for lr in base_lrs]

    criterion = CrackBinaryLoss(bce_weight=1.0, dice_weight=1.5)
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda'))

    # Multiplicative factor
    mult = (max_lr / min_lr) ** (1.0 / max(num_iter - 1, 1))

    lrs = []
    losses = []
    smoothed_losses = []
    best_loss = float('inf')
    avg_loss = 0.0
    beta = 0.98

    current_lr = min_lr
    step = 0
    iterator = iter(loader)

    print(f"\nStarting LR Range Test ({model_name}): {min_lr:.1e} -> {max_lr:.1e} across {num_iter} iterations...\n")
    print(f"{'Step':>5} | {'Base LR':>10} | {'Loss':>8} | {'Smooth Loss':>11}")
    print("-" * 45)

    while step < num_iter:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        images = batch['image'].to(device, non_blocking=True)
        masks = batch['label'].to(device, non_blocking=True) if 'label' in batch else batch['mask'].to(device, non_blocking=True)

        # Update optimizer LR for all param groups
        for i, grp in enumerate(optimizer.param_groups):
            grp['lr'] = current_lr * group_ratios[i]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            if hasattr(model, 'forward_with_routing_info'):
                forward_out = model.forward_with_routing_info(images)
                logits = forward_out['logits']
                routing_infos = forward_out['routing_infos']
                lb_loss = model.compute_total_load_balance_loss(routing_infos)
                seg_loss = criterion(logits, masks)
                loss = seg_loss + 1.0 * lb_loss
            else:
                logits = model(images)
                loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_val = loss.item()
        if not np.isfinite(loss_val):
            print(f"\n[Warning] Non-finite loss ({loss_val}) at step {step+1}. Stopping LR range test.")
            break

        avg_loss = beta * avg_loss + (1 - beta) * loss_val
        smooth = avg_loss / (1 - beta ** (step + 1))

        if smooth < best_loss:
            best_loss = smooth

        lrs.append(current_lr)
        losses.append(loss_val)
        smoothed_losses.append(smooth)

        if (step + 1) % 10 == 0 or step == 0:
            print(f"{step+1:5d} | {current_lr:10.2e} | {loss_val:8.4f} | {smooth:11.4f}")

        # Stop if loss explodes
        if smooth > 4 * best_loss and step > 10:
            print(f"\nLoss exploded ({smooth:.4f} > 4 * {best_loss:.4f}), stopping early at step {step+1}.")
            break

        current_lr *= mult
        step += 1

    if len(lrs) < 5:
        print("[Error] Not enough steps recorded to determine optimal LR.")
        return {}

    # Analysis: Find L* (point of steepest descent)
    log_lrs = np.log10(np.array(lrs))
    smooth_arr = np.array(smoothed_losses)

    # Compute numerical gradients
    gradients = np.gradient(smooth_arr, log_lrs)
    steepest_idx = int(np.argmin(gradients))
    min_loss_idx = int(np.argmin(smooth_arr))

    lr_steepest = lrs[steepest_idx]
    lr_min_loss = lrs[min_loss_idx]
    l_star = lr_steepest

    print("\n" + "=" * 50)
    print("=== LR RANGE TEST RESULTS ===")
    print("=" * 50)
    print(f"Minimum Loss: {smooth_arr[min_loss_idx]:.4f} at Base LR = {lr_min_loss:.2e}")
    print(f"Steepest Descent: d(Loss)/d(log LR) = {gradients[steepest_idx]:.4f} at Base LR = {lr_steepest:.2e}")
    print("-" * 50)
    print(f"Recommended L* (Base Decoder LR) : {l_star:.2e}")
    print(f"Recommended Backbone LR (0.1*L*) : {0.1 * l_star:.2e}")
    print(f"Recommended Decoder LR  (1.0*L*) : {1.0 * l_star:.2e}")
    print("=" * 50)

    # Save to CSV
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
    with open(output_csv, 'w') as f:
        f.write("step,base_lr,loss,smoothed_loss,gradient\n")
        for i in range(len(lrs)):
            f.write(f"{i+1},{lrs[i]:.6e},{losses[i]:.6f},{smoothed_losses[i]:.6f},{gradients[i]:.6f}\n")
    print(f"Results saved to {output_csv}")

    return {
        "l_star": l_star,
        "lr_backbone": 0.1 * l_star,
        "lr_decoder": 1.0 * l_star,
        "lr_min_loss": lr_min_loss,
        "best_loss": smooth_arr[min_loss_idx]
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LR Range Test for B0/B1/B2 models")
    parser.add_argument('--config', type=str, default='configs/b2_crack500_depth12.yaml')
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--max-lr', type=float, default=1e-1)
    parser.add_argument('--num-iter', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--data-root', type=str, default=None)
    parser.add_argument('--output-csv', type=str, default=None)
    args = parser.parse_args()

    run_lr_range_test(
        config_path=args.config,
        min_lr=args.min_lr,
        max_lr=args.max_lr,
        num_iter=args.num_iter,
        batch_size=args.batch_size,
        data_root=args.data_root,
        output_csv=args.output_csv
    )
