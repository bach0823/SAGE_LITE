import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import gc

from sage.networks import create_b0_unet
from sage.utils.dataloader import get_dataset_from_config
from sage.utils.training_utils import set_seed, seed_worker
from scripts.train_crack import CrackBinaryLoss, calculate_binary_metrics

def main():
    config_path = "configs/b0_crack500.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Real-Data Pilot on {device}...")
    
    img_size = config.get('img_size', 448)
    batch_size = 20
    num_workers = config.get('num_workers', 4)
    
    print("Loading datasets (Crack500 with real dataloader)...")
    train_dataset = get_dataset_from_config(config_path, split='train', image_size=img_size)
    val_dataset = get_dataset_from_config(config_path, split='val', image_size=img_size)
    
    g = torch.Generator()
    g.manual_seed(42)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    train_steps = len(train_loader)
    val_steps = len(val_loader)
    print(f"Datasets loaded! Train steps/epoch: {train_steps}, Val steps/epoch: {val_steps}")
    
    print("Initializing model and optimizer...")
    model = create_b0_unet(pretrained=True).to(device)
    criterion = CrackBinaryLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    
    epochs = 3
    
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        
    results = []
    
    print(f"\nStarting {epochs} Pilot Epochs...")
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        
        # Train Loop
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        
        torch.cuda.synchronize()
        train_start = time.time()
        
        for batch in train_loader:
            images = batch['image'].to(device, non_blocking=True)
            targets = batch['label'].to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            probs = torch.sigmoid(logits)
            _, dice, _ = calculate_binary_metrics(probs, targets)
            train_dice += dice
            
        torch.cuda.synchronize()
        train_end = time.time()
        train_time = train_end - train_start
        
        # Val Loop
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        
        torch.cuda.synchronize()
        val_start = time.time()
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device, non_blocking=True)
                targets = batch['label'].to(device, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    logits = model(images)
                    loss = criterion(logits, targets)
                    
                val_loss += loss.item()
                probs = torch.sigmoid(logits)
                _, dice, _ = calculate_binary_metrics(probs, targets)
                val_dice += dice
                
        torch.cuda.synchronize()
        val_end = time.time()
        val_time = val_end - val_start
        
        total_time = train_time + val_time
        
        # Averages
        train_loss /= train_steps
        train_dice /= train_steps
        val_loss /= val_steps
        val_dice /= val_steps
        
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3) if device.type == 'cuda' else 0
        
        res = {
            'epoch': epoch,
            'train_time': train_time,
            'val_time': val_time,
            'total_time': total_time,
            'train_loss': train_loss,
            'train_dice': train_dice,
            'val_loss': val_loss,
            'val_dice': val_dice,
            'vram': peak_vram
        }
        results.append(res)
        
        print(f"Train Time: {train_time:.1f}s | Val Time: {val_time:.1f}s | Total: {total_time:.1f}s")
        print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Dice:   {val_dice:.4f}")
        print(f"Peak VRAM:  {peak_vram:.2f} GB")

    print("\n" + "="*50)
    print("MARKDOWN REPORT (COPY THIS)")
    print("="*50)
    print("### Real-Data Pilot Benchmark Report")
    print(f"- **GPU Name:** {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    print(f"- **Train steps/epoch:** {train_steps}")
    print(f"- **Val steps/epoch:** {val_steps}")
    print(f"- **Batch Size:** {batch_size}")
    print(f"- **Image Size:** {img_size}x{img_size}")
    print("")
    print("| Epoch | Train Time | Val Time | Total Time | Train Loss/Dice | Val Loss/Dice | Peak VRAM |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['epoch']} | {r['train_time']:.1f}s | {r['val_time']:.1f}s | {r['total_time']:.1f}s | {r['train_loss']:.4f} / {r['train_dice']:.4f} | {r['val_loss']:.4f} / {r['val_dice']:.4f} | {r['vram']:.2f} GB |")
    print("="*50)

if __name__ == '__main__':
    main()
