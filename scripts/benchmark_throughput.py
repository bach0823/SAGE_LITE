import time
import torch
from sage.networks import create_b0_unet

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting T4 Pure Compute Throughput Benchmark on {device}...")
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        torch.cuda.reset_peak_memory_stats()
    else:
        gpu_name = "CPU"
        
    model = create_b0_unet().to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    criterion = torch.nn.BCEWithLogitsLoss()
    
    batch_size = 20
    img_size = 448
    
    print("Warming up for 10 steps (synthetic tensors, BCE-only)...")
    for _ in range(10):
        images = torch.randn(batch_size, 3, img_size, img_size, device=device)
        labels = torch.randint(0, 2, (batch_size, 1, img_size, img_size), device=device, dtype=torch.float)
        
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
    torch.cuda.synchronize()
    
    steps = 50
    print(f"Benchmarking for {steps} steps (Batch Size = {batch_size})...")
    
    start_time = time.time()
    for _ in range(steps):
        images = torch.randn(batch_size, 3, img_size, img_size, device=device)
        labels = torch.randint(0, 2, (batch_size, 1, img_size, img_size), device=device, dtype=torch.float)
        
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    time_per_step = total_time / steps
    img_per_sec = (batch_size * steps) / total_time
    
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3) if device.type == 'cuda' else 0
    
    dataset_sizes = [500, 1000, 1896, 3391]
    
    print("\n" + "="*50)
    print("MARKDOWN REPORT (COPY THIS)")
    print("="*50)
    print("### Pure Compute Benchmark Report")
    print("> *Note: This measures pure GPU compute (Forward+Backward) using synthetic tensors.*")
    print("> *It EXCLUDES DataLoader overhead, Augmentations, Validation loops, Checkpointing, and SoftDice computation.*")
    print(f"- **GPU:** {gpu_name}")
    print(f"- **Batch Size:** {batch_size}")
    print(f"- **Image Size:** {img_size}x{img_size}")
    print(f"- **Compute Time per step:** {time_per_step:.3f} s")
    print(f"- **Compute Throughput:** {img_per_sec:.1f} images/s")
    print(f"- **Peak VRAM (Compute):** {peak_vram_gb:.2f} GB")
    print("")
    print("#### Estimated Compute Time / Epoch")
    for d_size in dataset_sizes:
        steps_per_epoch = d_size / batch_size
        time_per_epoch_s = steps_per_epoch * time_per_step
        time_per_epoch_m = time_per_epoch_s / 60
        print(f"- **{d_size} images:** {time_per_epoch_m:.2f} min ({time_per_epoch_s:.1f} s)")
    print("="*50)
    
if __name__ == '__main__':
    main()
