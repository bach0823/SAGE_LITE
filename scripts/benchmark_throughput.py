import time
import torch
from sage.networks import create_b0_unet

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting T4 Throughput Benchmark on {device}...")
    
    model = create_b0_unet().to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    criterion = torch.nn.BCEWithLogitsLoss()
    
    batch_size = 20
    img_size = 448
    
    # Warmup
    print("Warming up for 10 steps...")
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
    
    # Benchmark
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
    
    # Giả sử dataset Crack500 có ~1896 ảnh train (hoặc tùy thực tế).
    # Chúng ta sẽ tính thời gian cho các mốc dataset khác nhau.
    dataset_sizes = [500, 1000, 1896, 3391]
    
    print("\n" + "="*40)
    print("BENCHMARK RESULTS (T4)")
    print("="*40)
    print(f"Time per step (BS={batch_size}): {time_per_step:.3f} s")
    print(f"Throughput: {img_per_sec:.1f} images/s")
    print("-" * 40)
    
    for d_size in dataset_sizes:
        steps_per_epoch = d_size / batch_size
        time_per_epoch_s = steps_per_epoch * time_per_step
        time_per_epoch_m = time_per_epoch_s / 60
        print(f"Estimated Time/Epoch for {d_size} images: {time_per_epoch_m:.2f} minutes ({time_per_epoch_s:.1f} s)")
    
if __name__ == '__main__':
    main()
