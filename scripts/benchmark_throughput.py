import time
import torch
import gc
from sage.networks import create_b0_unet

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting T4 Pure Compute Throughput Sweep on {device}...")
    
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = "CPU"

    batch_sizes = [16, 20, 24, 28]
    img_size = 448
    results = []

    print("\nStarting Batch Size Sweep...")
    for bs in batch_sizes:
        print(f"\n--- Testing Batch Size: {bs} ---")
        
        # Reset Memory Stats
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        gc.collect()

        # Tạo model/optimizer/scaler SẠCH RIÊNG cho mỗi batch size
        model = create_b0_unet().to(device)
        model.train()
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.cuda.amp.GradScaler()

        images, labels, logits, loss = None, None, None, None

        try:
            # Warmup
            for _ in range(10):
                images = torch.randn(bs, 3, img_size, img_size, device=device)
                labels = torch.randint(0, 2, (bs, 1, img_size, img_size), device=device, dtype=torch.float)
                
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
            start_time = time.time()
            for _ in range(steps):
                images = torch.randn(bs, 3, img_size, img_size, device=device)
                labels = torch.randint(0, 2, (bs, 1, img_size, img_size), device=device, dtype=torch.float)
                
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
            img_per_sec = (bs * steps) / total_time
            peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3) if device.type == 'cuda' else 0

            results.append({
                "BS": bs,
                "Time/Step": f"{time_per_step:.3f} s",
                "Throughput": f"{img_per_sec:.1f}",
                "VRAM": f"{peak_vram_gb:.2f} GB",
                "Status": "PASS"
            })
            print(f"Success: {img_per_sec:.1f} img/s, VRAM: {peak_vram_gb:.2f} GB")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM encountered for BS={bs}")
                results.append({
                    "BS": bs,
                    "Time/Step": "-",
                    "Throughput": "-",
                    "VRAM": "OOM",
                    "Status": "OOM"
                })
            else:
                raise e
        finally:
            # Cleanup iteration
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            del images, labels, logits, loss, optimizer, scaler, criterion, model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

    # Print Markdown Report
    print("\n" + "="*50)
    print("MARKDOWN REPORT (COPY THIS)")
    print("="*50)
    print("### Batch Size Sweep Benchmark Report")
    print("> *Note: Pure GPU compute (Forward+Backward) using synthetic tensors, BCE-only.*")
    print(f"- **GPU:** {gpu_name}")
    print(f"- **Image Size:** {img_size}x{img_size}")
    print("\n| Batch | Time/step | Throughput | Peak VRAM | Status |")
    print("|----:|--------:|---------:|--------:|:----:|")
    for res in results:
        print(f"| {res['BS']} | {res['Time/Step']} | {res['Throughput']} | {res['VRAM']} | {res['Status']} |")
    print("="*50)

if __name__ == '__main__':
    main()
