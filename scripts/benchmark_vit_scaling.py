"""
Isolated Micro-Benchmark: ViT Expert Scaling vs. Local/Window Attention Candidate (Tesla T4)

Objectives:
1. Benchmark Current Global ViT Expert (pretrained ViT-Tiny block, embed_dim=192, num_heads=3, AMP)
   across sequence lengths N in [196, 784, 3136, 12544] (spatial resolutions 14x14, 28x28, 56x56, 112x112).
2. Benchmark Local/Window Attention Candidate (SwinTransformerBlock, dim=192, num_heads=3, window_size=7 and 14)
   across the identical sequence lengths and batch sizes.
3. Stress test OOM boundaries across batch sizes at high resolution (N=12544).
4. Measure accurate forward ms, backward ms, total ms, peak VRAM (MB), and ms/token using CUDA Events.

Usage (Colab Tesla T4):
    python scripts/benchmark_vit_scaling.py --batch-sizes 2 4 --warmup 2 --measured 10
"""

import argparse
import gc
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple
import warnings

# Suppress amp deprecation warnings across torch versions
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn

try:
    import timm
    from timm.models.swin_transformer import SwinTransformerBlock
except ImportError:
    raise ImportError("timm is required. Please install timm via: pip install timm")


# ---------------------------------------------------------------------------
# Model Definitions
# ---------------------------------------------------------------------------

class GlobalViTBlock(nn.Module):
    """Current SAGE-Lite ViT Expert: Single ViT-Tiny Block (Global Self-Attention)."""
    def __init__(self, vit_model_name: str = "vit_tiny_patch16_224"):
        super().__init__()
        vit_full = timm.create_model(vit_model_name, pretrained=True)
        self.block = vit_full.blocks[0]
        self.dim = vit_full.embed_dim  # 192
        self.num_heads = getattr(vit_full, "num_heads", 3)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # x is (B, N, D)
        return self.block(x)


class WindowViTBlock(nn.Module):
    """Local/Window Attention Candidate: Swin-style Window Transformer Block."""
    def __init__(self, dim: int = 192, num_heads: int = 3, window_size: int = 7):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self._block_cache = nn.ModuleDict()

    def _get_block(self, h: int, w: int, device: torch.device) -> nn.Module:
        key = f"res_{h}x{w}"
        if key not in self._block_cache:
            block = SwinTransformerBlock(
                dim=self.dim,
                input_resolution=(h, w),
                num_heads=self.num_heads,
                window_size=self.window_size,
                shift_size=0,
            ).to(device)
            self._block_cache[key] = block
        return self._block_cache[key]

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # x: (B, N, D) -> reshape to (B, H, W, D) -> Swin block -> reshape to (B, N, D)
        B, N, D = x.shape
        block = self._get_block(h, w, x.device)
        x_spatial = x.view(B, h, w, D)
        out_spatial = block(x_spatial)
        return out_spatial.view(B, N, D)


# ---------------------------------------------------------------------------
# Benchmark Engine
# ---------------------------------------------------------------------------

def run_single_benchmark(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    h: int,
    w: int,
    embed_dim: int,
    device: torch.device,
    use_amp: bool,
    warmup: int,
    measured: int,
) -> Optional[Dict[str, float]]:
    """Runs isolated forward + backward benchmark for a specific (model, B, N) config.

    Returns dict with metrics or None if OOM.
    """
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Clean VRAM before running
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        # Synthetic input tensor (B, N, D)
        x = torch.randn(batch_size, seq_len, embed_dim, device=device, requires_grad=True)

        # Warmup loop
        for _ in range(warmup):
            model.zero_grad(set_to_none=True)
            if x.grad is not None:
                x.grad = None
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(x, h, w)
                loss = out.sum()
            scaler.scale(loss).backward()

        torch.cuda.synchronize()

        # Reset peak stats before measured iterations
        torch.cuda.reset_peak_memory_stats()

        fwd_times: List[float] = []
        bwd_times: List[float] = []

        start_fwd = torch.cuda.Event(enable_timing=True)
        end_fwd = torch.cuda.Event(enable_timing=True)
        start_bwd = torch.cuda.Event(enable_timing=True)
        end_bwd = torch.cuda.Event(enable_timing=True)

        for _ in range(measured):
            model.zero_grad(set_to_none=True)
            if x.grad is not None:
                x.grad = None

            # Measure Forward
            start_fwd.record()
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(x, h, w)
                loss = out.sum()
            end_fwd.record()

            # Measure Backward
            start_bwd.record()
            scaler.scale(loss).backward()
            end_bwd.record()

            torch.cuda.synchronize()

            fwd_times.append(start_fwd.elapsed_time(end_fwd))
            bwd_times.append(start_bwd.elapsed_time(end_bwd))

        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        avg_fwd = float(sum(fwd_times) / len(fwd_times))
        avg_bwd = float(sum(bwd_times) / len(bwd_times))
        total_time = avg_fwd + avg_bwd
        total_tokens = batch_size * seq_len
        ms_per_token = total_time / total_tokens
        us_per_token = (total_time * 1000.0) / total_tokens

        del x, out, loss
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "h": h,
            "w": w,
            "total_tokens": total_tokens,
            "fwd_ms": avg_fwd,
            "bwd_ms": avg_bwd,
            "total_ms": total_time,
            "peak_vram_mb": peak_vram_mb,
            "ms_per_token": ms_per_token,
            "us_per_token": us_per_token,
        }

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
            gc.collect()
            torch.cuda.empty_cache()
            return None
        raise e


# ---------------------------------------------------------------------------
# Formatting Helpers
# ---------------------------------------------------------------------------

def print_banner(text: str, ch: str = "=", width: int = 95):
    line = ch * width
    print(f"\n{line}\n{text}\n{line}")


def print_table(title: str, headers: List[str], rows: List[List[str]]):
    print(f"\n{title}")
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    header_line = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    print(header_line)
    print(sep_line)
    for row in rows:
        print(" | ".join(f"{str(val):<{col_widths[i]}}" for i, val in enumerate(row)))


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Micro-Benchmark ViT Scaling vs Local Window Attention")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[2, 4], help="Batch sizes to test (default: 2 4)")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations (default: 2)")
    parser.add_argument("--measured", type=int, default=10, help="Measured iterations (default: 10)")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP (run FP32)")
    parser.add_argument("--skip-oom-test", action="store_true", help="Skip OOM boundary search")
    args = parser.parse_args()

    use_amp = not args.no_amp
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_banner("MICRO-BENCHMARK: ViT EXPERT SCALING vs. LOCAL WINDOW ATTENTION (Tesla T4)")
    print(f"[Device] Target: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[Device] GPU: {gpu_name} ({vram_gb:.2f} GB)")
    print(f"[Precision] AMP Enabled: {use_amp}")
    print(f"[Protocol] Warmup: {args.warmup} | Measured: {args.measured} steps")
    print(f"[Test Batch Sizes] {args.batch_sizes}")

    # Standard resolutions corresponding to SAGE-Lite hierarchy:
    # 14x14 (N=196)   -> Stage 3 (ViT baseline)
    # 28x28 (N=784)   -> Stage 2
    # 56x56 (N=3136)  -> Stage 1
    # 112x112 (N=12544) -> Stage 0 (CNN high-res)
    resolutions = [
        (14, 14, 196, "Stage 3 (ViT baseline)"),
        (28, 28, 784, "Stage 2"),
        (56, 56, 3136, "Stage 1"),
        (112, 112, 12544, "Stage 0 (High-Res)"),
    ]

    # Instantiate models
    print("\n[Loading Models]...")
    global_vit = GlobalViTBlock("vit_tiny_patch16_224").to(device)
    window_vit_w7 = WindowViTBlock(dim=192, num_heads=3, window_size=7).to(device)
    window_vit_w14 = WindowViTBlock(dim=192, num_heads=3, window_size=14).to(device)
    print("  + GlobalViTBlock (vit_tiny_patch16_224 blocks[0], dim=192, heads=3) loaded.")
    print("  + WindowViTBlock (window_size=7, dim=192, heads=3) initialized.")
    print("  + WindowViTBlock (window_size=14, dim=192, heads=3) initialized.")

    # Storage for results
    global_results: Dict[Tuple[int, int], Dict[str, float]] = {}
    window7_results: Dict[Tuple[int, int], Dict[str, float]] = {}
    window14_results: Dict[Tuple[int, int], Dict[str, float]] = {}

    # -----------------------------------------------------------------------
    # Part 1: Global ViT Benchmark
    # -----------------------------------------------------------------------
    print_banner("PART 1: Benchmarking Current Global ViT Expert")
    for B in args.batch_sizes:
        for H, W, N, desc in resolutions:
            print(f"  Testing Global ViT: Batch={B}, N={N} ({H}x{W}, {desc})...", end="", flush=True)
            res = run_single_benchmark(
                model=global_vit,
                batch_size=B,
                seq_len=N,
                h=H,
                w=W,
                embed_dim=192,
                device=device,
                use_amp=use_amp,
                warmup=args.warmup,
                measured=args.measured,
            )
            if res is not None:
                global_results[(B, N)] = res
                print(f" Done! Fwd={res['fwd_ms']:.2f}ms, Bwd={res['bwd_ms']:.2f}ms, VRAM={res['peak_vram_mb']:.1f}MB")
            else:
                print(" OOM!")

    # -----------------------------------------------------------------------
    # Part 2: Local Window Attention (Window Size 7)
    # -----------------------------------------------------------------------
    print_banner("PART 2: Benchmarking Local/Window Attention Candidate (Window Size 7x7)")
    for B in args.batch_sizes:
        for H, W, N, desc in resolutions:
            print(f"  Testing Window ViT (W=7): Batch={B}, N={N} ({H}x{W}, {desc})...", end="", flush=True)
            res = run_single_benchmark(
                model=window_vit_w7,
                batch_size=B,
                seq_len=N,
                h=H,
                w=W,
                embed_dim=192,
                device=device,
                use_amp=use_amp,
                warmup=args.warmup,
                measured=args.measured,
            )
            if res is not None:
                window7_results[(B, N)] = res
                print(f" Done! Fwd={res['fwd_ms']:.2f}ms, Bwd={res['bwd_ms']:.2f}ms, VRAM={res['peak_vram_mb']:.1f}MB")
            else:
                print(" OOM!")

    # -----------------------------------------------------------------------
    # Part 3: Local Window Attention (Window Size 14)
    # -----------------------------------------------------------------------
    print_banner("PART 3: Benchmarking Local/Window Attention Candidate (Window Size 14x14)")
    for B in args.batch_sizes:
        for H, W, N, desc in resolutions:
            print(f"  Testing Window ViT (W=14): Batch={B}, N={N} ({H}x{W}, {desc})...", end="", flush=True)
            res = run_single_benchmark(
                model=window_vit_w14,
                batch_size=B,
                seq_len=N,
                h=H,
                w=W,
                embed_dim=192,
                device=device,
                use_amp=use_amp,
                warmup=args.warmup,
                measured=args.measured,
            )
            if res is not None:
                window14_results[(B, N)] = res
                print(f" Done! Fwd={res['fwd_ms']:.2f}ms, Bwd={res['bwd_ms']:.2f}ms, VRAM={res['peak_vram_mb']:.1f}MB")
            else:
                print(" OOM!")

    # -----------------------------------------------------------------------
    # Part 4: OOM & Scalability Boundary Search
    # -----------------------------------------------------------------------
    stress_batches = [2, 4, 8, 12, 16, 24, 32]
    oom_rows: List[List[str]] = []

    if not args.skip_oom_test:
        print_banner("PART 4: Stress Testing OOM & Scalability Boundaries at High-Res (N=12544, 112x112)")
        for B in stress_batches:
            # Test Global ViT
            print(f"  [Stress B={B}, N=12544] Testing Global ViT...", end="", flush=True)
            res_g = run_single_benchmark(
                model=global_vit, batch_size=B, seq_len=12544, h=112, w=112,
                embed_dim=192, device=device, use_amp=use_amp, warmup=1, measured=3
            )
            g_status = f"OK ({res_g['peak_vram_mb']:.1f} MB, {res_g['total_ms']:.1f} ms)" if res_g else "OOM"
            print(f" {g_status}")

            # Test Window ViT W=7
            print(f"  [Stress B={B}, N=12544] Testing Window ViT (W=7)...", end="", flush=True)
            res_w7 = run_single_benchmark(
                model=window_vit_w7, batch_size=B, seq_len=12544, h=112, w=112,
                embed_dim=192, device=device, use_amp=use_amp, warmup=1, measured=3
            )
            w7_status = f"OK ({res_w7['peak_vram_mb']:.1f} MB, {res_w7['total_ms']:.1f} ms)" if res_w7 else "OOM"
            print(f" {w7_status}")

            # Test Window ViT W=14
            print(f"  [Stress B={B}, N=12544] Testing Window ViT (W=14)...", end="", flush=True)
            res_w14 = run_single_benchmark(
                model=window_vit_w14, batch_size=B, seq_len=12544, h=112, w=112,
                embed_dim=192, device=device, use_amp=use_amp, warmup=1, measured=3
            )
            w14_status = f"OK ({res_w14['peak_vram_mb']:.1f} MB, {res_w14['total_ms']:.1f} ms)" if res_w14 else "OOM"
            print(f" {w14_status}")

            oom_rows.append([
                str(B),
                "12544 (112x112)",
                str(B * 12544),
                g_status,
                w7_status,
                w14_status
            ])

    # -----------------------------------------------------------------------
    # FORMATTED REPORTS
    # -----------------------------------------------------------------------
    print_banner("BENCHMARK REPORT & SUMMARY TABLES")

    # Table A: Global ViT Scaling
    headers_a = ["N", "Grid (HxW)", "Batch", "Total Tokens", "Forward ms", "Backward ms", "Total ms", "Peak VRAM", "ms/token", "Rel to N=196"]
    rows_a = []
    for B in args.batch_sizes:
        base_ms = global_results.get((B, 196), {}).get("total_ms", 1.0)
        for H, W, N, desc in resolutions:
            res = global_results.get((B, N))
            if res is not None:
                rel = f"{res['total_ms'] / base_ms:.1f}x"
                rows_a.append([
                    str(N),
                    f"{H}x{W}",
                    str(B),
                    str(res['total_tokens']),
                    f"{res['fwd_ms']:.2f}",
                    f"{res['bwd_ms']:.2f}",
                    f"{res['total_ms']:.2f}",
                    f"{res['peak_vram_mb']:.1f} MB",
                    f"{res['ms_per_token']:.6f}",
                    rel
                ])
            else:
                rows_a.append([str(N), f"{H}x{W}", str(B), str(B * N), "OOM", "OOM", "OOM", "OOM", "OOM", "N/A"])

    print_table("TABLE A: Current ViT Expert Scaling (Global Multi-Head Self-Attention, D=192, H=3)", headers_a, rows_a)

    # Table B: Window ViT (W=7) Scaling & Speedup vs Global ViT
    headers_b = ["N", "Grid (HxW)", "Batch", "Total Tokens", "Forward ms", "Backward ms", "Total ms", "Peak VRAM", "ms/token", "Speedup vs Global"]
    rows_b = []
    for B in args.batch_sizes:
        for H, W, N, desc in resolutions:
            res_w = window7_results.get((B, N))
            res_g = global_results.get((B, N))
            if res_w is not None:
                speedup_str = "N/A"
                if res_g is not None and res_g['total_ms'] > 0:
                    speedup = res_g['total_ms'] / res_w['total_ms']
                    speedup_str = f"{speedup:.2f}x faster"
                elif res_g is None:
                    speedup_str = "Inf (Global OOM)"
                rows_b.append([
                    str(N),
                    f"{H}x{W}",
                    str(B),
                    str(res_w['total_tokens']),
                    f"{res_w['fwd_ms']:.2f}",
                    f"{res_w['bwd_ms']:.2f}",
                    f"{res_w['total_ms']:.2f}",
                    f"{res_w['peak_vram_mb']:.1f} MB",
                    f"{res_w['ms_per_token']:.6f}",
                    speedup_str
                ])
            else:
                rows_b.append([str(N), f"{H}x{W}", str(B), str(B * N), "OOM", "OOM", "OOM", "OOM", "OOM", "N/A"])

    print_table("TABLE B1: Local Window Attention Candidate (Window Size 7x7, D=192, H=3)", headers_b, rows_b)

    # Table B2: Window ViT (W=14) Scaling
    rows_b2 = []
    for B in args.batch_sizes:
        for H, W, N, desc in resolutions:
            res_w = window14_results.get((B, N))
            res_g = global_results.get((B, N))
            if res_w is not None:
                speedup_str = "N/A"
                if res_g is not None and res_g['total_ms'] > 0:
                    speedup = res_g['total_ms'] / res_w['total_ms']
                    speedup_str = f"{speedup:.2f}x faster"
                elif res_g is None:
                    speedup_str = "Inf (Global OOM)"
                rows_b2.append([
                    str(N),
                    f"{H}x{W}",
                    str(B),
                    str(res_w['total_tokens']),
                    f"{res_w['fwd_ms']:.2f}",
                    f"{res_w['bwd_ms']:.2f}",
                    f"{res_w['total_ms']:.2f}",
                    f"{res_w['peak_vram_mb']:.1f} MB",
                    f"{res_w['ms_per_token']:.6f}",
                    speedup_str
                ])
            else:
                rows_b2.append([str(N), f"{H}x{W}", str(B), str(B * N), "OOM", "OOM", "OOM", "OOM", "OOM", "N/A"])

    print_table("TABLE B2: Local Window Attention Candidate (Window Size 14x14, D=192, H=3)", headers_b, rows_b2)

    # Table C: OOM Boundary
    if not args.skip_oom_test and oom_rows:
        headers_c = ["Batch Size", "Resolution (N)", "Total Tokens", "Global ViT Status", "Window ViT (W=7)", "Window ViT (W=14)"]
        print_table("TABLE C: Stress Testing & OOM Boundaries at High-Res (N=12544, 112x112)", headers_c, oom_rows)

    print("\n[Done] All micro-benchmarks completed successfully.")


if __name__ == "__main__":
    main()
