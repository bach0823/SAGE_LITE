"""
Feasibility Micro-Benchmark: Spatial Compression before Global ViT Expert (Tesla T4)

Objectives:
Evaluate whether downsampling high-resolution CNN features (112x112, N=12544) prior to entering
the current pretrained Global ViT block (vit_tiny_patch16_224, embed_dim=192, num_heads=3, AMP)
is a viable strategy to eliminate the sequence length bottleneck without altering baseline architecture.

Candidates:
- Candidate A (Direct):     112x112 -> ViT tokens N=12544
- Candidate B (Compressed): 112x112 -> downsample 56x56 -> ViT tokens N=3136
- Candidate C (Compressed): 112x112 -> downsample 28x28 -> ViT tokens N=784
- Candidate D (Compressed): 112x112 -> downsample 14x14 -> ViT tokens N=196

Isolated Pipeline:
input (B, 192, 112, 112) -> optional downsample -> flatten/transpose -> Global ViT block -> backward.

Usage (Colab Tesla T4):
    python scripts/benchmark_spatial_compression.py --batch-sizes 4 2 --warmup 2 --measured 10
"""

import argparse
import gc
import os
import sys
import time
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    raise ImportError("timm is required. Please install timm via: pip install timm")


# ---------------------------------------------------------------------------
# Downsampling Wrappers
# ---------------------------------------------------------------------------

class CompressedViTExpertPipeline(nn.Module):
    """Isolated pipeline: Spatial Input (B, C, H, W) -> Optional Downsample -> Global ViT Block."""
    def __init__(
        self,
        vit_model_name: str = "vit_tiny_patch16_224",
        target_grid: int = 112,
        downsample_mode: str = "adaptive_avg_pool",
    ):
        super().__init__()
        vit_full = timm.create_model(vit_model_name, pretrained=True)
        self.vit_block = vit_full.blocks[0]
        self.embed_dim = vit_full.embed_dim  # 192
        self.num_heads = getattr(vit_full, "num_heads", 3)
        self.target_grid = target_grid
        self.downsample_mode = downsample_mode

        if target_grid != 112:
            if downsample_mode == "adaptive_avg_pool":
                self.downsample = nn.AdaptiveAvgPool2d((target_grid, target_grid))
            elif downsample_mode == "bilinear":
                self.downsample = None  # Use F.interpolate in forward
            else:
                raise ValueError(f"Unknown downsample_mode: {downsample_mode}")
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is spatial tensor: (B, C, 112, 112)
        if self.target_grid != 112:
            if self.downsample_mode == "adaptive_avg_pool":
                x = self.downsample(x)
            elif self.downsample_mode == "bilinear":
                x = F.interpolate(x, size=(self.target_grid, self.target_grid), mode="bilinear", align_corners=False)

        # Flatten spatial dimensions to token sequence: (B, C, H, W) -> (B, H*W, C)
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)

        # Execute Global ViT Block
        out = self.vit_block(tokens)
        return out


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

def run_single_compression_benchmark(
    pipeline: nn.Module,
    batch_size: int,
    source_grid: int,
    embed_dim: int,
    device: torch.device,
    use_amp: bool,
    warmup: int,
    measured: int,
) -> Optional[Dict[str, float]]:
    pipeline.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        # Spatial input (B, C, 112, 112)
        x = torch.randn(batch_size, embed_dim, source_grid, source_grid, device=device, requires_grad=True)

        # Warmup loop
        for _ in range(warmup):
            pipeline.zero_grad(set_to_none=True)
            if x.grad is not None:
                x.grad = None
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = pipeline(x)
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
            pipeline.zero_grad(set_to_none=True)
            if x.grad is not None:
                x.grad = None

            # Measure Forward (including downsample + flatten + ViT)
            start_fwd.record()
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = pipeline(x)
                loss = out.sum()
            end_fwd.record()

            # Measure Backward (gradient back through ViT + downsample)
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
        vit_tokens = pipeline.target_grid * pipeline.target_grid
        total_tokens = batch_size * vit_tokens
        ms_per_token = total_time / total_tokens

        del x, out, loss
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "batch_size": batch_size,
            "source_grid": f"{source_grid}x{source_grid}",
            "vit_grid": f"{pipeline.target_grid}x{pipeline.target_grid}",
            "vit_tokens": vit_tokens,
            "total_tokens": total_tokens,
            "fwd_ms": avg_fwd,
            "bwd_ms": avg_bwd,
            "total_ms": total_time,
            "peak_vram_mb": peak_vram_mb,
            "ms_per_token": ms_per_token,
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
    parser = argparse.ArgumentParser(description="Feasibility Micro-Benchmark: Spatial Compression before Global ViT")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 2], help="Batch sizes to test (default: 4 2)")
    parser.add_argument("--downsample-mode", type=str, choices=["adaptive_avg_pool", "bilinear"], default="adaptive_avg_pool",
                        help="Downsampling method (default: adaptive_avg_pool)")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations (default: 2)")
    parser.add_argument("--measured", type=int, default=10, help="Measured iterations (default: 10)")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP (run FP32)")
    args = parser.parse_args()

    use_amp = not args.no_amp
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_banner("FEASIBILITY MICRO-BENCHMARK: SPATIAL COMPRESSION BEFORE GLOBAL ViT EXPERT")
    print(f"[Device] Target: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[Device] GPU: {gpu_name} ({vram_gb:.2f} GB)")
    print(f"[Precision] AMP Enabled: {use_amp}")
    print(f"[Downsample Method] {args.downsample_mode} (parameter-free, differentiable)")
    print(f"[Protocol] Warmup: {args.warmup} | Measured: {args.measured} steps")
    print(f"[Test Batch Sizes] {args.batch_sizes} (Prioritizing B=4)")

    candidates = [
        ("Candidate A (Direct)", 112, 12544, "Direct 112x112 -> ViT (N=12544)"),
        ("Candidate B (Compressed 56x56)", 56, 3136, "Downsample to 56x56 -> ViT (N=3136)"),
        ("Candidate C (Compressed 28x28)", 28, 784, "Downsample to 28x28 -> ViT (N=784)"),
        ("Candidate D (Compressed 14x14)", 14, 196, "Downsample to 14x14 -> ViT (N=196)"),
    ]

    all_results: Dict[Tuple[int, int], Dict[str, float]] = {}

    for B in args.batch_sizes:
        print_banner(f"BENCHMARKING BATCH SIZE B = {B}")
        for label, target_grid, n_tok, desc in candidates:
            print(f"  Testing {label} [{desc}]...", end="", flush=True)
            pipeline = CompressedViTExpertPipeline(
                vit_model_name="vit_tiny_patch16_224",
                target_grid=target_grid,
                downsample_mode=args.downsample_mode,
            ).to(device)

            res = run_single_compression_benchmark(
                pipeline=pipeline,
                batch_size=B,
                source_grid=112,
                embed_dim=192,
                device=device,
                use_amp=use_amp,
                warmup=args.warmup,
                measured=args.measured,
            )

            if res is not None:
                all_results[(B, target_grid)] = res
                print(f" Done! Fwd={res['fwd_ms']:.2f}ms, Bwd={res['bwd_ms']:.2f}ms, Total={res['total_ms']:.2f}ms, VRAM={res['peak_vram_mb']:.1f}MB")
            else:
                print(" OOM!")

    # -----------------------------------------------------------------------
    # FORMATTED REPORTS
    # -----------------------------------------------------------------------
    print_banner("COMPRESSION BENCHMARK REPORT TABLES")

    headers = [
        "Candidate", "Source Grid", "ViT Grid", "ViT Tokens", "Batch",
        "Forward ms", "Backward ms", "Total ms", "Peak VRAM", "ms/token", "Speedup vs Direct 12544"
    ]

    for B in args.batch_sizes:
        direct_ms = all_results.get((B, 112), {}).get("total_ms", 1.0)
        rows: List[List[str]] = []
        for label, target_grid, n_tok, desc in candidates:
            res = all_results.get((B, target_grid))
            if res is not None:
                speedup = direct_ms / res["total_ms"]
                speedup_str = f"{speedup:.2f}x faster" if target_grid != 112 else "1.00x (Baseline)"
                rows.append([
                    label,
                    res["source_grid"],
                    res["vit_grid"],
                    str(res["vit_tokens"]),
                    str(B),
                    f"{res['fwd_ms']:.2f}",
                    f"{res['bwd_ms']:.2f}",
                    f"{res['total_ms']:.2f}",
                    f"{res['peak_vram_mb']:.1f} MB",
                    f"{res['ms_per_token']:.6f}",
                    speedup_str,
                ])
            else:
                rows.append([label, "112x112", f"{target_grid}x{target_grid}", str(n_tok), str(B), "OOM", "OOM", "OOM", "OOM", "OOM", "N/A"])

        print_table(f"TABLE: Spatial Compression before Global ViT Expert (Batch Size = {B}, Method: {args.downsample_mode})", headers, rows)

    # -----------------------------------------------------------------------
    # SINGLE CONSOLIDATED TABLE (As explicitly requested by user)
    # -----------------------------------------------------------------------
    primary_b = args.batch_sizes[0]
    print_banner(f"FINAL CONSOLIDATED TABLE: FEASIBILITY SUMMARY (BATCH SIZE = {primary_b})")
    cons_headers = [
        "Configuration", "ViT Grid", "ViT Tokens", "Forward ms", "Backward ms", "Total ms", "Peak VRAM", "Speedup vs Direct"
    ]
    cons_rows: List[List[str]] = []
    primary_direct_ms = all_results.get((primary_b, 112), {}).get("total_ms", 1.0)

    name_mapping = {
        112: "Direct 12544",
        56:  "Compressed 3136 (56x56)",
        28:  "Compressed 784 (28x28)",
        14:  "Compressed 196 (14x14)",
    }

    for _, target_grid, n_tok, _ in candidates:
        res = all_results.get((primary_b, target_grid))
        cfg_name = name_mapping.get(target_grid, f"Target {target_grid}")
        if res is not None:
            speedup = primary_direct_ms / res["total_ms"]
            speedup_str = f"{speedup:.2f}x" if target_grid != 112 else "1.0x (Baseline)"
            cons_rows.append([
                cfg_name,
                res["vit_grid"],
                str(res["vit_tokens"]),
                f"{res['fwd_ms']:.2f}",
                f"{res['bwd_ms']:.2f}",
                f"{res['total_ms']:.2f}",
                f"{res['peak_vram_mb']:.1f} MB",
                speedup_str,
            ])
        else:
            cons_rows.append([cfg_name, f"{target_grid}x{target_grid}", str(n_tok), "OOM", "OOM", "OOM", "OOM", "N/A"])

    print_table(f"CONSOLIDATED SUMMARY TABLE: CNN (112x112) -> Downsample -> Global ViT Expert (B={primary_b}, T4)", cons_headers, cons_rows)
    print("\n[Done] Spatial compression micro-benchmarks completed successfully.")


if __name__ == "__main__":
    main()
