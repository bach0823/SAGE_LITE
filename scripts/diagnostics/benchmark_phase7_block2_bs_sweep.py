#!/usr/bin/env python3
"""
scripts/diagnostics/benchmark_phase7_block2_bs_sweep.py

Automated Batch Size Capacity & Throughput Sweep for Phase 7
S2-Gate Block 2 (Conv 3x3, Dilation 2) on Top of Baseline B2 Checkpoint (D=4, K=2)

Sweeps Batch Sizes: [16, 20, 24, 28, 32]
Protocol:
- Base model loaded from best_model_b2_global.pth (D=4, K=2, S2-Gate Block 1 active)
- S2-Gate Block 2 instantiated with Kernel 3x3, Dilation 2
- Strict Freeze: model.eval() -> s2_gate_block2.train() (only 41,601 trainable parameters)
- Optimizer: AdamW(s2_gate_block2.parameters(), lr=1e-3, weight_decay=1e-2)
- Measures Peak VRAM, Headroom, Training Throughput (samples/s)
- Stops on OOM and prints a clean comparison table.
"""

import argparse
import gc
import os
import sys
import time
import cv2
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
for p in [project_root, os.path.join(project_root, ".."), os.getcwd(), "/content/SAGE_LITE"]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from sage.networks import S2GateModule
from tools.run_phase6_c_topology_diagnostic import load_model_from_checkpoint
from sage.utils.training_utils import set_seed


def print_banner(text: str, ch: str = "="):
    line = ch * 80
    print(f"\n{line}\n{text}\n{line}")


class Crack500TrainDataset(Dataset):
    def __init__(self, img_dir: str, mask_dir: str, img_size: int = 448):
        self.img_size = img_size
        self.samples = []
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        for f in sorted(os.listdir(img_dir)):
            if not f.lower().endswith((".jpg", ".png")):
                continue
            base = os.path.splitext(f)[0]
            mp = os.path.join(mask_dir, base + ".png")
            if not os.path.exists(mp):
                mp = os.path.join(mask_dir, base + ".jpg")
            if os.path.exists(mp):
                self.samples.append((os.path.join(img_dir, f), mp, base))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path, stem = self.samples[idx]
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        H, W = img.shape[:2]
        crop_sz = self.img_size
        if H < crop_sz or W < crop_sz:
            img = cv2.resize(img, (max(W, crop_sz), max(H, crop_sz)), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (max(W, crop_sz), max(H, crop_sz)), interpolation=cv2.INTER_NEAREST)
            H, W = img.shape[:2]

        top = np.random.randint(0, H - crop_sz + 1)
        left = np.random.randint(0, W - crop_sz + 1)
        img_crop = img[top : top + crop_sz, left : left + crop_sz]
        mask_crop = mask[top : top + crop_sz, left : left + crop_sz]

        norm_img = (img_crop.astype(np.float32) / 255.0 - self.mean) / self.std
        t_img = torch.from_numpy(norm_img.transpose(2, 0, 1)).float()
        t_mask = torch.from_numpy((mask_crop > 127).astype(np.float32)).unsqueeze(0)
        return {"image": t_img, "mask": t_mask, "stem": stem}


def test_single_bs_block2(
    config_path: str,
    checkpoint_path: str,
    train_dataset,
    device,
    batch_size: int,
    kernel_size: int = 3,
    dilation: int = 2,
    lr: float = 1e-3,
    num_batches: int = 5,
    warmup_batches: int = 2,
):
    print_banner(f"TESTING S2-GATE BLOCK 2 (3x3 d=2) WITH BATCH SIZE: {batch_size}")
    total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2) if device.type == "cuda" else 0.0

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    model = None
    try:
        model = load_model_from_checkpoint(config_path, checkpoint_path, device=device)
        model.eval()

        s2_gate_b2 = S2GateModule(
            s2_channels=96,
            skip_channels=48,
            mid_channels=32,
            kernel_size=kernel_size,
            dilation=dilation,
        ).to(device)

        model.decoder.s2_gate_block2 = s2_gate_b2
        model.decoder.use_s2_gate_block2 = True

        for p in model.parameters():
            p.requires_grad = False
        for p in s2_gate_b2.parameters():
            p.requires_grad = True

        total_trainable = sum(p.numel() for p in s2_gate_b2.parameters() if p.requires_grad)
        assert total_trainable == 41601, f"Expected 41,601 params, got {total_trainable}"

        optimizer = torch.optim.AdamW(s2_gate_b2.parameters(), lr=lr, weight_decay=1e-2)

        data_iter = iter(train_loader)
        step_times = []

        for b_idx in range(1, num_batches + 1):
            t0 = time.perf_counter()
            model.eval()
            s2_gate_b2.train()

            batch = next(data_iter)
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            bce = F.binary_cross_entropy_with_logits(logits, masks)
            probs = torch.sigmoid(logits)
            inter = (probs.view(-1) * masks.view(-1)).sum()
            dice_l = 1.0 - (2.0 * inter + 1e-5) / (probs.view(-1).sum() + masks.view(-1).sum() + 1e-5)
            total_loss = bce + dice_l

            if not torch.isfinite(total_loss):
                raise RuntimeError(f"Non-finite loss at batch {b_idx}")

            total_loss.backward()
            optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if b_idx > warmup_batches:
                step_times.append(dt)

        throughput = (batch_size / (sum(step_times) / len(step_times))) if step_times else 0.0
        peak_alloc_gb = (torch.cuda.max_memory_allocated() / (1024 ** 3)) if device.type == "cuda" else 0.0
        peak_res_gb = (torch.cuda.max_memory_reserved() / (1024 ** 3)) if device.type == "cuda" else 0.0
        headroom_gb = (total_vram_mb / 1024.0) - peak_res_gb if device.type == "cuda" else 0.0

        print(f"  [RESULT] BS={batch_size} PASS! Peak Alloc: {peak_alloc_gb:.2f} GB | Peak Res: {peak_res_gb:.2f} GB | Headroom: {headroom_gb:.2f} GB")
        print(f"  [SPEED]  Throughput: {throughput:.2f} samples/s ({(sum(step_times)/len(step_times)):.2f}s per batch)")

        return {
            "batch_size": batch_size,
            "status": "PASS",
            "peak_alloc_gb": peak_alloc_gb,
            "peak_res_gb": peak_res_gb,
            "headroom_gb": headroom_gb,
            "headroom_pct": (headroom_gb / (total_vram_mb / 1024.0) * 100) if total_vram_mb > 0 else 0.0,
            "throughput": throughput,
        }

    except torch.cuda.OutOfMemoryError:
        print(f"  [RESULT] BS={batch_size} FAILED with CUDA OutOfMemoryError (OOM)!")
        return {
            "batch_size": batch_size,
            "status": "OOM",
            "peak_alloc_gb": total_vram_mb / 1024.0,
            "peak_res_gb": total_vram_mb / 1024.0,
            "headroom_gb": 0.0,
            "headroom_pct": 0.0,
            "throughput": 0.0,
        }
    except Exception as e:
        print(f"  [RESULT] BS={batch_size} FAILED with Error: {e}")
        return {
            "batch_size": batch_size,
            "status": f"ERROR: {str(e)[:30]}",
            "peak_alloc_gb": 0.0,
            "peak_res_gb": 0.0,
            "headroom_gb": 0.0,
            "headroom_pct": 0.0,
            "throughput": 0.0,
        }
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Sweep Batch Sizes for Phase 7 S2-Gate Block 2")
    parser.add_argument("--config", type=str, default="configs/p3_ablation/phase6_full_s1/a1_s2g_end_to_end.yaml")
    parser.add_argument("--checkpoint", type=str, default="results/phase6_combination/phase6_comb_a1_s2g_end_to_end/best_model_b2_global.pth")
    parser.add_argument("--data-root", type=str, default="/content/dataset/Crack500")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 20, 24, 28, 32])
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dilation", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-batches", type=int, default=5)
    parser.add_argument("--warmup-batches", type=int, default=2)
    args = parser.parse_args()

    # Path fallbacks
    for cand_ckpt in [args.checkpoint, os.path.join(project_root, args.checkpoint)]:
        if os.path.exists(cand_ckpt):
            args.checkpoint = cand_ckpt
            break
    for cand_cfg in [args.config, os.path.join(project_root, args.config)]:
        if os.path.exists(cand_cfg):
            args.config = cand_cfg
            break

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_banner(f"PHASE 7 S2-GATE BLOCK 2 (3x3 d=2) BATCH SIZE SWEEP (GPU: {torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})")

    train_img_dir = os.path.join(args.data_root, "train", "images")
    train_mask_dir = os.path.join(args.data_root, "train", "masks")
    train_dataset = Crack500TrainDataset(train_img_dir, train_mask_dir, img_size=448)
    print(f"[Dataset] Loaded {len(train_dataset)} training samples.")

    results = []
    for bs in args.batch_sizes:
        res = test_single_bs_block2(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            train_dataset=train_dataset,
            device=device,
            batch_size=bs,
            kernel_size=args.kernel_size,
            dilation=args.dilation,
            lr=args.lr,
            num_batches=args.num_batches,
            warmup_batches=args.warmup_batches,
        )
        results.append(res)
        if res["status"] == "OOM":
            print(f"\n[INFO] Hit hardware VRAM ceiling at BS={bs}. Stopping further sweep.")
            break

    print_banner("PHASE 7 S2-GATE BLOCK 2: BATCH SIZE & THROUGHPUT SUMMARY")
    print(f"{'BS':<6} | {'Status':<6} | {'Peak Alloc':<11} | {'Peak Res':<10} | {'Headroom':<15} | {'Throughput':<15}")
    print("-" * 80)
    for r in results:
        status = r["status"]
        alloc = f"{r['peak_alloc_gb']:.2f} GB" if status == "PASS" else "—"
        res_mem = f"{r['peak_res_gb']:.2f} GB" if status == "PASS" else "—"
        headroom = f"{r['headroom_gb']:.2f} GB ({r['headroom_pct']:.1f}%)" if status == "PASS" else "0 GB (OOM)"
        speed = f"{r['throughput']:.2f} s/s" if status == "PASS" else "—"
        print(f"{r['batch_size']:<6} | {status:<6} | {alloc:<11} | {res_mem:<10} | {headroom:<15} | {speed:<15}")


if __name__ == "__main__":
    main()
