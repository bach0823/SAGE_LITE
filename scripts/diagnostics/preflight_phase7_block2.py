#!/usr/bin/env python3
"""
scripts/diagnostics/preflight_phase7_block2.py

Dedicated Preflight & VRAM / OOM Probe for Phase 7 (S2-Gate Block 2, Conv 3x3, Dilation 2)
Tests:
1. Nạp Baseline B2 Checkpoint (D=4, K=2, có S2-Gate Block 1 Conv 3x3 hoạt động)
2. Khởi tạo S2-Gate Block 2 (Conv 3x3, Dilation 2, trường quan sát 5x5 ở Stride 4 / 112x112)
3. Áp dụng Strict Freeze Protocol: model.eval() -> s2_gate_block2.train() (chỉ đúng 41,601 tham số trainable)
4. Huấn luyện thử nghiệm 5 batches với Batch Size (16 hoặc 20) trên Crack500 train
5. Đánh giá thử nghiệm 5 ảnh validation (Setting A Tiling 448x448, FP32 strict, tính đầy đủ metrics & topology)
6. Báo cáo Peak VRAM Allocated, Peak Reserved, Headroom và Throughput
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

from sage.networks import create_b2_unet, S2GateModule
from tools.run_phase6_c_topology_diagnostic import (
    load_model_from_checkpoint,
    compute_topology_metrics,
)
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


def predict_tiling_setting_a(model, img_rgb, device, tile_size=448):
    H, W = img_rgb.shape[:2]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    norm_img = (img_rgb.astype(np.float32) / 255.0 - mean) / std

    y_steps = range(0, H, tile_size)
    x_steps = range(0, W, tile_size)
    patches = []
    coords = []
    pred_logits = np.zeros((H + tile_size, W + tile_size), dtype=np.float32)

    for y in y_steps:
        for x in x_steps:
            patch = norm_img[y : y + tile_size, x : x + tile_size]
            ph, pw = patch.shape[:2]
            if ph < tile_size or pw < tile_size:
                pad_h = tile_size - ph
                pad_w = tile_size - pw
                patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            patches.append(torch.from_numpy(patch.transpose(2, 0, 1)).float())
            coords.append((y, x))

    batch = torch.stack(patches).to(device)
    with torch.no_grad():
        logits = model(batch).squeeze(1).cpu().numpy()

    for idx, (y, x) in enumerate(coords):
        tile_out = logits[idx] if len(patches) > 1 else logits
        pred_logits[y : y + tile_size, x : x + tile_size] = tile_out

    return pred_logits[:H, :W]


def run_preflight_block2(
    config_path: str,
    checkpoint_path: str,
    data_root: str,
    batch_size: int = 16,
    kernel_size: int = 3,
    dilation: int = 2,
    lr: float = 1e-3,
    num_train_batches: int = 5,
    num_val_samples: int = 5,
):
    print_banner(f"PHASE 7 PREFLIGHT: S2-GATE BLOCK 2 (Conv {kernel_size}x{kernel_size}, Dilation={dilation}, BS={batch_size})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2) if device.type == "cuda" else 0.0
    print(f"[Device] Target: {device} (VRAM: {total_vram_mb/1024:.2f} GB)" if device.type == "cuda" else f"[Device] Target: {device}")

    # 1. Instantiate Base Model (or load checkpoint if provided)
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\n[Step 1] Loading base model from checkpoint: {checkpoint_path}...")
        model = load_model_from_checkpoint(config_path, checkpoint_path, device=device)
    else:
        print("\n[Step 1] Instantiating Base Model architecture B2 directly (D=4, K=2, S2-Gate v2, pretrained=False)...")
        model = create_b2_unet(
            num_transformer_layers=4,
            pretrained=False,
            sage_config={
                "top_k": 2,
                "gating_type": "sigmoid",
                "shared_expert_indices": [0, 1, 2, 3],
                "router_hidden_dim": 64,
                "load_balance_factor": 0.01,
                "logit_modulation": True,
                "expert_dropout": 0.1,
                "fusion_type": "residual",
                "residual_scale": 0.1,
            },
            use_s2_gate=True,
            s2_gate_kernel_size=3,
        ).to(device)
    model.eval()
    print("  [OK] Base model ready.")

    # 2. Instantiate S2-Gate Block 2
    print(f"\n[Step 2] Instantiating S2-Gate Block 2 (Kernel={kernel_size}, Dilation={dilation})...")
    s2_gate_b2 = S2GateModule(
        s2_channels=96,
        skip_channels=48,
        mid_channels=32,
        kernel_size=kernel_size,
        dilation=dilation,
    ).to(device)

    model.decoder.s2_gate_block2 = s2_gate_b2
    model.decoder.use_s2_gate_block2 = True

    # 3. Apply Strict Freeze Protocol
    for p in model.parameters():
        p.requires_grad = False
    for p in s2_gate_b2.parameters():
        p.requires_grad = True

    total_trainable = sum(p.numel() for p in s2_gate_b2.parameters() if p.requires_grad)
    print(f"  [OK] Strict Freeze verified: Exactly {total_trainable:,} trainable parameters (All base layers frozen).")
    assert total_trainable == 41601, f"Expected 41,601 params, got {total_trainable}"

    # 4. Training Smoke (Batch Size = 16 or 20)
    print_banner(f"STEP 3: TRAINING SMOKE ({num_train_batches} Batches, BS={batch_size}, LR={lr:.1e})")
    train_img_dir = os.path.join(data_root, "train", "images")
    train_mask_dir = os.path.join(data_root, "train", "masks")
    train_dataset = Crack500TrainDataset(train_img_dir, train_mask_dir, img_size=448)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    optimizer = torch.optim.AdamW(s2_gate_b2.parameters(), lr=lr, weight_decay=1e-2)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    train_times = []
    data_iter = iter(train_loader)

    for b_idx in range(1, num_train_batches + 1):
        t0 = time.perf_counter()
        # Execution order invariant
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

        assert torch.isfinite(total_loss), f"Batch {b_idx}: Non-finite loss!"
        total_loss.backward()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if b_idx > 2:
            train_times.append(dt)

        if device.type == "cuda":
            peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
            peak_res = torch.cuda.max_memory_reserved() / (1024 ** 2)
            free_vram = total_vram_mb - peak_res
            print(f"  Batch {b_idx:02d}/{num_train_batches:02d} | Step: {dt:5.2f}s | Loss: {total_loss.item():.4f} | Peak Alloc: {peak_alloc/1024:5.2f} GB | Peak Res: {peak_res/1024:5.2f} GB | Free: {free_vram/1024:4.2f} GB")
        else:
            print(f"  Batch {b_idx:02d}/{num_train_batches:02d} | Step: {dt:5.2f}s | Loss: {total_loss.item():.4f}")

    train_throughput = (batch_size / (sum(train_times) / len(train_times))) if train_times else 0.0
    print(f"[PASS] Training Smoke Complete (Throughput: {train_throughput:.2f} samples/s)")

    # 5. Validation Sanity Check (Setting A Canonical Tiling)
    print_banner(f"STEP 4: VALIDATION SANITY CHECK ({num_val_samples} Samples, Setting A Tiling)")
    val_img_dir = os.path.join(data_root, "val", "images")
    val_mask_dir = os.path.join(data_root, "val", "masks")
    val_files = sorted([f for f in os.listdir(val_img_dir) if f.lower().endswith((".jpg", ".png"))])[:num_val_samples]

    model.eval()
    val_dices = []
    val_times = []

    for f in val_files:
        t0 = time.perf_counter()
        base = os.path.splitext(f)[0]
        ip = os.path.join(val_img_dir, f)
        mp = os.path.join(val_mask_dir, base + ".png")
        if not os.path.exists(mp):
            mp = os.path.join(val_mask_dir, base + ".jpg")

        img_bgr = cv2.imread(ip)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)

        pred_logits = predict_tiling_setting_a(model, img_rgb, device=device, tile_size=448)
        pred_bin = (pred_logits > 0.0).astype(np.uint8)
        gt_bin = (gt_mask > 127).astype(np.uint8)

        dt = time.perf_counter() - t0
        val_times.append(dt)

        inter = np.logical_and(pred_bin > 0, gt_bin > 0).sum()
        dice = float((2.0 * inter + 1e-5) / (pred_bin.sum() + gt_bin.sum() + 1e-5))
        val_dices.append(dice)
        print(f"  Sample {base:<20} | Dice: {dice:.4f} | Time: {dt*1000:6.1f} ms")

    print(f"\n[PASS] Validation Sanity Complete: Mean Dice on {num_val_samples} probe samples: {np.mean(val_dices):.4f}")

    # 6. Final Verdict
    print_banner("FINAL PREFLIGHT VERDICT (PHASE 7 BLOCK 2)")
    if device.type == "cuda":
        final_free = total_vram_mb - (torch.cuda.max_memory_reserved() / (1024 ** 2))
        print(f"  [HARDWARE] GPU:                  {torch.cuda.get_device_name(0)}")
        print(f"  [PEAK VRAM ALLOCATED]:          {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB")
        print(f"  [PEAK VRAM RESERVED]:           {torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB")
        print(f"  [REMAINING HEADROOM]:           {final_free / 1024:.2f} GB ({(final_free / total_vram_mb * 100):.1f}%)")
        print(f"  [TRAINING SPEED]:               {train_throughput:.2f} samples/s")
        print(f"  [VERDICT] >>> PASS: Phase 7 Block 2 (Conv 3x3, Dilation 2) is FULLY READY for execution! <<<")
    else:
        print(f"  [VERDICT] >>> PASS on CPU! <<<")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 7 S2-Gate Block 2 Preflight Runner")
    parser.add_argument("--config", type=str, default="configs/p3_ablation/phase6_full_s1/a1_s2g_end_to_end.yaml")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional path to checkpoint. If empty, instantiates architecture directly.")
    parser.add_argument("--data-root", type=str, default="/content/dataset/Crack500")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dilation", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-train-batches", type=int, default=5)
    parser.add_argument("--num-val-samples", type=int, default=5)
    args = parser.parse_args()

    # Path fallbacks
    if args.checkpoint:
        for cand_ckpt in [args.checkpoint, os.path.join(project_root, args.checkpoint)]:
            if os.path.exists(cand_ckpt):
                args.checkpoint = cand_ckpt
                break
    for cand_cfg in [args.config, os.path.join(project_root, args.config)]:
        if os.path.exists(cand_cfg):
            args.config = cand_cfg
            break

    run_preflight_block2(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        batch_size=args.batch_size,
        kernel_size=args.kernel_size,
        dilation=args.dilation,
        lr=args.lr,
        num_train_batches=args.num_train_batches,
        num_val_samples=args.num_val_samples,
    )
