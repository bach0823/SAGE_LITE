"""
B2 (Full SAGE-Lite) Runtime Preflight & GPU Probe Script

Executes pre-training verification on target GPU (e.g. Google Colab T4 / Local):
1. OOM / VRAM Probe:
   - Tests batch size 20 (or config default) at 448x448 under AMP FP16.
   - Runs >= 3 iterations of forward_with_routing_info + seg loss + LB loss + backward + optimizer.step().
   - Measures memory allocated/reserved before and after, detects memory leaks.
   - Reports peak VRAM and free remaining VRAM.
2. Mini-Training Smoke:
   - Uses exact B2 optimizer groups (Backbone @ 1e-5, Decoder @ 1e-4, SAGE @ 1e-4, WD=0.05/0.0).
   - Verifies finite logits, seg loss, LB loss, total loss, and finite gradients.
   - Confirms optimizer and scaler steps.
3. Routing Diagnostics:
   - Logs expert usage counts across all 16 experts and all 16 routers (4 CNN + 12 ViT).
   - Verifies top_k=4 and finite load balance loss.
   - Warns if any expert has 0 selections across the mini-run without modifying hyperparameters.
4. Checkpoint Integrity:
   - Saves checkpoint dict and reloads into a fresh model with strict=True.
   - Asserts 0 missing / 0 unexpected keys and exact numerical match.

Usage:
    python scripts/preflight_b2.py --config configs/b2_crack500_depth12.yaml
"""

import argparse
import gc
import os
import sys
import tempfile
import time
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure sage_lite is on sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sage.networks.b2_unet import create_b2_unet, B2ConvNeXtViTUNet
from sage.components.sage_layer import SageLayer
from scripts.train_crack import get_optimizer_groups, get_scheduler, CrackBinaryLoss


def print_banner(text: str, ch: str = "="):
    line = ch * 70
    print(f"\n{line}\n{text}\n{line}")


def run_preflight(config_path: str, probe_batch_size: int = None, vram_iterations: int = 5, mini_batches: int = 5):
    print_banner(f"B2 RUNTIME PREFLIGHT: {config_path}")
    vram_iters = vram_iterations
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # -------------------------------------------------------------------------
    # Hardware & Device Setup
    # -------------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] Target Device: {device}")
    
    if device.type == 'cuda':
        dev_name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        total_vram_gb = total_vram_mb / 1024.0
        
        # Turing GTX 1650/1660 safeguard for local runs without Tensor Cores
        if '1650' in dev_name or '1660' in dev_name:
            torch.backends.cudnn.enabled = False
            print(f"[Device] Detected {dev_name} (CC {cap[0]}.{cap[1]}). Set cudnn.enabled=False for FP16 stability.")
        else:
            torch.backends.cudnn.benchmark = True
            
        print(f"[Device] GPU: {dev_name} | Compute Cap: {cap[0]}.{cap[1]} | Total VRAM: {total_vram_mb:.1f} MB ({total_vram_gb:.2f} GB)")
        print(f"[Device] cuDNN: {torch.backends.cudnn.version()} (enabled={torch.backends.cudnn.enabled}, benchmark={torch.backends.cudnn.benchmark})")
    else:
        total_vram_mb = 0.0
        total_vram_gb = 0.0
        print("[Device] WARNING: Running on CPU! VRAM metrics will not be measured.")

    # Model Configuration
    img_size = config.get('img_size', 448)
    vit_depth = int(config.get('num_transformer_layers', 12))
    sage_cfg = config.get('sage_config', {})
    bs = probe_batch_size if probe_batch_size is not None else int(config.get('batch_size', 20))
    base_lr = float(config.get('lr', 1e-4))
    lr_backbone = base_lr * 0.1
    lr_decoder = base_lr
    lr_sage = base_lr

    print(f"\n[Model Config] Model: B2 (Full SAGE-Lite) | ViT Depth: {vit_depth}")
    print(f"[Model Config] Batch Size: {bs} | Image Size: {img_size}x{img_size} | Top-K: {sage_cfg.get('top_k', 4)}")
    print(f"[Model Config] Gating: {sage_cfg.get('gating_type', 'sigmoid')} | Logit Mod: {sage_cfg.get('logit_modulation', True)}")

    # =========================================================================
    # STEP 1: OOM / VRAM PROBE
    # =========================================================================
    print_banner(f"STEP 1: OOM / VRAM PROBE (Batch Size = {bs}, {vram_iters} Iterations)")
    
    print("Instantiating B2 model for VRAM probe...")
    model = create_b2_unet(
        num_classes=1,
        img_size=img_size,
        num_transformer_layers=vit_depth,
        pretrained=False,
        sage_config=sage_cfg
    ).to(device)
    model.train()
    
    optimizer_groups = get_optimizer_groups(model, lr_backbone=lr_backbone, lr_decoder=lr_decoder, lr_sage=lr_sage)
    optimizer = torch.optim.AdamW(optimizer_groups)
    criterion = CrackBinaryLoss(bce_weight=1.0, dice_weight=1.5)
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda'))

    if device.type == 'cuda':
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Pre-generate synthetic batch with crack-like sparsity
    x_probe = torch.randn(bs, 3, img_size, img_size, device=device)
    # Binary ground truth with ~2% foreground pixels (typical crack ratio)
    y_probe = (torch.rand(bs, 1, img_size, img_size, device=device) > 0.98).float()

    vram_history = []
    oom_occurred = False

    try:
        for iter_idx in range(1, vram_iters + 1):
            if device.type == 'cuda':
                mem_alloc_before = torch.cuda.memory_allocated() / (1024 ** 2)
                mem_res_before = torch.cuda.memory_reserved() / (1024 ** 2)
            else:
                mem_alloc_before, mem_res_before = 0.0, 0.0

            t0 = time.time()
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16):
                ret = model.forward_with_routing_info(x_probe)
                logits = ret['logits']
                seg_loss = criterion(logits, y_probe)
                lb_loss = model.compute_total_load_balance_loss(ret['routing_infos'])
                total_loss = seg_loss + 1.0 * lb_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.time() - t0

            if device.type == 'cuda':
                mem_alloc_after = torch.cuda.memory_allocated() / (1024 ** 2)
                mem_res_after = torch.cuda.memory_reserved() / (1024 ** 2)
                cur_peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
                cur_peak_res = torch.cuda.max_memory_reserved() / (1024 ** 2)
                free_remaining = total_vram_mb - cur_peak_res
            else:
                mem_alloc_after, mem_res_after, cur_peak_alloc, cur_peak_res, free_remaining = 0, 0, 0, 0, 0

            vram_history.append({
                'iter': iter_idx,
                'time_sec': elapsed,
                'loss': total_loss.item(),
                'seg_loss': seg_loss.item(),
                'lb_loss': lb_loss.item(),
                'alloc_before': mem_alloc_before,
                'alloc_after': mem_alloc_after,
                'peak_alloc': cur_peak_alloc,
                'peak_res': cur_peak_res,
                'free_vram': free_remaining,
            })

            print(
                f"  Iter {iter_idx}/{vram_iters} ({elapsed:.2f}s) | "
                f"Loss: {total_loss.item():.4f} (Seg: {seg_loss.item():.4f}, LB: {lb_loss.item():.4f}) | "
                f"Alloc: {mem_alloc_after:.1f} MB | Peak Alloc: {cur_peak_alloc:.1f} MB ({cur_peak_alloc/1024:.2f} GB) | "
                f"Peak Res: {cur_peak_res:.1f} MB ({cur_peak_res/1024:.2f} GB) | Free: {free_remaining/1024:.2f} GB"
            )

    except torch.cuda.OutOfMemoryError as e:
        oom_occurred = True
        print(f"\n[OOM ERROR] Out of Memory encountered at batch size {bs}!\nDetails: {e}")
        raise RuntimeError(f"VRAM Probe FAILED: Batch size {bs} caused OutOfMemory on {device}.") from e

    # Memory Leak Check across iterations 2 to N
    if len(vram_history) >= 3 and device.type == 'cuda':
        alloc_diff = vram_history[-1]['alloc_after'] - vram_history[1]['alloc_after']
        print(f"\n[VRAM Memory Stability Check]")
        print(f"  Allocated Memory Iter 2: {vram_history[1]['alloc_after']:.2f} MB")
        print(f"  Allocated Memory Iter {vram_iters}: {vram_history[-1]['alloc_after']:.2f} MB")
        print(f"  Difference (Iter {vram_iters} - Iter 2): {alloc_diff:+.2f} MB")
        assert alloc_diff < 50.0, f"Possible memory leak detected! Allocated memory increased by {alloc_diff:.2f} MB after warmup."
        print("  [PASS] No monotonically increasing memory leak detected.")

    final_peak_alloc = vram_history[-1]['peak_alloc']
    final_peak_res = vram_history[-1]['peak_res']
    final_free = vram_history[-1]['free_vram']
    print(f"\n[VRAM Summary for Batch={bs}]")
    print(f"  - Peak Memory Allocated: {final_peak_alloc:.1f} MB ({final_peak_alloc/1024:.2f} GB)")
    print(f"  - Peak Memory Reserved:  {final_peak_res:.1f} MB ({final_peak_res/1024:.2f} GB)")
    print(f"  - Free VRAM Remaining:   {final_free:.1f} MB ({final_free/1024:.2f} GB) out of {total_vram_gb:.2f} GB")
    print("  [PASS] Step 1: OOM / VRAM Probe successfully passed without OOM.")

    # =========================================================================
    # STEP 2: MINI-TRAINING SMOKE
    # =========================================================================
    print_banner(f"STEP 2: MINI-TRAINING SMOKE ({mini_batches} Training Batches)")
    
    scheduler = get_scheduler(optimizer, epochs=30, warmup_epochs=3)
    
    for batch_i in range(1, mini_batches + 1):
        x_mb = torch.randn(bs, 3, img_size, img_size, device=device)
        y_mb = (torch.rand(bs, 1, img_size, img_size, device=device) > 0.98).float()

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16):
            ret = model.forward_with_routing_info(x_mb)
            logits = ret['logits']
            seg_loss = criterion(logits, y_mb)
            lb_loss = model.compute_total_load_balance_loss(ret['routing_infos'])
            loss = seg_loss + 1.0 * lb_loss

        # Verify finite values
        assert torch.isfinite(logits).all(), f"Batch {batch_i}: Logits contain NaN or Inf!"
        assert torch.isfinite(seg_loss), f"Batch {batch_i}: Seg loss is NaN or Inf!"
        assert torch.isfinite(lb_loss), f"Batch {batch_i}: LB loss is NaN or Inf!"
        assert torch.isfinite(loss), f"Batch {batch_i}: Total loss is NaN or Inf!"

        scaler.scale(loss).backward()

        # Verify finite gradients across all parameters with grad
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Batch {batch_i}: Gradient for {name} contains NaN or Inf!"

        scaler.step(optimizer)
        scaler.update()
        scheduler.step(0)

        print(f"  Batch {batch_i}/{mini_batches} | Total Loss: {loss.item():.4f} (Seg: {seg_loss.item():.4f}, LB: {lb_loss.item():.4f}) | Logits & Grads: FINITE [OK]")

    print("  [PASS] Step 2: Mini-training smoke passed (all losses, logits, and gradients strictly finite).")

    # =========================================================================
    # STEP 3: ROUTING DIAGNOSTICS
    # =========================================================================
    print_banner("STEP 3: ROUTING DIAGNOSTICS & EXPERT USAGE INSPECTION")
    
    all_routers = []
    for i, stage in enumerate(model.backbone.convnext.stages):
        if isinstance(stage, SageLayer):
            all_routers.append((f"CNN Stage {i}", stage.router, stage.my_index))
    for j, blk in enumerate(model.backbone.transformer_blocks):
        if isinstance(blk, SageLayer):
            all_routers.append((f"ViT Block {j:02d}", blk.router, blk.my_index))

    total_routers = len(all_routers)
    assert total_routers == 4 + vit_depth, f"Expected {4 + vit_depth} routers, got {total_routers}"
    print(f"  Total Injected Routers: {total_routers} (4 CNN Stages + {vit_depth} ViT Blocks)")

    # Expert pool size
    expert_pool_size = all_routers[0][1].expert_pool_size
    print(f"  Expert Pool Size:       {expert_pool_size}")
    print(f"  Target Top-K:           {sage_cfg.get('top_k', 4)}")

    # Verify top_k across all routers
    for name, r, my_idx in all_routers:
        assert r.top_k == 4, f"Router {name} top_k={r.top_k}, expected 4"

    # Aggregate usage counts across all routers
    # Matrix of shape (Num_Routers, Expert_Pool_Size)
    usage_matrix = torch.zeros(total_routers, expert_pool_size, dtype=torch.long)
    for r_idx, (name, r, my_idx) in enumerate(all_routers):
        usage_matrix[r_idx] = r.expert_usage_count.cpu()

    # Sum selections per expert across the entire network
    total_per_expert = usage_matrix.sum(dim=0).tolist()
    total_routing_events = sum(total_per_expert)

    print("\n--- Per-Expert Selection Breakdown (Entire Mini-Run) ---")
    zero_usage_experts = []
    for exp_idx, count in enumerate(total_per_expert):
        pct = (count / max(total_routing_events, 1)) * 100.0
        exp_type = "CNN Stage" if exp_idx < 4 else f"ViT Block {exp_idx-4:02d}"
        shared_tag = "[SHARED]" if exp_idx in sage_cfg.get('shared_expert_indices', [0, 1, 2, 3]) else "        "
        print(f"  Expert {exp_idx:02d} ({exp_type:>12}) {shared_tag}: {count:6d} selections ({pct:5.1f}%)")
        if count == 0:
            zero_usage_experts.append(exp_idx)

    # Policy Check: Warning on 0-usage without changing hyperparameters
    if zero_usage_experts:
        print(
            f"\n[ROUTING WARNING] The following experts had 0 selections during the mini-run: {zero_usage_experts}."
            f"\n  (Per project rules: No hyperparameters are modified to artificially 'fix' routing)."
        )
    else:
        print(f"\n  [PASS] All {expert_pool_size} experts received at least 1 routing selection.")

    print(f"  Total routing selection events recorded: {total_routing_events}")
    print("  [PASS] Step 3: Routing diagnostics verified (top_k=4 verified, expert usage recorded).")

    # =========================================================================
    # STEP 4: CHECKPOINT INTEGRITY
    # =========================================================================
    print_banner("STEP 4: CHECKPOINT SAVE & LOAD INTEGRITY")
    
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # 1. Save checkpoint dict
        ckpt_dict = {
            'epoch': 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_type': 'B2',
            'num_transformer_layers': vit_depth,
            'sage_config': sage_cfg,
            'best_dice': 0.7420,
        }
        torch.save(ckpt_dict, tmp_path)
        ckpt_size_mb = os.path.getsize(tmp_path) / (1024 ** 2)
        print(f"  Saved test checkpoint to: {tmp_path} ({ckpt_size_mb:.2f} MB)")

        # 2. Instantiate a fresh model
        fresh_model = create_b2_unet(
            num_classes=1,
            img_size=img_size,
            num_transformer_layers=vit_depth,
            pretrained=False,
            sage_config=sage_cfg
        ).to(device)

        # 3. Load checkpoint with strict=True
        loaded_ckpt = torch.load(tmp_path, map_location=device)
        load_res = fresh_model.load_state_dict(loaded_ckpt['model_state_dict'], strict=True)
        assert len(load_res.missing_keys) == 0, f"Missing keys on load: {load_res.missing_keys}"
        assert len(load_res.unexpected_keys) == 0, f"Unexpected keys on load: {load_res.unexpected_keys}"
        print(f"  Loaded state_dict strictly: 0 missing keys, 0 unexpected keys.")

        # 4. Verify exact output match
        model.eval()
        fresh_model.eval()
        with torch.no_grad():
            x_test_ckpt = torch.randn(2, 3, img_size, img_size, device=device)
            out_orig = model(x_test_ckpt)
            out_loaded = fresh_model(x_test_ckpt)
            max_diff = (out_orig - out_loaded).abs().max().item()

        print(f"  Max absolute difference between original and loaded model: {max_diff:.8e}")
        assert max_diff < 1e-6, f"Loaded model output deviates from original: {max_diff}"
        print("  [PASS] Step 4: Checkpoint save & load integrity 100% verified.")

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # =========================================================================
    # PREFLIGHT VERDICT
    # =========================================================================
    print_banner("PREFLIGHT VERDICT: ALL 4 CHECKS PASSED [READY FOR TRAINING]")
    print(f"1. OOM / VRAM Probe:   PASS (Batch={bs}, Peak Alloc={final_peak_alloc/1024:.2f} GB, Free={final_free/1024:.2f} GB)")
    print(f"2. Mini-Training Smoke: PASS ({mini_batches} batches, finite losses, valid gradients)")
    print(f"3. Routing Diagnostics: PASS (16 routers, top_k=4 verified, usage matrix logged)")
    print(f"4. Checkpoint Roundtrip: PASS (0 missing/unexpected keys, exact numerical match)")
    print("\nYou can now safely launch full B2 training on Colab using:")
    print(f"  python scripts/train_crack.py --config {config_path}")
    print("=" * 70)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="B2 Runtime Preflight & GPU Probe")
    parser.add_argument('--config', type=str, default='configs/b2_crack500_depth12.yaml', help='Path to B2 config YAML')
    parser.add_argument('--probe_batch_size', type=int, default=None, help='Override batch size for VRAM probe')
    parser.add_argument('--vram_iterations', type=int, default=5, help='Number of iterations for VRAM probe (>= 3)')
    parser.add_argument('--mini_batches', type=int, default=5, help='Number of mini-training smoke batches')
    args = parser.parse_args()

    success = run_preflight(
        config_path=args.config,
        probe_batch_size=args.probe_batch_size,
        vram_iterations=args.vram_iterations,
        mini_batches=args.mini_batches
    )
    sys.exit(0 if success else 1)
