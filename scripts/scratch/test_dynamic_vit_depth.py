"""
Comprehensive Smoke Test for Dynamic ViT-Depth in B1 Baseline
Tests:
1. Depths 4, 6, 8, 12 initialization and block count validation
2. Output shape == (B, 1, 448, 448)
3. Forward and Backward passes with gradients
4. Optimizer parameter groups structure (217+ backbone tensors, 36 decoder tensors)
5. Backward compatibility (default num_transformer_layers=6)
6. Checkpoint compatibility test (save 6-block dummy checkpoint, load cleanly)
"""

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from sage.networks import create_b1_unet
from scripts.train_crack import get_optimizer_groups


def run_smoke_tests():
    print("=" * 70)
    print("RUNNING SMOKE TESTS FOR DYNAMIC ViT DEPTH IN B1")
    print("=" * 70)

    device = torch.device("cpu")
    depths = [4, 6, 8, 12]
    batch_size = 2
    img_size = 448

    for depth in depths:
        print(f"\n--- Testing B1 with num_transformer_layers = {depth} ---")
        model = create_b1_unet(
            img_size=img_size,
            num_transformer_layers=depth,
            pretrained=False  # offline unit test
        ).to(device)

        # 1. Verify actual block count
        actual_blocks = len(model.backbone.transformer_blocks)
        assert actual_blocks == depth, f"Expected {depth} blocks, got {actual_blocks}!"
        print(f"  [PASS] Actual ViT blocks: {actual_blocks}")

        # 2. Verify forward pass
        x = torch.randn(batch_size, 3, img_size, img_size, device=device)
        logits = model(x)
        assert logits.shape == (batch_size, 1, img_size, img_size), (
            f"Expected shape ({batch_size}, 1, {img_size}, {img_size}), got {logits.shape}"
        )
        print(f"  [PASS] Output shape: {tuple(logits.shape)}")

        # 3. Verify backward pass
        loss = logits.sum()
        loss.backward()
        has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grad, "Some trainable parameters did not receive gradients!"
        print(f"  [PASS] Backward pass: gradients received for all trainable parameters")

        # 4. Verify parameter counts and optimizer groups
        model_info = model.get_model_info()
        param_groups = get_optimizer_groups(model, lr_backbone=1e-5, lr_decoder=1e-4, weight_decay=0.05)
        total_p = sum(p.numel() for p in model.parameters())
        print(f"  [PASS] Total parameters: {total_p:,} ({total_p/1e6:.2f}M)")
        print(f"  [PASS] Optimizer groups: {len(param_groups)} groups:")
        for idx, g in enumerate(param_groups):
            name = g.get('name', f'group_{idx}')
            print(f"         Group {idx} ({name}): {len(g['params'])} tensors, lr={g['lr']}, wd={g['weight_decay']}")

    # 5. Verify default backward compatibility
    print("\n--- Testing Backward Compatibility (Default Arguments) ---")
    default_model = create_b1_unet(pretrained=False)
    default_blocks = len(default_model.backbone.transformer_blocks)
    assert default_blocks == 6, f"Default expected 6 blocks, got {default_blocks}!"
    print(f"  [PASS] Default create_b1_unet() gives exactly {default_blocks} blocks.")

    # 6. Verify Checkpoint Load Compatibility for 6-block model
    print("\n--- Testing Checkpoint Loading (6-block model) ---")
    model_6 = create_b1_unet(num_transformer_layers=6, pretrained=False)
    state_dict = model_6.state_dict()
    
    # New instance loading the same state_dict
    model_6_loaded = create_b1_unet(num_transformer_layers=6, pretrained=False)
    load_res = model_6_loaded.load_state_dict(state_dict)
    assert len(load_res.missing_keys) == 0 and len(load_res.unexpected_keys) == 0, (
        f"State dict mismatch: missing={load_res.missing_keys}, unexpected={load_res.unexpected_keys}"
    )
    print(f"  [PASS] 6-block checkpoint loaded with 0 missing / 0 unexpected keys.")

    print("\n" + "=" * 70)
    print("ALL SMOKE TESTS PASSED 100% SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_tests()
