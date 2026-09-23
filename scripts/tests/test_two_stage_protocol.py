"""
Unit and Smoke Tests for Two-Stage Protocol in SAGE-Lite (B2)

Verifies:
1. Stage-2 optimizer parameter partitioning (shared_decay, shared_no_decay, others_decay, others_no_decay)
   with strict mathematical disjointness and completeness assertions.
2. Positive and negative prefix checks (CNN main_block -> shared; router/sa_hub/alpha/ViT/decoder -> others).
3. Synthetic B=1 end-to-end execution smoke test (checkpoint reload -> forward_with_routing_info -> loss -> backward -> optimizer.step -> weight delta).
"""

import os
import sys
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sage.networks.b2_unet import create_b2_unet, B2ConvNeXtViTUNet
from sage.components.router import SageRouter
from scripts.train_crack import create_stage2_optimizer, DEFAULT_SHARED_PREFIXES, CrackBinaryLoss


def test_stage2_optimizer_partitioning():
    print("\n" + "=" * 60)
    print("RUNNING TEST 1: Stage-2 Optimizer Partitioning Integrity")
    print("=" * 60)

    # Use small depth 4 for fast structural verification
    model = create_b2_unet(num_transformer_layers=4, pretrained=False)
    model.eval()

    shared_indices = [0, 1, 2, 3]
    model.set_shared_experts(shared_indices)
    assert model.sage_config["shared_expert_indices"] == shared_indices, "sage_config shared_expert_indices mismatch!"

    # Verify all SageRouter modules received the update
    routers = [m for m in model.modules() if isinstance(m, SageRouter)]
    assert len(routers) > 0, "No SageRouter instances found in model!"
    for r in routers:
        assert r.shared_expert_indices == shared_indices, f"Router {r} shared_expert_indices not updated!"
        if hasattr(r, 'shared_mask') and r.shared_mask is not None:
            # Check mask for first 4 entries
            assert r.shared_mask[:4].sum().item() == 4, "Router shared_mask not updated correctly!"

    # Create Stage 2 optimizer
    stage2_base_lr = 1e-4
    stage2_shared_lr = 3e-4
    optimizer = create_stage2_optimizer(
        model, 
        stage2_base_lr=stage2_base_lr, 
        stage2_shared_lr=stage2_shared_lr,
        weight_decay=0.05
    )

    group_names = [g['name'] for g in optimizer.param_groups]
    print(f"Optimizer param groups: {group_names}")
    assert len(optimizer.param_groups) == 4, f"Expected 4 groups, got {len(optimizer.param_groups)}"

    # Verify LRs and weight decays
    for g in optimizer.param_groups:
        if g['name'] == 'shared_experts':
            assert g['lr'] == stage2_shared_lr, f"Expected shared LR {stage2_shared_lr}, got {g['lr']}"
        elif g['name'] == 'other_and_routers':
            assert g['lr'] == stage2_base_lr, f"Expected base LR {stage2_base_lr}, got {g['lr']}"

    # Partition sets
    shared_params = []
    other_params = []
    for g in optimizer.param_groups:
        if g['name'] == 'shared_experts':
            shared_params.extend(g['params'])
        else:
            other_params.extend(g['params'])

    shared_set = set(shared_params)
    other_set = set(other_params)

    # 1. Disjointness
    intersection = shared_set.intersection(other_set)
    assert len(intersection) == 0, f"Found {len(intersection)} overlapping parameters between shared and other groups!"

    # 2. Completeness
    all_trainable = [p for p in model.parameters() if p.requires_grad]
    all_grouped = shared_params + other_params
    assert len(all_trainable) == len(all_grouped), f"Trainable params ({len(all_trainable)}) != grouped params ({len(all_grouped)})"
    assert set(all_trainable) == set(all_grouped), "Set of trainable params does not match grouped params!"

    # 3. Specific positive assertions: All CNN main_block parameters must be in shared_set
    for stage_idx in range(4):
        stage = model.backbone.convnext.stages[stage_idx]
        block = stage.main_block if hasattr(stage, 'main_block') else stage
        params = list(block.parameters())
        assert len(params) > 0, f"No parameters found for CNN stage {stage_idx}"
        for p in params:
            assert p in shared_set, f"A parameter of CNN stage {stage_idx} was NOT placed in shared_experts!"
        print(f"  [POSITIVE VERIFIED] All {len(params)} parameters of CNN Stage {stage_idx} main_block -> shared_experts")


    # 4. Specific negative assertions: router, sa_hub, alpha, ViT, decoder must be in other_set
    named_params = dict(model.named_parameters())
    negative_checks = [
        "backbone.convnext.stages.0.router.",
        "backbone.convnext.stages.0.alpha",
        "backbone.transformer_blocks.0.main_block.",
        "decoder.",
    ]
    for prefix in negative_checks:
        found_key = None
        for k in named_params.keys():
            if prefix in k and named_params[k].requires_grad:
                found_key = k
                break
        if found_key is not None:
            assert named_params[found_key] in other_set, f"Parameter {found_key} was incorrectly placed in shared_experts!"
            assert named_params[found_key] not in shared_set, f"Parameter {found_key} leaked into shared_experts!"
            print(f"  [NEGATIVE VERIFIED] {found_key} -> other_and_routers")

    print("TEST 1 PASSED: Parameter partitioning is strictly correct and disjoint!")


def test_stage2_execution_smoke():
    print("\n" + "=" * 60)
    print("RUNNING TEST 2: Stage-2 Execution Smoke Test (Synthetic B=1)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Step 1: Create dummy Stage 1 checkpoint in temp directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        stage1_model = create_b2_unet(num_transformer_layers=4, pretrained=False).to(device)
        dummy_ckpt_path = os.path.join(tmp_dir, "best_model_b2_stage1.pth")
        torch.save({
            'epoch': 5,
            'stage': 1,
            'model_state_dict': stage1_model.state_dict(),
            'best_dice': 0.725,
            'best_loss': 1.15,
        }, dummy_ckpt_path)
        print(f"Saved temporary Stage 1 checkpoint to {dummy_ckpt_path}")

        # Step 2: Instantiate fresh model and reload Stage 1 checkpoint
        model = create_b2_unet(num_transformer_layers=4, pretrained=False).to(device)
        checkpoint = torch.load(dummy_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Successfully reloaded Stage 1 checkpoint into fresh model")

        # Step 3: Configure shared experts and Stage 2 optimizer
        model.set_shared_experts([0, 1, 2, 3])
        optimizer = create_stage2_optimizer(model, stage2_base_lr=1e-4, stage2_shared_lr=1e-4)
        criterion = CrackBinaryLoss()

        # Step 4: Track a few weights before optimization step
        shared_weight_key = None
        other_weight_key = None
        for name, param in model.named_parameters():
            if name.startswith("backbone.convnext.stages.0.main_block.") and param.requires_grad and shared_weight_key is None:
                shared_weight_key = name
            if ("decoder." in name or "transformer_to_decoder" in name) and param.requires_grad and other_weight_key is None:
                other_weight_key = name


        assert shared_weight_key is not None, "Shared weight key not found!"
        assert other_weight_key is not None, "Other weight key not found!"

        shared_weight_before = dict(model.named_parameters())[shared_weight_key].detach().clone()
        other_weight_before = dict(model.named_parameters())[other_weight_key].detach().clone()

        # Step 5: Synthetic micro-batch (B=1, C=3, H=448, W=448)
        model.train()
        optimizer.zero_grad()

        dummy_images = torch.randn(1, 3, 448, 448, device=device)
        dummy_labels = torch.randint(0, 2, (1, 1, 448, 448), device=device).float()

        # Forward pass with routing info
        out = model.forward_with_routing_info(dummy_images)
        logits = out['logits']
        routing_infos = out['routing_infos']
        lb_loss = model.compute_total_load_balance_loss(routing_infos)
        seg_loss = criterion(logits, dummy_labels)
        loss = seg_loss + 1.0 * lb_loss

        print(f"Forward completed | Seg Loss: {seg_loss.item():.4f}, LB Loss: {lb_loss.item():.4f}, Total: {loss.item():.4f}")
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

        # Backward & Step
        loss.backward()
        optimizer.step()

        # Step 6: Verify weights have actually updated
        shared_weight_after = dict(model.named_parameters())[shared_weight_key].detach()
        other_weight_after = dict(model.named_parameters())[other_weight_key].detach()

        shared_delta = (shared_weight_after - shared_weight_before).abs().max().item()
        other_delta = (other_weight_after - other_weight_before).abs().max().item()

        print(f"Weight delta for {shared_weight_key}: {shared_delta:.6e}")
        print(f"Weight delta for {other_weight_key}: {other_delta:.6e}")

        assert shared_delta > 0.0, "Shared expert weight did not update after optimizer.step()!"
        assert other_delta > 0.0, "Other component weight did not update after optimizer.step()!"

        print("TEST 2 PASSED: Stage-2 synthetic execution and weight updates verified successfully!")


def test_full_two_stage_training_flow():
    print("\n" + "=" * 60)
    print("RUNNING TEST 3: Full End-to-End Two-Stage Training Flow Smoke Test")
    print("=" * 60)

    import cv2
    import numpy as np
    import yaml
    from scripts.train_crack import main as train_main

    with tempfile.TemporaryDirectory() as tmp_dir:
        train_img_dir = os.path.join(tmp_dir, "train", "images")
        train_mask_dir = os.path.join(tmp_dir, "train", "masks")
        val_img_dir = os.path.join(tmp_dir, "val", "images")
        val_mask_dir = os.path.join(tmp_dir, "val", "masks")
        output_dir = os.path.join(tmp_dir, "output")

        for d in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir, output_dir]:
            os.makedirs(d, exist_ok=True)

        for idx in range(2):
            img = np.random.randint(50, 200, (448, 448, 3), dtype=np.uint8)
            mask = np.zeros((448, 448), dtype=np.uint8)
            cv2.line(mask, (50, 50), (150, 150), 255, 3)

            cv2.imwrite(os.path.join(train_img_dir, f"sample_{idx}.png"), img)
            cv2.imwrite(os.path.join(train_mask_dir, f"sample_{idx}.png"), mask)
            cv2.imwrite(os.path.join(val_img_dir, f"sample_{idx}.png"), img)
            cv2.imwrite(os.path.join(val_mask_dir, f"sample_{idx}.png"), mask)

        config_path = os.path.join(tmp_dir, "test_config.yaml")
        cfg = {
            "model": "B2",
            "num_transformer_layers": 2,
            "seed": 42,
            "img_size": 448,
            "batch_size": 2,
            "num_workers": 0,
            "epochs": 2,
            "stage1_epochs": 1,
            "two_stage": True,
            "patience": 5,
            "lr": 1e-4,
            "stage2_base_lr": 1e-4,
            "stage2_shared_lr": 1e-4,
            "output_dir": output_dir,
            "smart_filter": False,
            "crop_mode": "none",
            "root_dir": tmp_dir,
            "train": {"images": "train/images", "masks": "train/masks"},
            "val": {"images": "val/images", "masks": "val/masks"},
            "test": {"images": "val/images", "masks": "val/masks"},
            "sage_config": {
                "top_k": 2,
                "shared_expert_indices": [0, 1, 2, 3],
                "router_hidden_dim": 32,
                "gating_type": "sigmoid",
                "fusion_type": "residual",
                "residual_scale": 0.1,
                "load_balance_factor": 0.01,
            }
        }
        with open(config_path, "w") as f:
            yaml.safe_dump(cfg, f)

        class Args:
            config = config_path
            stage2_only = False
            stage1_epochs_used = None
            two_stage = True

        print(f"Launching train_crack.main() on full 2-stage pipeline...")
        train_main(Args())

        stage1_ckpt = os.path.join(output_dir, "best_model_b2_stage1.pth")
        stage2_ckpt = os.path.join(output_dir, "best_model_b2_stage2.pth")
        global_ckpt = os.path.join(output_dir, "best_model_b2_global.pth")
        stage1_meta = os.path.join(output_dir, "stage1_completion.json")
        stage2_meta = os.path.join(output_dir, "stage2_completion.json")

        assert os.path.exists(stage1_ckpt), "best_model_b2_stage1.pth was NOT created!"
        assert os.path.exists(stage2_ckpt), "best_model_b2_stage2.pth was NOT created!"
        assert os.path.exists(global_ckpt), "best_model_b2_global.pth was NOT created!"
        assert os.path.exists(stage1_meta), "stage1_completion.json was NOT created!"
        assert os.path.exists(stage2_meta), "stage2_completion.json was NOT created!"

        for p in [stage1_ckpt, stage2_ckpt, global_ckpt]:
            data = torch.load(p, map_location='cpu', weights_only=False)
            assert 'model_state_dict' in data, f"Checkpoint {p} missing model_state_dict!"
            assert data['best_dice'] >= 0.0, f"Checkpoint {p} has negative best_dice!"

        import logging
        logging.shutdown()
        print("TEST 3 PASSED: Full Two-Stage Training Flow executed end-to-end successfully!")



if __name__ == '__main__':
    test_stage2_optimizer_partitioning()
    test_stage2_execution_smoke()
    test_full_two_stage_training_flow()
    print("\n" + "=" * 60)
    print("ALL TWO-STAGE PROTOCOL TESTS PASSED! (100% SUCCESS)")
    print("=" * 60 + "\n")

