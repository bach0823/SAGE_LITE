"""
Phase 6E Verification Suite: PointRend Adaptive Point Refinement Head
Verification of architectural hard locks, shape contracts, parameter counts,
gradient flows, optimizer partitioning, loss mechanics, and checkpoint provenance.

Invariants Verified:
1. Shape contract:
   - point_sample: (B, C, H, W) + (B, N, 2) -> (B, C, N)
   - sample_training_points: (B, 1, 112, 112) -> (B, 2048, 2) coords in [-1, 1]
   - PointRendHead: in_channels=48 + 1 = 49 -> 3-layer 1D Conv MLP (mid=128) -> (B, 1, N)
   - subdivide_and_refine: 112x112 -> 224x224 -> 448x448, returns (B, 1, 448, 448)
2. Parameter count:
   - PointRendHead: exactly 23,041 added parameters (+0.228% relative to Candidate B 10,118,955).
   - Candidate B + PointRend total: exactly 10,141,996 parameters.
3. Loss mechanics:
   - Decoupled two-tier loss: Coarse CrackBinaryLoss (BCE 1.0 + Dice 1.5) on 112x112 +
     Point-level BCEWithLogitsLoss (weight=1.0) on N=2048 boundary points.
   - Eval mode: transparent passthrough to base CrackBinaryLoss on 448x448 refined predictions.
4. UNetDecoder & B2ConvNeXtViTUNet integration:
   - Training mode with return_point_rend_dict=True returns dict containing point tensors.
   - Eval mode returns standard (B, 1, 448, 448) tensor (zero downstream script modification).
5. Gradient flow:
   - Full backpropagation from PointRendLoss updates PointRendHead, decoder, and upstream backbone.
6. Optimizer partitioning:
   - Stage 1: PointRend params belong strictly to 'decoder' tier (lr=1e-4).
   - Stage 2: PointRend params belong strictly to 'other_non_experts' tier (lr=1e-4).
   - Zero parameter omission, zero duplication across groups.
7. Checkpoint provenance & ancestral loading:
   - Candidate B Stage 1 checkpoint loads with 0 other missing, 0 unexpected, and exactly 6
     PointRend head tensors (3 weights + 3 biases) freshly initialized.
8. Canonical YAML configuration validation:
   - configs/p3_ablation/b2_p3_run_c_d4_k2_h64_phase6e_pointrend.yaml matches strict specification.
"""

import os
import sys
import tempfile
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure SAGE_LITE and repo root are in path
test_dir = os.path.dirname(os.path.abspath(__file__))
sage_lite_dir = os.path.abspath(os.path.join(test_dir, "..", ".."))
repo_root = os.path.abspath(os.path.join(sage_lite_dir, ".."))
for p in [sage_lite_dir, repo_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from sage.networks.point_rend import (
    point_sample,
    sample_training_points,
    get_uncertain_point_coords_on_grid,
    PointRendHead,
    subdivide_and_refine,
    PointRendLoss,
)
from sage.networks.decoder_block import DecoderBlock, UNetDecoder
from sage.networks.b2_unet import create_b2_unet, B2ConvNeXtViTUNet
from scripts.train_crack import get_optimizer_groups, create_stage2_optimizer, CrackBinaryLoss


def test_1_point_rend_isolated_primitives():
    print("\n--- Test 1: Isolated PointRend Primitives & Mathematical Contract ---")
    B, C, H, W = 2, 48, 112, 112
    N = 2048

    fine_features = torch.randn(B, C, H, W)
    coarse_logits = torch.randn(B, 1, H, W)

    # 1. Point sampling continuous interpolation
    test_coords = torch.rand(B, N, 2) * 2.0 - 1.0  # [-1, 1]
    sampled = point_sample(fine_features, test_coords)
    assert sampled.shape == (B, C, N), f"Expected (B, C, N) = ({B}, {C}, {N}), got {sampled.shape}"
    print(f"  [PASS] point_sample output shape: {sampled.shape}")

    # 2. Training point sampling (uncertainty + coverage + uniform)
    sampled_coords = sample_training_points(
        coarse_logits,
        num_points=N,
        oversample_ratio=3,
        importance_ratio=0.75,
    )
    assert sampled_coords.shape == (B, N, 2), f"Expected (B, N, 2) = ({B}, {N}, 2), got {sampled_coords.shape}"
    assert sampled_coords.min() >= -1.0 and sampled_coords.max() <= 1.0, "Point coords out of [-1, 1] range!"
    print(f"  [PASS] sample_training_points output shape: {sampled_coords.shape}, range: [{sampled_coords.min().item():.3f}, {sampled_coords.max().item():.3f}]")

    # 3. Grid uncertainty ranking
    grid_coords, flat_indices = get_uncertain_point_coords_on_grid(coarse_logits, num_points=1024)
    assert grid_coords.shape == (B, 1024, 2), f"Expected (B, 1024, 2), got {grid_coords.shape}"
    assert flat_indices.shape == (B, 1024), f"Expected (B, 1024), got {flat_indices.shape}"
    print(f"  [PASS] get_uncertain_point_coords_on_grid shape: {grid_coords.shape}, indices: {flat_indices.shape}")

    # 4. PointRendHead architecture & parameter count
    head = PointRendHead(in_channels=48, num_classes=1, mid_channels=128)
    head_params = sum(p.numel() for p in head.parameters())
    # Layer 1: Conv1d(49, 128, 1) -> 49*128 + 128 = 6400
    # Layer 2: Conv1d(128, 128, 1) -> 128*128 + 128 = 16512
    # Layer 3: Conv1d(128, 1, 1) -> 128*1 + 1 = 129
    # Total = 6400 + 16512 + 129 = 23041
    assert head_params == 23041, f"Expected exactly 23,041 head parameters, got {head_params}"
    print(f"  [PASS] PointRendHead parameter count: {head_params:,} (expected 23,041)")

    # 5. PointRendHead forward pass
    point_logits = head(fine_features, coarse_logits, sampled_coords)
    assert point_logits.shape == (B, 1, N), f"Expected (B, 1, N) = ({B}, 1, {N}), got {point_logits.shape}"
    print(f"  [PASS] PointRendHead forward output shape: {point_logits.shape}")

    # 6. Adaptive subdivision and refinement (eval mode)
    with torch.no_grad():
        refined_logits = subdivide_and_refine(
            fine_features=fine_features,
            coarse_logits=coarse_logits,
            point_head=head,
            target_size=(448, 448),
            num_subdivision_points=2048,
        )
    assert refined_logits.shape == (B, 1, 448, 448), f"Expected (B, 1, 448, 448), got {refined_logits.shape}"
    print(f"  [PASS] subdivide_and_refine output shape: {refined_logits.shape}")


def test_2_pointrend_loss_mechanics():
    print("\n--- Test 2: PointRend Decoupled Loss Mechanics ---")
    B = 2
    base_criterion = CrackBinaryLoss(bce_weight=1.0, dice_weight=1.5)
    point_criterion = PointRendLoss(base_criterion=base_criterion, point_loss_weight=1.0)

    # 1. Training mode forward (dict input)
    train_pred_dict = {
        "logits": torch.randn(B, 1, 448, 448, requires_grad=True),
        "coarse_logits": torch.randn(B, 1, 112, 112, requires_grad=True),
        "point_logits": torch.randn(B, 1, 2048, requires_grad=True),
        "point_coords": torch.rand(B, 2048, 2) * 2.0 - 1.0,
    }
    target = (torch.rand(B, 1, 448, 448) > 0.8).float()

    train_loss = point_criterion(train_pred_dict, target)
    assert train_loss.ndim == 0, f"Expected scalar loss, got shape {train_loss.shape}"
    assert train_loss.item() > 0.0, f"Expected positive loss, got {train_loss.item()}"
    train_loss.backward()

    assert train_pred_dict["coarse_logits"].grad is not None and train_pred_dict["coarse_logits"].grad.norm() > 0
    assert train_pred_dict["point_logits"].grad is not None and train_pred_dict["point_logits"].grad.norm() > 0
    print(f"  [PASS] Training loss value: {train_loss.item():.4f}, gradients confirmed on coarse & point logits")

    # 2. Eval mode forward (4D tensor passthrough)
    eval_pred = torch.randn(B, 1, 448, 448)
    eval_loss = point_criterion(eval_pred, target)
    expected_eval_loss = base_criterion(eval_pred, target)
    assert torch.isclose(eval_loss, expected_eval_loss), "Eval loss must be identical to base_criterion on 4D prediction!"
    print(f"  [PASS] Eval loss transparent passthrough confirmed: {eval_loss.item():.4f} == {expected_eval_loss.item():.4f}")


def test_3_unet_decoder_integration():
    print("\n--- Test 3: UNetDecoder Integration & Execution Modes ---")
    B = 2
    enc_channels = [48, 96, 192, 384]

    decoder = UNetDecoder(
        encoder_channels=enc_channels,
        use_point_rend=True,
        point_rend_mid_channels=128,
        point_rend_train_points=2048,
        point_rend_subdivision_points=4096,
    )
    assert decoder.point_rend_head is not None, "PointRend head must be instantiated"

    bottleneck = torch.randn(B, 384, 14, 14)
    skips = [
        torch.randn(B, 48, 112, 112),
        torch.randn(B, 96, 56, 56),
        torch.randn(B, 192, 28, 28),
    ]

    # Mode A: Training with return_point_rend_dict=True
    decoder.train()
    out_dict = decoder(bottleneck, skips, target_size=(448, 448), return_point_rend_dict=True)
    assert isinstance(out_dict, dict), f"Expected dict in training mode, got {type(out_dict)}"
    for key in ["logits", "coarse_logits", "point_logits", "point_coords"]:
        assert key in out_dict, f"Missing key '{key}' in training output dict"
    assert out_dict["point_logits"].shape == (B, 1, 2048)
    assert out_dict["coarse_logits"].shape == (B, 1, 112, 112)
    assert out_dict["logits"].shape == (B, 1, 448, 448)
    print(f"  [PASS] Training mode output dict verified: point_logits={out_dict['point_logits'].shape}")

    # Mode B: Eval with subdivision
    decoder.eval()
    with torch.no_grad():
        out_eval = decoder(bottleneck, skips, target_size=(448, 448))
    assert isinstance(out_eval, torch.Tensor), f"Expected Tensor in eval mode, got {type(out_eval)}"
    assert out_eval.shape == (B, 1, 448, 448), f"Expected (B, 1, 448, 448), got {out_eval.shape}"
    print(f"  [PASS] Eval mode subdivision output verified: {out_eval.shape}")


def test_4_full_b2_model_contract_and_param_count():
    print("\n--- Test 4: Full B2ConvNeXtViTUNet Contract & Parameter Count ---")
    sage_cfg = {
        "top_k": 2,
        "gating_type": "sigmoid",
        "shared_expert_indices": [0, 1, 2, 3],
        "router_hidden_dim": 64,
        "load_balance_factor": 0.01,
        "logit_modulation": True,
        "expert_dropout": 0.1,
        "fusion_type": "residual",
        "residual_scale": 0.1,
    }

    # 1. Baseline Candidate B
    model_base = create_b2_unet(
        num_classes=1,
        img_size=448,
        num_transformer_layers=4,
        pretrained=False,
        sage_config=sage_cfg,
        p3_mode="C",
        use_plu_head=False,
        use_cgsr=False,
        use_point_rend=False,
    )
    base_params = sum(p.numel() for p in model_base.parameters())
    assert base_params == 10118955, f"Expected 10,118,955 params for Candidate B, got {base_params}"
    print(f"  [PASS] Baseline Candidate B param count: {base_params:,} (10,118,955)")

    # 2. Candidate B + PointRend
    model_pr = create_b2_unet(
        num_classes=1,
        img_size=448,
        num_transformer_layers=4,
        pretrained=False,
        sage_config=sage_cfg,
        p3_mode="C",
        use_plu_head=False,
        use_cgsr=False,
        use_point_rend=True,
        point_rend_mid_channels=128,
        point_rend_train_points=2048,
        point_rend_subdivision_points=8192,
    )
    pr_params = sum(p.numel() for p in model_pr.parameters())
    expected_pr_total = 10118955 + 23041  # 10,141,996
    assert pr_params == expected_pr_total, f"Expected {expected_pr_total} params, got {pr_params}"
    param_delta = pr_params - base_params
    assert param_delta == 23041, f"Expected delta +23,041 params, got {param_delta}"
    print(f"  [PASS] Candidate B + PointRend param count: {pr_params:,} (delta: +{param_delta:,} = +0.228%)")

    # 3. Model forward in eval mode
    x = torch.randn(2, 3, 448, 448)
    model_pr.eval()
    with torch.no_grad():
        out_eval = model_pr(x)
    assert out_eval.shape == (2, 1, 448, 448), f"Expected (2, 1, 448, 448), got {out_eval.shape}"
    print(f"  [PASS] Full B2 PointRend eval forward output shape: {out_eval.shape}")

    # 4. Model forward_with_routing_info in training mode
    model_pr.train()
    routing_out = model_pr.forward_with_routing_info(x)
    assert "logits" in routing_out
    assert "routing_infos" in routing_out
    assert "coarse_logits" in routing_out and routing_out["coarse_logits"] is not None
    assert "point_logits" in routing_out and routing_out["point_logits"] is not None
    assert "point_coords" in routing_out and routing_out["point_coords"] is not None
    print(f"  [PASS] forward_with_routing_info in train mode harvested PointRend point tensors successfully")


def test_5_gradient_flow_and_backprop():
    print("\n--- Test 5: Gradient Flow & End-to-End Backprop ---")
    sage_cfg = {
        "top_k": 2,
        "gating_type": "sigmoid",
        "shared_expert_indices": [0, 1, 2, 3],
        "router_hidden_dim": 64,
        "load_balance_factor": 0.01,
        "logit_modulation": True,
        "expert_dropout": 0.1,
        "fusion_type": "residual",
        "residual_scale": 0.1,
    }

    model = create_b2_unet(
        num_classes=1,
        img_size=128,  # small resolution for fast gradient test
        num_transformer_layers=2,
        pretrained=False,
        sage_config=sage_cfg,
        p3_mode="C",
        use_point_rend=True,
        point_rend_mid_channels=64,
        point_rend_train_points=512,
    )
    model.train()

    head = model.decoder.point_rend_head
    assert head is not None

    base_criterion = CrackBinaryLoss(bce_weight=1.0, dice_weight=1.5)
    criterion = PointRendLoss(base_criterion=base_criterion, point_loss_weight=1.0)

    x = torch.randn(2, 3, 128, 128, requires_grad=True)
    target = (torch.rand(2, 1, 128, 128) > 0.8).float()

    out = model.forward_with_routing_info(x)
    pred_dict = {
        "logits": out["logits"],
        "coarse_logits": out["coarse_logits"],
        "point_logits": out["point_logits"],
        "point_coords": out["point_coords"],
    }
    seg_loss = criterion(pred_dict, target)
    lb_loss = model.compute_total_load_balance_loss(out["routing_infos"])
    total_loss = seg_loss + 0.01 * lb_loss
    total_loss.backward()

    # Verify PointRendHead gradients
    for idx, layer in enumerate(head.mlp):
        if isinstance(layer, nn.Conv1d):
            w_grad = layer.weight.grad
            b_grad = layer.bias.grad
            assert w_grad is not None and w_grad.norm().item() > 0, f"Head layer {idx} weight grad missing/zero!"
            assert b_grad is not None and b_grad.norm().item() > 0, f"Head layer {idx} bias grad missing/zero!"

    # Verify decoder and input image gradients
    dec_grad = model.decoder.decoder_blocks[2].conv2[0].weight.grad
    assert dec_grad is not None and dec_grad.norm().item() > 0, "Decoder conv grad missing/zero!"
    assert x.grad is not None and x.grad.norm().item() > 0, "Input image gradient missing/zero!"

    print("  [PASS] Full gradient backpropagation confirmed across PointRendHead, Decoder, and Backbone")


def test_6_optimizer_partitioning():
    print("\n--- Test 6: Optimizer Parameter Partition Invariants ---")
    sage_cfg = {
        "top_k": 2,
        "gating_type": "sigmoid",
        "shared_expert_indices": [0, 1, 2, 3],
        "router_hidden_dim": 64,
        "load_balance_factor": 0.01,
        "logit_modulation": True,
        "expert_dropout": 0.1,
        "fusion_type": "residual",
        "residual_scale": 0.1,
    }

    model = create_b2_unet(
        num_classes=1,
        img_size=128,
        num_transformer_layers=2,
        pretrained=False,
        sage_config=sage_cfg,
        p3_mode="C",
        use_point_rend=True,
    )

    # 1. Stage 1 Optimizer Groups
    stage1_groups = get_optimizer_groups(model, lr_backbone=1e-5, lr_decoder=1e-4, lr_sage=2e-4, lr_p3=1e-4)
    stage1_param_count = sum(len(g['params']) for g in stage1_groups)
    trainable_params = sum(1 for p in model.parameters() if p.requires_grad)
    assert stage1_param_count == trainable_params, (
        f"Stage 1 partition mismatch! {stage1_param_count} grouped vs {trainable_params} trainable."
    )

    pr_head = model.decoder.point_rend_head
    pr_param_ids = {id(p) for p in pr_head.parameters()}
    decoder_group_param_ids = set()
    for g in stage1_groups:
        if g['name'] == 'decoder':
            decoder_group_param_ids.update(id(p) for p in g['params'])
    assert pr_param_ids.issubset(decoder_group_param_ids), "PointRend head params must belong to 'decoder' group in Stage 1!"
    print(f"  [PASS] Stage 1 optimizer: all {len(pr_param_ids)} PointRend tensors strictly in 'decoder' tier")

    # 2. Stage 2 Optimizer Groups
    stage2_opt = create_stage2_optimizer(model, stage2_base_lr=1e-4, stage2_shared_lr=1e-4, stage2_sage_lr=1e-4, stage2_p3_lr=1e-4)
    stage2_param_count = sum(len(g['params']) for g in stage2_opt.param_groups)
    assert stage2_param_count == trainable_params, (
        f"Stage 2 partition mismatch! {stage2_param_count} grouped vs {trainable_params} trainable."
    )

    others_group_param_ids = set()
    for g in stage2_opt.param_groups:
        if g['name'] == 'other_non_experts':
            others_group_param_ids.update(id(p) for p in g['params'])
    assert pr_param_ids.issubset(others_group_param_ids), "PointRend head params must belong to 'other_non_experts' group in Stage 2!"
    print(f"  [PASS] Stage 2 optimizer: all {len(pr_param_ids)} PointRend tensors strictly in 'other_non_experts' tier")


def test_7_ancestor_checkpoint_loading():
    print("\n--- Test 7: Ancestor Checkpoint Loading & State Dict Provenance ---")
    sage_cfg = {
        "top_k": 2,
        "gating_type": "sigmoid",
        "shared_expert_indices": [0, 1, 2, 3],
        "router_hidden_dim": 64,
        "load_balance_factor": 0.01,
        "logit_modulation": True,
        "expert_dropout": 0.1,
        "fusion_type": "residual",
        "residual_scale": 0.1,
    }

    # 1. Test with synthetic ancestor checkpoint
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_ancestor = create_b2_unet(
            num_classes=1,
            img_size=128,
            num_transformer_layers=2,
            pretrained=False,
            sage_config=sage_cfg,
            p3_mode="C",
            use_point_rend=False,
        )
        ancestor_sd = model_ancestor.state_dict()

        model_pr = create_b2_unet(
            num_classes=1,
            img_size=128,
            num_transformer_layers=2,
            pretrained=False,
            sage_config=sage_cfg,
            p3_mode="C",
            use_point_rend=True,
        )
        missing, unexpected = model_pr.load_stage1_state_dict(ancestor_sd)
        assert len(unexpected) == 0, f"Expected 0 unexpected keys, got {unexpected}"
        pr_missing = [k for k in missing if "decoder.point_rend_head." in k]
        other_missing = [k for k in missing if "decoder.point_rend_head." not in k]
        assert len(other_missing) == 0, f"Expected 0 non-PointRend missing keys, got {other_missing}"
        assert len(pr_missing) == 6, f"Expected exactly 6 PointRend missing keys (3 weights + 3 biases), got {len(pr_missing)}"
        print(f"  [PASS] Synthetic ancestor loading verified (6 PointRend head tensors initialized, 0 other missing, 0 unexpected)")

    # 2. Test with real canonical Stage 1 Candidate B checkpoint if present on disk
    canonical_ckpt = os.path.join(repo_root, "results", "checkpoints", "P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_stage1.pth")
    if os.path.exists(canonical_ckpt):
        print(f"  Testing canonical Candidate B Stage 1 checkpoint at: {canonical_ckpt}")
        full_pr_model = create_b2_unet(
            num_classes=1,
            img_size=448,
            num_transformer_layers=4,
            pretrained=False,
            sage_config=sage_cfg,
            p3_mode="C",
            use_point_rend=True,
        )
        raw_data = torch.load(canonical_ckpt, map_location="cpu", weights_only=False)
        real_sd = raw_data["model_state_dict"] if "model_state_dict" in raw_data else raw_data
        missing_real, unexpected_real = full_pr_model.load_stage1_state_dict(real_sd)
        assert len(unexpected_real) == 0, f"Real ckpt: expected 0 unexpected keys, got {unexpected_real}"
        real_pr_missing = [k for k in missing_real if "decoder.point_rend_head." in k]
        real_other_missing = [k for k in missing_real if "decoder.point_rend_head." not in k]
        assert len(real_other_missing) == 0, f"Real ckpt: expected 0 non-PointRend missing keys, got {real_other_missing}"
        assert len(real_pr_missing) == 6, f"Real ckpt: expected 6 PointRend missing keys, got {len(real_pr_missing)}"
        print(f"  [PASS] REAL Canonical Candidate B Stage 1 checkpoint loaded successfully into PointRend model!")
    else:
        print(f"  [SKIP] Canonical checkpoint not found at {canonical_ckpt} (synthetic test passed)")


def test_8_yaml_config_validation():
    print("\n--- Test 8: Canonical YAML Config Contract Validation ---")
    yaml_path = os.path.join(sage_lite_dir, "configs", "p3_ablation", "b2_p3_run_c_d4_k2_h64_phase6e_pointrend.yaml")
    assert os.path.exists(yaml_path), f"YAML config not found at: {yaml_path}"

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Invariants
    model_name = cfg.get("model") or cfg.get("model_type")
    assert model_name == "B2", f"Expected model B2, got {model_name}"
    assert cfg.get("p3_mode") == "C", f"Expected p3_mode C, got {cfg.get('p3_mode')}"
    assert cfg.get("num_transformer_layers") == 4, f"Expected 4 ViT layers (D4), got {cfg.get('num_transformer_layers')}"
    assert cfg.get("use_point_rend") is True, "use_point_rend must be True"
    assert cfg.get("point_loss_weight") == 1.0, f"Expected point_loss_weight 1.0, got {cfg.get('point_loss_weight')}"
    assert cfg.get("point_rend_train_points") == 2048, f"Expected 2048 train points, got {cfg.get('point_rend_train_points')}"
    assert cfg.get("point_rend_subdivision_points") == 8192, f"Expected 8192 subdivision points, got {cfg.get('point_rend_subdivision_points')}"
    assert cfg.get("point_rend_mid_channels") == 128, f"Expected 128 mid channels, got {cfg.get('point_rend_mid_channels')}"

    # Provenance
    valid_ancestors = [
        "/content/checkpoints/best_model_b2_stage1.pth",
        "results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_stage1.pth",
    ]
    actual_ckpt = cfg.get("checkpoint") or cfg.get("locked_base_checkpoint")
    assert any(actual_ckpt.endswith(os.path.basename(va)) for va in valid_ancestors), (
        f"Expected checkpoint pointing to Stage 1 Candidate B ancestor, got '{actual_ckpt}'"
    )

    print(f"  [PASS] YAML configuration at {yaml_path} satisfies 100% of Phase 6E invariants")


def main():
    print("=" * 75)
    print("PHASE 6E PREFLIGHT VERIFICATION: POINTREND REFINEMENT HEAD")
    print("=" * 75)
    test_1_point_rend_isolated_primitives()
    test_2_pointrend_loss_mechanics()
    test_3_unet_decoder_integration()
    test_4_full_b2_model_contract_and_param_count()
    test_5_gradient_flow_and_backprop()
    test_6_optimizer_partitioning()
    test_7_ancestor_checkpoint_loading()
    test_8_yaml_config_validation()
    print("\n" + "=" * 75)
    print("ALL 8 PREFLIGHT INVARIANT TESTS PASSED (100% SUCCESS)!")
    print("=" * 75)


if __name__ == "__main__":
    main()
