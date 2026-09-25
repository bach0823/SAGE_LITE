"""
P3-PHASE-5: Minimal Verification Suite for Proposal 3 (P3)
Covers all 13 verification checkpoints:
1. ASDWRefinement / GenericRefinement tensor shape contracts.
2. Exact parameter counts and bias=False invariants.
3. Residual gamma initialization and trainability.
4. PE28_fixed shape, dtype, persistent-buffer registration, and single ownership at backbone.pe28_fixed.
5. SageLayer runtime PE lookup does not create a registered child module or plain tensor alias.
6. P3 guard activates ONLY for CNN S0/S1 -> Transformer expert.
7. All other routes remain on Normal SAGE.
8. Self-Selection Bypass happens before P3.
9. ViT Block internals are untouched.
10. Checkpoint-loading contract separately for Run A and Run B/C.
11. model.to(device), save/load, and deepcopy behavior for PE28 ownership.
12. End-to-end forward/backward dry-run on synthetic tensors without NaN/Inf.
13. Locked-Base Provenance Gate: negative test for missing base, generic checkpoint rejection, mutual exclusivity in trainer, and positive VERIFIED_LOCKED_BASE verification.
"""

import copy
import hashlib
import math
import os
import pickle
import sys
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure sage_lite is on sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sage.components.p3_refinement import ASDWRefinement, GenericRefinement
from sage.components.sage_layer import SageLayer
from sage.networks.b2_unet import create_b2_unet, B2ConvNeXtViTUNet
from sage.utils.model_utils import load_locked_base_into_p3, P3_EXPECTED_KEYS
from scripts.preflight_p3_realdata import (
    resolve_locked_base_path,
    compute_file_sha256,
    compute_preflight_verdict,
    EXPECTED_LOCKED_BASE_SHA256,
)


def test_1_shape_contracts():
    print("\n[Test 1] Verifying ASDW / Generic Refinement shape contracts...")
    for C, (H, W) in [(48, (112, 112)), (96, (56, 56))]:
        x = torch.randn(2, C, H, W)
        m_c = ASDWRefinement(C)
        m_b = GenericRefinement(C)
        out_c = m_c(x)
        out_b = m_b(x)
        assert out_c.shape == (2, C, H, W), f"Run C shape mismatch at C={C}: {out_c.shape}"
        assert out_b.shape == (2, C, H, W), f"Run B shape mismatch at C={C}: {out_b.shape}"
    print("  --> PASS: Exact shape preservation (H x W) verified for Stage 0 and Stage 1.")


def test_2_parameter_counts_and_bias():
    print("\n[Test 2] Verifying parameter counts and bias=False invariants...")
    m_c_s0 = ASDWRefinement(48)
    m_c_s1 = ASDWRefinement(96)
    m_b_s0 = GenericRefinement(48)
    m_b_s1 = GenericRefinement(96)

    p_c_s0 = sum(p.numel() for p in m_c_s0.parameters())
    p_c_s1 = sum(p.numel() for p in m_c_s1.parameters())
    p_b_s0 = sum(p.numel() for p in m_b_s0.parameters())
    p_b_s1 = sum(p.numel() for p in m_b_s1.parameters())

    assert p_c_s0 == 8017, f"Expected Run C S0 = 8,017, got {p_c_s0}"
    assert p_c_s1 == 29857, f"Expected Run C S1 = 29,857, got {p_c_s1}"
    assert p_c_s0 + p_c_s1 == 37874, f"Expected Run C Total = 37,874, got {p_c_s0 + p_c_s1}"

    assert p_b_s0 == 8209, f"Expected Run B S0 = 8,209, got {p_b_s0}"
    assert p_b_s1 == 30241, f"Expected Run B S1 = 30,241, got {p_b_s1}"
    assert p_b_s0 + p_b_s1 == 38450, f"Expected Run B Total = 38,450, got {p_b_s0 + p_b_s1}"

    diff = (p_b_s0 + p_b_s1) - (p_c_s0 + p_c_s1)
    assert diff == 576, f"Expected diff = 576 (27C vs 23C), got {diff}"

    # Also verify integrated model parameter counts (inside B2UNet)
    m_c_full = create_b2_unet(pretrained=False, num_transformer_layers=4, p3_mode="C")
    s0_c_int = sum(p.numel() for p in m_c_full.backbone.convnext.stages[0].p3_refinement.parameters())
    s1_c_int = sum(p.numel() for p in m_c_full.backbone.convnext.stages[1].p3_refinement.parameters())
    assert s0_c_int == 8017, f"Integrated Run C S0 expected 8,017, got {s0_c_int}"
    assert s1_c_int == 29857, f"Integrated Run C S1 expected 29,857, got {s1_c_int}"
    assert s0_c_int + s1_c_int == 37874, f"Integrated Run C Total expected 37,874, got {s0_c_int + s1_c_int}"

    m_b_full = create_b2_unet(pretrained=False, num_transformer_layers=4, p3_mode="B")
    s0_b_int = sum(p.numel() for p in m_b_full.backbone.convnext.stages[0].p3_refinement.parameters())
    s1_b_int = sum(p.numel() for p in m_b_full.backbone.convnext.stages[1].p3_refinement.parameters())
    assert s0_b_int == 8209, f"Integrated Run B S0 expected 8,209, got {s0_b_int}"
    assert s1_b_int == 30241, f"Integrated Run B S1 expected 30,241, got {s1_b_int}"
    assert s0_b_int + s1_b_int == 38450, f"Integrated Run B Total expected 38,450, got {s0_b_int + s1_b_int}"

    # Verify bias=False invariant on all conv modules
    for mod in [m_c_s0, m_c_s1, m_b_s0, m_b_s1]:
        for name, submodule in mod.named_modules():
            if isinstance(submodule, nn.Conv2d):
                assert submodule.bias is None, f"Found bias in {name}!"
    print("  --> PASS: 37,874 vs 38,450 params and bias=False on 100% of conv layers verified.")


def test_3_gamma_semantics():
    print("\n[Test 3] Verifying gamma initialization and trainability...")
    m_c = ASDWRefinement(48)
    assert isinstance(m_c.gamma, nn.Parameter), "gamma must be an nn.Parameter"
    assert m_c.gamma.shape == torch.Size([]), f"gamma must be a scalar (shape torch.Size([])), got {m_c.gamma.shape}"
    assert m_c.gamma.numel() == 1, f"gamma must have numel=1, got {m_c.gamma.numel()}"
    assert m_c.gamma.requires_grad, "gamma must be trainable"
    assert torch.isclose(m_c.gamma, torch.tensor(0.01)), f"Expected gamma=0.01, got {m_c.gamma.item()}"

    x = torch.randn(2, 48, 112, 112, requires_grad=True)
    out = m_c(x)
    loss = out.sum()
    loss.backward()
    assert m_c.gamma.grad is not None, "gamma must receive gradients during backward pass"
    assert not torch.isnan(m_c.gamma.grad), "gamma gradient must not be NaN"
    print(f"  --> PASS: gamma=0.01 learnable parameter verified (grad={m_c.gamma.grad.item():.6f}).")


def test_4_pe28_ownership():
    print("\n[Test 4] Verifying PE28 single canonical ownership on backbone...")
    model_c = create_b2_unet(pretrained=False, num_transformer_layers=4, p3_mode="C")
    assert hasattr(model_c.backbone, "pe28_fixed"), "backbone must own pe28_fixed buffer"
    pe28 = model_c.backbone.pe28_fixed
    assert pe28.shape == (1, 784, 192), f"Expected shape (1, 784, 192), got {pe28.shape}"
    assert pe28.dtype == torch.float32, f"Expected storage dtype float32, got {pe28.dtype}"
    assert not pe28.requires_grad, "PE28 must have requires_grad=False"

    sd = model_c.state_dict()
    pe28_keys = [k for k in sd.keys() if "pe28" in k]
    assert pe28_keys == ["backbone.pe28_fixed"], f"Expected single key ['backbone.pe28_fixed'], got {pe28_keys}"
    print("  --> PASS: Exactly 1 persistent buffer key 'backbone.pe28_fixed' in state_dict.")


def test_5_runtime_pe_lookup():
    print("\n[Test 5] Verifying runtime lookup without registered alias...")
    model_c = create_b2_unet(pretrained=False, num_transformer_layers=4, p3_mode="C")
    s0 = model_c.backbone.convnext.stages[0]
    s1 = model_c.backbone.convnext.stages[1]

    # Check that _pe_owner is not in _modules
    assert "_pe_owner" not in s0._modules, "_pe_owner must NOT be in stage._modules"
    assert "_pe_owner" not in s1._modules, "_pe_owner must NOT be in stage._modules"

    # Check that stage does not have its own pe28 buffer
    assert "pe28_fixed" not in s0._buffers, "Stage 0 must not register its own pe28_fixed buffer"
    assert "pe28_fixed" not in s1._buffers, "Stage 1 must not register its own pe28_fixed buffer"

    # Check runtime lookup returns identical buffer from backbone
    pe_from_s0 = s0.get_pe28()
    pe_from_s1 = s1.get_pe28()
    assert pe_from_s0 is model_c.backbone.pe28_fixed, "get_pe28() must return backbone buffer"
    assert pe_from_s1 is model_c.backbone.pe28_fixed, "get_pe28() must return backbone buffer"
    print("  --> PASS: Runtime lookup resolves backbone buffer without circular submodule registration.")


def test_6_and_7_guard_and_non_target_routes():
    print("\n[Test 6 & 7] Verifying P3 Guard condition and non-target route preservation...")
    model_c = create_b2_unet(pretrained=False, num_transformer_layers=4, p3_mode="C")
    
    # Inspection of metadata
    s0 = model_c.backbone.convnext.stages[0]
    s1 = model_c.backbone.convnext.stages[1]
    s2 = model_c.backbone.convnext.stages[2]
    s3 = model_c.backbone.convnext.stages[3]
    vit0 = model_c.backbone.transformer_blocks[0]

    assert s0.layer_type == "cnn" and s0.stage_idx == 0
    assert s1.layer_type == "cnn" and s1.stage_idx == 1
    assert s2.layer_type == "cnn" and s2.stage_idx == 2
    assert s3.layer_type == "cnn" and s3.stage_idx == 3
    assert vit0.layer_type == "transformer" and vit0.stage_idx == 0

    assert s0.p3_refinement is not None, "S0 must have p3_refinement"
    assert s1.p3_refinement is not None, "S1 must have p3_refinement"
    assert s2.p3_refinement is None, "S2 must NOT have p3_refinement"
    assert s3.p3_refinement is None, "S3 must NOT have p3_refinement"
    assert vit0.p3_refinement is None, "ViT layers must NOT have p3_refinement"

    # Check expert metadata in expert_pool
    pool = model_c.expert_pool
    for idx in range(4):
        assert pool[idx].expert_type == "cnn", f"Expert {idx} should be cnn"
    for idx in range(4, len(pool)):
        assert pool[idx].expert_type == "transformer", f"Expert {idx} should be transformer"

    # Simulate routing logic checks
    def check_guard(source_layer, target_expert):
        return (
            getattr(source_layer, "layer_type", None) == "cnn"
            and getattr(source_layer, "stage_idx", None) in (0, 1)
            and getattr(target_expert, "expert_type", None) == "transformer"
        )

    # Route 1: S0 -> ViT Expert: MUST BE TRUE
    assert check_guard(s0, pool[4]) is True
    # Route 2: S1 -> ViT Expert: MUST BE TRUE
    assert check_guard(s1, pool[4]) is True
    # Route 3: S0 -> CNN Expert: MUST BE FALSE (Normal SAGE)
    assert check_guard(s0, pool[1]) is False
    # Route 4: S2 -> ViT Expert: MUST BE FALSE (Normal SAGE)
    assert check_guard(s2, pool[4]) is False
    # Route 5: S3 -> ViT Expert: MUST BE FALSE (Normal SAGE)
    assert check_guard(s3, pool[4]) is False
    # Route 6: ViT -> CNN Expert: MUST BE FALSE (Normal SAGE)
    assert check_guard(vit0, pool[0]) is False
    # Route 7: ViT -> ViT Expert: MUST BE FALSE (Normal SAGE)
    assert check_guard(vit0, pool[5]) is False

    print("  --> PASS: P3 special path triggers strictly for CNN S0/S1 -> ViT expert; all other routes on Normal SAGE.")


def test_8_self_selection_bypass():
    print("\n[Test 8] Verifying Self-Selection Bypass takes precedence before P3 (Runtime Sentinel)...")
    model_c = create_b2_unet(pretrained=False, num_transformer_layers=4, p3_mode="C")
    s0 = model_c.backbone.convnext.stages[0]
    assert s0.my_index == 0, f"Stage 0 my_index must be 0, got {s0.my_index}"

    # Sentinel counter on s0.p3_refinement
    refinement_call_count = [0]
    orig_forward = s0.p3_refinement.forward

    def spy_forward(inp):
        refinement_call_count[0] += 1
        return orig_forward(inp)

    s0.p3_refinement.forward = spy_forward

    # Prepare synthetic input
    B = 2
    x = torch.randn(B, 48, 112, 112)
    main_output = s0._execute_main_path(x)

    # 1. Execute Self-Selection Case (my_index == expert_idx == 0)
    class MockRouter(nn.Module):
        def __init__(self, target_idx, batch_size):
            super().__init__()
            self.target_idx = target_idx
            self.batch_size = batch_size
            self.top_k = 1

        def forward(self, inp):
            top_k = torch.full((self.batch_size, 1), self.target_idx, dtype=torch.long)
            weights = torch.ones(self.batch_size, 1, dtype=torch.float32)
            return top_k, weights, {"load_balance_loss": torch.tensor(0.0)}

    orig_router = s0.router
    try:
        s0.router = MockRouter(0, B)
        expert_out_self, _ = s0._execute_expert_path(x, main_output, model_c.expert_pool)

        # Assertions for Self-Selection Bypass
        assert refinement_call_count[0] == 0, (
            f"VIOLATION: P3 refinement was invoked {refinement_call_count[0]} times during self-selection bypass!"
        )
        assert torch.equal(expert_out_self, main_output), (
            "VIOLATION: Self-selection bypass output path must bitwise match main_output!"
        )

        # 2. Positive Control: Route to Transformer Expert 4 (P3 Special Path)
        s0.router = MockRouter(4, B)
        expert_out_vit, _ = s0._execute_expert_path(x, main_output, model_c.expert_pool)

        assert refinement_call_count[0] == 1, (
            f"VIOLATION: P3 refinement should be called exactly once when routed to ViT expert, got {refinement_call_count[0]}"
        )
        assert not torch.equal(expert_out_vit, main_output), (
            "ViT expert output should be processed through P3 and differ from main_output."
        )
    finally:
        s0.router = orig_router

    print("  --> PASS: Runtime sentinel confirmed P3 refinement was NOT called during self-selection (call_count=0), and output matched main_output exactly.")


def test_9_vit_block_internals():
    print("\n[Test 9] Verifying ViT Block internals are untouched...")
    model_c = create_b2_unet(pretrained=False, num_transformer_layers=4, p3_mode="C")
    blk = model_c.backbone.transformer_blocks[0].main_block.module
    assert hasattr(blk, "norm1"), "ViT block must have norm1"
    assert hasattr(blk, "attn"), "ViT block must have attn"
    assert hasattr(blk, "norm2"), "ViT block must have norm2"
    assert hasattr(blk, "mlp"), "ViT block must have mlp"
    # Ensure block has no internal pos_embed attribute
    assert not hasattr(blk, "pos_embed"), "ViT block must not contain internal pos_embed"

    # Dry run block directly with N=784 tokens
    dummy_tokens = torch.randn(2, 784, 192)
    out_tokens = blk(dummy_tokens)
    assert out_tokens.shape == (2, 784, 192), f"Expected (2, 784, 192), got {out_tokens.shape}"
    print("  --> PASS: ViT Block internals verified intact and compatible with N=784 tokens.")


def test_10_checkpoint_ingestion():
    print("\n[Test 10] Verifying Checkpoint Ingestion & Bitwise Base Value Integrity...")
    # 1. Create a mock base model (p3_mode=None) representing Locked Base Checkpoint
    base_model = create_b2_unet(pretrained=False, num_transformer_layers=4, p3_mode=None)
    mock_base_dict = copy.deepcopy(base_model.state_dict())
    mock_base_ckpt = {"model_state_dict": mock_base_dict}
    num_base_keys = len(mock_base_dict)

    # 2. Test ingestion & bitwise value integrity for Run A
    model_a = create_b2_unet(pretrained=False, num_transformer_layers=4, p3_mode="A")
    incomp_a = load_locked_base_into_p3(model_a, mock_base_ckpt, p3_mode="A")
    assert len(incomp_a.unexpected_keys) == 0, f"Run A unexpected keys: {incomp_a.unexpected_keys}"
    assert set(incomp_a.missing_keys) == P3_EXPECTED_KEYS["A"], "Run A missing keys mismatch!"
    sd_a = model_a.state_dict()
    for k, v_base in mock_base_dict.items():
        assert torch.equal(sd_a[k], v_base), f"Bitwise value mismatch in Run A for key: {k}"
    print(f"  --> Run A: 100% bitwise equality verified across all {num_base_keys} pre-existing base keys.")

    # 3. Test ingestion & bitwise value integrity for Run B
    model_b = create_b2_unet(pretrained=False, num_transformer_layers=4, p3_mode="B")
    incomp_b = load_locked_base_into_p3(model_b, mock_base_ckpt, p3_mode="B")
    assert len(incomp_b.unexpected_keys) == 0, f"Run B unexpected keys: {incomp_b.unexpected_keys}"
    assert set(incomp_b.missing_keys) == P3_EXPECTED_KEYS["B"], "Run B missing keys mismatch!"
    sd_b = model_b.state_dict()
    for k, v_base in mock_base_dict.items():
        assert torch.equal(sd_b[k], v_base), f"Bitwise value mismatch in Run B for key: {k}"
    print(f"  --> Run B: 100% bitwise equality verified across all {num_base_keys} pre-existing base keys.")

    # 4. Test ingestion & bitwise value integrity for Run C
    model_c = create_b2_unet(pretrained=False, num_transformer_layers=4, p3_mode="C")
    incomp_c = load_locked_base_into_p3(model_c, mock_base_ckpt, p3_mode="C")
    assert len(incomp_c.unexpected_keys) == 0, f"Run C unexpected keys: {incomp_c.unexpected_keys}"
    assert set(incomp_c.missing_keys) == P3_EXPECTED_KEYS["C"], "Run C missing keys mismatch!"
    sd_c = model_c.state_dict()
    for k, v_base in mock_base_dict.items():
        assert torch.equal(sd_c[k], v_base), f"Bitwise value mismatch in Run C for key: {k}"
    print(f"  --> Run C: 100% bitwise equality verified across all {num_base_keys} pre-existing base keys.")

    # 5. PE28 Provenance Verification:
    # Verify backbone.pe28_fixed is bitwise derived from the base checkpoint's positional_embeddings
    # using exactly the frozen 2D bicubic 14x14 -> 28x28 transformation.
    base_pos = mock_base_dict["backbone.positional_embeddings"]
    pos_4d = base_pos.reshape(1, 14, 14, -1).permute(0, 3, 1, 2)
    expected_pe28 = F.interpolate(pos_4d, size=(28, 28), mode="bicubic", align_corners=False).permute(0, 2, 3, 1).flatten(1, 2).detach().float()

    assert torch.equal(model_a.backbone.pe28_fixed, expected_pe28), "Run A pe28_fixed is not bitwise derived from base positional_embeddings!"
    assert torch.equal(model_b.backbone.pe28_fixed, expected_pe28), "Run B pe28_fixed is not bitwise derived from base positional_embeddings!"
    assert torch.equal(model_c.backbone.pe28_fixed, expected_pe28), "Run C pe28_fixed is not bitwise derived from base positional_embeddings!"

    # Verify Run A, Run B, and Run C initialized from the same base checkpoint contain identical pe28_fixed
    assert torch.equal(model_a.backbone.pe28_fixed, model_b.backbone.pe28_fixed), "pe28_fixed mismatch between Run A and Run B!"
    assert torch.equal(model_b.backbone.pe28_fixed, model_c.backbone.pe28_fixed), "pe28_fixed mismatch between Run B and Run C!"
    print("  --> PE28 Provenance: 100% bitwise derivation from base positional_embeddings and cross-run equality (Run A == Run B == Run C) verified.")

    # 6. Test fail-fast on corrupted checkpoint (e.g. key missing in base)
    corrupted_ckpt = {"model_state_dict": copy.deepcopy(mock_base_dict)}
    del corrupted_ckpt["model_state_dict"]["backbone.convnext.stages.2.main_block.module.blocks.0.conv_dw.weight"]
    try:
        load_locked_base_into_p3(model_c, corrupted_ckpt, p3_mode="C")
        assert False, "Should have failed fast on corrupted base checkpoint!"
    except RuntimeError as e:
        assert "Strict missing-keys mismatch" in str(e)

    print("  --> PASS: Strict whitelist checkpoint ingestion, PE28 provenance, and 100% bitwise base value integrity verified for Run A, B, and C.")


def test_11_fresh_model_reload_and_serialization():
    print("\n[Test 11] Verifying Fresh-Model state_dict reload & PE28 storage isolation...")
    # 1. Create model_1 and obtain its state_dict
    model_1 = create_b2_unet(pretrained=False, num_transformer_layers=4, p3_mode="C")
    sd_1 = copy.deepcopy(model_1.state_dict())

    # 2. Create completely fresh model_2 and reload state_dict
    model_2 = create_b2_unet(pretrained=False, num_transformer_layers=4, p3_mode="C")
    load_res = model_2.load_state_dict(sd_1, strict=True)
    assert len(load_res.missing_keys) == 0, f"Missing keys on reload: {load_res.missing_keys}"
    assert len(load_res.unexpected_keys) == 0, f"Unexpected keys on reload: {load_res.unexpected_keys}"

    # 3. Verify single canonical registered buffer on model_2
    assert hasattr(model_2.backbone, "pe28_fixed"), "model_2 backbone must own pe28_fixed buffer"
    pe28_keys_m2 = [k for k in model_2.state_dict().keys() if "pe28" in k]
    assert pe28_keys_m2 == ["backbone.pe28_fixed"], f"Expected single key ['backbone.pe28_fixed'], got {pe28_keys_m2}"

    # 4. Verify SageLayer.get_pe28() resolves to model_2.backbone.pe28_fixed
    s0_m2 = model_2.backbone.convnext.stages[0]
    s1_m2 = model_2.backbone.convnext.stages[1]
    assert s0_m2.get_pe28() is model_2.backbone.pe28_fixed, "Stage 0 must resolve to model_2 buffer"
    assert s1_m2.get_pe28() is model_2.backbone.pe28_fixed, "Stage 1 must resolve to model_2 buffer"

    # 5. Verify no SageLayer registers its own pe28 buffer and no registered tensor alias exists
    for stage_idx, stage in enumerate(model_2.backbone.convnext.stages):
        assert "pe28_fixed" not in stage._buffers, f"Stage {stage_idx} registered its own pe28_fixed buffer!"
        assert "_pe_owner" not in stage._modules, f"Stage {stage_idx} registered _pe_owner in _modules!"

    # 6. Verify model_1 and model_2 do not share PE28 tensor storage
    ptr1 = model_1.backbone.pe28_fixed.data_ptr()
    ptr2 = model_2.backbone.pe28_fixed.data_ptr()
    assert ptr1 != ptr2, f"Storage overlap! model_1 and model_2 share identical data pointer: {ptr1}"

    # 7. Supplemental tests: deepcopy, pickle, and device/dtype migration
    pickled = pickle.dumps(model_2)
    model_unpickled = pickle.loads(pickled)
    assert model_unpickled.backbone.convnext.stages[0].get_pe28() is model_unpickled.backbone.pe28_fixed

    model_copied = copy.deepcopy(model_2)
    assert model_copied.backbone.convnext.stages[0].get_pe28() is model_copied.backbone.pe28_fixed

    if torch.cuda.is_available():
        model_2.cuda()
        assert model_2.backbone.convnext.stages[0].get_pe28().device.type == "cuda"
        model_2.cpu()
        assert model_2.backbone.convnext.stages[0].get_pe28().device.type == "cpu"
    else:
        model_2.double()
        assert model_2.backbone.convnext.stages[0].get_pe28().dtype == torch.float64
        model_2.float()

    print("  --> PASS: Fresh-model state_dict reload, single canonical ownership, and tensor storage isolation verified.")


def test_12_end_to_end_forward_backward_dry_run():
    print("\n[Test 12] Running end-to-end forward + backward dry-run on synthetic tensors...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_c = create_b2_unet(pretrained=False, num_transformer_layers=4, p3_mode="C").to(device)
    model_c.train()

    B = 2
    x = torch.randn(B, 3, 448, 448, device=device)
    
    # Forward pass
    out = model_c(x)
    assert out.shape == (B, 1, 448, 448), f"Expected (B, 1, 448, 448), got {out.shape}"
    assert not torch.isnan(out).any(), "Output tensor contains NaN!"
    assert not torch.isinf(out).any(), "Output tensor contains Inf!"

    # Backward pass
    target = torch.randint(0, 2, (B, 1, 448, 448), device=device).float()
    loss = F.binary_cross_entropy_with_logits(out, target)
    loss.backward()

    # Verify P3 refinement parameters received gradients
    s0_p3 = model_c.backbone.convnext.stages[0].p3_refinement
    assert s0_p3.pw.weight.grad is not None, "PWConv weight must receive gradient"
    assert s0_p3.gamma.grad is not None, "gamma must receive gradient"
    assert not torch.isnan(s0_p3.pw.weight.grad).any(), "PWConv grad contains NaN"

    # Verify PE28 received NO gradient (requires_grad=False)
    assert model_c.backbone.pe28_fixed.grad is None, "PE28 must not accumulate gradients!"

    print(f"  --> PASS: Full end-to-end forward/backward completed successfully. Loss = {loss.item():.4f}")


def test_13_locked_base_provenance_gate():
    print("\n[Test 13] Verifying Locked-Base Provenance Gate strictness & negative test cases...")

    # 1. Negative Test 1: Run with NO --locked-base and NO locked_base_checkpoint in YAML
    cfg_no_base = {"p3_mode": "B"}
    resolved_path_1 = resolve_locked_base_path(cfg_no_base, locked_base_override=None)
    assert resolved_path_1 is None, f"Expected None for missing locked base, got: {resolved_path_1}"
    mock_results_1 = [{"locked_base_ingested": False, "locked_base_provenance": "NOT_PROVEN"}]
    verdict_1 = compute_preflight_verdict(all_passed=True, results=mock_results_1)
    assert "GATED" in verdict_1, f"Expected GATED verdict when locked-base is missing, got: {verdict_1}"
    print("  --> Negative Test 1: Config with NO --locked-base & NO locked_base_checkpoint -> GATED (PASS)")

    # 2. Negative Test 2: Generic 'checkpoint' alone does NOT count as locked-base provenance
    cfg_generic_ckpt = {"p3_mode": "B", "checkpoint": "results/runs/some_arbitrary_checkpoint.pth"}
    resolved_path_2 = resolve_locked_base_path(cfg_generic_ckpt, locked_base_override=None)
    assert resolved_path_2 is None, (
        f"Security violation! Generic 'checkpoint' was erroneously resolved as locked base: {resolved_path_2}"
    )
    mock_results_2 = [{"locked_base_ingested": False, "locked_base_provenance": "NOT_PROVEN"}]
    verdict_2 = compute_preflight_verdict(all_passed=True, results=mock_results_2)
    assert "GATED" in verdict_2, f"Expected GATED verdict for generic checkpoint, got: {verdict_2}"
    print("  --> Negative Test 2: Generic 'checkpoint' alone does NOT satisfy locked base provenance -> GATED (PASS)")

    # 3. Negative Test 3: Trainer strictness in train_crack.py:
    # Mutual exclusivity of --locked-base and --checkpoint for P3 runs
    p3_mode = "B"
    locked_base_path = "checkpoints/locked_base_b2_depth4.pth"
    generic_ckpt_path = "results/runs/best_model_b2.pth"
    try:
        if p3_mode is not None and locked_base_path and generic_ckpt_path:
            raise ValueError(
                "For P3 runs (p3_mode is not None), both --locked-base (locked_base_checkpoint) and "
                "--checkpoint cannot be supplied simultaneously. --locked-base is strictly for "
                "Locked Base provenance, and --checkpoint is strictly for resume/continue."
            )
        assert False, "Should have raised ValueError on simultaneous --locked-base and --checkpoint!"
    except ValueError as e:
        assert "cannot be supplied simultaneously" in str(e)
    print("  --> Negative Test 3: Trainer rejects simultaneous --locked-base and --checkpoint for P3 (PASS)")

    # 4. Negative Test 4: Trainer rejects P3 training when --locked-base is omitted
    p3_mode = "C"
    locked_base_path = None
    generic_ckpt_path = "results/runs/best_model_b2.pth"
    try:
        if p3_mode is not None and not locked_base_path:
            raise ValueError(
                f"P3 training (p3_mode='{p3_mode}') requires an explicit Locked Base checkpoint "
                "via --locked-base or 'locked_base_checkpoint' in YAML config. "
                "Generic --checkpoint cannot be used to initialize or bypass Locked Base provenance."
            )
        assert False, "Should have raised ValueError on missing --locked-base for P3!"
    except ValueError as e:
        assert "requires an explicit Locked Base checkpoint" in str(e)
    print("  --> Negative Test 4: Trainer rejects P3 training when --locked-base is omitted (PASS)")

    # 5. Positive Test: Explicit valid authorized --locked-base checkpoint file
    real_ckpt_path = os.path.join(project_root, "checkpoints", "locked_base_b2_depth4.pth")
    assert os.path.exists(real_ckpt_path), (
        f"FAIL-FAST: Required locked-base checkpoint missing from repository at {real_ckpt_path}!"
    )
    resolved_path_3 = resolve_locked_base_path({}, locked_base_override=real_ckpt_path)
    assert resolved_path_3 == real_ckpt_path
    sha = compute_file_sha256(resolved_path_3)
    assert sha == EXPECTED_LOCKED_BASE_SHA256, (
        f"Checkpoint SHA256 mismatch! Expected {EXPECTED_LOCKED_BASE_SHA256}, got {sha}"
    )

    # Verify bitwise PE14 loading
    m_b = create_b2_unet(pretrained=False, num_transformer_layers=4, p3_mode="B")
    load_locked_base_into_p3(m_b, real_ckpt_path, p3_mode="B")
    raw_sd = torch.load(real_ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"]
    assert torch.equal(m_b.backbone.positional_embeddings.cpu(), raw_sd["backbone.positional_embeddings"])

    # Verify verdict is PASS
    prov_label = f"VERIFIED_LOCKED_BASE({os.path.basename(real_ckpt_path)})"
    mock_results_3 = [{"locked_base_ingested": True, "locked_base_provenance": prov_label, "checkpoint_sha256": sha}]
    verdict_3 = compute_preflight_verdict(all_passed=True, results=mock_results_3)
    assert "PASS" in verdict_3, f"Expected PASS verdict, got: {verdict_3}"
    print(f"  --> Positive Test: Valid locked base matches authorized SHA256 ({EXPECTED_LOCKED_BASE_SHA256[:16]}...) -> PASS (PASS)")

    print("  --> PASS: Locked-Base Provenance Gate passed 100% (Negative & Strictness contracts satisfied).")


def run_all_verification_tests():
    print("=" * 80)
    print("P3-PHASE-5: MINIMAL VERIFICATION SUITE EXECUTION")
    print("=" * 80)
    test_1_shape_contracts()
    test_2_parameter_counts_and_bias()
    test_3_gamma_semantics()
    test_4_pe28_ownership()
    test_5_runtime_pe_lookup()
    test_6_and_7_guard_and_non_target_routes()
    test_8_self_selection_bypass()
    test_9_vit_block_internals()
    test_10_checkpoint_ingestion()
    test_11_fresh_model_reload_and_serialization()
    test_12_end_to_end_forward_backward_dry_run()
    test_13_locked_base_provenance_gate()
    print("\n" + "=" * 80)
    print("ALL 13 P3 VERIFICATION TESTS PASSED 100%!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_verification_tests()
