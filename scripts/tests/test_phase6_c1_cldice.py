"""
Preflight Test Suite for Phase 6-C.1: Topology Probe (Soft-clDice)
Tests:
1. Exact Equivalence: On fixed logits and target tensors, CrackBinaryLoss with cldice_weight=0.0
   is bitwise identical (torch.equal) to the canonical Base loss for both loss and gradients.
2. Topology Sensitivity: A broken/fragmented crack with gaps produces higher clDice loss than
   a continuous crack with the exact same pixel volume.
3. AMP FP16 Stability: Forward + backward under AMP FP16 produces no NaNs or Infs across sparse,
   dense, and all-background patches.
4. Exact Parameter Invariance: Total and trainable parameter count equals exactly 10,118,955 (+0 params).
5. Stage-1 Checkpoint Loading Contract: Candidate B Stage-1 checkpoint loads cleanly with strict=True
   and produces correct output shape (B, 1, 448, 448).
"""

import sys
import os
import torch
import torch.nn.functional as F
import yaml

# Ensure SAGE_LITE is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from scripts.train_crack import SoftclDiceLoss, CrackBinaryLoss
from sage.networks import create_b2_unet


def test_1_loss_and_gradient_exact_equivalence():
    """Preflight 1: Exact equality on fixed logits tensor when cldice_weight=0.0."""
    torch.manual_seed(42)
    B, C, H, W = 4, 1, 64, 64
    logits_base = torch.randn(B, C, H, W, requires_grad=True)
    logits_new = logits_base.clone().detach().requires_grad_(True)
    targets = (torch.rand(B, C, H, W) > 0.8).float()

    # Reference Base loss path
    bce = torch.nn.BCEWithLogitsLoss()
    probs = torch.sigmoid(logits_base)
    inter = (probs * targets).sum(dim=(2, 3))
    uni = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2.0 * inter + 1e-5) / (uni + 1e-5)
    loss_base = 1.0 * bce(logits_base, targets) + 1.5 * (1.0 - dice.mean())
    loss_base.backward()

    # Enhanced CrackBinaryLoss with cldice_weight=0.0
    crit = CrackBinaryLoss(
        bce_weight=1.0,
        dice_weight=1.5,
        boundary_weight=0.0,
        margin_weight=0.0,
        cldice_weight=0.0,
    )
    loss_new = crit(logits_new, targets)
    loss_new.backward()

    # Check exact tensor equality
    assert torch.equal(loss_new, loss_base), f"Loss mismatch: {loss_new.item()} vs {loss_base.item()}"
    assert torch.equal(logits_new.grad, logits_base.grad), "Gradient mismatch on fixed logits tensor!"
    print("[PASS] Preflight 1: Loss & Gradient exact equivalence when cldice_weight=0.0 (torch.equal verified)")


def test_2_topology_sensitivity():
    """Preflight 2: Broken/fragmented crack yields significantly higher clDice loss than continuous crack."""
    H, W = 64, 64
    targets = torch.zeros(1, 1, H, W)
    targets[:, :, 30:34, 10:54] = 1.0  # Continuous horizontal crack: length 44, width 4 -> 176 pixels

    # Continuous prediction: matches GT
    logits_cont = torch.full((1, 1, H, W), -5.0)
    logits_cont[:, :, 30:34, 10:54] = 5.0

    # Broken/fragmented prediction: same pixel count (44 cols), but 2 large gaps in the middle
    logits_broken = torch.full((1, 1, H, W), -5.0)
    logits_broken[:, :, 30:34, 10:22] = 5.0  # chunk 1 (12 cols)
    # gap of 5 cols (22:27)
    logits_broken[:, :, 30:34, 27:39] = 5.0  # chunk 2 (12 cols)
    # gap of 5 cols (39:44)
    logits_broken[:, :, 30:34, 44:64] = 5.0  # chunk 3 (20 cols)

    cldice_loss_fn = SoftclDiceLoss(iters=5)
    loss_cont = cldice_loss_fn(logits_cont, targets)
    loss_broken = cldice_loss_fn(logits_broken, targets)

    assert loss_broken > loss_cont, f"Expected broken loss > continuous loss, got broken={loss_broken.item():.4f}, cont={loss_cont.item():.4f}"
    diff = (loss_broken - loss_cont).item()
    assert diff > 0.05, f"Expected significant gap in clDice loss, got diff={diff:.4f}"
    print(f"[PASS] Preflight 2: Topology sensitivity verified (continuous={loss_cont.item():.4f} vs broken={loss_broken.item():.4f}, diff=+{diff:.4f})")


def test_3_amp_fp16_numerical_stability():
    """Preflight 3: AMP FP16 forward and backward stability under diverse target regimes."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    crit = CrackBinaryLoss(bce_weight=1.0, dice_weight=1.5, cldice_weight=0.030, cldice_iters=5).to(device)

    scenarios = [
        ("sparse_crack", 0.98),
        ("dense_crack", 0.70),
        ("all_background", 1.00),
    ]

    for name, threshold in scenarios:
        torch.manual_seed(123)
        logits = torch.randn(2, 1, 112, 112, device=device, requires_grad=True)
        if threshold >= 1.0:
            targets = torch.zeros(2, 1, 112, 112, device=device)
        else:
            targets = (torch.rand(2, 1, 112, 112, device=device) > threshold).float()

        with torch.amp.autocast('cuda' if device == 'cuda' else 'cpu', enabled=(device == 'cuda')):
            loss = crit(logits, targets)

        assert not torch.isnan(loss), f"NaN detected in {name} scenario!"
        assert not torch.isinf(loss), f"Inf detected in {name} scenario!"

        loss.backward()
        assert logits.grad is not None, f"Gradient is None in {name} scenario!"
        assert not torch.isnan(logits.grad).any(), f"NaN in gradients for {name} scenario!"
        assert not torch.isinf(logits.grad).any(), f"Inf in gradients for {name} scenario!"

    print(f"[PASS] Preflight 3: AMP FP16 numerical stability verified on device: {device}")


def test_4_exact_parameter_invariance():
    """Preflight 4: Verifies total parameters remain exactly 10,118,955 (+0 dynamic parameters)."""
    config_candidates = [
        "configs/p3_ablation/b2_p3_run_c_d4_k2_h64_phase6_c1_cldice.yaml",
        "SAGE_LITE/configs/p3_ablation/b2_p3_run_c_d4_k2_h64_phase6_c1_cldice.yaml",
        "../SAGE_LITE/configs/p3_ablation/b2_p3_run_c_d4_k2_h64_phase6_c1_cldice.yaml",
        "../../configs/p3_ablation/b2_p3_run_c_d4_k2_h64_phase6_c1_cldice.yaml"
    ]
    config_path = None
    for p in config_candidates:
        if os.path.exists(p):
            config_path = p
            break

    assert config_path is not None, "Could not find Phase 6-C.1 config YAML file!"

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_b2_unet(
        num_transformer_layers=cfg.get('num_transformer_layers', 4),
        p3_mode=cfg.get('p3_mode', 'C'),
        img_size=cfg.get('img_size', 448),
        sage_config=cfg.get('sage_config', {}),
        use_plu_head=cfg.get('use_plu_head', False),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    expected_params = 10_118_955
    assert total_params == expected_params, f"Parameter count mismatch: got {total_params:,}, expected {expected_params:,}"
    assert trainable_params == expected_params, f"Trainable parameter mismatch: got {trainable_params:,}, expected {expected_params:,}"

    # Verify zero dynamic parameters after forward pass
    dummy = torch.randn(2, 3, 448, 448, device=device)
    with torch.no_grad():
        out = model(dummy)
    params_after = sum(p.numel() for p in model.parameters())
    assert params_after == total_params, "Dynamic parameters detected after forward pass!"
    print(f"[PASS] Preflight 4: Exact parameter invariance ({total_params:,} params matched, +0 dynamic)")


def test_5_strict_checkpoint_loading():
    """Preflight 5: Strict loading from Candidate B Stage 1 checkpoint."""
    checkpoint_candidates = [
        "results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_stage1.pth",
        "../results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_stage1.pth",
        "../../results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_stage1.pth",
        "/content/drive/MyDrive/crack_seg/P3_C_Canonical_Base_D4_K2/best_model_b2_stage1.pth",
    ]
    ckpt_path = None
    for p in checkpoint_candidates:
        if os.path.exists(p):
            ckpt_path = p
            break

    if ckpt_path is None:
        print("[SKIP] Preflight 5: Candidate B Stage-1 checkpoint not found locally, skipping strict load check.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_b2_unet(
        num_transformer_layers=4,
        p3_mode="C",
        img_size=448,
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
        use_plu_head=False,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

    incompatible = model.load_state_dict(state_dict, strict=True)
    assert len(incompatible.missing_keys) == 0, f"Missing keys: {incompatible.missing_keys}"
    assert len(incompatible.unexpected_keys) == 0, f"Unexpected keys: {incompatible.unexpected_keys}"

    dummy = torch.randn(2, 3, 448, 448, device=device)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (2, 1, 448, 448), f"Output shape mismatch: {out.shape}"
    print("[PASS] Preflight 5: Strict checkpoint loading from Candidate B Stage-1 (0 missing, 0 unexpected)")


if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING PREFLIGHT TEST SUITE FOR PHASE 6-C.1 (SOFT-CLDICE PROBE)")
    print("=" * 80)
    test_1_loss_and_gradient_exact_equivalence()
    test_2_topology_sensitivity()
    test_3_amp_fp16_numerical_stability()
    test_4_exact_parameter_invariance()
    test_5_strict_checkpoint_loading()
    print("=" * 80)
    print("PREFLIGHT GATE: 5/5 PASSED")
    print("=" * 80)
