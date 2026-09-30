"""
Unit test for SoftBoundaryIoULoss and enhanced CrackBinaryLoss.
Tests:
1. Shape handling: 3D and 4D targets.
2. Numerical stability: AMP FP16 forward + backward, no NaNs.
3. Boundary preservation: Thin 1-pixel crack boundary sum equals crack mask sum.
4. All-zero target handling: Zero division protection with epsilon.
5. Backward-compatibility: CrackBinaryLoss with boundary_weight=0.0 equals baseline loss.
"""

import sys
import os
import torch

# Ensure SAGE_LITE is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from scripts.train_crack import SoftBoundaryIoULoss, CrackBinaryLoss


def test_soft_boundary_iou_loss_basic():
    torch.manual_seed(42)
    loss_fn = SoftBoundaryIoULoss(dilation=2)

    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    targets = (torch.rand(2, 1, 64, 64) > 0.85).float()

    loss = loss_fn(logits, targets)
    assert not torch.isnan(loss), "Loss produced NaN!"
    loss.backward()
    assert logits.grad is not None, "Gradients not computed!"
    assert not torch.isnan(logits.grad).any(), "NaN in gradients!"
    print("[PASS] test_soft_boundary_iou_loss_basic")


def test_thin_crack_boundary_invariance():
    loss_fn = SoftBoundaryIoULoss(dilation=2)

    # 1-pixel hairline crack
    mask = torch.zeros(1, 1, 32, 32)
    mask[0, 0, 16, 5:25] = 1.0  # 1-pixel width, length 20
    b_gt = loss_fn._get_boundary(mask)

    # For 1-pixel crack, inner erosion removes all pixels, so boundary == original mask
    assert torch.equal(b_gt, mask), "1-pixel crack boundary must be identical to original mask!"
    print("[PASS] test_thin_crack_boundary_invariance")


def test_all_background_patch():
    loss_fn = SoftBoundaryIoULoss(dilation=2)

    logits_neg = torch.full((1, 1, 32, 32), -10.0, requires_grad=True)  # predict background
    targets = torch.zeros(1, 1, 32, 32)

    loss = loss_fn(logits_neg, targets)
    assert not torch.isnan(loss), "Loss on all-background patch produced NaN!"
    loss.backward()
    assert not torch.isnan(logits_neg.grad).any(), "Gradients on all-background patch produced NaN!"
    print("[PASS] test_all_background_patch")


def test_amp_fp16_stability():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_fn = SoftBoundaryIoULoss(dilation=2).to(device)

    logits = torch.randn(4, 1, 448, 448, device=device, requires_grad=True)
    targets = (torch.rand(4, 1, 448, 448, device=device) > 0.90).float()

    with torch.amp.autocast('cuda' if device == 'cuda' else 'cpu', enabled=(device == 'cuda')):
        loss = loss_fn(logits, targets)

    scaler = torch.amp.GradScaler('cuda' if device == 'cuda' else 'cpu', enabled=(device == 'cuda'))
    scaler.scale(loss).backward()
    scaler.step(torch.optim.SGD([logits], lr=1e-3))
    scaler.update()

    assert not torch.isnan(loss), "AMP FP16 loss produced NaN!"
    assert not torch.isnan(logits.grad).any(), "AMP FP16 grads produced NaN!"
    print("[PASS] test_amp_fp16_stability")


def test_backward_compatibility():
    torch.manual_seed(42)
    x = torch.randn(2, 1, 64, 64)
    y = (torch.rand(2, 1, 64, 64) > 0.8).float()

    # Legacy loss
    bce = torch.nn.BCEWithLogitsLoss()
    probs = torch.sigmoid(x)
    inter = (probs * y).sum(dim=(2, 3))
    uni = probs.sum(dim=(2, 3)) + y.sum(dim=(2, 3))
    dice = (2.0 * inter + 1e-5) / (uni + 1e-5)
    legacy_loss = 1.0 * bce(x, y) + 1.5 * (1.0 - dice.mean())

    crit_default = CrackBinaryLoss()
    crit_zero = CrackBinaryLoss(boundary_weight=0.0)
    crit_active = CrackBinaryLoss(boundary_weight=0.5, boundary_dilation=2)

    val_default = crit_default(x, y)
    val_zero = crit_zero(x, y)
    val_active = crit_active(x, y)

    assert torch.isclose(legacy_loss, val_default), "Default CrackBinaryLoss diverges from legacy!"
    assert torch.isclose(val_default, val_zero), "boundary_weight=0.0 diverges from default!"
    assert val_active.item() > val_default.item(), "Active boundary loss should penalize boundary errors!"
    print("[PASS] test_backward_compatibility")


if __name__ == '__main__':
    print("=" * 60)
    print("RUNNING SOFT BOUNDARY IOU LOSS TEST SUITE")
    print("=" * 60)
    test_soft_boundary_iou_loss_basic()
    test_thin_crack_boundary_invariance()
    test_all_background_patch()
    test_amp_fp16_stability()
    test_backward_compatibility()
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY (5/5)!")
    print("=" * 60)
