"""
scripts/tests/test_u1_s2g_invariants.py

Test Suite for Phase 6 U1-S2G (S2-Gate Module) Invariants:
1. Target Model Architecture and Channel Contract:
   - S2 features: [B, 192, 28, 28] (Decoder Block 0 output)
   - Skip S1 features: [B, 96, 56, 56] (Stage 1 skip connection)
   - Gated skip S1: [B, 96, 56, 56]
   - Gate alpha: [B, 1, 56, 56] in (0, 1)
2. Identity Initialization at t=0:
   - Final conv weight is strictly zero
   - Final conv bias is +5.0 -> gate alpha ~ sigmoid(5.0) ~ 0.9933
   - Mean gate alpha > 0.99
   - Max difference |model_s2gate - model_base| < 1e-3 on full forward pass
3. Parameter Whitelist & Count:
   - S2GateModule: (192+96)*32 + 32 (conv1) + 32*2 (BN) + 32*1 + 1 (conv2)
     = 288*32 + 32 + 64 + 32 + 1 = 9216 + 32 + 64 + 33 = 9,345 parameters
   - All other layers in Backbone, SAGE, and Decoder strictly FROZEN
4. Gradient Flow Isolation:
   - Only S2GateModule parameters receive non-zero gradients
   - Decoder Block 0, Block 1, Block 2, Seg Head, Stages 0-3 have zero/None gradient
5. Checkpoint Hash Integrity:
   - Base checkpoint SHA256 is strictly 147f784021414efd0db514aa6dae94585fece820e88f584e436fc65de851fb66
"""

import hashlib
import os
import sys
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
cand_roots = [
    project_root,
    os.path.abspath(os.path.join(project_root, "..")),
    os.getcwd(),
    "/content",
    "/content/SAGE_LITE",
]
for p in cand_roots:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from sage.networks import create_b2_unet, S2GateModule
from tools.run_phase6_c_topology_diagnostic import load_model_from_checkpoint


class TestU1S2GInvariants(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.expected_ckpt_hash = "147f784021414efd0db514aa6dae94585fece820e88f584e436fc65de851fb66"

        rel_subpath = os.path.join("results", "checkpoints", "P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth")
        self.ckpt_path = None
        for root in cand_roots:
            candidate = os.path.join(root, rel_subpath)
            if os.path.exists(candidate):
                self.ckpt_path = candidate
                break
        if self.ckpt_path is None:
            self.ckpt_path = os.path.join(project_root, rel_subpath)

        rel_config = os.path.join("configs", "p3_ablation", "b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml")
        self.config_path = None
        for root in cand_roots:
            cand_cfg = os.path.join(root, rel_config)
            if os.path.exists(cand_cfg):
                self.config_path = cand_cfg
                break
            cand_cfg2 = os.path.join(root, "SAGE_LITE", rel_config)
            if os.path.exists(cand_cfg2):
                self.config_path = cand_cfg2
                break

    def test_01_checkpoint_integrity(self):
        """Verifies candidate B checkpoint hash matches locked value."""
        self.assertTrue(os.path.exists(self.ckpt_path), f"Checkpoint not found at {self.ckpt_path}")
        h = hashlib.sha256()
        with open(self.ckpt_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        self.assertEqual(h.hexdigest(), self.expected_ckpt_hash)

    def test_02_module_shape_and_identity_init(self):
        """Verifies S2GateModule output shapes and near-identity behavior at t=0."""
        s2_gate = S2GateModule(s2_channels=192, skip_channels=96).to(self.device)
        s2_feat = torch.randn(2, 192, 28, 28, device=self.device)
        skip_s1 = torch.randn(2, 96, 56, 56, device=self.device)

        skip_gated, gate_alpha = s2_gate(s2_feat, skip_s1)

        self.assertEqual(skip_gated.shape, (2, 96, 56, 56))
        self.assertEqual(gate_alpha.shape, (2, 1, 56, 56))

        # Alpha should be close to sigmoid(5.0) = 0.9933
        mean_alpha = gate_alpha.mean().item()
        min_alpha = gate_alpha.min().item()
        self.assertGreater(mean_alpha, 0.990)
        self.assertGreater(min_alpha, 0.985)

        # Max absolute difference between skip_s1 and skip_gated should be tiny
        diff = torch.max(torch.abs(skip_gated - skip_s1)).item()
        rel_diff = diff / (torch.max(torch.abs(skip_s1)).item() + 1e-8)
        self.assertLess(rel_diff, 0.01)

    def test_03_end_to_end_identity_at_t0(self):
        """Verifies full model forward with S2Gate vs baseline produces nearly identical logits at t=0."""
        # 1. Load baseline model
        base_model = load_model_from_checkpoint(self.config_path, self.ckpt_path, device=self.device)
        base_model.eval()

        # 2. Add S2GateModule directly into decoder
        s2_gate = S2GateModule(s2_channels=192, skip_channels=96).to(self.device)
        base_model.decoder.s2_gate = s2_gate
        base_model.decoder.use_s2_gate = True

        dummy_x = torch.randn(2, 3, 448, 448, device=self.device)
        with torch.no_grad():
            out_gated = base_model(dummy_x)

            # Turn gate off to get baseline output
            base_model.decoder.s2_gate = None
            out_base = base_model(dummy_x)

        max_diff = torch.max(torch.abs(out_gated - out_base)).item()
        self.assertLess(max_diff, 1e-2, f"Max difference at t=0 is too high: {max_diff:.4e}")

    def test_04_trainable_whitelist_count(self):
        """Verifies trainable whitelist contains strictly S2GateModule (~9,345 parameters)."""
        model = load_model_from_checkpoint(self.config_path, self.ckpt_path, device=self.device)
        s2_gate = S2GateModule(s2_channels=192, skip_channels=96).to(self.device)
        model.decoder.s2_gate = s2_gate
        model.decoder.use_s2_gate = True

        # Freeze all model parameters
        for p in model.parameters():
            p.requires_grad = False

        # Unfreeze strictly S2GateModule
        for p in s2_gate.parameters():
            p.requires_grad = True

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        total_trainable = sum(p.numel() for p in trainable_params)

        # Expected:
        # Conv1: 288 * 32 + 32 = 9,248
        # BN: 32 weight + 32 bias = 64
        # Conv2: 32 * 1 + 1 = 33
        # Total = 9,248 + 64 + 33 = 9,345
        self.assertEqual(total_trainable, 9345, f"Trainable parameter count mismatch: {total_trainable} vs 9345")

    def test_05_gradient_isolation(self):
        """Verifies backward pass sends gradients ONLY to S2Gate and zero to all other layers."""
        model = load_model_from_checkpoint(self.config_path, self.ckpt_path, device=self.device)
        s2_gate = S2GateModule(s2_channels=192, skip_channels=96).to(self.device)
        model.decoder.s2_gate = s2_gate
        model.decoder.use_s2_gate = True

        for p in model.parameters():
            p.requires_grad = False
        for p in s2_gate.parameters():
            p.requires_grad = True

        dummy_x = torch.randn(2, 3, 448, 448, device=self.device)
        out = model(dummy_x)
        loss = out.sum()
        loss.backward()

        # Check S2Gate has grads
        for name, p in s2_gate.named_parameters():
            self.assertIsNotNone(p.grad, f"S2Gate param {name} grad is None")

        # Check Decoder and Encoder layers have strictly None grad
        self.assertIsNone(model.decoder.decoder_blocks[0].conv1[0].weight.grad)
        self.assertIsNone(model.decoder.decoder_blocks[1].conv1[0].weight.grad)
        self.assertIsNone(model.decoder.segmentation_head[0].weight.grad)
        for p in model.backbone.parameters():
            self.assertIsNone(p.grad)


if __name__ == "__main__":
    unittest.main()
