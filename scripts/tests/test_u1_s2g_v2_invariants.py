"""
scripts/tests/test_u1_s2g_v2_invariants.py

Test Suite for Phase 6 U1-S2G-v2 (Conv3x3 Gate Module) Invariants:
1. Target Model Architecture and Channel Contract:
   - S2 features: [B, 192, 28, 28] (Decoder Block 0 output)
   - Skip S1 features: [B, 96, 56, 56] (Stage 1 skip connection)
   - Gated skip S1: [B, 96, 56, 56]
   - Gate alpha: [B, 1, 56, 56] in (0, 1)
   - Conv3x3 RF: kernel_size=3, padding=1 (RF expands from 1px -> 3px)
2. Identity Initialization & Warm-Start from v1:
   - Final conv weight is strictly zero-init, bias = +5.0 -> alpha ~ 0.9933
   - warm_start_from_v1 embeds 1x1 kernel at center (1, 1) with zero spatial neighbors
   - Max output difference |gate_v2(warm_start) - gate_v1| < 1e-4 at t=0
3. Parameter Whitelist & Count:
   - S2GateModule (kernel_size=3):
     Conv1 (3x3): 288 * 32 * 3 * 3 + 32 = 82,944 + 32 = 82,976
     BN: 32 weight + 32 bias = 64
     Conv2 (1x1): 32 * 1 + 1 = 33
     Total = 82,976 + 64 + 33 = 83,073 parameters
   - All other layers strictly FROZEN.
4. Gradient Flow Isolation:
   - Only S2GateModule (Conv3x3) receives non-zero gradients.
5. Base Checkpoint SHA256 Integrity:
   - 147f784021414efd0db514aa6dae94585fece820e88f584e436fc65de851fb66
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


class TestU1S2Gv2Invariants(unittest.TestCase):
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

    def test_02_v2_shape_and_warm_start_preservation(self):
        """Verifies S2GateModule v2 (Conv3x3) shapes and exact warm-start preservation."""
        s2_gate_v1 = S2GateModule(s2_channels=192, skip_channels=96, kernel_size=1).to(self.device)
        s2_gate_v2 = S2GateModule(s2_channels=192, skip_channels=96, kernel_size=3).to(self.device)

        s2_feat = torch.randn(2, 192, 28, 28, device=self.device)
        skip_s1 = torch.randn(2, 96, 56, 56, device=self.device)

        # Warm start v2 from v1
        s2_gate_v2.warm_start_from_v1(s2_gate_v1.state_dict())

        with torch.no_grad():
            skip_gated_v1, gate_alpha_v1 = s2_gate_v1(s2_feat, skip_s1)
            skip_gated_v2, gate_alpha_v2 = s2_gate_v2(s2_feat, skip_s1)

        self.assertEqual(skip_gated_v2.shape, (2, 96, 56, 56))
        self.assertEqual(gate_alpha_v2.shape, (2, 1, 56, 56))

        # Check exact numerical preservation between v1 and v2 at t=0
        diff_alpha = torch.abs(gate_alpha_v1 - gate_alpha_v2).max().item()
        diff_skip = torch.abs(skip_gated_v1 - skip_gated_v2).max().item()
        self.assertLess(diff_alpha, 1e-4, f"Warm-start alpha mismatch: {diff_alpha}")
        self.assertLess(diff_skip, 1e-4, f"Warm-start gated skip mismatch: {diff_skip}")

    def test_03_trainable_whitelist_count(self):
        """Verifies trainable whitelist contains strictly S2GateModule v2 (83,073 parameters)."""
        model = load_model_from_checkpoint(self.config_path, self.ckpt_path, device=self.device)
        s2_gate_v2 = S2GateModule(s2_channels=192, skip_channels=96, kernel_size=3).to(self.device)
        model.decoder.s2_gate = s2_gate_v2
        model.decoder.use_s2_gate = True

        for p in model.parameters():
            p.requires_grad = False
        for p in s2_gate_v2.parameters():
            p.requires_grad = True

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        total_trainable = sum(p.numel() for p in trainable_params)

        # Expected:
        # Conv3x3: 288 * 32 * 9 + 32 = 82,976
        # BN: 32 + 32 = 64
        # Conv1x1: 32 * 1 + 1 = 33
        # Total = 82,976 + 64 + 33 = 83,073
        self.assertEqual(total_trainable, 83073, f"Trainable param count mismatch: {total_trainable} vs 83,073")

    def test_04_gradient_isolation(self):
        """Verifies backward pass sends gradients ONLY to S2Gate v2 and zero to all other layers."""
        model = load_model_from_checkpoint(self.config_path, self.ckpt_path, device=self.device)
        s2_gate_v2 = S2GateModule(s2_channels=192, skip_channels=96, kernel_size=3).to(self.device)
        model.decoder.s2_gate = s2_gate_v2
        model.decoder.use_s2_gate = True

        for p in model.parameters():
            p.requires_grad = False
        for p in s2_gate_v2.parameters():
            p.requires_grad = True

        dummy_x = torch.randn(2, 3, 448, 448, device=self.device)
        out = model(dummy_x)
        loss = out.sum()
        loss.backward()

        for name, p in s2_gate_v2.named_parameters():
            self.assertIsNotNone(p.grad, f"S2Gate param {name} grad is None")

        self.assertIsNone(model.decoder.decoder_blocks[0].conv1[0].weight.grad)
        self.assertIsNone(model.decoder.decoder_blocks[1].conv1[0].weight.grad)
        self.assertIsNone(model.decoder.segmentation_head[0].weight.grad)
        for p in model.backbone.parameters():
            self.assertIsNone(p.grad)


if __name__ == "__main__":
    unittest.main()
