#!/usr/bin/env python3
"""
scripts/tests/test_u0_c3_invariants.py

Unit test suite verifying Phase 6 U0-C3 (DC-Init Stem) Invariants:
1. Shape Contract: Stem output shape == (B, 48, 112, 112)
2. Zero-Init Identity (t=0): Correlation between DCInitStem and True DC convolution > 0.99
3. Parameter Whitelist: Only DCInitStem + Stage 0 LayerNorms are trainable (~2,784 params)
4. Gradient Flow: Gradients exist only on whitelist parameters; all downstream layers have grad == None
5. Base Checkpoint Invariance: SHA256 matches expected locked checkpoint hash
"""

import hashlib
import os
import sys
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sage_lite_dir = os.path.join(project_root, "SAGE_LITE")
for p in [project_root, sage_lite_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from sage.networks import create_b2_unet, DCInitStem, init_dc_stem_from_pretrained


class TestU0C3Invariants(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.expected_ckpt_hash = "147f784021414efd0db514aa6dae94585fece820e88f584e436fc65de851fb66"
        self.ckpt_path = os.path.join(
            project_root,
            "results",
            "checkpoints",
            "P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth",
        )

    def test_01_checkpoint_hash_integrity(self):
        """Verifies base checkpoint SHA256 matches expected locked hash."""
        self.assertTrue(os.path.exists(self.ckpt_path), f"Checkpoint not found at: {self.ckpt_path}")
        with open(self.ckpt_path, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(
            actual_hash,
            self.expected_ckpt_hash,
            f"Checkpoint SHA256 mismatch! Expected {self.expected_ckpt_hash}, got {actual_hash}",
        )

    def test_02_stem_shape_contract(self):
        """Verifies DCInitStem output shape is strictly (B, 48, 112, 112)."""
        stem = DCInitStem().to(self.device)
        dummy_x = torch.randn(2, 3, 448, 448, device=self.device)
        out = stem(dummy_x)
        self.assertEqual(out.shape, (2, 48, 112, 112))

    def test_03_zero_init_dc_identity(self):
        """Verifies DCInitStem output at t=0 matches True DC convolution with correlation > 0.999."""
        model = create_b2_unet(num_transformer_layers=4, p3_mode="C", pretrained=False).to(self.device)
        ckpt = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        orig_stem = model.backbone.convnext.stem
        dc_stem = DCInitStem().to(self.device)
        init_dc_stem_from_pretrained(dc_stem, orig_stem)

        dummy_x = torch.randn(2, 3, 448, 448, device=self.device)
        with torch.no_grad():
            out_dc_stem = dc_stem(dummy_x)

            # Mathematical True DC convolution: spatial mean kernel + original bias + original LayerNorm
            W_full = orig_stem[0].weight.data
            b_full = orig_stem[0].bias.data
            W_dc = W_full.mean(dim=(2, 3), keepdim=True).repeat(1, 1, 4, 4)
            conv_dc = F.conv2d(dummy_x, W_dc, b_full, stride=4)
            out_true_dc = orig_stem[1](conv_dc)

        max_diff = (out_dc_stem - out_true_dc).abs().max().item()
        corr = torch.corrcoef(torch.stack([out_dc_stem.flatten(), out_true_dc.flatten()]))[0, 1].item()

        self.assertLess(max_diff, 1e-4, f"DCInitStem vs True DC max diff too high: {max_diff}")
        self.assertGreater(corr, 0.999, f"DCInitStem vs True DC correlation too low: {corr}")

    def test_04_parameter_whitelist_and_freeze(self):
        """Verifies that only DCInitStem and Stage 0 LayerNorms are trainable."""
        model = create_b2_unet(num_transformer_layers=4, p3_mode="C", pretrained=False).to(self.device)
        ckpt = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        orig_stem = model.backbone.convnext.stem
        dc_stem = DCInitStem().to(self.device)
        init_dc_stem_from_pretrained(dc_stem, orig_stem)
        model.backbone.convnext.stem = dc_stem

        # Freeze entire model first
        for p in model.parameters():
            p.requires_grad = False

        # Unfreeze DCInitStem
        for p in dc_stem.parameters():
            p.requires_grad = True

        # Unfreeze Stage 0 LayerNorms (ConvNeXt stage is wrapped in SageLayer -> main_block.module)
        stage0 = model.backbone.convnext.stages[0]
        stage0_convnext = stage0.main_block.module if hasattr(stage0, "main_block") else stage0
        stage0_ln_params = []
        for block in stage0_convnext.blocks:
            if hasattr(block, "norm"):
                if block.norm.weight is not None:
                    block.norm.weight.requires_grad = True
                    stage0_ln_params.append(block.norm.weight)
                if block.norm.bias is not None:
                    block.norm.bias.requires_grad = True
                    stage0_ln_params.append(block.norm.bias)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        total_trainable = sum(p.numel() for p in trainable_params)

        # Expected: dc_proj (144+48=192) + ac_delta (2304) + norm (48+48=96) + 2x Stage0 LNs (2x96=192) = 2,784
        self.assertEqual(total_trainable, 2784, f"Trainable params mismatch: {total_trainable} vs expected 2784")

    def test_05_gradient_flow(self):
        """Verifies backward pass sends gradients ONLY to trainable whitelist and none to frozen layers."""
        model = create_b2_unet(num_transformer_layers=4, p3_mode="C", pretrained=False).to(self.device)
        ckpt = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        orig_stem = model.backbone.convnext.stem
        dc_stem = DCInitStem().to(self.device)
        init_dc_stem_from_pretrained(dc_stem, orig_stem)
        model.backbone.convnext.stem = dc_stem

        for p in model.parameters():
            p.requires_grad = False

        for p in dc_stem.parameters():
            p.requires_grad = True

        stage0 = model.backbone.convnext.stages[0]
        stage0_convnext = stage0.main_block.module if hasattr(stage0, "main_block") else stage0
        for block in stage0_convnext.blocks:
            if hasattr(block, "norm"):
                if block.norm.weight is not None:
                    block.norm.weight.requires_grad = True
                if block.norm.bias is not None:
                    block.norm.bias.requires_grad = True

        dummy_x = torch.randn(2, 3, 448, 448, device=self.device)
        out = model(dummy_x)
        loss = out.sum()
        loss.backward()

        # Check whitelist gradients are not None and non-zero
        self.assertIsNotNone(dc_stem.dc_proj.weight.grad)
        self.assertIsNotNone(dc_stem.ac_delta.weight.grad)
        self.assertIsNotNone(dc_stem.norm.weight.grad)

        # Check frozen layer gradients are strictly None
        self.assertIsNone(stage0_convnext.blocks[0].conv_dw.weight.grad)
        stage1 = model.backbone.convnext.stages[1]
        stage1_convnext = stage1.main_block.module if hasattr(stage1, "main_block") else stage1
        self.assertIsNone(stage1_convnext.blocks[0].norm.weight.grad)
        self.assertIsNone(model.decoder.segmentation_head[0].weight.grad)


if __name__ == "__main__":
    unittest.main()
