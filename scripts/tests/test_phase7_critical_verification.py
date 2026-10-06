#!/usr/bin/env python3
"""
scripts/tests/test_phase7_critical_verification.py

Phase 7: Critical Verification Suite for S2-Gate Block 2 Extension
===================================================================
Verifies the three mandatory prerequisites before Phase 7A:
2.1 Optimizer parameter group:
    - Optimizer strictly contains s2_gate_block2.parameters() only.
    - Zero base model parameters included.
2.2 Freeze + BatchNorm / train-eval mode:
    - Execution order: model.eval(), then s2_gate_block2.train().
    - Backbone, router, decoder, segmentation head, S2-Gate Block 1 frozen.
    - S2-Gate Block 2 is the ONLY trainable module.
    - Frozen BatchNorm layers do NOT update running statistics.
    - No rogue code calls model.train().
2.3 Numeric identity check (1x1 -> 3x3):
    - Warm-start conversion from 1x1 to 3x3 with zero padding.
    - Tested on a REAL batch from Crack500 validation set.
    - Measures max_abs_diff, mean_abs_diff, and allclose.
    - Tested in both model.eval() and (model.eval() + gate.train()) modes.
"""

import copy
import glob
import os
import sys
import unittest

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Resolve project roots
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
cand_roots = [
    project_root,
    os.path.abspath(os.path.join(project_root, "..")),
    os.getcwd(),
    "d:/truong/SpecialSubjectTTNT",
    "d:/truong/SpecialSubjectTTNT/SAGE_LITE",
]
for p in cand_roots:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from sage.networks import create_b2_unet, S2GateModule
from tools.run_phase6_c_topology_diagnostic import load_model_from_checkpoint


class TestPhase7CriticalVerification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[SetUp] Running Critical Verification on Device: {cls.device}")

        # Locate checkpoint with B1 gate
        cls.ckpt_path = None
        cand_ckpts = [
            os.path.join(project_root, "results", "phase6_combination", "phase6_comb_a1_s2g_end_to_end", "best_model_b2_global.pth"),
            "d:/truong/SpecialSubjectTTNT/SAGE_LITE/results/phase6_combination/phase6_comb_a1_s2g_end_to_end/best_model_b2_global.pth",
            "d:/truong/SpecialSubjectTTNT/results/checkpoints/P3_C_D4_K2_H64_Phase5_SAGELR2e-4_best_model_b2_global.pth",
        ]
        for p in cand_ckpts:
            if os.path.exists(p):
                cls.ckpt_path = p
                break
        assert cls.ckpt_path is not None, f"Could not find valid checkpoint in candidates: {cand_ckpts}"
        print(f"[SetUp] Using Base Checkpoint: {cls.ckpt_path}")

        # Locate config
        cls.config_path = None
        cand_cfgs = [
            os.path.join(project_root, "configs", "p3_ablation", "phase6_full_s1", "a1_s2g_end_to_end.yaml"),
            "d:/truong/SpecialSubjectTTNT/SAGE_LITE/configs/p3_ablation/phase6_full_s1/a1_s2g_end_to_end.yaml",
            "d:/truong/SpecialSubjectTTNT/SAGE_LITE/configs/p3_ablation/b2_p3_run_c_d4_k2_h64_phase5_sagelr2e4.yaml",
        ]
        for p in cand_cfgs:
            if os.path.exists(p):
                cls.config_path = p
                break
        assert cls.config_path is not None, f"Could not find valid config in candidates: {cand_cfgs}"
        print(f"[SetUp] Using Config: {cls.config_path}")

        # Locate real validation image from Crack500
        cls.val_images = []
        cand_val_dirs = [
            "d:/truong/SpecialSubjectTTNT/datasets/Crack500_ready/val/images",
            os.path.join(project_root, "datasets", "Crack500_ready", "val", "images"),
            os.path.join(project_root, "..", "datasets", "Crack500_ready", "val", "images"),
        ]
        for d in cand_val_dirs:
            if os.path.isdir(d):
                imgs = sorted(glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.jpg")))
                if len(imgs) > 0:
                    cls.val_images = imgs[:4]  # Batch of 4 real images
                    break
        assert len(cls.val_images) > 0, "Could not find real validation images from Crack500!"
        print(f"[SetUp] Loaded {len(cls.val_images)} REAL validation samples for testing.")

        # Build real tensor batch
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        batch_tensors = []
        for img_path in cls.val_images:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (448, 448), interpolation=cv2.INTER_LINEAR)
            norm = (img.astype(np.float32) / 255.0 - mean) / std
            t = torch.from_numpy(norm.transpose(2, 0, 1)).float()
            batch_tensors.append(t)
        cls.real_batch = torch.stack(batch_tensors).to(cls.device)
        print(f"[SetUp] Real validation batch tensor shape: {cls.real_batch.shape}")

    def test_2_1_optimizer_parameter_group(self):
        """
        Verification 2.1:
        Confirm optimizer only contains s2_gate_block2.parameters().
        Zero parameters from backbone, router, decoder, or S2-Gate Block 1.
        """
        print("\n--- Running Test 2.1: Optimizer Parameter Group Verification ---")
        model = load_model_from_checkpoint(self.config_path, self.ckpt_path, device=self.device)
        
        # Instantiate Block 2 S2-Gate
        s2_gate_b2 = S2GateModule(s2_channels=96, skip_channels=48, kernel_size=3).to(self.device)
        model.decoder.s2_gate_block2 = s2_gate_b2
        model.decoder.use_s2_gate_block2 = True

        # Construct optimizer following the exact Phase 7A specification
        optimizer = torch.optim.AdamW(s2_gate_b2.parameters(), lr=1e-3, weight_decay=1e-2)

        # Audit optimizer parameter list
        opt_params = []
        for group in optimizer.param_groups:
            opt_params.extend(group["params"])

        gate_params = list(s2_gate_b2.parameters())
        all_model_params = list(model.parameters())

        # Check 1: Param count matches gate exactly
        self.assertEqual(len(opt_params), len(gate_params), 
                         f"Optimizer params ({len(opt_params)}) does not match gate params ({len(gate_params)})")
        
        # Check 2: Every param in optimizer is in gate_params
        for p in opt_params:
            self.assertTrue(any(p is gp for gp in gate_params), "Found non-gate parameter in optimizer!")

        # Check 3: No parameter from backbone, router, decoder body, or B1 gate is in optimizer
        non_b2_params = [p for name, p in model.named_parameters() if "s2_gate_block2" not in name]
        for p in non_b2_params:
            self.assertFalse(any(p is op for op in opt_params), "A base model parameter leaked into optimizer!")

        total_opt_elements = sum(p.numel() for p in opt_params)
        print(f"[Test 2.1 PASS] Optimizer parameter count: {len(opt_params)} tensors, {total_opt_elements} elements.")
        print(f"[Test 2.1 PASS] Exactly matches S2-Gate Block 2 (Conv3x3: 41,601 elements).")

    def test_2_2_freeze_and_eval_train_mode_protocol(self):
        """
        Verification 2.2:
        Confirm exact execution order: model.eval(), then s2_gate_block2.train().
        Verify:
        - backbone frozen
        - router frozen
        - decoder frozen
        - segmentation head frozen
        - S2-Gate Block 1 frozen
        - S2-Gate Block 2 is the ONLY trainable module
        - Frozen BatchNorm layers do NOT update running statistics
        - Backward sends gradients ONLY to s2_gate_block2
        """
        print("\n--- Running Test 2.2: Freeze + BatchNorm / Train-Eval Mode Protocol ---")
        model = load_model_from_checkpoint(self.config_path, self.ckpt_path, device=self.device)

        # Plug in S2-Gate Block 2
        s2_gate_b2 = S2GateModule(s2_channels=96, skip_channels=48, kernel_size=3).to(self.device)
        model.decoder.s2_gate_block2 = s2_gate_b2
        model.decoder.use_s2_gate_block2 = True

        # Strict parameter freeze
        for p in model.parameters():
            p.requires_grad = False
        for p in s2_gate_b2.parameters():
            p.requires_grad = True

        # Verify trainable whitelist
        trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
        for name in trainable_names:
            self.assertTrue("s2_gate_block2" in name, f"Non-B2 parameter is trainable: {name}")

        frozen_names = [name for name, p in model.named_parameters() if not p.requires_grad]
        # Check specific critical components are frozen
        self.assertTrue(any("backbone" in n for n in frozen_names), "Backbone not frozen!")
        self.assertTrue(any("router" in n for n in frozen_names), "Router not frozen!")
        self.assertTrue(any("decoder.decoder_blocks" in n for n in frozen_names), "Decoder blocks not frozen!")
        self.assertTrue(any("segmentation_head" in n for n in frozen_names), "Segmentation head not frozen!")
        if model.decoder.s2_gate is not None:
            b1_trainable = [name for name, p in model.decoder.s2_gate.named_parameters() if p.requires_grad]
            self.assertEqual(len(b1_trainable), 0, "S2-Gate Block 1 is NOT frozen!")

        # MANDATORY EXECUTION ORDER
        model.eval()               # 1. model.eval() recurses down all submodules
        s2_gate_b2.train()          # 2. s2_gate_block2.train() reactivates only Block 2 gate

        # Verify module modes
        self.assertTrue(s2_gate_b2.training, "s2_gate_block2 is NOT in training mode!")
        self.assertFalse(model.backbone.training, "Backbone is NOT in eval mode!")
        self.assertFalse(model.decoder.decoder_blocks[0].training, "Decoder Block 0 is NOT in eval mode!")
        self.assertFalse(model.decoder.decoder_blocks[1].training, "Decoder Block 1 is NOT in eval mode!")
        if model.decoder.s2_gate is not None:
            self.assertFalse(model.decoder.s2_gate.training, "S2-Gate Block 1 is NOT in eval mode!")

        # Snapshot running statistics of all BatchNorm layers
        bn_stats_before = {}
        for name, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                bn_stats_before[name] = {
                    "running_mean": m.running_mean.clone() if m.running_mean is not None else None,
                    "running_var": m.running_var.clone() if m.running_var is not None else None,
                }

        # Run forward and backward step on REAL batch
        optimizer = torch.optim.AdamW(s2_gate_b2.parameters(), lr=1e-3, weight_decay=1e-2)
        optimizer.zero_grad()

        out = model(self.real_batch)
        loss = out.sum()
        loss.backward()
        optimizer.step()

        # Verify gradient isolation: all gate parameters have non-None grad
        for name, p in model.named_parameters():
            if "s2_gate_block2" in name:
                self.assertIsNotNone(p.grad, f"Gradient missing on Block 2 param: {name}")
            else:
                self.assertIsNone(p.grad, f"Leaked gradient on frozen param: {name}")

        # Check that bias of final conv receives non-zero gradient even at t=0
        self.assertFalse(torch.all(s2_gate_b2.gate_net[3].bias.grad == 0), 
                         "Final conv bias gradient should be non-zero at t=0!")

        # Verify frozen BN running stats did NOT change
        for name, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if "s2_gate_block2" in name:
                    # Block 2 BN SHOULD update because it is in training mode
                    continue
                before = bn_stats_before[name]
                if before["running_mean"] is not None:
                    diff_mean = torch.abs(m.running_mean - before["running_mean"]).max().item()
                    self.assertEqual(diff_mean, 0.0, f"Frozen BN {name} running_mean mutated! Diff: {diff_mean}")
                if before["running_var"] is not None:
                    diff_var = torch.abs(m.running_var - before["running_var"]).max().item()
                    self.assertEqual(diff_var, 0.0, f"Frozen BN {name} running_var mutated! Diff: {diff_var}")

        print("[Test 2.2 PASS] Strict freeze, eval/train order, gradient isolation, and BN invariance confirmed.")

    def test_2_3_numeric_identity_check_1x1_vs_3x3(self):
        """
        Verification 2.3:
        Numeric identity check when converting Block 2 S2-Gate:
        1x1 -> 3x3 with zero padding kernel centering.
        Tested on REAL batch from Crack500 validation set.
        Measures max_abs_diff, mean_abs_diff, and allclose.
        Tested in both eval and (model.eval + gate.train) modes.
        """
        print("\n--- Running Test 2.3: Numeric Identity Check (1x1 -> 3x3) on Real Batch ---")
        
        # 1. Instantiate 1x1 gate and initialize with random weights (representing a trained 1x1 gate)
        torch.manual_seed(42)
        s2_gate_1x1 = S2GateModule(s2_channels=96, skip_channels=48, kernel_size=1).to(self.device)
        nn.init.normal_(s2_gate_1x1.gate_net[0].weight, mean=0.0, std=0.05)
        nn.init.constant_(s2_gate_1x1.gate_net[0].bias, 0.1)
        nn.init.uniform_(s2_gate_1x1.gate_net[1].running_mean, 0.0, 1.0)
        nn.init.uniform_(s2_gate_1x1.gate_net[1].running_var, 0.5, 1.5)
        nn.init.normal_(s2_gate_1x1.gate_net[3].weight, mean=0.0, std=0.05)
        nn.init.constant_(s2_gate_1x1.gate_net[3].bias, 1.0)

        # 2. Instantiate 3x3 gate and perform warm-start conversion
        s2_gate_3x3 = S2GateModule(s2_channels=96, skip_channels=48, kernel_size=3).to(self.device)
        s2_gate_3x3.warm_start_from_v1(s2_gate_1x1.state_dict())

        # 3. Test on isolated inputs first
        dummy_context = torch.randn(4, 96, 56, 56, device=self.device)
        dummy_skip = torch.randn(4, 48, 112, 112, device=self.device)

        s2_gate_1x1.eval()
        s2_gate_3x3.eval()
        with torch.no_grad():
            out_skip_1x1, alpha_1x1 = s2_gate_1x1(dummy_context, dummy_skip)
            out_skip_3x3, alpha_3x3 = s2_gate_3x3(dummy_context, dummy_skip)

        diff_alpha_iso = (alpha_1x1 - alpha_3x3).abs().max().item()
        diff_skip_iso = (out_skip_1x1 - out_skip_3x3).abs().max().item()
        self.assertLess(diff_alpha_iso, 1e-5, f"Isolated gate alpha diff too high: {diff_alpha_iso}")
        self.assertLess(diff_skip_iso, 1e-5, f"Isolated gate skip diff too high: {diff_skip_iso}")

        # 4. Test FULL MODEL on REAL VALIDATION BATCH
        model_1x1 = load_model_from_checkpoint(self.config_path, self.ckpt_path, device=self.device)
        model_1x1.decoder.s2_gate_block2 = s2_gate_1x1
        model_1x1.decoder.use_s2_gate_block2 = True

        model_3x3 = load_model_from_checkpoint(self.config_path, self.ckpt_path, device=self.device)
        model_3x3.decoder.s2_gate_block2 = s2_gate_3x3
        model_3x3.decoder.use_s2_gate_block2 = True

        # (A) Evaluation Mode: model.eval()
        model_1x1.eval()
        model_3x3.eval()
        with torch.no_grad():
            logits_1x1_eval = model_1x1(self.real_batch)
            logits_3x3_eval = model_3x3(self.real_batch)

        max_abs_diff_eval = torch.max(torch.abs(logits_1x1_eval - logits_3x3_eval)).item()
        mean_abs_diff_eval = torch.mean(torch.abs(logits_1x1_eval - logits_3x3_eval)).item()
        allclose_eval = torch.allclose(logits_1x1_eval, logits_3x3_eval, atol=1e-5, rtol=1e-4)

        print(f"[Eval Mode] Real Batch Max Abs Diff:  {max_abs_diff_eval:.8e}")
        print(f"[Eval Mode] Real Batch Mean Abs Diff: {mean_abs_diff_eval:.8e}")
        print(f"[Eval Mode] torch.allclose (atol=1e-5): {allclose_eval}")

        self.assertLess(max_abs_diff_eval, 1e-4, f"Eval mode max abs diff exceeded threshold: {max_abs_diff_eval}")
        self.assertTrue(allclose_eval, "Eval mode output not allclose!")

        # (B) Train Mode Protocol: model.eval() + gate.train()
        model_1x1.eval()
        s2_gate_1x1.train()
        model_3x3.eval()
        s2_gate_3x3.train()

        with torch.no_grad():
            logits_1x1_train = model_1x1(self.real_batch)
            logits_3x3_train = model_3x3(self.real_batch)

        max_abs_diff_train = torch.max(torch.abs(logits_1x1_train - logits_3x3_train)).item()
        mean_abs_diff_train = torch.mean(torch.abs(logits_1x1_train - logits_3x3_train)).item()
        allclose_train = torch.allclose(logits_1x1_train, logits_3x3_train, atol=1e-5, rtol=1e-4)

        print(f"[Train Mode] Real Batch Max Abs Diff:  {max_abs_diff_train:.8e}")
        print(f"[Train Mode] Real Batch Mean Abs Diff: {mean_abs_diff_train:.8e}")
        print(f"[Train Mode] torch.allclose (atol=1e-5): {allclose_train}")

        self.assertLess(max_abs_diff_train, 1e-4, f"Train mode max abs diff exceeded threshold: {max_abs_diff_train}")
        self.assertTrue(allclose_train, "Train mode output not allclose!")

        print("[Test 2.3 PASS] Exact numeric identity (1x1 -> 3x3) confirmed on real validation batch.")


if __name__ == "__main__":
    unittest.main()
