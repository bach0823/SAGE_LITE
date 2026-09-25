import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath("."))
from sage_lite.sage.networks.b2_unet import create_b2_unet
from sage_lite.sage.utils.model_utils import P3_EXPECTED_KEYS, load_locked_base_into_p3
from sage_lite.sage.components.p3_refinement import ASDWRefinement, GenericRefinement

print("=" * 80)
print("AUDIT: GAMMA IMPLEMENTATION")
print("=" * 80)

asdw = ASDWRefinement(48)
print(f"asdw.gamma: {asdw.gamma}")
print(f"asdw.gamma shape: {asdw.gamma.shape}, numel: {asdw.gamma.numel()}, is_leaf: {asdw.gamma.is_leaf}, val: {asdw.gamma.item()}")

gen = GenericRefinement(48)
print(f"gen.gamma: {gen.gamma}")
print(f"gen.gamma shape: {gen.gamma.shape}, numel: {gen.gamma.numel()}, is_leaf: {gen.gamma.is_leaf}, val: {gen.gamma.item()}")

print("\n" + "=" * 80)
print("AUDIT: PARAMETER COUNTS (S0, S1, TOTAL)")
print("=" * 80)

m_b = create_b2_unet(pretrained=False, num_transformer_layers=4, p3_mode="B")
s0_b_p3 = m_b.backbone.convnext.stages[0].p3_refinement
s1_b_p3 = m_b.backbone.convnext.stages[1].p3_refinement
s0_b_params = sum(p.numel() for p in s0_b_p3.parameters())
s1_b_params = sum(p.numel() for p in s1_b_p3.parameters())
total_b_params = s0_b_params + s1_b_params
print(f"Run B (GenericRefinement):")
print(f"  Stage 0 (C=48): {s0_b_params:,} parameters (Formula 3*48^2 + 27*48 + 1 = {3*48**2 + 27*48 + 1:,})")
print(f"  Stage 1 (C=96): {s1_b_params:,} parameters (Formula 3*96^2 + 27*96 + 1 = {3*96**2 + 27*96 + 1:,})")
print(f"  Total (S0+S1):  {total_b_params:,} parameters (Target = 38,450)")

m_c = create_b2_unet(pretrained=False, num_transformer_layers=4, p3_mode="C")
s0_c_p3 = m_c.backbone.convnext.stages[0].p3_refinement
s1_c_p3 = m_c.backbone.convnext.stages[1].p3_refinement
s0_c_params = sum(p.numel() for p in s0_c_p3.parameters())
s1_c_params = sum(p.numel() for p in s1_c_p3.parameters())
total_c_params = s0_c_params + s1_c_params
print(f"\nRun C (ASDWRefinement):")
print(f"  Stage 0 (C=48): {s0_c_params:,} parameters (Formula 3*48^2 + 23*48 + 1 = {3*48**2 + 23*48 + 1:,})")
print(f"  Stage 1 (C=96): {s1_c_params:,} parameters (Formula 3*96^2 + 23*96 + 1 = {3*96**2 + 23*96 + 1:,})")
print(f"  Total (S0+S1):  {total_c_params:,} parameters (Target = 37,874)")
print(f"Delta (Run B - Run C): {total_b_params - total_c_params} parameters (Target = 576)")

print("\n" + "=" * 80)
print("AUDIT: RUN B STATE_DICT & MISSING KEYS")
print("=" * 80)

base = create_b2_unet(pretrained=False, num_transformer_layers=4, p3_mode=None)
base_sd = base.state_dict()
incomp_b = m_b.load_state_dict(base_sd, strict=False)

missing_b = sorted(incomp_b.missing_keys)
expected_b = sorted(list(P3_EXPECTED_KEYS["B"]))
newly_added_b = sorted(list(set(m_b.state_dict().keys()) - set(base_sd.keys())))

print(f"Count of missing_keys: {len(missing_b)}")
print("exact sorted missing_keys:")
for k in missing_b:
    print(f"  '{k}'")

print(f"\nCount of P3_EXPECTED_KEYS['B']: {len(expected_b)}")
print("exact sorted P3_EXPECTED_KEYS['B']:")
for k in expected_b:
    print(f"  '{k}'")

print(f"\nCount of newly-added state_dict keys: {len(newly_added_b)}")
print("exact sorted newly-added state_dict keys:")
for k in newly_added_b:
    print(f"  '{k}'")

assert missing_b == expected_b == newly_added_b, "MISMATCH DETECTED!"
print("\n>>> AUDIT PASSED: missing_keys == P3_EXPECTED_KEYS['B'] == newly_added_keys (exactly 11 keys) <<<")
