import sys
sys.path.insert(0, '.')
import torch
from sage.networks import create_b1_unet
from scripts.train_crack import get_optimizer_groups

print("Initializing B1 model (pretrained=False)...")
model = create_b1_unet(pretrained=False)
x = torch.randn(2, 3, 448, 448)
print("Running forward pass with shape:", x.shape)
y = model(x)
print("Output shape:", y.shape)
assert y.shape == (2, 1, 448, 448), f"Bad shape: {y.shape}"

print("Running backward pass...")
loss = y.sum()
loss.backward()
print("Backward pass: SUCCESS!")

print("Checking optimizer groups...")
groups = get_optimizer_groups(model, lr_backbone=1e-5, lr_decoder=1e-4, weight_decay=0.05)
bb_params = [g for g in groups if g['name'] == 'backbone']
dec_params = [g for g in groups if g['name'] == 'decoder']
print(f"Total parameter tensors: {len(groups)}")
print(f"Backbone parameter tensors: {len(bb_params)}, LR: {bb_params[0]['lr']}, WD: {bb_params[0]['weight_decay']}")
print(f"Decoder parameter tensors: {len(dec_params)}, LR: {dec_params[0]['lr']}, WD: {dec_params[0]['weight_decay']}")
print("ALL PRE-FLIGHT CHECKS PASSED!")
