"""
sage/networks/dc_init_stem.py

DC-Init Stem Architecture (Phase 6 — U0-C3):
Replaces the standard Conv2d(3, 48, kernel_size=4, stride=4) patchify stem with:
  - DC path: AvgPool2d(4, 4) -> Conv2d(3, 48, kernel_size=1, bias=True)
             Initialized to spatial sum of pretrained kernel (16 * mean) to match DC convolution.
  - AC residual: Conv2d(3, 48, kernel_size=4, stride=4, bias=False)
                 Initialized to zero so AC output = 0 at t=0.
  - LayerNorm2d: LayerNorm2d(48, eps=1e-6) initialized from pretrained stem norm.

Output shape: [B, 48, 112, 112], fully identical in dimensionality to the original stem.
"""

import torch
import torch.nn as nn
from timm.layers.norm import LayerNorm2d


class DCInitStem(nn.Module):
    """
    DC-Init Stem module for ConvNeXt-V2-Femto.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 48):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=4)
        self.dc_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.ac_delta = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=4, bias=False)
        self.norm = LayerNorm2d(out_channels, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dc = self.dc_proj(self.avg_pool(x))
        ac = self.ac_delta(x)
        return self.norm(dc + ac)


def init_dc_stem_from_pretrained(dc_stem: DCInitStem, pretrained_stem: nn.Sequential):
    """
    Initializes DCInitStem from pretrained ConvNeXt stem:
      - dc_proj weights: spatial sum (16 * mean) of pretrained 4x4 kernel
      - dc_proj bias: pretrained bias
      - ac_delta weights: strictly zero
      - norm weight and bias: copied from pretrained LayerNorm2d
    """
    pretrained_conv = pretrained_stem[0]
    pretrained_norm = pretrained_stem[1]

    with torch.no_grad():
        W_full = pretrained_conv.weight.data  # [48, 3, 4, 4]
        b_full = pretrained_conv.bias.data    # [48]

        # Spatial sum over 4x4 positions matches Conv2d with constant DC kernel on AvgPool output
        W_dc = W_full.sum(dim=(2, 3), keepdim=True)  # [48, 3, 1, 1]

        dc_stem.dc_proj.weight.copy_(W_dc)
        dc_stem.dc_proj.bias.copy_(b_full)

        # AC residual begins at zero
        nn.init.zeros_(dc_stem.ac_delta.weight)

        # Copy LayerNorm parameters
        dc_stem.norm.weight.copy_(pretrained_norm.weight)
        dc_stem.norm.bias.copy_(pretrained_norm.bias)
