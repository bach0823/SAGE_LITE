"""
sage/networks/strip_pooling.py

Oriented Strip Pooling module with per-pixel learnable gating.
Designed for Phase 6 topology ablation to capture anisotropic crack context
without introducing orthogonal false bridges.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class OrientedStripPooling(nn.Module):
    """
    Oriented Strip Pooling module.
    Instead of fixed addition y = y_h + y_w (which risks horizontal false bridging),
    uses a learned per-pixel gate g(x, y) in [0, 1] to select the directional mixing weight.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv_h = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.conv_v = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.gate = nn.Conv2d(channels, 1, kernel_size=1)  # Predicts g in [0, 1] per pixel
        self.fusion = nn.Conv2d(channels, channels, kernel_size=1)
        self._last_gate: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        y_h = F.adaptive_avg_pool2d(x, (1, W)).expand(-1, -1, H, -1)  # Horizontal strip
        y_v = F.adaptive_avg_pool2d(x, (H, 1)).expand(-1, -1, -1, W)  # Vertical strip

        g = torch.sigmoid(self.gate(x))  # Shape: (B, 1, H, W)
        if self.training:
            self._last_gate = g.detach()

        y_mix = g * self.conv_h(y_h) + (1.0 - g) * self.conv_v(y_v)
        return x + self.fusion(y_mix)

    def get_gate_stats(self) -> Tuple[Optional[float], Optional[float]]:
        """Returns (gate_mean, gate_std) from the most recent forward pass."""
        if self._last_gate is not None:
            return float(self._last_gate.mean().item()), float(self._last_gate.std().item())
        return None, None
