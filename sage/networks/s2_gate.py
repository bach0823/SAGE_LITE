"""
sage/networks/s2_gate.py

S2-Gate Module (Phase 6 — U1-S2G):
Uses high-level context features from S2 (28x28, 192 channels) after Decoder Block 0
to dynamically construct a spatial gate alpha in (0, 1) that modulates the Stage 1
skip connection (56x56, 96 channels) prior to concatenation into Decoder Block 1.

Identity initialization at t=0:
- Final 1x1 conv layer weights are strictly initialized to 0.0.
- Final 1x1 conv layer bias is initialized to +5.0 (sigmoid(5.0) ~ 0.9933).
- Ensures bitwise near-identity pass-through at epoch 0 (skip_gated ~ skip_s1).
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class S2GateModule(nn.Module):
    """
    S2-Gate: Uses features from S2 (28x28) to create a spatial gate
    modulating skip S1 (56x56) before entering Decoder Block 1.

    Identity initialization: at t=0, gate_alpha ~ 1.0 -> skip_gated = skip_S1.
    """
    def __init__(
        self,
        s2_channels: int = 192,
        skip_channels: int = 96,
        mid_channels: int = 32,
        kernel_size: int = 1,
    ):
        super().__init__()
        self.s2_channels = s2_channels
        self.skip_channels = skip_channels
        self.mid_channels = mid_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        # Gate MLP: compress s2_up (192) + skip_s1 (96) = 288 channels -> mid (32) -> 1 spatial gate
        self.gate_net = nn.Sequential(
            nn.Conv2d(s2_channels + skip_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=True),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True),
        )
        self._init_identity()

    def _init_identity(self):
        """
        Initializes final projection so gate_alpha ~ 1.0 at t=0 (identity).
        Zero-init final conv weight, set bias = +5.0 (sigmoid(5.0) ~ 0.9933).
        """
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, 5.0)

    def warm_start_from_v1(self, v1_state_dict: dict):
        """
        Warm-start weights from trained U1-S2G v1 (1x1 conv):
        - For gate_net.0.weight:
          If v1 is 1x1 [32, 288, 1, 1] and self is 3x3 [32, 288, 3, 3],
          place 1x1 kernel at the center (1, 1) and zero-initialize spatial neighbors.
        - For gate_net.0.bias: directly copy.
        - For gate_net.1 (BatchNorm): copy weight, bias, running_mean, running_var.
        - For gate_net.3 (Conv1x1 32->1): copy weight, bias.
        """
        with torch.no_grad():
            # 1. First conv layer
            w1 = v1_state_dict["gate_net.0.weight"]  # [32, 288, 1, 1]
            b1 = v1_state_dict["gate_net.0.bias"]    # [32]
            if self.kernel_size == 3 and w1.shape[2:] == (1, 1):
                self.gate_net[0].weight.zero_()
                self.gate_net[0].weight[:, :, 1:2, 1:2] = w1
            else:
                self.gate_net[0].weight.copy_(w1)
            self.gate_net[0].bias.copy_(b1)

            # 2. BatchNorm layer
            for k in ["weight", "bias", "running_mean", "running_var"]:
                key = f"gate_net.1.{k}"
                if key in v1_state_dict:
                    getattr(self.gate_net[1], k).copy_(v1_state_dict[key])

            # 3. Final projection layer
            self.gate_net[3].weight.copy_(v1_state_dict["gate_net.3.weight"])
            self.gate_net[3].bias.copy_(v1_state_dict["gate_net.3.bias"])

    def forward(self, s2_feat: torch.Tensor, skip_s1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s2_feat (torch.Tensor): [B, 192, 28, 28] - S2 features from Decoder Block 0 output
            skip_s1 (torch.Tensor): [B, 96, 56, 56] - Stage 1 skip connection
        Returns:
            skip_gated (torch.Tensor): [B, 96, 56, 56]
            gate_alpha (torch.Tensor): [B, 1, 56, 56] in (0, 1)
        """
        # Upsample S2 context to 56x56 matching skip S1
        s2_up = F.interpolate(
            s2_feat,
            size=skip_s1.shape[2:],
            mode="bilinear",
            align_corners=False,
        )  # [B, 192, 56, 56]

        gate_input = torch.cat([s2_up, skip_s1], dim=1)  # [B, 288, 56, 56]
        gate_logit = self.gate_net(gate_input)           # [B, 1, 56, 56]
        gate_alpha = torch.sigmoid(gate_logit)           # [B, 1, 56, 56]

        skip_gated = gate_alpha * skip_s1                # broadcast over channel dim
        return skip_gated, gate_alpha
