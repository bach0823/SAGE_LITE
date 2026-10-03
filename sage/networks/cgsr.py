"""
Context-Guided Stage-1 Skip Refinement (CGSR) Module for SAGE-Lite Phase 6D.

Scope & Mechanism:
- Local decoder-side learned spatial/channel gate at DecoderBlock 1 (56x56 resolution, Stage-1 skip).
- Input:
    - upsampled_deep: (B, 192, 56, 56) from ConvTranspose2d of deeper decoder feature
    - skip: (B, 96, 56, 56) from ConvNeXt Stage-1 skip
- Processing:
    1. context = Conv1x1(192 -> 96)(upsampled_deep) -> (B, 96, 56, 56)
    2. gate_input = cat([skip, context], dim=1) -> (B, 192, 56, 56)
    3. gate = sigmoid(Conv1x1(192 -> 96)(gate_input)) -> (B, 96, 56, 56), 0 <= G <= 1
    4. skip_refined = gate * skip -> (B, 96, 56, 56)
- Initialization:
    - context_proj: Kaiming normal, bias = 0.0
    - gate_conv: weight = 0.0, bias = init_bias (default +3.0 => sigmoid(3.0) ~ 0.9526)
    - Ensures gate initially starts near identity (G ~ 1) matching Candidate B baseline.
- Parameter Count:
    - context_proj: 192 * 96 + 96 = 18,528 params
    - gate_conv: 192 * 96 + 96 = 18,528 params
    - Total: 37,056 params (+0.36% relative to Candidate B 10.12M).

Author: Special Subject AI Team
Date: October 2026
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextGuidedStage1SkipRefinement(nn.Module):
    """
    Context-Guided Stage-1 Skip Refinement (CGSR) for DecoderBlock 1 (56x56 resolution).
    Selectively suppresses non-discriminative skip features using deep semantic context.
    """

    def __init__(
        self,
        in_channels: int = 192,
        skip_channels: int = 96,
        init_bias: float = 3.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.init_bias = float(init_bias)

        # 1. Project deep context: (B, 192, 56, 56) -> (B, 96, 56, 56)
        self.context_proj = nn.Conv2d(
            in_channels,
            skip_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # 2. Gate projection: (B, 96 + 96 = 192, 56, 56) -> (B, 96, 56, 56)
        combined_channels = skip_channels + skip_channels
        self.gate_conv = nn.Conv2d(
            combined_channels,
            skip_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # Internal attribute to cache last computed gate for diagnostics/logging without backward hooks
        self.last_gate: Optional[torch.Tensor] = None

        self._init_weights()

    def _init_weights(self):
        """
        Near-identity initialization:
        - context_proj: Kaiming normal, zero bias
        - gate_conv: small Gaussian weights (std=1e-3) ensuring gradients flow through context_proj
          from step 0, while positive bias (+init_bias) guarantees gate starts near identity (G ~ 0.95-0.98).
        """
        nn.init.kaiming_normal_(self.context_proj.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.context_proj.bias)

        nn.init.normal_(self.gate_conv.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.gate_conv.bias, self.init_bias)

    def forward(
        self,
        upsampled_deep: torch.Tensor,
        skip: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of CGSR.

        Args:
            upsampled_deep: (B, 192, H, W) feature map from decoder upsample.
            skip: (B, 96, H, W) feature map from ConvNeXt Stage-1 skip.

        Returns:
            skip_refined: (B, 96, H, W) = gate * skip
            gate: (B, 96, H, W) in [0, 1]
        """
        # Spatial alignment guard if resolutions slightly differ
        if upsampled_deep.shape[2:] != skip.shape[2:]:
            upsampled_deep = F.interpolate(
                upsampled_deep, size=skip.shape[2:], mode="bilinear", align_corners=False
            )

        context = self.context_proj(upsampled_deep)
        gate_input = torch.cat([skip, context], dim=1)
        gate = torch.sigmoid(self.gate_conv(gate_input))
        skip_refined = gate * skip

        # Cache gate detach for diagnostics
        if not self.training:
            self.last_gate = gate.detach()

        return skip_refined, gate
