"""
UNet Decoder Blocks and Decoder Architecture for SAGE-Lite (SK3)

Scope: SK3 (Milestone 1)
- Standard 3x3 convolutions with batch normalization and ReLU.
- Optional Depthwise Separable Convolution (`use_dwsc=False` default, togglable for smoke tests).
- Progressive upsampling:
    Stage 1: in=384, skip=192 (Stage 2) -> (B, 192, 28, 28)
    Stage 2: in=192, skip=96  (Stage 1) -> (B, 96, 56, 56)
    Stage 3: in=96,  skip=48  (Stage 0) -> (B, 48, 112, 112)
- Segmentation Head: 48 -> 24 -> num_classes (default 1) + 4x Bilinear Upsample -> (B, 1, 448, 448).
- No Router, No SAGE, No SA-Hub, No Injection.

Author: Special Subject AI Team
Date: September 2026
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def make_conv3x3(in_ch: int, out_ch: int, use_dwsc: bool = False) -> nn.Module:
    """
    Constructs a 3x3 convolution layer.
    Default: Standard 3x3 Conv2d (bias=True, 100% identical to original SAGE DecoderBlock).
    Optional: Depthwise Separable Convolution (DWSC) for smoke tests / extreme compression.
    """
    if use_dwsc:
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True),
        )
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)


from .cgsr import ContextGuidedStage1SkipRefinement
from .point_rend import (
    PointRendHead,
    sample_training_points,
    subdivide_and_refine,
)


class DecoderBlock(nn.Module):
    """
    Single UNet Decoder Block with 2x upsampling, skip concatenation, and double conv3x3.

    Args:
        in_channels (int): Input channel dimension from deeper layer.
        skip_channels (int): Channel dimension of the matching encoder skip connection.
        out_channels (int): Output channel dimension.
        use_dwsc (bool): If True, uses Depthwise Separable Conv (default: False).
        use_cgsr (bool): If True, applies Context-Guided Stage-1 Skip Refinement (default: False).
        cgsr_init_bias (float): Initial bias for CGSR gate (default: 3.0 => G ~ 0.9526).
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_dwsc: bool = False,
        use_cgsr: bool = False,
        cgsr_init_bias: float = 3.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.use_dwsc = use_dwsc
        self.use_cgsr = use_cgsr

        # 2x Transposed Convolution for spatial upsampling
        self.upsample = nn.ConvTranspose2d(
            in_channels,
            in_channels,
            kernel_size=2,
            stride=2,
        )

        if use_cgsr:
            self.cgsr = ContextGuidedStage1SkipRefinement(
                in_channels=in_channels,
                skip_channels=skip_channels,
                init_bias=cgsr_init_bias,
            )
        else:
            self.cgsr = None

        combined_channels = in_channels + skip_channels

        self.conv1 = nn.Sequential(
            make_conv3x3(combined_channels, out_channels, use_dwsc=use_dwsc),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            make_conv3x3(out_channels, out_channels, use_dwsc=use_dwsc),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder block.

        Args:
            x (torch.Tensor): Deeper feature map (B, in_channels, H, W).
            skip (torch.Tensor): Skip connection from encoder (B, skip_channels, 2H, 2W).

        Returns:
            torch.Tensor: Upsampled and fused feature map (B, out_channels, 2H, 2W).
        """
        x = self.upsample(x)

        # Align spatial resolution in case of rounding differences
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

        if self.cgsr is not None:
            skip, _ = self.cgsr(x, skip)

        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ProgressiveLearnedUpsamplingHead(nn.Module):
    """
    Progressive Learned Upsampling Head (PLU-Head) for Phase 6-A.2 Representation Probe.

    Replaces the non-parametric 4x bilinear interpolation with two progressive
    transposed convolution stages:
      48x112^2 -> Conv3x3 (48->24) -> BN/ReLU -> ConvTranspose2d (24->16, k=4, s=2, p=1, bias=False)
      -> BN/ReLU -> ConvTranspose2d (16->num_classes, k=4, s=2, p=1, bias=True) -> 448^2.

    Cấu hình kernel=4, stride=2 có overlap đều theo không gian và được sử dụng để giảm nguy cơ
    checkerboard artifact; không coi việc loại bỏ artifact là một giả định đã được chứng minh.
    """

    def __init__(
        self,
        in_channels: int = 48,
        mid_channels: int = 24,
        up_channels: int = 16,
        num_classes: int = 1,
        use_dwsc: bool = False,
    ):
        super().__init__()
        self.conv112 = make_conv3x3(in_channels, mid_channels, use_dwsc=use_dwsc)
        self.norm112 = nn.BatchNorm2d(mid_channels)
        self.act112 = nn.ReLU(inplace=True)

        self.up224 = nn.ConvTranspose2d(
            mid_channels,
            up_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.norm224 = nn.BatchNorm2d(up_channels)
        self.act224 = nn.ReLU(inplace=True)

        self.up448 = nn.ConvTranspose2d(
            up_channels,
            num_classes,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=True,
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.up224.weight, mode="fan_out", nonlinearity="relu")
        nn.init.ones_(self.norm224.weight)
        nn.init.zeros_(self.norm224.bias)
        nn.init.kaiming_normal_(self.up448.weight, mode="fan_out", nonlinearity="linear")
        nn.init.zeros_(self.up448.bias)

    def forward(
        self,
        x: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = (448, 448),
    ) -> torch.Tensor:
        x = self.act112(self.norm112(self.conv112(x)))
        x = self.act224(self.norm224(self.up224(x)))
        logits = self.up448(x)

        if target_size is not None and logits.shape[2:] != target_size:
            logits = F.interpolate(
                logits,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        return logits


class UNetDecoder(nn.Module):
    """
    Full UNet Decoder for SAGE-Lite.

    Takes:
      - bottleneck: (B, 384, 14, 14) from ConvNeXtV2-ViT bottleneck
      - skips: [
            Stage 0: (B, 48, 112, 112),
            Stage 1: (B, 96, 56, 56),
            Stage 2: (B, 192, 28, 28)
        ]
    Produces:
      - logits: (B, num_classes, 448, 448) (default num_classes=1)

    Args:
        encoder_channels (List[int]): Channels of the 4 encoder stages (default: [48, 96, 192, 384]).
        num_classes (int): Number of segmentation classes (default: 1 for crack).
        use_dwsc (bool): Whether to use Depthwise Separable Convolutions (default: False).
        use_plu_head (bool): Whether to use Progressive Learned Upsampling Head (default: False).
        use_cgsr (bool): Whether to use Context-Guided Stage-1 Skip Refinement (default: False).
        cgsr_init_bias (float): Initial bias for CGSR gate (default: 3.0).
    """

    def __init__(
        self,
        encoder_channels: List[int] = [48, 96, 192, 384],
        num_classes: int = 1,
        use_dwsc: bool = False,
        use_plu_head: bool = False,
        use_cgsr: bool = False,
        cgsr_init_bias: float = 3.0,
        use_point_rend: bool = False,
        point_rend_mid_channels: int = 128,
        point_rend_train_points: int = 2048,
        point_rend_subdivision_points: int = 8192,
        use_oriented_strip_pooling: bool = False,
        use_tangent_head: bool = False,
    ):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.num_classes = num_classes
        self.use_dwsc = use_dwsc
        self.use_plu_head = use_plu_head
        self.use_cgsr = use_cgsr
        self.cgsr_init_bias = cgsr_init_bias
        self.use_point_rend = use_point_rend
        self.point_rend_mid_channels = point_rend_mid_channels
        self.point_rend_train_points = point_rend_train_points
        self.point_rend_subdivision_points = point_rend_subdivision_points
        self.use_oriented_strip_pooling = use_oriented_strip_pooling
        self.use_tangent_head = use_tangent_head

        # Reversed channels: [384, 192, 96, 48]
        reversed_channels = list(reversed(encoder_channels))
        self.decoder_blocks = nn.ModuleList()

        # Build 3 progressive upsampling blocks:
        # Block 0: 384 -> 192 (with skip 192, Stage 2)
        # Block 1: 192 -> 96  (with skip 96,  Stage 1) -> CGSR intervention site
        # Block 2: 96  -> 48  (with skip 48,  Stage 0)
        for i in range(len(reversed_channels) - 1):
            in_ch = reversed_channels[i]
            skip_ch = reversed_channels[i + 1]
            out_ch = reversed_channels[i + 1]

            # CGSR is strictly applied only to Block 1 (56x56 resolution, Stage-1 skip)
            block_use_cgsr = use_cgsr and (i == 1)

            block = DecoderBlock(
                in_channels=in_ch,
                skip_channels=skip_ch,
                out_channels=out_ch,
                use_dwsc=use_dwsc,
                use_cgsr=block_use_cgsr,
                cgsr_init_bias=cgsr_init_bias,
            )
            self.decoder_blocks.append(block)

        # Final Segmentation Head:
        # Input: (B, 48, 112, 112) -> 48 -> 24 -> num_classes
        final_channels = reversed_channels[-1]  # 48
        head_mid_channels = max(final_channels // 2, 16)  # 24

        if self.use_plu_head:
            self.segmentation_head = ProgressiveLearnedUpsamplingHead(
                in_channels=final_channels,
                mid_channels=head_mid_channels,
                up_channels=16,
                num_classes=num_classes,
                use_dwsc=use_dwsc,
            )
        else:
            self.segmentation_head = nn.Sequential(
                make_conv3x3(final_channels, head_mid_channels, use_dwsc=use_dwsc),
                nn.BatchNorm2d(head_mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_mid_channels, num_classes, kernel_size=1),
            )

        if self.use_point_rend:
            self.point_rend_head = PointRendHead(
                in_channels=final_channels,
                num_classes=num_classes,
                mid_channels=point_rend_mid_channels,
            )
        else:
            self.point_rend_head = None

        # Experiment A: Oriented Strip Pooling at Block 1 (56x56 resolution, Stage 1 skip, 96 channels)
        if self.use_oriented_strip_pooling:
            from .strip_pooling import OrientedStripPooling
            self.strip_pool_56 = OrientedStripPooling(channels=reversed_channels[2])
        else:
            self.strip_pool_56 = None

        # Experiment B: Tangent Field Auxiliary Head at Final 112x112 features (48 channels)
        if self.use_tangent_head:
            self.tangent_head = nn.Sequential(
                make_conv3x3(final_channels, head_mid_channels, use_dwsc=use_dwsc),
                nn.BatchNorm2d(head_mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_mid_channels, 2, kernel_size=1),
            )
        else:
            self.tangent_head = None

    def forward(
        self,
        bottleneck: torch.Tensor,
        skips: List[torch.Tensor],
        target_size: Optional[Tuple[int, int]] = (448, 448),
        return_point_rend_dict: bool = False,
        return_tangent: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through UNet Decoder.

        Args:
            bottleneck (torch.Tensor): Feature map of shape (B, 384, 14, 14).
            skips (List[torch.Tensor]): List of 3 skip tensors in shallow-to-deep order:
                - skips[0]: Stage 0 (B, 48, 112, 112)
                - skips[1]: Stage 1 (B, 96, 56, 56)
                - skips[2]: Stage 2 (B, 192, 28, 28)
            target_size (Tuple[int, int], optional): Final image resolution (default: (448, 448)).
            return_point_rend_dict (bool): If True, returns dict with point training tensors.
            return_tangent (bool): If True, returns dict with predicted tangent field.

        Returns:
            torch.Tensor or Dict: Refined logits or training point / tangent dictionary.
        """
        # Reverse skips to deep-to-shallow order: [Stage 2 (192), Stage 1 (96), Stage 0 (48)]
        reversed_skips = list(reversed(skips))
        assert len(reversed_skips) == len(self.decoder_blocks), (
            f"Expected {len(self.decoder_blocks)} skip connections, got {len(reversed_skips)}"
        )

        x_dec = bottleneck

        for i, block in enumerate(self.decoder_blocks):
            skip = reversed_skips[i]
            x_dec = block(x_dec, skip)
            # Experiment A: Oriented Strip Pooling at Block 1 (56x56 resolution)
            if i == 1 and self.strip_pool_56 is not None:
                x_dec = self.strip_pool_56(x_dec)

        # Experiment B: Tangent Field prediction from final decoder features (112x112)
        pred_tangent = None
        if self.use_tangent_head and self.tangent_head is not None:
            raw_tan = self.tangent_head(x_dec)
            if target_size is not None and raw_tan.shape[2:] != target_size:
                raw_tan = F.interpolate(
                    raw_tan,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            # Normalize to unit vector in double-angle space
            pred_tangent = F.normalize(raw_tan, p=2, dim=1)

        if self.use_point_rend:
            # 1. Coarse prediction at 112x112
            coarse_logits = self.segmentation_head(x_dec)

            if self.training:
                # Training mode: sample uncertain points
                point_coords = sample_training_points(
                    coarse_logits,
                    num_points=self.point_rend_train_points,
                )
                point_logits = self.point_rend_head(x_dec, coarse_logits, point_coords)

                coarse_upsampled = F.interpolate(
                    coarse_logits,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
                if return_point_rend_dict:
                    res = {
                        "logits": coarse_upsampled,
                        "coarse_logits": coarse_logits,
                        "point_logits": point_logits,
                        "point_coords": point_coords,
                    }
                    if return_tangent and pred_tangent is not None:
                        res["pred_tangent"] = pred_tangent
                    return res
                if return_tangent and pred_tangent is not None:
                    return {"logits": coarse_upsampled, "pred_tangent": pred_tangent}
                return coarse_upsampled
            else:
                # Evaluation mode: adaptive subdivision refinement
                logits = subdivide_and_refine(
                    fine_features=x_dec,
                    coarse_logits=coarse_logits,
                    point_head=self.point_rend_head,
                    target_size=target_size,
                    num_subdivision_points=self.point_rend_subdivision_points,
                )
                if return_tangent and pred_tangent is not None:
                    return {"logits": logits, "pred_tangent": pred_tangent}
                return logits

        if self.use_plu_head:
            logits = self.segmentation_head(x_dec, target_size=target_size)
        else:
            # Apply segmentation head: (B, 48, 112, 112) -> (B, num_classes, 112, 112)
            logits = self.segmentation_head(x_dec)

            # Bilinear 4x upsampling to match input resolution (448, 448)
            if target_size is not None and logits.shape[2:] != target_size:
                logits = F.interpolate(
                    logits,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )

        if return_tangent and pred_tangent is not None:
            return {"logits": logits, "pred_tangent": pred_tangent}
        return logits


if __name__ == "__main__":
    print("=" * 65)
    print("=== SK3 Self-Test: UNet Decoder (SAGE-Lite) ===")
    print("=" * 65)

    batch_size = 2
    num_classes = 1

    # 1. Synthesize inputs matching contract from SK2 Backbone
    bottleneck = torch.randn(batch_size, 384, 14, 14)
    skips = [
        torch.randn(batch_size, 48, 112, 112),  # Stage 0
        torch.randn(batch_size, 96, 56, 56),    # Stage 1
        torch.randn(batch_size, 192, 28, 28),   # Stage 2
    ]

    print("\n[Test 1] Inputs to Decoder:")
    print(f"  Bottleneck shape: {tuple(bottleneck.shape)}")
    for i, s in enumerate(skips):
        print(f"  Skip {i} shape:    {tuple(s.shape)}")

    # 2. Test Standard 3x3 Decoder
    print("\n[Test 2] Testing Standard 3x3 Decoder:")
    decoder_std = UNetDecoder(
        encoder_channels=[48, 96, 192, 384],
        num_classes=num_classes,
        use_dwsc=False,
    )
    logits_std = decoder_std(bottleneck, skips, target_size=(448, 448))
    print(f"  Output logits shape: {tuple(logits_std.shape)}")
    assert logits_std.shape == (batch_size, num_classes, 448, 448), (
        f"Shape mismatch: expected ({batch_size}, {num_classes}, 448, 448), got {logits_std.shape}"
    )
    print("  [OK] Standard 3x3 Decoder output shape verified successfully.")

    # 3. Test DWSC Decoder (Smoke-test option)
    print("\n[Test 3] Testing DWSC Decoder (use_dwsc=True):")
    decoder_dwsc = UNetDecoder(
        encoder_channels=[48, 96, 192, 384],
        num_classes=num_classes,
        use_dwsc=True,
    )
    logits_dwsc = decoder_dwsc(bottleneck, skips, target_size=(448, 448))
    print(f"  Output logits shape: {tuple(logits_dwsc.shape)}")
    assert logits_dwsc.shape == (batch_size, num_classes, 448, 448)
    print("  [OK] DWSC Decoder output shape verified successfully.")

    # 4. Test Gradient Flow (Backward Pass)
    print("\n[Test 4] Verifying Gradient Flow:")
    bottleneck.requires_grad = True
    for s in skips:
        s.requires_grad = True
    logits = decoder_std(bottleneck, skips, target_size=(448, 448))
    loss = logits.sum()
    loss.backward()
    assert bottleneck.grad is not None, "Bottleneck gradients not computed!"
    for i, s in enumerate(skips):
        assert s.grad is not None, f"Skip {i} gradients not computed!"
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Bottleneck grad shape: {tuple(bottleneck.grad.shape)}")
    print("  [OK] Gradient flow through all skip connections and bottleneck verified.")

    print("\n" + "=" * 65)
    print("[SUCCESS] All SK3 Decoder Self-Tests Passed Successfully!")
    print("=" * 65)
