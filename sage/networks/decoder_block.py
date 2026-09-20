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
from typing import List, Optional, Tuple

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


class DecoderBlock(nn.Module):
    """
    Single UNet Decoder Block with 2x upsampling, skip concatenation, and double conv3x3.

    Args:
        in_channels (int): Input channel dimension from deeper layer.
        skip_channels (int): Channel dimension of the matching encoder skip connection.
        out_channels (int): Output channel dimension.
        use_dwsc (bool): If True, uses Depthwise Separable Conv (default: False).
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_dwsc: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.use_dwsc = use_dwsc

        # 2x Transposed Convolution for spatial upsampling
        self.upsample = nn.ConvTranspose2d(
            in_channels,
            in_channels,
            kernel_size=2,
            stride=2,
        )

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

        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


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
    """

    def __init__(
        self,
        encoder_channels: List[int] = [48, 96, 192, 384],
        num_classes: int = 1,
        use_dwsc: bool = False,
    ):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.num_classes = num_classes
        self.use_dwsc = use_dwsc

        # Reversed channels: [384, 192, 96, 48]
        reversed_channels = list(reversed(encoder_channels))
        self.decoder_blocks = nn.ModuleList()

        # Build 3 progressive upsampling blocks:
        # Block 0: 384 -> 192 (with skip 192)
        # Block 1: 192 -> 96  (with skip 96)
        # Block 2: 96  -> 48  (with skip 48)
        for i in range(len(reversed_channels) - 1):
            in_ch = reversed_channels[i]
            skip_ch = reversed_channels[i + 1]
            out_ch = reversed_channels[i + 1]

            block = DecoderBlock(
                in_channels=in_ch,
                skip_channels=skip_ch,
                out_channels=out_ch,
                use_dwsc=use_dwsc,
            )
            self.decoder_blocks.append(block)

        # Final Segmentation Head:
        # Input: (B, 48, 112, 112) -> 48 -> 24 -> num_classes
        final_channels = reversed_channels[-1]  # 48
        head_mid_channels = max(final_channels // 2, 16)  # 24

        self.segmentation_head = nn.Sequential(
            make_conv3x3(final_channels, head_mid_channels, use_dwsc=use_dwsc),
            nn.BatchNorm2d(head_mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_mid_channels, num_classes, kernel_size=1),
        )

    def forward(
        self,
        bottleneck: torch.Tensor,
        skips: List[torch.Tensor],
        target_size: Optional[Tuple[int, int]] = (448, 448),
    ) -> torch.Tensor:
        """
        Forward pass through UNet Decoder.

        Args:
            bottleneck (torch.Tensor): Feature map of shape (B, 384, 14, 14).
            skips (List[torch.Tensor]): List of 3 skip tensors in shallow-to-deep order:
                - skips[0]: Stage 0 (B, 48, 112, 112)
                - skips[1]: Stage 1 (B, 96, 56, 56)
                - skips[2]: Stage 2 (B, 192, 28, 28)
            target_size (Tuple[int, int], optional): Final image resolution (default: (448, 448)).

        Returns:
            torch.Tensor: Segmentation logits of shape (B, num_classes, 448, 448).
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
