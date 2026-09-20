"""
Baseline Ladder B0: ConvNeXtV2-ViT Hybrid UNet (Pure Baseline without SAGE)

Scope: SK3 (Milestone 1)
- Combines SK2 Pure Backbone (ConvNeXtV2-Femto + 6 ViT-Tiny blocks) with SK3 UNet Decoder.
- Zero SAGE components: No Router, No SageLayer, No SA-Hub, No Load-Balancing Loss.
- Hard-locked 6 ViT blocks, CLS token dropped, pure 196 spatial tokens at bottleneck.
- Input:  (B, 3, 448, 448)
- Output: (B, 1, 448, 448) (logits)
- Serves as the Baseline B0 model for the Baseline Ladder (B0 -> B1 -> B2 -> B3).

Author: Special Subject AI Team
Date: September 2026
"""

import logging
from typing import Dict, Optional, Union

import torch
import torch.nn as nn

try:
    from .convnextv2_vit_hybrid import ConvNeXtV2ViTHybrid, create_convnextv2_vit_hybrid
    from .decoder_block import UNetDecoder
except ImportError:
    from sage.networks.convnextv2_vit_hybrid import ConvNeXtV2ViTHybrid, create_convnextv2_vit_hybrid
    from sage.networks.decoder_block import UNetDecoder


logger = logging.getLogger(__name__)


class B0ConvNeXtViTUNet(nn.Module):
    """
    Baseline B0 Model: ConvNeXtV2-ViT Hybrid UNet.

    Args:
        num_classes (int): Number of segmentation classes (default: 1 for crack).
        img_size (int): Standard input image resolution (default: 448).
        freeze_encoder (bool): Whether to freeze ConvNeXt parameters (default: False).
        freeze_transformer (bool): Whether to freeze ViT parameters (default: False).
        use_dwsc (bool): Whether to use Depthwise Separable Conv in decoder (default: False).
        pretrained (bool): Whether to load ImageNet pretrained weights (default: True).
    """

    def __init__(
        self,
        num_classes: int = 1,
        img_size: int = 448,
        freeze_encoder: bool = False,
        freeze_transformer: bool = False,
        use_dwsc: bool = False,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size

        # 1. SK2 Backbone: ConvNeXt-Femto + 6 ViT-Tiny blocks (hard-locked)
        self.backbone = ConvNeXtV2ViTHybrid(
            img_size=img_size,
            convnext_model_name="convnextv2_femto.fcmae",
            vit_model_name="vit_tiny_patch16_224",
            freeze_encoder=freeze_encoder,
            freeze_transformer=freeze_transformer,
            pretrained=pretrained,
        )

        # 2. SK3 Decoder: Standard 3x3 UNet Decoder with skip connections
        self.decoder = UNetDecoder(
            encoder_channels=self.backbone.encoder_channels,
            num_classes=num_classes,
            use_dwsc=use_dwsc,
        )

    @property
    def num_sage_experts(self) -> int:
        """
        Number of shared CNN stages (fixed at 4).
        Contract property for training script compatibility.
        """
        return self.backbone.num_sage_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        End-to-end forward pass for Baseline B0.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Segmentation logits of shape (B, num_classes, H, W).
        """
        target_size = (x.shape[2], x.shape[3])

        # 1. Extract skips and bottleneck through backbone
        feat_dict = self.backbone(x)
        skips = feat_dict["skips"]             # [Stage 0 (48), Stage 1 (96), Stage 2 (192)]
        bottleneck = feat_dict["bottleneck"]   # (B, 384, 14, 14)

        # 2. Decode features with progressive upsampling
        logits = self.decoder(
            bottleneck=bottleneck,
            skips=skips,
            target_size=target_size,
        )

        return logits

    def get_model_info(self) -> Dict[str, Union[int, str]]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model_name": "B0ConvNeXtViTUNet (Baseline Ladder B0)",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "num_classes": self.num_classes,
            "img_size": self.img_size,
            "num_shared_experts": self.num_sage_experts,
        }


def create_b0_unet(
    num_classes: int = 1,
    img_size: int = 448,
    freeze_encoder: bool = False,
    freeze_transformer: bool = False,
    use_dwsc: bool = False,
    pretrained: bool = True,
) -> B0ConvNeXtViTUNet:
    """
    Factory function for Baseline B0 Model.
    """
    return B0ConvNeXtViTUNet(
        num_classes=num_classes,
        img_size=img_size,
        freeze_encoder=freeze_encoder,
        freeze_transformer=freeze_transformer,
        use_dwsc=use_dwsc,
        pretrained=pretrained,
    )


if __name__ == "__main__":
    print("=" * 65)
    print("=== SK3 Self-Test: End-to-End Baseline B0 Model ===")
    print("=" * 65)

    batch_size = 2
    img_size = 448
    num_classes = 1

    # 1. Initialize B0 Model (pretrained=False for offline smoke test)
    model = create_b0_unet(
        num_classes=num_classes,
        img_size=img_size,
        pretrained=False,
    )

    info = model.get_model_info()
    for k, v in info.items():
        print(f"  {k}: {v}")

    # 2. Input Tensor (B=2, C=3, H=448, W=448)
    x = torch.randn(batch_size, 3, img_size, img_size, requires_grad=True)
    print(f"\n[Test 1] Input Shape: {tuple(x.shape)}")

    # 3. Forward Pass
    logits = model(x)
    print(f"[Test 2] Output Logits Shape: {tuple(logits.shape)}")
    assert logits.shape == (batch_size, num_classes, img_size, img_size), (
        f"Shape mismatch: expected ({batch_size}, {num_classes}, {img_size}, {img_size}), got {logits.shape}"
    )
    print("  [OK] End-to-end output shape (B, 1, 448, 448) verified successfully.")

    # 4. Backward Pass & Gradient Flow
    loss = logits.sum()
    loss.backward()
    assert x.grad is not None, "Gradients failed to flow back to input image!"
    print(f"[Test 3] Backward Pass OK: Loss={loss.item():.4f}, Input grad shape={tuple(x.grad.shape)}")

    # Verify parameters received gradients
    trainable_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"[Test 4] Parameters with gradient: {trainable_with_grad}/{total_trainable}")
    assert trainable_with_grad == total_trainable, "Some trainable parameters did not receive gradients!"
    print("  [OK] 100% trainable parameters received valid gradients.")

    print("\n" + "=" * 65)
    print("[SUCCESS] All SK3 Baseline B0 Self-Tests Passed Successfully!")
    print("=" * 65)
