"""
Baseline Ladder B0: ConvNeXtV2-Femto U-Net (Pure ConvNeXt, No ViT)
"""

import logging
from typing import Dict, Union, List

import torch
import torch.nn as nn
import timm

try:
    from .decoder_block import UNetDecoder
except ImportError:
    from sage.networks.decoder_block import UNetDecoder

logger = logging.getLogger(__name__)

class B0ConvNeXtUNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        img_size: int = 448,
        freeze_encoder: bool = False,
        use_dwsc: bool = False,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size

        logger.info(f"Loading Pure ConvNeXtV2-Femto (pretrained={pretrained})")
        self.backbone = timm.create_model(
            "convnextv2_femto.fcmae",
            pretrained=pretrained,
            features_only=True,
        )

        if freeze_encoder:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("ConvNeXt encoder parameters frozen")

        self.encoder_channels = [48, 96, 192, 384]

        self.decoder = UNetDecoder(
            encoder_channels=self.encoder_channels,
            num_classes=num_classes,
            use_dwsc=use_dwsc,
        )

    @property
    def num_sage_experts(self) -> int:
        return 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_size = (x.shape[2], x.shape[3])

        features = self.backbone(x)
        skips = features[:-1]
        bottleneck = features[-1]

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
            "model_name": "B0ConvNeXtUNet (Baseline Ladder B0 - Pure ConvNeXt)",
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
) -> B0ConvNeXtUNet:
    return B0ConvNeXtUNet(
        num_classes=num_classes,
        img_size=img_size,
        freeze_encoder=freeze_encoder,
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

    model = create_b0_unet(
        num_classes=num_classes,
        img_size=img_size,
        pretrained=False,
    )

    info = model.get_model_info()
    for k, v in info.items():
        print(f"  {k}: {v}")

    x = torch.randn(batch_size, 3, img_size, img_size, requires_grad=True)
    print(f"\n[Test 1] Input Shape: {tuple(x.shape)}")

    logits = model(x)
    print(f"[Test 2] Output Logits Shape: {tuple(logits.shape)}")
    assert logits.shape == (batch_size, num_classes, img_size, img_size)
    print("  [OK] End-to-end output shape (B, 1, 448, 448) verified successfully.")

    loss = logits.sum()
    loss.backward()
    assert x.grad is not None
    print(f"[Test 3] Backward Pass OK: Loss={loss.item():.4f}")
    
    trainable_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    assert trainable_with_grad == total_trainable
    print("  [OK] 100% trainable parameters received valid gradients.")
