"""
ConvNeXtV2-ViT Hybrid Backbone for Crack Segmentation (SAGE-Lite)

Scope: SK2 (Milestone 1)
- Pure hybrid backbone (ConvNeXtV2-Femto + 6 blocks ViT-Tiny).
- No SAGE components (Router, SageLayer, SA-Hub, injection) - deferred to SK4/SK5.
- No Decoder - deferred to SK3.
- Hard-locked to 6 ViT blocks (50% depth of pretrained ViT-Tiny).
- CLS token dropped; positional embedding aligned to 14x14 grid.
- Shape contract:
    Input: (B, 3, 448, 448)
    Stages: [ (B, 48, 112, 112), (B, 96, 56, 56), (B, 192, 28, 28), (B, 384, 14, 14) ]
    Bottleneck tokens: 196 tokens, embed_dim=192
    Bottleneck output: (B, 384, 14, 14)
    num_sage_experts = 4 (for downstream SAGE compatibility)

Author: Special Subject AI Team
Date: September 2026
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Hard-lock constants for SAGE-Lite architecture
NUM_TRANSFORMER_BLOCKS: int = 6  # First 6 blocks of 12-block ViT-Tiny
CONVNEXT_FEMTO_CHANNELS: List[int] = [48, 96, 192, 384]
VIT_TINY_EMBED_DIM: int = 192
DEFAULT_IMG_SIZE: int = 448
BOTTLENECK_GRID_SIZE: int = 14  # 448 / 32 = 14 -> 196 tokens


class ConvNeXtV2ViTHybrid(nn.Module):
    """
    Hybrid Backbone combining ConvNeXtV2-Femto with 6 blocks of ViT-Tiny.

    Provides multi-scale feature extraction for UNet skip connections and
    global self-attention modeling at the bottleneck.

    Args:
        img_size (int): Expected input resolution (default: 448).
        convnext_model_name (str): timm model name for ConvNeXt-V2 (default: 'convnextv2_femto.fcmae').
        vit_model_name (str): timm model name for ViT (default: 'vit_tiny_patch16_224').
        freeze_encoder (bool): Whether to freeze ConvNeXt parameters.
        freeze_transformer (bool): Whether to freeze ViT transformer blocks.
        pretrained (bool): Whether to load ImageNet pretrained weights.
    """

    def __init__(
        self,
        img_size: int = DEFAULT_IMG_SIZE,
        convnext_model_name: str = "convnextv2_femto.fcmae",
        vit_model_name: str = "vit_tiny_patch16_224",
        freeze_encoder: bool = False,
        freeze_transformer: bool = False,
        pretrained: bool = True,
    ):
        super().__init__()

        self.img_size = img_size
        self.num_transformer_layers = NUM_TRANSFORMER_BLOCKS  # Hard-locked to 6 blocks

        # ====================================================================
        # 1. ENCODER: ConvNeXt-V2 Femto Backbone
        # ====================================================================
        logger.info(f"Loading ConvNeXt-V2: {convnext_model_name} (pretrained={pretrained})")
        self.convnext = timm.create_model(
            convnext_model_name,
            pretrained=pretrained,
            features_only=False,
        )

        # Profile encoder channels and spatial dims using dummy pass
        dummy_input = torch.randn(1, 3, img_size, img_size)
        self.encoder_channels: List[int] = []
        self.encoder_spatial_sizes: List[int] = []

        with torch.no_grad():
            x = self.convnext.stem(dummy_input)
            self.stem_channels = x.shape[1]
            for stage in self.convnext.stages:
                x = stage(x)
                self.encoder_channels.append(x.shape[1])
                self.encoder_spatial_sizes.append(x.shape[2])

        # Validate channel contract
        assert self.encoder_channels == CONVNEXT_FEMTO_CHANNELS, (
            f"Channel contract mismatch: Expected {CONVNEXT_FEMTO_CHANNELS}, got {self.encoder_channels}"
        )

        # Explicitly remove ConvNeXt classification head (not used in segmentation)
        if hasattr(self.convnext, "head"):
            del self.convnext.head
            logger.info("Removed ConvNeXt classification head")

        if freeze_encoder:
            for param in self.convnext.parameters():
                param.requires_grad = False
            logger.info("ConvNeXt encoder parameters frozen")

        # ====================================================================
        # 2. BOTTLENECK: ViT-Tiny (Hard-locked: first 6 blocks)
        # ====================================================================
        logger.info(f"Loading ViT-Tiny: {vit_model_name} (pretrained={pretrained})")
        vit_full = timm.create_model(vit_model_name, pretrained=pretrained)

        self.transformer_dim = vit_full.embed_dim  # 192
        self.num_heads = getattr(vit_full, "num_heads", 3)

        if not hasattr(vit_full, "blocks"):
            raise AttributeError(f"Cannot find 'blocks' in {vit_model_name}")

        all_blocks = vit_full.blocks
        # Slice exactly 6 blocks, preserving pretrained ImageNet weights
        self.transformer_blocks = nn.ModuleList([
            all_blocks[i] for i in range(NUM_TRANSFORMER_BLOCKS)
        ])
        logger.info(f"Hard-locked {len(self.transformer_blocks)} ViT blocks (embed_dim={self.transformer_dim})")

        # ====================================================================
        # Explicit CLS Token Handling & Positional Embedding Separation
        # Contract: vit_tiny_patch16_224 has 1 CLS token + 196 patch tokens = 197 tokens.
        # For SAGE-Lite dense prediction, discard/skip CLS token explicitly and retain
        # exclusively the 196 patch tokens for the spatial grid (14x14).
        # ====================================================================
        raw_pos_embed = vit_full.pos_embed.detach().clone()
        has_cls = getattr(vit_full, "has_class_token", True) and raw_pos_embed.shape[1] > 1

        if has_cls:
            self.cls_pos_embed = raw_pos_embed[:, :1, :]   # (1, 1, 192) - Isolated CLS positional embedding
            patch_pos_embed = raw_pos_embed[:, 1:, :]      # (1, 196, 192) - 196 spatial patch tokens only
            logger.info(
                f"[Explicit CLS Handling] Detected {raw_pos_embed.shape[1]} tokens in ViT pos_embed. "
                f"Discarded CLS token (1 token), retained exactly {patch_pos_embed.shape[1]} spatial patch tokens."
            )
        else:
            self.cls_pos_embed = None
            patch_pos_embed = raw_pos_embed

        # Register EXCLUSIVELY the 196 spatial patch positional embeddings as model parameter
        self.positional_embeddings = nn.Parameter(patch_pos_embed)
        self.pretrained_grid_size = int(math.isqrt(patch_pos_embed.shape[1]))
        assert self.positional_embeddings.shape[1] == 196, (
            f"Expected exactly 196 spatial patch tokens, got {self.positional_embeddings.shape[1]}!"
        )
        logger.info(
            f"Loaded patch_pos: shape={tuple(self.positional_embeddings.shape)} "
            f"(grid: {self.pretrained_grid_size}x{self.pretrained_grid_size})"
        )

        del vit_full  # Clean up unneeded ViT parts

        if freeze_transformer:
            for block in self.transformer_blocks:
                for param in block.parameters():
                    param.requires_grad = False
            logger.info("Transformer blocks frozen")

        # ====================================================================
        # 3. INTERFACE PROJECTIONS: ConvNeXt <-> ViT
        # ====================================================================
        self.bottleneck_channels = self.encoder_channels[-1]  # 384
        self.bottleneck_spatial_size = self.encoder_spatial_sizes[-1]  # 14

        # Linear projection: ConvNeXt channels (384) -> ViT dimension (192)
        self.convnext_to_transformer = nn.Linear(self.bottleneck_channels, self.transformer_dim)
        self.pre_transformer_norm = nn.LayerNorm(self.transformer_dim)

        # Reprojection: ViT dimension (192) -> Bottleneck channels (384)
        self.transformer_to_decoder = nn.Linear(self.transformer_dim, self.bottleneck_channels)
        self.post_transformer_norm = nn.LayerNorm(self.bottleneck_channels)

    @property
    def num_sage_experts(self) -> int:
        """
        Number of CNN stages that act as shared experts (fixed at 4).
        Required contract for downstream SAGE components (SK4/SK5).
        """
        return len(self.encoder_channels)

    def _get_interpolated_pos_embed(self, h: int, w: int) -> torch.Tensor:
        """
        Align positional embeddings with bottleneck spatial grid (H, W).
        At 448x448, H=14, W=14 -> 196 tokens -> Identity / no-op.
        """
        num_patches = h * w
        if num_patches == self.positional_embeddings.shape[1]:
            return self.positional_embeddings

        # 2D Bicubic interpolation for non-448 resolutions
        orig_grid = self.pretrained_grid_size
        pos = self.positional_embeddings.reshape(1, orig_grid, orig_grid, -1).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(h, w), mode="bicubic", align_corners=False)
        return pos.permute(0, 2, 3, 1).flatten(1, 2)

    def get_encoder_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale hierarchical features from ConvNeXt-V2.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W).

        Returns:
            List[torch.Tensor]: Feature maps from 4 stages:
                - Stage 0: (B, 48, H/4, W/4)
                - Stage 1: (B, 96, H/8, W/8)
                - Stage 2: (B, 192, H/16, W/16)
                - Stage 3: (B, 384, H/32, W/32)
        """
        features = []
        x_feat = self.convnext.stem(x)
        for stage in self.convnext.stages:
            x_feat = stage(x_feat)
            features.append(x_feat)
        return features

    def forward_bottleneck(self, stage3_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process the Stage 3 bottleneck feature map through the 6-block ViT.

        Strict Contract:
        - Input is spatial 2D feature map from Stage 3 (B, 384, H, W).
        - Sequence path operates EXCLUSIVELY on H*W patch tokens (196 tokens at 448x448).
        - CLS token is strictly excluded from spatial path.
        - Any reshape back to (B, C, H, W) is performed exclusively on the H*W patch tokens.
        - Never reshape a 197-token sequence into the spatial grid.

        Args:
            stage3_feat (torch.Tensor): Stage 3 feature map of shape (B, 384, H_bn, W_bn).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - bottleneck_spatial: Reprojected 2D feature map of shape (B, 384, H_bn, W_bn).
                - tokens: Transformer token sequence of shape (B, H_bn*W_bn, 192).
        """
        b, c, h, w = stage3_feat.shape
        num_spatial_patches = h * w  # Exactly 14 * 14 = 196 at 448x448

        # 1. Flatten to spatial tokens: (B, 384, H, W) -> (B, H*W, 384)
        tokens = stage3_feat.flatten(2).transpose(1, 2)
        assert tokens.shape[1] == num_spatial_patches, (
            f"Expected {num_spatial_patches} spatial tokens, got {tokens.shape[1]}"
        )

        # 2. Linear projection: 384 -> 192
        tokens = self.convnext_to_transformer(tokens)

        # 3. Add positional embeddings (Exclusively 196 patch tokens; CLS discarded)
        pos_embed = self._get_interpolated_pos_embed(h, w)
        assert pos_embed.shape[1] == num_spatial_patches, (
            f"Positional embedding mismatch: expected {num_spatial_patches} tokens, got {pos_embed.shape[1]}"
        )
        tokens = tokens + pos_embed
        tokens = self.pre_transformer_norm(tokens)

        # 4. Forward through the 6 hard-locked ViT blocks
        for blk in self.transformer_blocks:
            tokens = blk(tokens)

        # 5. Strict Fail-Fast Guard: Verify sequence is not contaminated with CLS token
        if tokens.shape[1] == num_spatial_patches + 1:
            raise RuntimeError(
                f"[Fatal CLS Error] Detected {tokens.shape[1]} tokens (CLS token leaked into sequence)! "
                f"SAGE-Lite strictly discards CLS at initialization. Expected exactly {num_spatial_patches} patch tokens."
            )
        if tokens.shape[1] != num_spatial_patches:
            raise RuntimeError(
                f"[Fatal Spatial Shape Error] Expected {num_spatial_patches} spatial patch tokens "
                f"for ({h}x{w}) grid, but got {tokens.shape[1]}! Never reshape invalid token counts into spatial grid."
            )

        # 6. Reproject back to bottleneck channels: 192 -> 384
        out_tokens = self.transformer_to_decoder(tokens)
        out_tokens = self.post_transformer_norm(out_tokens)

        # 7. Reshape EXCLUSIVELY the 196 patch tokens back to 2D spatial grid: (B, 384, H, W)
        bottleneck_spatial = out_tokens.transpose(1, 2).reshape(b, c, h, w)

        return bottleneck_spatial, tokens

    def forward(self, x: torch.Tensor) -> Dict[str, Union[List[torch.Tensor], torch.Tensor]]:
        """
        Forward pass through the hybrid backbone.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, H, W).

        Returns:
            Dict containing:
                - 'skips': List of 3 skip connection feature maps (Stage 0, 1, 2).
                - 'bottleneck': 2D feature map after 6 ViT blocks (B, 384, H/32, W/32).
                - 'tokens': Transformer token embeddings (B, 196, 192).
        """
        encoder_features = self.get_encoder_features(x)
        skips = encoder_features[:-1]  # Stage 0 (48), Stage 1 (96), Stage 2 (192)
        stage3_feat = encoder_features[-1]  # Stage 3 (384)

        bottleneck_spatial, tokens = self.forward_bottleneck(stage3_feat)

        return {
            "skips": skips,
            "bottleneck": bottleneck_spatial,
            "tokens": tokens,
        }

    def get_model_info(self) -> Dict[str, Union[int, str, List[int]]]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model_name": "ConvNeXtV2ViTHybrid (SAGE-Lite Backbone)",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "encoder_channels": self.encoder_channels,
            "num_transformer_layers": len(self.transformer_blocks),
            "num_sage_experts": self.num_sage_experts,
        }


def create_convnextv2_vit_hybrid(
    img_size: int = DEFAULT_IMG_SIZE,
    freeze_encoder: bool = False,
    freeze_transformer: bool = False,
    pretrained: bool = True,
) -> ConvNeXtV2ViTHybrid:
    """
    Factory function for SAGE-Lite Hybrid Backbone.
    """
    return ConvNeXtV2ViTHybrid(
        img_size=img_size,
        convnext_model_name="convnextv2_femto.fcmae",
        vit_model_name="vit_tiny_patch16_224",
        freeze_encoder=freeze_encoder,
        freeze_transformer=freeze_transformer,
        pretrained=pretrained,
    )


if __name__ == "__main__":
    print("=" * 65)
    print("=== SK2 Self-Test: ConvNeXtV2-ViT Hybrid Backbone (SAGE-Lite) ===")
    print("=" * 65)

    # 1. Model Initialization
    model = create_convnextv2_vit_hybrid(
        img_size=448,
        pretrained=False,  # Offline smoke test
    )

    info = model.get_model_info()
    for k, v in info.items():
        print(f"  {k}: {v}")

    # 2. Input Tensor (Contract: 448x448)
    batch_size = 2
    x = torch.randn(batch_size, 3, 448, 448)
    print(f"\n[Test 1] Input Shape: {tuple(x.shape)}")

    # 3. Test get_encoder_features()
    encoder_feats = model.get_encoder_features(x)
    print(f"\n[Test 2] get_encoder_features() Output ({len(encoder_feats)} stages):")
    for i, feat in enumerate(encoder_feats):
        print(f"  Stage {i}: shape={tuple(feat.shape)} (channels={feat.shape[1]})")
    assert len(encoder_feats) == 4, f"Expected 4 stages, got {len(encoder_feats)}"
    assert [f.shape[1] for f in encoder_feats] == [48, 96, 192, 384]

    # 4. Test forward_bottleneck()
    stage3_in = encoder_feats[-1]
    bottleneck_spatial, tokens = model.forward_bottleneck(stage3_in)
    print(f"\n[Test 3] forward_bottleneck() Output:")
    print(f"  Spatial 2D: shape={tuple(bottleneck_spatial.shape)}")
    print(f"  Tokens:     shape={tuple(tokens.shape)}")
    assert bottleneck_spatial.shape == (batch_size, 384, 14, 14), f"Shape mismatch: {bottleneck_spatial.shape}"
    assert tokens.shape == (batch_size, 196, 192), f"Tokens mismatch: {tokens.shape}"

    # 5. Test forward() full pass
    out = model(x)
    print(f"\n[Test 4] forward() Output Dict:")
    print(f"  Skips count: {len(out['skips'])}")
    for i, skip in enumerate(out["skips"]):
        print(f"    Skip {i}: {tuple(skip.shape)}")
    print(f"  Bottleneck: {tuple(out['bottleneck'].shape)}")
    print(f"  Tokens:     {tuple(out['tokens'].shape)}")

    # 6. Verify gradient flow
    loss = out["bottleneck"].sum() + out["tokens"].sum()
    loss.backward()
    print(f"\n[Test 5] Gradient flow check: loss={loss.item():.4f} -> Backward pass OK!")

    # 7. Explicit CLS Discard & Reshape Invariance Verification
    print(f"\n[Test 6] Explicit CLS Handling & Spatial Reshape Invariance:")
    print(f"  Model positional_embeddings shape: {tuple(model.positional_embeddings.shape)}")
    assert model.positional_embeddings.shape == (1, 196, 192), (
        f"Positional embedding must have exactly 196 tokens (got {model.positional_embeddings.shape})"
    )
    assert model.positional_embeddings.shape[1] == 196, "CLS token must be excluded from positional_embeddings!"
    print("  [OK] Confirmed positional_embeddings contains strictly 196 spatial tokens (CLS dropped).")

    # Verify that reshaping 197 tokens directly into (14, 14) grid is mathematically impossible
    try:
        dummy_197 = torch.randn(batch_size, 197, 384)
        dummy_197.transpose(1, 2).reshape(batch_size, 384, 14, 14)
        raise AssertionError("Reshaping 197 tokens into 14x14 should have failed!")
    except RuntimeError:
        print("  [OK] Confirmed RuntimeError when attempting to reshape 197 tokens into 14x14 grid.")

    # Verify that forward_bottleneck raises RuntimeError immediately (fail-fast) if tokens contain 197 tokens
    # by temporarily hooking or testing the guard
    orig_blocks = model.transformer_blocks
    class LeakyBlock(torch.nn.Module):
        def forward(self, x):
            # Leaks a CLS token
            cls = torch.zeros(x.shape[0], 1, x.shape[2], device=x.device)
            return torch.cat([cls, x], dim=1)
    model.transformer_blocks = torch.nn.ModuleList([LeakyBlock()])
    try:
        model.forward_bottleneck(stage3_in)
        raise AssertionError("Fail-fast guard should have raised RuntimeError on 197 tokens!")
    except RuntimeError as e:
        assert "Fatal CLS Error" in str(e), f"Unexpected error: {e}"
        print("  [OK] Confirmed Fail-Fast RuntimeError when 197 tokens are detected in bottleneck.")
    finally:
        model.transformer_blocks = orig_blocks

    print("\n" + "=" * 65)
    print("[SUCCESS] All SK2 Backbone Self-Tests Passed Successfully!")
    print("=" * 65)
