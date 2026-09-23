"""
Baseline Ladder B2: Full SAGE-Lite Model (ConvNeXtV2-ViT Hybrid UNet + SAGE)

Scope: Milestone 2 (B2 Implementation)
- Combines B1 backbone (ConvNeXtV2-Femto + configurable ViT-Tiny blocks) with UNet Decoder.
- Full SAGE-Lite integration:
  - SageRouter with SAR, Shared Gating (g_s), Top-K selection, and Load Balancing Loss.
  - Heterogeneous Expert Pool (4 CNN stages + N_vit ViT blocks).
  - Shape-Adapting Hub (SA-Hub) with O(D^2) Pairwise Learnable Projections.
  - SageLayer with Pure Residual Fusion (fused = main + dropout(expert)) and Zero-Cost Self-Selection Bypass.
  - Full Injection: All 4 ConvNeXt stages + all N_vit ViT blocks (total 4 + N_vit routers).
- Technical Locks:
  - Lock #1: forward() returns pure torch.Tensor; routing metadata cached internally for forward_with_routing_info().
  - Lock #2: Static router channels; zero dynamic parameter creation during forward pass.
- Default resolution: 448x448.

Author: Special Subject AI Team
Date: September 2026
"""

import logging
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

try:
    from .convnextv2_vit_hybrid import ConvNeXtV2ViTHybrid
    from .decoder_block import UNetDecoder
    from .sage_injection import inject_sage_layers, pre_populate_sa_hubs
    from ..components.sage_layer import SageLayer
    from ..components.router import SageRouter
except ImportError:
    from sage.networks.convnextv2_vit_hybrid import ConvNeXtV2ViTHybrid
    from sage.networks.decoder_block import UNetDecoder
    from sage.networks.sage_injection import inject_sage_layers, pre_populate_sa_hubs
    from sage.components.sage_layer import SageLayer
    from sage.components.router import SageRouter

logger = logging.getLogger(__name__)

DEFAULT_SAGE_CONFIG: Dict[str, Any] = {
    "top_k": 4,
    "gating_type": "sigmoid",
    "shared_expert_indices": [0, 1, 2, 3],
    "router_hidden_dim": 64,
    "load_balance_factor": 0.01,
    "logit_modulation": True,
    "expert_dropout": 0.1,
    "fusion_type": "residual",
    "residual_scale": 0.1,
    "adaptive_alpha": 0.9,
}


class B2ConvNeXtViTUNet(nn.Module):
    """
    Baseline B2 Model: Full SAGE-Lite ConvNeXtV2-ViT Hybrid UNet.

    Args:
        num_classes (int): Number of segmentation classes (default: 1 for crack).
        img_size (int): Standard input image resolution (default: 448).
        num_transformer_layers (int): Number of ViT blocks (default: 12, supports depth sweep).
        freeze_encoder (bool): Whether to freeze ConvNeXt parameters (default: False).
        freeze_transformer (bool): Whether to freeze ViT parameters (default: False).
        use_dwsc (bool): Whether to use Depthwise Separable Conv in decoder (default: False).
        pretrained (bool): Whether to load ImageNet pretrained weights (default: True).
        sage_config (dict): Optional custom dictionary to override DEFAULT_SAGE_CONFIG.
    """

    def __init__(
        self,
        num_classes: int = 1,
        img_size: int = 448,
        num_transformer_layers: int = 12,
        freeze_encoder: bool = False,
        freeze_transformer: bool = False,
        use_dwsc: bool = False,
        pretrained: bool = True,
        sage_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.num_transformer_layers = num_transformer_layers

        # 1. Merge SAGE config with defaults
        self.sage_config = dict(DEFAULT_SAGE_CONFIG)
        if sage_config is not None:
            self.sage_config.update(sage_config)

        # 2. Backbone: ConvNeXt-Femto + ViT-Tiny (configurable depth)
        self.backbone = ConvNeXtV2ViTHybrid(
            img_size=img_size,
            convnext_model_name="convnextv2_femto.fcmae",
            vit_model_name="vit_tiny_patch16_224",
            num_transformer_layers=num_transformer_layers,
            freeze_encoder=freeze_encoder,
            freeze_transformer=freeze_transformer,
            pretrained=pretrained,
        )

        # 3. Decoder: Standard 3x3 UNet Decoder with skip connections
        self.decoder = UNetDecoder(
            encoder_channels=self.backbone.encoder_channels,
            num_classes=num_classes,
            use_dwsc=use_dwsc,
        )

        # 4. Inject SAGE wrappers in-place and construct shared expert pool
        self.expert_pool = inject_sage_layers(
            convnext=self.backbone.convnext,
            transformer_blocks=self.backbone.transformer_blocks,
            encoder_channels=self.backbone.encoder_channels,
            stem_channels=self.backbone.stem_channels,
            transformer_dim=self.backbone.transformer_dim,
            sage_config=self.sage_config,
        )

        # 5. Pre-populate all SA-Hub adapters eagerly (O(D^2) pairwise)
        pre_populate_sa_hubs(
            model=self,
            encoder_channels=self.backbone.encoder_channels,
            stem_channels=self.backbone.stem_channels,
            transformer_dim=self.backbone.transformer_dim,
        )

    @property
    def num_sage_experts(self) -> int:
        """Number of shared CNN stages (fixed at 4)."""
        return self.backbone.num_sage_experts

    def set_shared_experts(self, shared_expert_indices: List[int]) -> None:
        """
        Update shared expert indices across all SageRouters in the model
        and trigger router shared mask recomputation.
        """
        self.sage_config["shared_expert_indices"] = list(shared_expert_indices)
        for module in self.modules():
            if isinstance(module, SageRouter):
                module.shared_expert_indices = list(shared_expert_indices)
                if hasattr(module, "_update_shared_mask"):
                    module._update_shared_mask()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass satisfying Strict Tensor Contract (Lock #1).
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W).
            
        Returns:
            torch.Tensor: Segmentation logits of shape (B, num_classes, H, W).
        """
        target_size = (x.shape[2], x.shape[3])

        # 1. Forward through hybrid backbone with injected SAGE layers
        feat_dict = self.backbone(x)
        skips = feat_dict["skips"]
        bottleneck = feat_dict["bottleneck"]

        # 2. Decode features with progressive upsampling
        logits = self.decoder(
            bottleneck=bottleneck,
            skips=skips,
            target_size=target_size,
        )

        return logits

    def forward_with_routing_info(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Forward pass that captures routing information from all SAGE layers.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W).
            
        Returns:
            dict containing:
                - 'logits': Segmentation logits (B, num_classes, H, W).
                - 'routing_infos': Dict with 'cnn', 'transformer', and 'all' lists.
        """
        # 1. Normal forward pass (caches routing info inside each SageLayer)
        logits = self.forward(x)

        # 2. Harvest routing info from all CNN stages and ViT blocks
        cnn_routing_infos = []
        for stage in self.backbone.convnext.stages:
            if isinstance(stage, SageLayer):
                cnn_routing_infos.append(stage.get_last_routing_info())

        transformer_routing_infos = []
        for blk in self.backbone.transformer_blocks:
            if isinstance(blk, SageLayer):
                transformer_routing_infos.append(blk.get_last_routing_info())

        all_routing_infos = cnn_routing_infos + transformer_routing_infos

        return {
            "logits": logits,
            "routing_infos": {
                "cnn": cnn_routing_infos,
                "transformer": transformer_routing_infos,
                "all": all_routing_infos,
            },
        }

    def compute_total_load_balance_loss(
        self,
        routing_infos: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> torch.Tensor:
        """
        Compute total load balance loss aggregated across all active SAGE routers.
        
        Args:
            routing_infos: Dict returned by forward_with_routing_info or list of routing dicts.
            
        Returns:
            torch.Tensor: Scalar load balance loss tensor.
        """
        if isinstance(routing_infos, dict):
            if "all" in routing_infos:
                infos = routing_infos["all"]
            else:
                infos = routing_infos.get("cnn", []) + routing_infos.get("transformer", [])
        elif isinstance(routing_infos, list):
            infos = routing_infos
        else:
            infos = []

        total_loss = None
        for info in infos:
            if info and isinstance(info, dict) and "load_balance_loss" in info:
                lb = info["load_balance_loss"]
                if isinstance(lb, torch.Tensor):
                    total_loss = lb if total_loss is None else (total_loss + lb)

        if total_loss is None:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)
        return total_loss

    def get_expert_usage_statistics(self) -> Dict[str, Any]:
        """
        Collect expert usage statistics from all SAGE layers.
        
        Returns:
            dict: Grouped statistics for CNN stages and Transformer blocks.
        """
        stats: Dict[str, List[Dict[str, Any]]] = {
            "cnn_stages": [],
            "transformer_blocks": [],
        }

        for stage in self.backbone.convnext.stages:
            if isinstance(stage, SageLayer) and hasattr(stage, "get_stats"):
                stats["cnn_stages"].append(stage.get_stats())

        for blk in self.backbone.transformer_blocks:
            if isinstance(blk, SageLayer) and hasattr(blk, "get_stats"):
                stats["transformer_blocks"].append(blk.get_stats())

        return stats

    def get_model_info(self) -> Dict[str, Union[int, str, List[int]]]:
        """Get model parameter count and structural information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        sage_layers = [m for m in self.modules() if isinstance(m, SageLayer)]

        return {
            "model_name": "B2ConvNeXtViTUNet (Full SAGE-Lite)",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "num_classes": self.num_classes,
            "img_size": self.img_size,
            "num_transformer_layers": self.num_transformer_layers,
            "num_sage_experts": self.num_sage_experts,
            "num_injected_routers": len(sage_layers),
            "expert_pool_size": len(self.expert_pool) if self.expert_pool is not None else 0,
            "top_k": self.sage_config.get("top_k", 4),
            "router_hidden_dim": self.sage_config.get("router_hidden_dim", 64),
            "gating_type": self.sage_config.get("gating_type", "sigmoid"),
            "fusion_type": self.sage_config.get("fusion_type", "residual"),
            "residual_scale": self.sage_config.get("residual_scale", 0.1),
        }


def create_b2_unet(
    num_classes: int = 1,
    img_size: int = 448,
    num_transformer_layers: int = 12,
    freeze_encoder: bool = False,
    freeze_transformer: bool = False,
    use_dwsc: bool = False,
    pretrained: bool = True,
    sage_config: Optional[Dict[str, Any]] = None,
) -> B2ConvNeXtViTUNet:
    """
    Factory function for Full SAGE-Lite Model (Baseline Ladder B2).
    """
    return B2ConvNeXtViTUNet(
        num_classes=num_classes,
        img_size=img_size,
        num_transformer_layers=num_transformer_layers,
        freeze_encoder=freeze_encoder,
        freeze_transformer=freeze_transformer,
        use_dwsc=use_dwsc,
        pretrained=pretrained,
        sage_config=sage_config,
    )
