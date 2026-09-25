"""
This module implements the Mixture of Local Experts layer architecture,
which combines a main processing path with a dynamic expert selection mechanism.

Architecture:
    - SageLayer: Orchestrates dual-path processing (main + expert paths)
    - Router: Handles dynamic expert selection via SAGE routing
    - SAHub: Manages shape adaptation between different tensor formats

Key Features:
    - Dual-path processing with learnable mixing weight (alpha)
    - Dynamic expert selection based on input characteristics
    - Automatic shape adaptation for heterogeneous expert architectures
    - Built-in statistics tracking for monitoring and analysis
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
from .router import SageRouter
from .sa_hub import SAHub


class SageLayer(nn.Module):
    """
    SAGE Layer - Mixture of Local Experts Layer
    
    A dual-path neural network layer that combines:
    1. Main Path: Original block processing for stable feature extraction
    2. Expert Path: Dynamic expert selection and execution for specialized processing
    
    The outputs are combined using a learnable mixing weight (alpha):
        output = alpha * main_output + (1 - alpha) * expert_output
    """
    
    def __init__(
        self,
        main_block: nn.Module,
        router: SageRouter,
        sa_hub: SAHub,
        expert_dropout: float = 0.0,
        my_index: Optional[int] = None,
        expert_pool: Optional[nn.ModuleList] = None,
        alpha: Optional[float] = None,
        residual_scale: float = 0.1,
        fusion_type: str = "residual",
        adaptive_alpha: float = 0.9,
        layer_type: Optional[str] = None,
        stage_idx: Optional[int] = None,
        p3_refinement: Optional[nn.Module] = None,
    ):
        """
        Initialize SAGE Layer for SAGE-Lite.
        
        Args:
            main_block: Original block being wrapped (e.g., ConvNeXt stage / Transformer block)
            router: Router for dynamic expert selection (implements SAGE routing)
            sa_hub: Shape-Adapting Hub for tensor format conversion
            expert_dropout: Dropout rate applied to expert path output (default: 0.0)
            my_index: Index of this layer in the expert pool for zero-cost self-selection bypass
            expert_pool: Optional reference to the shared expert pool
            alpha: Deprecated in SAGE-Lite (pure residual fusion: main + adapter is used)
            residual_scale: Scaling factor for residual expert path (default: 0.1) to preserve
                stability and numerical range under AMP FP16 across deep networks.
        """
        super().__init__()
        self.main_block = main_block
        self.router = router
        self.sa_hub = sa_hub
        self.my_index = my_index
        self.expert_pool = expert_pool
        self.fusion_type = fusion_type
        if self.fusion_type == "adaptive":
            self.alpha = nn.Parameter(torch.tensor(adaptive_alpha))
        elif self.fusion_type == "residual":
            self.residual_scale = float(residual_scale)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
        
        self.expert_dropout = (
            nn.Dropout(expert_dropout) if expert_dropout > 0 else nn.Identity()
        )

        self.register_buffer('forward_calls', torch.tensor(0))
        self.register_buffer('expert_successes', torch.tensor(0))

        # Cached outputs from the most recent forward pass (Tensor Contract)
        self._last_lb_loss: Optional[torch.Tensor] = None
        self._last_routing_info: Optional[Dict[str, Any]] = None
        self.layer_type = layer_type
        self.stage_idx = stage_idx
        self.p3_refinement = p3_refinement
        object.__setattr__(self, "_pe_owner", None)

        self.logger = logging.getLogger(self.__class__.__name__)

    def set_pe_owner(self, owner: nn.Module) -> None:
        """Set canonical PE28 owner without registering as child module in _modules."""
        object.__setattr__(self, "_pe_owner", owner)

    def get_pe28(self) -> Optional[torch.Tensor]:
        """Dynamically retrieve current device-local PE28 buffer from canonical owner."""
        if hasattr(self, "_pe_owner") and self._pe_owner is not None:
            return getattr(self._pe_owner, "pe28_fixed", None)
        return None

    def forward(
        self,
        x: torch.Tensor,
        expert_pool: Optional[nn.ModuleList] = None
    ) -> torch.Tensor:
        """
        SAGE Layer forward pass obeying Strict Tensor Contract.
        
        Flow:
            1. Main Path: x -> main_block -> main_output
            2. Expert Path: x -> router -> experts (via SA-Hub) -> expert_output
            3. Pure Residual Fusion: main_output + Dropout(expert_output)
            4. Cache routing_info internally for forward_with_routing_info collection.
        
        Args:
            x: Input tensor of shape (B, C, H, W) or (B, N, D)
            expert_pool: Optional ModuleList of experts. If None, uses self.expert_pool.
            
        Returns:
            torch.Tensor: final_output (pure Tensor, ensuring unbroken backbone flow)
        """
        self.forward_calls += 1
        pool = expert_pool if expert_pool is not None else self.expert_pool
        if pool is None:
            raise ValueError(
                f"expert_pool is not provided and not set on SageLayer (my_index={self.my_index})"
            )
        
        # Step 1: Execute main processing path
        main_output = self._execute_main_path(x)
        
        # Step 2: Execute expert processing path
        expert_output, routing_info = self._execute_expert_path(
            x, main_output, pool
        )
        
        # Step 3: Fusion
        if self.fusion_type == "adaptive":
            alpha = torch.clamp(self.alpha, 0.1, 1.0)
            final_output = alpha * main_output + (1.0 - alpha) * self.expert_dropout(expert_output)
        else:
            final_output = main_output + self.expert_dropout(self.residual_scale * expert_output)
        
        # Step 4: Update and cache routing statistics internally (Tensor Contract)
        routing_info.update({
            'forward_call_count': self.forward_calls.item(),
            'expert_success_rate': (
                self.expert_successes / max(self.forward_calls, 1)
            ).item(),
            'my_index': self.my_index,
        })
        if self.fusion_type == "adaptive":
            routing_info['alpha'] = self.alpha.item()
        self._last_routing_info = routing_info
        
        return final_output

    def get_last_routing_info(self) -> Optional[Dict[str, Any]]:
        """Retrieve routing info cached from the most recent forward pass."""
        return self._last_routing_info

    def _execute_main_path(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute main block processing with error handling.
        
        Args:
            x: Input tensor
            
        Returns:
            Processed tensor from main block
            Falls back to identity mapping if main block fails
        """
        try:
            main_output = self.main_block(x)
            if isinstance(main_output, tuple):
                return main_output[0]
            return main_output
        except torch.cuda.OutOfMemoryError as e:
            self.logger.error("OOM encountered in main path, propagating...")
            raise e
        except Exception as e:
            self.logger.warning(f"Main block failed: {e}")
            return x
    

    def _execute_expert_path(
        self,
        x: torch.Tensor,
        main_output: torch.Tensor,
        expert_pool: nn.ModuleList
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Execute expert processing path with SAGE routing and SA-Hub adaptation.
        Args:
            x: Input tensor (B, C, H, W) or (B, N, D)
            main_output: Output from main path (for shape reference)
            expert_pool: ModuleList of available expert modules
            
        Returns:
            Tuple containing:
                - aggregated_expert_output: Weighted sum of expert outputs
                - routing_info: Dictionary with routing decisions and metadata
        """
        B, *dims = main_output.shape
        device = x.device
        
        try:
            # Step 1: Get routing decisions from SAGE router
            top_k_indices, gating_weights, routing_info = self.router(x)

            # Cache lb_loss so training loop can retrieve it without extra forward
            self._last_lb_loss = routing_info.get('load_balance_loss', None)
            
            # Add sample-level logging information for debugging
            if B > 0:
                routing_info['active_experts_sample_0'] = top_k_indices[0].tolist()
                routing_info['gating_weights_sample_0'] = gating_weights[0].tolist()
            
            # Step 2: Prepare batch-expert dispatch mapping
            # Flatten batch and top-k dimensions for efficient processing
            flat_indices = top_k_indices.flatten()  # Shape: [B*K]
            flat_weights = gating_weights.flatten()  # Shape: [B*K]
            
            # Create mapping from flattened items to original batch indices
            batch_map = torch.arange(B, device=device).repeat_interleave(
                self.router.top_k
            )
            
            # Step 3: Initialize output accumulator
            final_expert_output = torch.zeros_like(main_output)
            
            # Step 4: Process each expert (iterate over M experts, not B batches)
            for expert_idx, expert in enumerate(expert_pool):
                # Find which samples are routed to this expert
                expert_mask = (flat_indices == expert_idx)
                if not expert_mask.any():
                    continue  # Skip unused experts
                
                # Get original batch indices for this expert's inputs
                original_batch_indices = batch_map[expert_mask]
                
                # Zero-cost Self-Selection Bypass
                if self.my_index is not None and expert_idx == self.my_index:
                    adapted_expert_output = main_output[original_batch_indices]
                elif (
                    self.get_pe28() is not None
                    and getattr(self, "layer_type", None) == "cnn"
                    and getattr(self, "stage_idx", None) in (0, 1)
                    and getattr(expert, "expert_type", None) == "transformer"
                ):
                    # Step 4a (P3): High-Resolution CNN (S0/S1) -> ViT Expert
                    # S0 feature is x (C=48, 112x112); S1 feature is main_output (C=96, 56x56)
                    feat_sub = main_output[original_batch_indices] if getattr(self, "stage_idx", None) == 1 else x[original_batch_indices]
                    x_refined = self.p3_refinement(feat_sub) if self.p3_refinement is not None else feat_sub
                    x_compressed = F.adaptive_avg_pool2d(x_refined, (28, 28))
                    x_tokens = x_compressed.flatten(2).transpose(1, 2).contiguous()
                    adapted_input = self.sa_hub._adapt_channels(x_tokens, 192)

                    # Fixed Shared PE28 via Runtime Lookup
                    pe28 = self.get_pe28()
                    assert pe28 is not None, f"SageLayer (idx={self.my_index}) missing pe_owner link!"
                    tokens_with_pe = adapted_input + pe28.to(device=adapted_input.device, dtype=adapted_input.dtype)

                    expert_raw_output = expert(tokens_with_pe)
                    if isinstance(expert_raw_output, tuple):
                        expert_raw_output = expert_raw_output[0]

                    main_path_shape_subset = main_output[original_batch_indices].shape
                    adapted_expert_output, _ = self.sa_hub.adapt(
                        expert_raw_output,
                        self.main_block,
                        main_path_shape=main_path_shape_subset
                    )
                else:
                    # Normal SAGE Path
                    # Step 4a: Adapt input tensor shape for this expert
                    adapted_input, _ = self.sa_hub.adapt(
                        x[original_batch_indices],
                        expert
                    )
                    
                    # Step 4b: Execute expert forward pass
                    expert_raw_output = expert(adapted_input)
                    if isinstance(expert_raw_output, tuple):
                        expert_raw_output = expert_raw_output[0]
                    
                    # Step 4c: Adapt expert output to match main path shape
                    main_path_shape_subset = main_output[original_batch_indices].shape
                    adapted_expert_output, _ = self.sa_hub.adapt(
                        expert_raw_output,
                        self.main_block,
                        main_path_shape=main_path_shape_subset
                    )
                
                # Step 4d: Apply gating weights (reshape for broadcasting)
                weights_for_expert = flat_weights[expert_mask]
                reshape_dims = [-1] + [1] * (adapted_expert_output.dim() - 1)
                weighted_output = adapted_expert_output * weights_for_expert.view(
                    reshape_dims
                )
                
                # Step 5: Accumulate weighted outputs to correct batch positions
                final_expert_output.index_add_(
                    0, original_batch_indices, weighted_output.to(final_expert_output.dtype)
                )
            
            self.expert_successes += 1
            return final_expert_output, routing_info
            
        except torch.cuda.OutOfMemoryError as e:
            self.logger.error("OOM encountered in expert path, propagating...")
            raise e
        except Exception as e:
            self.logger.error(f"Expert path failed: {e}", exc_info=True)
            routing_info = {'error': str(e)}
            self._last_lb_loss = None
            return torch.zeros_like(main_output), routing_info
    
    # ========================================================================
    # Public Methods - Statistics and Monitoring
    # ========================================================================

    def get_stats(self) -> Dict[str, Any]:
        stats = {
            'forward_calls': self.forward_calls.item(),
            'expert_successes': self.expert_successes.item(),
            'success_rate': (
                self.expert_successes / max(self.forward_calls, 1)
            ).item(),
            'fusion': self.fusion_type,
            'my_index': self.my_index,
            'router_stats': self.router.get_usage_statistics()
        }
        if self.fusion_type == "adaptive":
            stats['alpha'] = self.alpha.item()
        return stats


# ============================================================================
# Factory Functions
# ============================================================================

def create_sage_layer(
    main_block: nn.Module,
    router: SageRouter,
    sa_hub: SAHub,
    config: Optional[Dict[str, Any]] = None,
    my_index: Optional[int] = None,
    expert_pool: Optional[nn.ModuleList] = None,
    layer_type: Optional[str] = None,
    stage_idx: Optional[int] = None,
    p3_refinement: Optional[nn.Module] = None,
) -> SageLayer:
    """
    Factory function to create a SAGE layer with configuration.
    
    This provides a convenient way to instantiate SageLayer with default
    values and optional configuration overrides.
    
    Args:
        main_block: Original block to be wrapped
        router: Router module for expert selection
        sa_hub: Shape adapter hub for tensor format conversion
        config: Optional configuration dictionary with keys:
            - 'expert_dropout' (float): Dropout rate (default: 0.0)
        my_index: Optional index in the shared expert pool
        expert_pool: Optional reference to the shared expert pool
    
    Returns:
        Configured SageLayer instance
    """
    if config is None:
        config = {}
    
    return SageLayer(
        main_block=main_block,
        router=router,
        sa_hub=sa_hub,
        expert_dropout=config.get('expert_dropout', 0.0),
        my_index=my_index,
        expert_pool=expert_pool,
        residual_scale=config.get('residual_scale', 0.1),
        fusion_type=config.get('fusion_type', 'residual'),
        adaptive_alpha=config.get('adaptive_alpha', 0.9),
        layer_type=layer_type,
        stage_idx=stage_idx,
        p3_refinement=p3_refinement,
    )

