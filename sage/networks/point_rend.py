"""
PointRend: Image Segmentation as Rendering (Kirillov et al., CVPR 2020)
Adapted for SAGE-Lite Decoder Refinement.

This module provides:
1. `point_sample`: Bilinear feature/logit sampling at continuous coordinates in [0, 1].
2. `sample_training_points`: Uncertainty-biased point sampling during training.
3. `get_uncertain_point_coords_on_grid`: Top-K uncertain point extraction during inference.
4. `PointRendHead`: Lightweight 3-layer MLP operating on concatenated fine features + coarse logits.
5. `subdivide_and_refine`: 2-step adaptive subdivision inference (112 -> 224 -> 448).
6. `PointRendLoss`: Decoupled loss (Coarse CrackBinaryLoss + Point-level BCEWithLogitsLoss).
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def point_sample(features: torch.Tensor, point_coords: torch.Tensor) -> torch.Tensor:
    """
    Sample 2D feature map at continuous point coordinates.

    Args:
        features (torch.Tensor): Feature map of shape (B, C, H, W).
        point_coords (torch.Tensor): Coordinates in [0, 1] range of shape (B, N, 2),
            where point_coords[..., 0] is x (width/col) and point_coords[..., 1] is y (height/row).

    Returns:
        torch.Tensor: Sampled features of shape (B, C, N).
    """
    B, N, _ = point_coords.shape
    # Map [0, 1] -> [-1, 1] for F.grid_sample
    grid = point_coords.view(B, N, 1, 2) * 2.0 - 1.0
    output = F.grid_sample(
        features,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return output.squeeze(-1)


def sample_training_points(
    coarse_logits: torch.Tensor,
    num_points: int = 2048,
    oversample_ratio: int = 3,
    importance_ratio: float = 0.75,
) -> torch.Tensor:
    """
    Sample training points using uncertainty-biased oversampling.

    Args:
        coarse_logits (torch.Tensor): Coarse prediction logits (B, 1, H, W).
        num_points (int): Total points to sample per image (default: 2048).
        oversample_ratio (int): Candidate point multiplier (default: 3 => 3 * 2048 = 6144).
        importance_ratio (float): Fraction of points selected from highest uncertainty (default: 0.75).

    Returns:
        torch.Tensor: Sampled point coordinates in [0, 1] range of shape (B, num_points, 2).
    """
    B = coarse_logits.shape[0]
    device = coarse_logits.device
    num_candidates = int(oversample_ratio * num_points)
    num_imp = int(importance_ratio * num_points)
    num_rand = num_points - num_imp

    # 1. Generate random candidate coordinates uniformly in [0, 1]
    candidate_coords = torch.rand(B, num_candidates, 2, device=device)

    # 2. Sample coarse logits at candidate coordinates
    candidate_logits = point_sample(coarse_logits, candidate_coords)  # (B, 1, num_candidates)

    # 3. Calculate uncertainty: for binary segmentation, highest when |p - 0.5| is lowest
    candidate_probs = torch.sigmoid(candidate_logits)
    candidate_uncertainty = -torch.abs(candidate_probs - 0.5).squeeze(1)  # (B, num_candidates)

    # 4. Top-K uncertain points
    _, topk_idx = torch.topk(candidate_uncertainty, k=num_imp, dim=1)  # (B, num_imp)
    topk_coords = torch.gather(
        candidate_coords,
        dim=1,
        index=topk_idx.unsqueeze(-1).expand(-1, -1, 2),
    )

    # 5. Remaining points sampled uniformly at random
    rand_coords = torch.rand(B, num_rand, 2, device=device)

    return torch.cat([topk_coords, rand_coords], dim=1)


def get_uncertain_point_coords_on_grid(
    coarse_logits: torch.Tensor,
    num_points: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find top-K most uncertain grid locations for subdivision inference.

    Args:
        coarse_logits (torch.Tensor): Logits map (B, 1, H, W).
        num_points (int): Maximum points to refine.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - coords: Point coordinates in [0, 1] range of shape (B, K, 2).
            - flat_indices: 1D spatial indices in [0, H*W - 1] of shape (B, K).
    """
    B, _, H, W = coarse_logits.shape
    total_grid = H * W
    k = min(num_points, total_grid)

    uncertainty = -torch.abs(torch.sigmoid(coarse_logits) - 0.5).view(B, total_grid)
    _, flat_indices = torch.topk(uncertainty, k=k, dim=1)  # (B, k)

    # Convert 1D indices to row, col
    r = flat_indices // W
    c = flat_indices % W

    x = (c.float() + 0.5) / float(W)
    y = (r.float() + 0.5) / float(H)
    coords = torch.stack([x, y], dim=-1)  # (B, k, 2)

    return coords, flat_indices


class PointRendHead(nn.Module):
    """
    PointRend Multi-Layer Perceptron (MLP) head.

    Refines a set of points by concatenating:
      - fine_features: Bilinearly sampled from Decoder's finest feature map (B, C, N).
      - coarse_logits: Bilinearly sampled from coarse prediction (B, num_classes, N).
    Passes the concatenated (C + num_classes) vector through a 3-layer 1x1 Conv1d MLP.

    Args:
        in_channels (int): Channels in fine-grained feature map (default: 48).
        num_classes (int): Segmentation output channels (default: 1).
        mid_channels (int): Hidden dimension of MLP (default: 128).
    """

    def __init__(
        self,
        in_channels: int = 48,
        num_classes: int = 1,
        mid_channels: int = 128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.mid_channels = mid_channels

        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels + num_classes, mid_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, mid_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, num_classes, kernel_size=1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        fine_features: torch.Tensor,
        coarse_logits: torch.Tensor,
        point_coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of Point Head.

        Args:
            fine_features (torch.Tensor): Feature map of shape (B, C, H_f, W_f).
            coarse_logits (torch.Tensor): Logits of shape (B, num_classes, H_c, W_c).
            point_coords (torch.Tensor): Point coordinates in [0, 1] of shape (B, N, 2).

        Returns:
            torch.Tensor: Refined point logits of shape (B, num_classes, N).
        """
        f_point = point_sample(fine_features, point_coords)  # (B, C, N)
        z_point = point_sample(coarse_logits, point_coords)  # (B, num_classes, N)
        h = torch.cat([f_point, z_point], dim=1)  # (B, C + num_classes, N)
        return self.mlp(h)


def subdivide_and_refine(
    fine_features: torch.Tensor,
    coarse_logits: torch.Tensor,
    point_head: PointRendHead,
    target_size: Tuple[int, int] = (448, 448),
    num_subdivision_points: int = 8192,
) -> torch.Tensor:
    """
    Subdivision inference: progressively upsamples coarse logits and refines
    the most uncertain points at each resolution level.

    Args:
        fine_features (torch.Tensor): (B, 48, 112, 112)
        coarse_logits (torch.Tensor): (B, 1, 112, 112)
        point_head (PointRendHead): Trained MLP module.
        target_size (Tuple[int, int]): Final image size (448, 448).
        num_subdivision_points (int): Max points to refine per subdivision step (default: 8192).

    Returns:
        torch.Tensor: Refined full-resolution logits (B, 1, 448, 448).
    """
    B, num_classes, _, _ = coarse_logits.shape
    current = coarse_logits

    while current.shape[2] < target_size[0] or current.shape[3] < target_size[1]:
        new_H = min(current.shape[2] * 2, target_size[0])
        new_W = min(current.shape[3] * 2, target_size[1])

        # 1. Bilinear upsample current prediction 2x
        current = F.interpolate(
            current,
            size=(new_H, new_W),
            mode="bilinear",
            align_corners=False,
        )

        # 2. Find most uncertain points on new grid
        coords, flat_indices = get_uncertain_point_coords_on_grid(
            current,
            num_points=num_subdivision_points,
        )

        # 3. Refine with Point Head
        refined = point_head(fine_features, current, coords)  # (B, num_classes, k)

        # 4. Scatter update refined points back into grid
        current_flat = current.view(B, num_classes, new_H * new_W).clone()
        current_flat.scatter_(
            dim=2,
            index=flat_indices.unsqueeze(1).expand(-1, num_classes, -1),
            src=refined,
        )
        current = current_flat.view(B, num_classes, new_H, new_W)

    return current


class PointRendLoss(nn.Module):
    """
    Decoupled Two-Tier Loss for PointRend:
      L_total = L_coarse + lambda_point * L_point

    Where:
      - L_coarse: Standard CrackBinaryLoss (BCE 1.0 + SoftDice 1.5) on coarse 112x112 map.
      - L_point: BCEWithLogitsLoss on N sampled boundary/uncertain points.

    Args:
        base_criterion (nn.Module): Base segmentation criterion (e.g. CrackBinaryLoss).
        point_loss_weight (float): Weight for point loss (default: 1.0).
    """

    def __init__(
        self,
        base_criterion: nn.Module,
        point_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.base_criterion = base_criterion
        self.point_loss_weight = float(point_loss_weight)
        self.bce_point = nn.BCEWithLogitsLoss()

    def forward(
        self,
        logits: Union[torch.Tensor, Dict[str, torch.Tensor]],
        targets: torch.Tensor,
        point_logits: Optional[torch.Tensor] = None,
        point_coords: Optional[torch.Tensor] = None,
        coarse_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute decoupled coarse + point loss during training,
        or delegate directly to base_criterion during validation/testing.

        Args:
            logits: Refined logits tensor (B, 1, H, W) OR dict with training point info.
            targets: Ground truth masks (B, 1, 448, 448).
            point_logits: Optional point prediction logits (B, 1, N).
            point_coords: Optional point coordinates in [0, 1] (B, N, 2).
            coarse_logits: Optional coarse prediction logits (B, 1, 112, 112).

        Returns:
            torch.Tensor: Total loss scalar tensor.
        """
        # If input is a dictionary containing point training info
        if isinstance(logits, dict):
            coarse_logits = logits.get("coarse_logits", coarse_logits)
            point_logits = logits.get("point_logits", point_logits)
            point_coords = logits.get("point_coords", point_coords)
            full_logits = logits.get("logits", None)
        else:
            full_logits = logits

        # If point supervision is provided (Training phase)
        if point_logits is not None and point_coords is not None and coarse_logits is not None:
            if targets.dim() == 3:
                targets = targets.unsqueeze(1)
            targets = targets.float()

            # 1. Coarse Loss on downsampled targets matching coarse resolution (112x112)
            targets_coarse = F.interpolate(
                targets,
                size=coarse_logits.shape[2:],
                mode="nearest",
            )
            coarse_loss = self.base_criterion(coarse_logits, targets_coarse)

            # 2. Point Loss: sample continuous ground truth at point coordinates
            point_targets = point_sample(targets, point_coords)  # (B, 1, N)
            point_loss = self.bce_point(point_logits, point_targets)

            total_loss = coarse_loss + self.point_loss_weight * point_loss
            return total_loss

        # Evaluation / Standard phase: compute base criterion on full resolution tensor
        return self.base_criterion(full_logits, targets)

