"""
Proposal 3 (P3) High-Resolution Refinement Modules:
- ASDWRefinement (Run C): Anisotropic Strip Depthwise Refinement (1x7 + 7x1 + 3x3)
- GenericRefinement (Run B): Generic Isotropic Depthwise Refinement (3x 3x3)

Invariants strictly preserved:
1. bias=False on all DWConv and PWConv layers.
2. Padding: (0, 3) for 1x7, (3, 0) for 7x1, 1 for 3x3 (exact spatial preservation).
3. Gamma: learnable nn.Parameter initialized to 0.01 (near-identity residual initialization).
4. No normalization layers (no BN, no LN, no GN).
5. Exact parameter counts:
   - S0 (C=48): Run C = 8,017; Run B = 8,209
   - S1 (C=96): Run C = 29,857; Run B = 30,241
   - Total (S0+S1): Run C = 37,874; Run B = 38,450 (Diff = 576 params)
"""

import torch
import torch.nn as nn


class ASDWRefinement(nn.Module):
    """
    Proposal 3 (Run C) Anisotropic Strip Depthwise Refinement Module.
    
    Dataflow:
        H = DWConv_{1x7}(x)  [in=C, out=C, k=(1, 7), p=(0, 3), groups=C, bias=False]
        V = DWConv_{7x1}(x)  [in=C, out=C, k=(7, 1), p=(3, 0), groups=C, bias=False]
        L = DWConv_{3x3}(x)  [in=C, out=C, k=(3, 3), p=1,      groups=C, bias=False]
        C_concat = Concat([H, V, L], dim=1)  # Tensor: (B, 3C, H, W)
        C_act = GELU(C_concat)
        F = PWConv_{1x1}(C_act)              # Tensor: (B, C, H, W), bias=False
        out = x + gamma * F                  # gamma learnable scalar (init=0.01)
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.dw_h = nn.Conv2d(
            channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels, bias=False
        )
        self.dw_v = nn.Conv2d(
            channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels, bias=False
        )
        self.dw_l = nn.Conv2d(
            channels, channels, kernel_size=(3, 3), padding=1, groups=channels, bias=False
        )
        self.act = nn.GELU()
        self.pw = nn.Conv2d(3 * channels, channels, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.tensor(0.01))
        self._init_weights()

    def _init_weights(self):
        for m in [self.dw_h, self.dw_v, self.dw_l]:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.pw.weight, mode="fan_out", nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dw_h(x)
        v = self.dw_v(x)
        l = self.dw_l(x)
        c = torch.cat([h, v, l], dim=1)
        c_act = self.act(c)
        f = self.pw(c_act)
        return x + self.gamma * f


class GenericRefinement(nn.Module):
    """
    Proposal 3 (Run B) Generic Isotropic Depthwise Refinement Module.
    Near-Matched-Capacity Control for Run C.
    
    Dataflow:
        H1 = DWConv_{3x3}(x)  [in=C, out=C, k=(3, 3), p=1, groups=C, bias=False]
        H2 = DWConv_{3x3}(x)  [in=C, out=C, k=(3, 3), p=1, groups=C, bias=False]
        H3 = DWConv_{3x3}(x)  [in=C, out=C, k=(3, 3), p=1, groups=C, bias=False]
        C_concat = Concat([H1, H2, H3], dim=1)  # Tensor: (B, 3C, H, W)
        C_act = GELU(C_concat)
        F = PWConv_{1x1}(C_act)                 # Tensor: (B, C, H, W), bias=False
        out = x + gamma * F                     # gamma learnable scalar (init=0.01)
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.dw_1 = nn.Conv2d(
            channels, channels, kernel_size=(3, 3), padding=1, groups=channels, bias=False
        )
        self.dw_2 = nn.Conv2d(
            channels, channels, kernel_size=(3, 3), padding=1, groups=channels, bias=False
        )
        self.dw_3 = nn.Conv2d(
            channels, channels, kernel_size=(3, 3), padding=1, groups=channels, bias=False
        )
        self.act = nn.GELU()
        self.pw = nn.Conv2d(3 * channels, channels, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.tensor(0.01))
        self._init_weights()

    def _init_weights(self):
        for m in [self.dw_1, self.dw_2, self.dw_3]:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        nn.init.kaiming_normal_(self.pw.weight, mode="fan_out", nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.dw_1(x)
        h2 = self.dw_2(x)
        h3 = self.dw_3(x)
        c = torch.cat([h1, h2, h3], dim=1)
        c_act = self.act(c)
        f = self.pw(c_act)
        return x + self.gamma * f
