from __future__ import annotations

import torch
from torch import nn


class MambaLiteBlock(nn.Module):
    """A pure PyTorch sequence mixer inspired by gated state-space blocks."""

    def __init__(self, dim: int, kernel_size: int = 7, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        inner = dim * expansion
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, inner * 2)
        self.dwconv = nn.Conv1d(inner, inner, kernel_size, padding=kernel_size // 2, groups=inner)
        self.ema_gate = nn.Sequential(nn.Linear(inner, inner), nn.Sigmoid())
        self.out_proj = nn.Linear(inner, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        v, gate = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        v = self.dwconv(v.transpose(1, 2)).transpose(1, 2)
        alpha = self.ema_gate(v)
        mixed = torch.cumsum((1 - alpha) * v, dim=1) / torch.arange(1, x.shape[1] + 1, device=x.device).view(1, -1, 1)
        y = torch.nn.functional.silu(gate) * (alpha * v + (1 - alpha) * mixed)
        return residual + self.drop(self.out_proj(y))
