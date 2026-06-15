from __future__ import annotations

import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, kernel_size: int = 15, dropout: float = 0.1):
        super().__init__()
        self.ff1 = FeedForward(dim, dropout=dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.conv_norm = nn.LayerNorm(dim)
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim * 2, 1),
            nn.GLU(dim=1),
            nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 1),
            nn.Dropout(dropout),
        )
        self.ff2 = FeedForward(dim, dropout=dropout)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        a, _ = self.attn(self.attn_norm(x), self.attn_norm(x), self.attn_norm(x), key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        c = self.conv(self.conv_norm(x).transpose(1, 2)).transpose(1, 2)
        x = x + c
        x = x + 0.5 * self.ff2(x)
        return self.out_norm(x)
