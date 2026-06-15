from __future__ import annotations

import torch
from torch import nn

from .blocks import ConvBlock, Downsample, Upsample
from .conformer import ConformerBlock
from .mamba_lite import MambaLiteBlock


class SleepMambaUNet(nn.Module):
    def __init__(self, in_channels: int, base_dim: int = 96, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        d = base_dim
        self.proj = nn.Conv1d(in_channels, d, 1)
        self.stem = ConvBlock(d, dropout=dropout)
        self.enc0 = nn.Sequential(ConvBlock(d, dropout=dropout), _Mamba1d(d, dropout))
        self.down1 = Downsample(d, d * 2)
        self.enc1_conv = ConvBlock(d * 2, dilation=2, dropout=dropout)
        self.enc1_conf = _Conformer1d(d * 2, num_heads, dropout)
        self.down2 = Downsample(d * 2, d * 4)
        self.enc2 = nn.Sequential(_Mamba1d(d * 4, dropout), ConvBlock(d * 4, dilation=4, dropout=dropout))
        self.down3 = Downsample(d * 4, d * 4)
        self.bottleneck = _Conformer1d(d * 4, num_heads, dropout)
        self.up2 = Upsample(d * 4, d * 4, d * 2)
        self.up1 = Upsample(d * 2, d * 2, d)
        self.up0 = Upsample(d, d, d)
        self.head = nn.Conv1d(d, 4, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        orig_t = x.shape[1]
        x = x.transpose(1, 2)
        s0 = self.stem(self.proj(x))
        e0 = self.enc0(s0)
        e1 = self.enc1_conf(self.enc1_conv(self.down1(e0)))
        e2 = self.enc2(self.down2(e1))
        b = self.bottleneck(self.down3(e2))
        y = self.up2(b, e2)
        y = self.up1(y, e1)
        y = self.up0(y, e0)
        logits = self.head(y)[..., :orig_t].transpose(1, 2)
        return {
            "onset": logits[..., 0],
            "wakeup": logits[..., 1],
            "sleep": logits[..., 2],
            "invalid": logits[..., 3],
        }


class _Mamba1d(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = MambaLiteBlock(dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x.transpose(1, 2)).transpose(1, 2)


class _Conformer1d(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.block = ConformerBlock(dim, heads, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x.transpose(1, 2)).transpose(1, 2)
