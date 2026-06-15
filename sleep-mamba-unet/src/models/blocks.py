from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = dilation * (kernel_size // 2)
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation, groups=channels),
            nn.Conv1d(channels, channels * 2, 1),
            nn.GLU(dim=1),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class Downsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=2 * stride, stride=stride, padding=stride // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv1d(in_ch + skip_ch, out_ch, 1),
            nn.GELU(),
            ConvBlock(out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))
