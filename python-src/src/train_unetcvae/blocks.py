"""
U-Net 基础组件模块
实现 ResNet Block, FiLM-ResNet Block, FiLM Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ResidualBlock(nn.Module):
    """标准 ResNet Block

    Architecture:
    x → Conv3x3 → BN → ReLU → Conv3x3 → BN → (+ shortcut) → ReLU → out
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class FiLMResidualBlock(nn.Module):
    """带 FiLM 条件化的 ResNet Block

    FiLM modulation: γ·h + β
    其中 γ (gamma) 是缩放参数，β (beta) 是偏移参数

    Architecture:
    x → Conv3x3 → BN → ReLU → FiLM(γ,β) → Conv3x3 → BN → (+ shortcut) → ReLU → out
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        condition_dim: int = 256,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.film_gamma = nn.Linear(condition_dim, out_channels)
        self.film_beta = nn.Linear(condition_dim, out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor, style_code: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))

        gamma = self.film_gamma(style_code).view(-1, out.size(1), 1, 1)
        beta = self.film_beta(style_code).view(-1, out.size(1), 1, 1)
        out = gamma * out + beta

        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class FiLMNetwork(nn.Module):
    """条件向量 → Style Code

    将 S/R/C 评分向量映射为高维 style code，用于 FiLM 调制
    """

    def __init__(
        self, condition_dim: int = 3, hidden_dim: int = 256, output_dim: int = 256
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return self.network(condition)


class DownsampleBlock(nn.Module):
    """下采样 Block (可选，用于更激进的下采样)"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 2, 1)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class UpsampleBlock(nn.Module):
    """上采样 Block (使用 ConvTranspose)"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))
