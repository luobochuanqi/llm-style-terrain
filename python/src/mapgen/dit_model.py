"""
Diffusion Transformer（DiT）模型定义
实现基于 Transformer 的扩散模型骨干网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class DiTConfig:
    """DiT 模型配置"""

    # 图像配置
    image_size: int = 64  # 输入图像尺寸（假设为正方形）
    in_channels: int = 3  # 输入通道数
    out_channels: int = 3  # 输出通道数（预测噪声）

    # Transformer 配置
    hidden_size: int = 768  # 隐藏层维度
    num_layers: int = 12  # Transformer 块数量
    num_attention_heads: int = 12  # 注意力头数
    mlp_ratio: float = 4.0  # MLP 隐藏层维度比例

    # 位置嵌入
    patch_size: int = 4  # 图像分块大小
    max_positions: int = 1024  # 最大位置数

    # 文本条件
    text_embedding_dim: int = 768  # CLIP 文本嵌入维度

    # Dropout
    dropout: float = 0.1


class PatchEmbed(nn.Module):
    """将图像分割为 patch 并嵌入到高维空间"""

    def __init__(
        self, image_size: int, patch_size: int, in_channels: int, hidden_size: int
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像，shape: (batch, in_channels, H, W)

        Returns:
            patch 嵌入，shape: (batch, num_patches, hidden_size)
        """
        x = self.proj(x)  # (batch, hidden_size, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (batch, num_patches, hidden_size)
        return x


class PositionalEmbedding(nn.Module):
    """可学习的位置嵌入"""

    def __init__(self, max_positions: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(max_positions, hidden_size)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices: 位置索引，shape: (num_positions,)

        Returns:
            位置嵌入，shape: (num_positions, hidden_size)
        """
        return self.embedding(indices)


class TransformerBlock(nn.Module):
    """单个 Transformer 块，包含自注意力和 MLP"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        # 自注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # MLP
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(dropout),
        )

        # 交叉注意力（用于文本条件）
        self.norm_cross = nn.LayerNorm(hidden_size)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        text_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: 输入特征，shape: (batch, seq_len, hidden_size)
            text_embeddings: 文本嵌入，shape: (batch, text_len, hidden_size)

        Returns:
            输出特征，shape: (batch, seq_len, hidden_size)
        """
        # 自注意力
        normed = self.norm1(x)
        attn_output, _ = self.attention(normed, normed, normed, need_weights=False)
        x = x + attn_output

        # 交叉注意力（如果有文本条件）
        if text_embeddings is not None:
            normed = self.norm_cross(x)
            cross_attn_output, _ = self.cross_attention(
                normed, text_embeddings, text_embeddings, need_weights=False
            )
            x = x + cross_attn_output

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


class DiTModel(nn.Module):
    """
    Diffusion Transformer 模型

    输入：带噪图像 + 时间步嵌入 + 文本嵌入
    输出：预测的噪声
    """

    def __init__(self, config: Optional[DiTConfig] = None):
        super().__init__()
        self.config = config or DiTConfig()

        # Patch 嵌入
        self.patch_embed = PatchEmbed(
            image_size=self.config.image_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            hidden_size=self.config.hidden_size,
        )

        # 位置嵌入
        self.pos_embed = PositionalEmbedding(
            max_positions=self.config.max_positions,
            hidden_size=self.config.hidden_size,
        )

        # 时间步嵌入（使用正弦位置编码 + MLP）
        self.time_embed = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size * 4),
            nn.SiLU(),
            nn.Linear(self.config.hidden_size * 4, self.config.hidden_size),
        )

        # 文本嵌入投影（将 CLIP 嵌入投影到隐藏维度）
        self.text_proj = nn.Linear(
            self.config.text_embedding_dim, self.config.hidden_size
        )

        # Transformer 块
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=self.config.hidden_size,
                    num_heads=self.config.num_attention_heads,
                    mlp_ratio=self.config.mlp_ratio,
                    dropout=self.config.dropout,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        # 输出投影（从隐藏维度投影回 patch 维度）
        self.norm_out = nn.LayerNorm(self.config.hidden_size)
        self.out_proj = nn.Linear(
            self.config.hidden_size,
            self.config.patch_size**2 * self.config.out_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播：预测噪声

        Args:
            x: 带噪图像，shape: (batch, in_channels, H, W)
            t: 时间步，shape: (batch_size,)
            text_embeddings: CLIP 文本嵌入，shape: (batch, text_len, text_dim)

        Returns:
            预测的噪声，shape: (batch, out_channels, H, W)
        """
        batch_size = x.shape[0]
        device = x.device

        # Patch 嵌入
        x = self.patch_embed(x)  # (batch, num_patches, hidden_size)

        # 位置嵌入
        num_patches = self.patch_embed.num_patches
        pos_indices = (
            torch.arange(num_patches, device=device).unsqueeze(0).expand(batch_size, -1)
        )
        x = x + self.pos_embed(pos_indices)

        # 时间步嵌入
        time_embed = self.time_embed(
            t.float().unsqueeze(-1).expand(-1, self.config.hidden_size)
        )
        x = x + time_embed.unsqueeze(1)

        # 文本嵌入投影
        if text_embeddings is not None:
            text_embed = self.text_proj(
                text_embeddings
            )  # (batch, text_len, hidden_size)
        else:
            text_embed = None

        # Transformer 块
        for block in self.transformer_blocks:
            x = block(x, text_embed)

        # 输出投影
        x = self.norm_out(x)
        x = self.out_proj(x)  # (batch, num_patches, patch_size^2 * out_channels)

        # 重构为图像形状
        patch_size = self.config.patch_size
        h = w = self.config.image_size // patch_size
        x = x.reshape(
            batch_size, h, w, patch_size * patch_size * self.config.out_channels
        )
        x = x.permute(0, 3, 1, 2)  # (batch, out_channels, h, w)
        x = F.pixel_shuffle(x, patch_size)  # (batch, out_channels, H, W)

        return x


def create_dit_model(
    image_size: int = 64,
    in_channels: int = 3,
    out_channels: int = 3,
    text_embedding_dim: int = 768,
    hidden_size: int = 768,
    num_layers: int = 12,
    num_heads: int = 12,
) -> DiTModel:
    """
    创建 DiT 模型的便捷函数

    Args:
        image_size: 输入图像尺寸
        in_channels: 输入通道数
        out_channels: 输出通道数
        text_embedding_dim: CLIP 文本嵌入维度
        hidden_size: 隐藏层维度
        num_layers: Transformer 块数量
        num_heads: 注意力头数

    Returns:
        DiTModel 实例
    """
    config = DiTConfig(
        image_size=image_size,
        in_channels=in_channels,
        out_channels=out_channels,
        text_embedding_dim=text_embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_heads,
    )
    return DiTModel(config)
