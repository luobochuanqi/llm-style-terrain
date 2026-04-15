"""
cVAE 模型模块
实现带 FiLM 条件化的变分自编码器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
from dataclasses import dataclass


class ConditionNormalizer(nn.Module):
    """条件向量归一化层

    对 C_score 应用对数变换，然后标准化
    """

    def __init__(self, condition_dim: int = 3):
        super().__init__()
        self.condition_dim = condition_dim

        # 从训练集统计的均值和标准差
        # S_score: mean=2.56, std=1.68
        # R_score: mean=4.04, std=1.89
        # C_score: mean=1.05, std=1.58 (严重偏态)
        self.register_buffer("mean", torch.tensor([2.56, 4.04, 1.05]))
        self.register_buffer("std", torch.tensor([1.68, 1.89, 1.58]))

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """归一化条件向量

        Args:
            cond: 原始条件向量 (batch, 3)

        Returns:
            归一化后的条件向量
        """
        cond = cond.clone()

        # 对 C_score 应用对数变换
        cond[:, 2] = torch.log(cond[:, 2] + 1e-5)

        # 重新计算对数变换后的 C_score 统计量
        mean = self.mean.clone()
        mean[2] = torch.log(torch.tensor(1.05) + 1e-5)
        std = self.std.clone()
        std[2] = 1.5  # 近似值

        # 标准化
        cond = (cond - mean) / (std + 1e-6)

        return cond


class Encoder(nn.Module):
    """cVAE 编码器

    将 256x256 图像压缩为 128 维隐空间向量

    Architecture:
    Input: (1, 256, 256)
    → Conv2d(1, 32, 4, 2, 1) + BN + ReLU → (32, 128, 128)
    → Conv2d(32, 64, 4, 2, 1) + BN + ReLU → (64, 64, 64)
    → Conv2d(64, 128, 4, 2, 1) + BN + ReLU → (128, 32, 32)
    → Conv2d(128, 256, 4, 2, 1) + BN + ReLU → (256, 16, 16)
    → Conv2d(256, 512, 4, 2, 1) + BN + ReLU → (512, 8, 8)
    → Flatten → Linear(512*8*8, 256)
    → Output: mu (128), logvar (128)
    """

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        # 卷积编码器
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # 压缩到 latent dim
        self.fc_mu = nn.Linear(512 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(512 * 8 * 8, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入图像 (batch, 1, H, W)

        Returns:
            mu: 隐空间均值 (batch, latent_dim)
            logvar: 隐空间对数方差 (batch, latent_dim)
        """
        # 卷积编码
        h = self.conv_layers(x)  # (batch, 512, 8, 8)
        h = h.view(h.size(0), -1)  # (batch, 512*8*8)

        # 预测 mu 和 logvar
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


class FiLMDecoder(nn.Module):
    """带 FiLM 条件化的解码器

    使用 Feature-wise Linear Modulation 将条件向量融入解码过程

    Architecture:
    Input: z (128) + condition (3)
    → 投影 condition → 256D style code
    → Linear(128, 512*8*8) → Reshape(512, 8, 8)
    → ConvTranspose2d(512, 256, 4, 2, 1) + BN + ReLU + FiLM → (256, 16, 16)
    → ConvTranspose2d(256, 128, 4, 2, 1) + BN + ReLU + FiLM → (128, 32, 32)
    → ConvTranspose2d(128, 64, 4, 2, 1) + BN + ReLU + FiLM → (64, 64, 64)
    → ConvTranspose2d(64, 32, 4, 2, 1) + BN + ReLU + FiLM → (32, 128, 128)
    → ConvTranspose2d(32, 1, 4, 2, 1) + Sigmoid → (1, 256, 256)
    """

    def __init__(
        self,
        latent_dim: int = 128,
        condition_dim: int = 3,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # 条件向量投影到 style code
        self.fc_cond = nn.Linear(condition_dim, hidden_dim)

        # 隐向量投影到初始特征图
        self.fc_z = nn.Linear(latent_dim, 512 * 8 * 8)

        # 上采样块
        self.up_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(512, 256, 4, 2, 1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(256, 128, 4, 2, 1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(128, 64, 4, 2, 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(64, 32, 4, 2, 1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                ),
            ]
        )

        # FiLM 参数生成器 (gamma, beta)
        self.film_gammas = nn.ModuleList(
            [nn.Linear(hidden_dim, channels) for channels in [256, 128, 64, 32]]
        )
        self.film_betas = nn.ModuleList(
            [nn.Linear(hidden_dim, channels) for channels in [256, 128, 64, 32]]
        )

        # 输出层
        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z: 隐向量 (batch, latent_dim)
            condition: 条件向量 (batch, condition_dim)

        Returns:
            重建图像 (batch, 1, H, W)
        """
        batch_size = z.size(0)
        device = z.device

        # 生成 style code
        style = self.fc_cond(condition)  # (batch, hidden_dim)

        # 初始特征图
        h = self.fc_z(z)  # (batch, 512*8*8)
        h = h.view(batch_size, 512, 8, 8)

        # FiLM 调制的上采样
        for i, block in enumerate(self.up_blocks):
            h = block(h)

            # FiLM: h = gamma * h + beta
            gamma = self.film_gammas[i](style).view(batch_size, h.size(1), 1, 1)
            beta = self.film_betas[i](style).view(batch_size, h.size(1), 1, 1)
            h = gamma * h + beta

        # 输出
        out = self.output_layer(h)

        return out


class cVAE(nn.Module):
    """条件变分自编码器

    结合 Encoder 和 FiLMDecoder，实现完整的 cVAE 模型
    """

    def __init__(
        self,
        latent_dim: int = 128,
        condition_dim: int = 3,
        film_hidden_dim: int = 256,
        beta: float = 1.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        # 组件
        self.encoder = Encoder(latent_dim)
        self.decoder = FiLMDecoder(latent_dim, condition_dim, film_hidden_dim)
        self.condition_norm = ConditionNormalizer(condition_dim)

        # 优化器
        self.optimizer: Optional[torch.optim.Optimizer] = None

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        编码阶段

        Returns:
            mu: 隐空间均值
            logvar: 隐空间对数方差
            z: 采样得到的隐向量
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z

    def decode(
        self,
        z: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        解码阶段

        Args:
            z: 隐向量
            condition: 原始条件向量 (未归一化)

        Returns:
            重建图像
        """
        # 归一化条件
        condition_norm = self.condition_norm(condition)

        # 解码
        recon = self.decoder(z, condition_norm)

        return recon

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        前向传播

        Args:
            x: 输入图像 (batch, 1, H, W)
            condition: 原始条件向量 (batch, 3) [S, R, C]

        Returns:
            recon: 重建图像
            losses: 损失字典
        """
        # 编码
        mu, logvar, z = self.encode(x)

        # 解码
        recon = self.decode(z, condition)

        # 计算损失
        recon_loss = F.mse_loss(recon, x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # 当前 beta 值
        beta = self.beta

        loss_total = recon_loss + beta * kl_loss

        return recon, {
            "loss_total": loss_total,
            "loss_recon": recon_loss,
            "loss_kl": kl_loss,
            "beta": beta,
        }

    def generate(
        self,
        condition: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        生成模式：从条件向量生成图像

        Args:
            condition: 条件向量 (batch, 3)
            z: 隐向量 (可选，如果为 None 则从标准正态分布采样)

        Returns:
            生成的图像
        """
        if z is None:
            z = torch.randn(condition.size(0), self.latent_dim, device=condition.device)

        return self.decode(z, condition)

    def configure_optimizers(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        """配置优化器"""
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        return self.optimizer

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        optimizer_state: Optional[dict] = None,
    ):
        """保存检查点"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": (
                optimizer_state
                if optimizer_state is not None
                else self.optimizer.state_dict()
            ),
            "latent_dim": self.latent_dim,
            "beta": self.beta,
        }
        torch.save(checkpoint, path)
        print(f"✅ 检查点已保存：{path}")

    def load_checkpoint(self, path: str, device: torch.device):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint["model_state_dict"])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"✅ 已加载检查点：{path} (epoch {checkpoint['epoch']})")
        return checkpoint["epoch"]


def create_model(
    latent_dim: int = 128,
    condition_dim: int = 3,
    film_hidden_dim: int = 256,
    beta: float = 1.0,
) -> cVAE:
    """创建 cVAE 模型的便捷函数"""
    return cVAE(
        latent_dim=latent_dim,
        condition_dim=condition_dim,
        film_hidden_dim=film_hidden_dim,
        beta=beta,
    )
