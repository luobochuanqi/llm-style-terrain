"""
Residual U-Net cVAE 模型定义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional, List
from .blocks import ResidualBlock, FiLMResidualBlock, FiLMNetwork


class ConditionNormalizer(nn.Module):
    """条件向量归一化层

    对 C_score 应用对数变换，然后标准化
    """

    def __init__(self, condition_dim: int = 3):
        super().__init__()
        self.condition_dim = condition_dim

        self.register_buffer("mean", torch.tensor([2.56, 4.04, 1.05]))
        self.register_buffer("std", torch.tensor([1.68, 1.89, 1.58]))

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        cond = cond.clone()

        cond[:, 2] = torch.log(cond[:, 2] + 1e-5)

        mean = self.mean.clone()
        mean[2] = torch.log(torch.tensor(1.05) + 1e-5)
        std = self.std.clone()
        std[2] = 1.5

        cond = (cond - mean) / (std + 1e-6)

        return cond


class Encoder(nn.Module):
    """U-Net Encoder (下采样路径)

    返回 skip connections 用于 Decoder 拼接

    Architecture:
    Input: (1, 256, 256)
    → Conv2d(1, 64, 3, 1, 1) + BN + ReLU → (64, 256, 256)
    → ResBlock(64, 64) → (64, 256, 256) [skip1]
    → ResBlock(64, 128, stride=2) → (128, 128, 128) [skip2]
    → ResBlock(128, 256, stride=2) → (256, 64, 64) [skip3]
    → ResBlock(256, 512, stride=2) → (512, 32, 32) [skip4]
    → ResBlock(512, 512, stride=2) → (512, 16, 16) [skip5]
    → Bottleneck ResBlock → (512, 8, 8)
    → Flatten → FC → μ, logvar (128)
    """

    def __init__(self, latent_dim: int = 128, channels: tuple = (64, 128, 256, 512)):
        super().__init__()
        self.latent_dim = latent_dim
        self.channels = channels

        self.input_conv = nn.Sequential(
            nn.Conv2d(1, channels[0], 3, 1, 1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        self.enc1 = ResidualBlock(channels[0], channels[0])
        self.enc2 = ResidualBlock(channels[0], channels[1], stride=2)
        self.enc3 = ResidualBlock(channels[1], channels[2], stride=2)
        self.enc4 = ResidualBlock(channels[2], channels[3], stride=2)

        self.bottleneck = ResidualBlock(channels[3], channels[3], stride=2)

        bottleneck_size = channels[3] * 16 * 16
        self.fc_mu = nn.Linear(bottleneck_size, latent_dim)
        self.fc_logvar = nn.Linear(bottleneck_size, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_conv(x)

        h = self.enc1(h)
        h = self.enc2(h)
        h = self.enc3(h)
        h = self.enc4(h)

        h = self.bottleneck(h)

        h = h.view(h.size(0), -1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def get_skip_connections(self, x: torch.Tensor) -> List[torch.Tensor]:
        """获取所有 skip connections

        Returns:
            [input, enc1_out, enc2_out, enc3_out, enc4_out]
        """
        skips = []

        h = self.input_conv(x)
        skips.append(h)

        h = self.enc1(h)
        skips.append(h)

        h = self.enc2(h)
        skips.append(h)

        h = self.enc3(h)
        skips.append(h)

        h = self.enc4(h)
        skips.append(h)

        return skips


class Decoder(nn.Module):
    """U-Net Decoder (上采样路径) 带 FiLM 条件化和 Skip Connection

    Architecture:
    Input: z (128) + condition (3)
    → Style Code (256) via FiLMNetwork
    → Linear(128, 512*16*16) → Reshape(512, 16, 16)
    → FiLM-ResBlock(512, 512) → (512, 16, 16)
    → Upsample → (512, 32, 32)
    → Concat Skip → Conv(1024, 256) → FiLM-ResBlock(256, 256) → (256, 32, 32)
    → Upsample → (256, 64, 64)
    → Concat Skip → Conv(512, 128) → FiLM-ResBlock(128, 128) → (128, 64, 64)
    → Upsample → (128, 128, 128)
    → Concat Skip → Conv(256, 64) → FiLM-ResBlock(64, 64) → (64, 128, 128)
    → Upsample → (64, 256, 256)
    → Concat Skip → Conv(128, 64) → FiLM-ResBlock(64, 64) → (64, 256, 256)
    → Conv2d(64, 1, 3, 1, 1) + Sigmoid → (1, 256, 256)
    """

    def __init__(
        self,
        latent_dim: int = 128,
        condition_dim: int = 3,
        film_hidden_dim: int = 256,
        channels: tuple = (64, 128, 256, 512),
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.channels = channels

        self.film_network = FiLMNetwork(condition_dim, film_hidden_dim, film_hidden_dim)

        bottleneck_size = channels[3] * 16 * 16
        self.fc_z = nn.Linear(latent_dim, bottleneck_size)

        self.dec1 = FiLMResidualBlock(
            channels[3], channels[3], stride=1, condition_dim=film_hidden_dim
        )

        self.upsample1 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.dec2_conv = nn.Conv2d(channels[3] * 2, channels[2], 1)
        self.dec2 = FiLMResidualBlock(
            channels[2], channels[2], stride=1, condition_dim=film_hidden_dim
        )

        self.upsample2 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.dec3_conv = nn.Conv2d(channels[2] * 2, channels[1], 1)
        self.dec3 = FiLMResidualBlock(
            channels[1], channels[1], stride=1, condition_dim=film_hidden_dim
        )

        self.upsample3 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.dec4_conv = nn.Conv2d(channels[1] * 2, channels[0], 1)
        self.dec4 = FiLMResidualBlock(
            channels[0], channels[0], stride=1, condition_dim=film_hidden_dim
        )

        self.upsample4 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        self.dec5_conv = nn.Conv2d(channels[0] * 2, channels[0], 1)
        self.dec5 = FiLMResidualBlock(
            channels[0], channels[0], stride=1, condition_dim=film_hidden_dim
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(channels[0], 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z: torch.Tensor,
        condition: torch.Tensor,
        skip_connections: List[torch.Tensor],
    ) -> torch.Tensor:
        style = self.film_network(condition)

        h = self.fc_z(z)
        h = h.view(-1, self.channels[3], 16, 16)

        h = self.dec1(h, style)

        h = self.upsample1(h)
        h = torch.cat([h, skip_connections[4]], dim=1)
        h = self.dec2_conv(h)
        h = self.dec2(h, style)

        h = self.upsample2(h)
        h = torch.cat([h, skip_connections[3]], dim=1)
        h = self.dec3_conv(h)
        h = self.dec3(h, style)

        h = self.upsample3(h)
        h = torch.cat([h, skip_connections[2]], dim=1)
        h = self.dec4_conv(h)
        h = self.dec4(h, style)

        h = self.upsample4(h)
        h = torch.cat([h, skip_connections[1]], dim=1)
        h = self.dec5_conv(h)
        h = self.dec5(h, style)

        out = self.output_conv(h)

        return out


class UNetcVAE(nn.Module):
    """Residual U-Net cVAE 完整模型"""

    def __init__(
        self,
        latent_dim: int = 128,
        condition_dim: int = 3,
        film_hidden_dim: int = 256,
        channels: tuple = (64, 128, 256, 512),
        beta: float = 1.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.channels = channels

        self.encoder = Encoder(latent_dim, channels)
        self.decoder = Decoder(latent_dim, condition_dim, film_hidden_dim, channels)
        self.condition_norm = ConditionNormalizer(condition_dim)

        self.optimizer: Optional[torch.optim.Optimizer] = None

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z

    def decode(
        self,
        z: torch.Tensor,
        condition: torch.Tensor,
        skip_connections: List[torch.Tensor],
    ) -> torch.Tensor:
        condition_norm = self.condition_norm(condition)
        return self.decoder(z, condition_norm, skip_connections)

    @torch.no_grad()
    def generate(
        self,
        condition: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        dummy_image: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """从条件向量生成地形（无需真实输入图像）

        Args:
            condition: 条件向量 (batch, 3)
            z: 隐向量 (可选，如果为 None 则随机采样)
            dummy_image: 用于获取 skip connections 的虚拟图像 (可选)
            seed: 随机种子 (可选，用于复现结果)

        Returns:
            生成的地形 (batch, 1, 256, 256)
        """
        self.eval()

        if seed is not None:
            torch.manual_seed(seed)

        if z is None:
            z = torch.randn(condition.size(0), self.latent_dim, device=condition.device)

        if dummy_image is None:
            # 使用随机噪声而不是全零，因为 Encoder 期望有信号的输入
            # 全零会导致 skip connections 无有效信息，生成结果异常
            dummy_image = (
                torch.randn(condition.size(0), 1, 256, 256, device=condition.device)
                * 0.5
            )

        skip_connections = self.encoder.get_skip_connections(dummy_image)
        return self.decode(z, condition, skip_connections)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        mu, logvar, z = self.encode(x)

        skip_connections = self.encoder.get_skip_connections(x)

        recon = self.decode(z, condition, skip_connections)

        recon_loss = F.mse_loss(recon, x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        loss_total = recon_loss + self.beta * kl_loss

        return recon, {
            "loss_total": loss_total,
            "loss_recon": recon_loss,
            "loss_kl": kl_loss,
            "beta": self.beta,
        }

    def configure_optimizers(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
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

    def load_checkpoint(self, path: str, device: torch.device) -> int:
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
    channels: tuple = (64, 128, 256, 512),
    beta: float = 1.0,
) -> UNetcVAE:
    """创建模型的便捷函数"""
    return UNetcVAE(
        latent_dim=latent_dim,
        condition_dim=condition_dim,
        film_hidden_dim=film_hidden_dim,
        channels=channels,
        beta=beta,
    )
