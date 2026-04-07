"""
地形高度图生成器（HeightMapGen）
从语义布局图生成精确的地形高度场图
使用 DiT + BBDM 架构，配备梯度损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np
from dataclasses import dataclass

from .bbddm_scheduler import BBDMScheduler
from .dit_model import create_dit_model


@dataclass
class HeightMapGenConfig:
    """HeightMapGen 配置"""

    # 模型配置
    image_size: int = 64  # 生成图像尺寸
    hidden_size: int = 512  # DiT 隐藏层维度
    num_layers: int = 8  # Transformer 层数
    num_heads: int = 8  # 注意力头数

    # 输入输出通道
    in_channels: int = 1  # 输入：语义布局图（单通道索引或灰度）
    out_channels: int = 1  # 输出：高度图（单通道灰度）

    # BBDM 配置
    num_train_timesteps: int = 1000
    sigma_min: float = 1e-4
    sigma_max: float = 1.0

    # 文本条件
    text_embedding_dim: int = 768  # CLIP 文本嵌入维度

    # 损失函数权重
    lambda_gradient: float = 0.5  # 梯度损失权重

    # 训练配置
    learning_rate: float = 1e-4
    gradient_clip: float = 1.0

    # 推理配置
    num_inference_timesteps: int = 50  # 推理步数


class GradientLoss(nn.Module):
    """
    梯度损失（Gradient Loss）

    惩罚预测高度图与真实高度图在水平和垂直方向的梯度差异，
    保证地形高程变化连续平滑，避免断崖、尖刺等不自然形态
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        predicted_height: torch.Tensor,
        target_height: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算梯度损失

        Args:
            predicted_height: 预测高度图，shape: (batch, 1, H, W)
            target_height: 真实高度图，shape: (batch, 1, H, W)

        Returns:
            梯度损失值（标量）
        """
        # 水平梯度（Sobel 算子近似）
        kernel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=predicted_height.dtype,
            device=predicted_height.device,
        ).view(1, 1, 3, 3)

        # 垂直梯度
        kernel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=predicted_height.dtype,
            device=predicted_height.device,
        ).view(1, 1, 3, 3)

        # 计算预测高度图的梯度
        grad_x_pred = F.conv2d(predicted_height, kernel_x, padding=1)
        grad_y_pred = F.conv2d(predicted_height, kernel_y, padding=1)

        # 计算真实高度图的梯度
        grad_x_target = F.conv2d(target_height, kernel_x, padding=1)
        grad_y_target = F.conv2d(target_height, kernel_y, padding=1)

        # L1 损失
        loss_x = F.l1_loss(grad_x_pred, grad_x_target)
        loss_y = F.l1_loss(grad_y_pred, grad_y_target)

        return loss_x + loss_y


class HeightMapGen(nn.Module):
    """
    地形高度图生成器

    输入：语义布局图 M_l + CLIP 文本嵌入
    输出：地形高度场图 M_h（单通道灰度图）
    """

    def __init__(self, config: Optional[HeightMapGenConfig] = None):
        super().__init__()
        self.config = config or HeightMapGenConfig()

        # 初始化 DiT 模型
        self.dit = create_dit_model(
            image_size=self.config.image_size,
            in_channels=self.config.in_channels,  # 输入：语义布局图
            out_channels=self.config.out_channels,  # 输出：预测噪声
            text_embedding_dim=self.config.text_embedding_dim,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
        )

        # BBDM 调度器
        self.scheduler = BBDMScheduler(
            num_train_timesteps=self.config.num_train_timesteps,
            sigma_min=self.config.sigma_min,
            sigma_max=self.config.sigma_max,
        )

        # 梯度损失
        self.gradient_loss = GradientLoss()

        # 优化器（在 train_step 中初始化）
        self.optimizer: Optional[torch.optim.Optimizer] = None

    def forward(
        self,
        x: torch.Tensor,
        text_embeddings: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播：预测噪声

        Args:
            x: 输入图像（带噪状态），shape: (batch, in_channels, H, W)
            text_embeddings: CLIP 文本嵌入，shape: (batch, text_len, text_dim)
            t: 时间步，shape: (batch_size,) 或 None（训练时随机采样）

        Returns:
            预测的噪声，shape: (batch, out_channels, H, W)
        """
        batch_size = x.shape[0]
        device = x.device

        # 训练时随机采样时间步
        if t is None:
            t = self.scheduler.sample_timestep(batch_size, device)

        # 通过 DiT 预测噪声
        noise_pred = self.dit(x, t, text_embeddings)

        return noise_pred

    def compute_loss(
        self,
        semantic_layout: torch.Tensor,
        target_height: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        计算训练损失

        Args:
            semantic_layout: 语义布局图，shape: (batch, in_channels, H, W)
            target_height: 真实高度图，shape: (batch, out_channels, H, W)
            text_embeddings: CLIP 文本嵌入，shape: (batch, text_len, text_dim)

        Returns:
            包含各损失项的字典
        """
        batch_size = semantic_layout.shape[0]
        device = semantic_layout.device

        # 采样时间步
        t = self.scheduler.sample_timestep(batch_size, device)

        # 前向扩散：添加噪声
        diffusion_output = self.scheduler.add_noise(
            source_image=semantic_layout,
            target_image=target_height,
            t=t,
        )

        # 预测噪声
        noise_pred = self.forward(diffusion_output.x_t, text_embeddings, t)

        # 基础 BBDM 损失（MSE）
        loss_bbddm = F.mse_loss(noise_pred, diffusion_output.noise)

        # 预测高度图（通过去噪得到）
        t_normalized = t / self.scheduler.num_train_timesteps
        alpha = (1 - t_normalized).view(-1, 1, 1, 1)
        beta = t_normalized.view(-1, 1, 1, 1)
        predicted_height = (
            diffusion_output.x_t - alpha * semantic_layout - noise_pred
        ) / (beta + 1e-6)

        # 梯度损失
        loss_grad = self.gradient_loss(predicted_height, target_height)

        # 总损失
        loss_total = loss_bbddm + self.config.lambda_gradient * loss_grad

        return {
            "loss_total": loss_total,
            "loss_bbddm": loss_bbddm,
            "loss_gradient": loss_grad,
        }

    def train_step(
        self,
        semantic_layout: torch.Tensor,
        target_height: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> Dict[str, float]:
        """
        执行一步训练

        Args:
            semantic_layout: 语义布局图，shape: (batch, in_channels, H, W)
            target_height: 真实高度图，shape: (batch, out_channels, H, W)
            text_embeddings: CLIP 文本嵌入，shape: (batch, text_len, text_dim)

        Returns:
            损失值字典
        """
        # 初始化优化器（如果还未初始化）
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config.learning_rate,
            )

        self.train()
        self.optimizer.zero_grad()

        # 计算损失
        losses = self.compute_loss(semantic_layout, target_height, text_embeddings)

        # 反向传播
        losses["loss_total"].backward()

        # 梯度裁剪
        if self.config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.parameters(),
                self.config.gradient_clip,
            )

        # 更新参数
        self.optimizer.step()

        # 返回损失值（转换为 Python 浮点数）
        return {k: v.item() for k, v in losses.items()}

    @torch.no_grad()
    def generate(
        self,
        semantic_layout: torch.Tensor,
        text_embeddings: torch.Tensor,
        num_timesteps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        推理：生成地形高度图

        Args:
            semantic_layout: 语义布局图，shape: (batch, in_channels, H, W)
            text_embeddings: CLIP 文本嵌入，shape: (batch, text_len, text_dim)
            num_timesteps: 推理步数（使用默认配置如果为 None）

        Returns:
            生成的地形高度图，shape: (batch, out_channels, H, W)
        """
        self.eval()

        batch_size = semantic_layout.shape[0]
        device = semantic_layout.device
        num_timesteps = num_timesteps or self.config.num_inference_timesteps

        # 初始化：将语义布局图作为扩散初始状态
        x_t = semantic_layout.clone()

        # 推理时间步（从 T 到 1）
        timesteps = torch.linspace(
            self.scheduler.num_train_timesteps - 1,
            0,
            num_timesteps,
            device=device,
            dtype=torch.float32,
        )

        # 逐步去噪
        for t in timesteps:
            t_batch = t.unsqueeze(0).expand(batch_size)

            # 预测噪声
            noise_pred = self.dit(x_t, t_batch, text_embeddings)

            # 执行一步去噪
            x_t = self.scheduler.step_denoise(
                model_output=noise_pred,
                source_image=semantic_layout,
                x_t=x_t,
                t=t_batch,
            )

        # 输出最终的高度图（确保为单通道）
        height_map = x_t

        return height_map

    def save_pretrained(self, save_dir: Path) -> None:
        """保存模型权重"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "config": self.config,
            },
            save_dir / "height_mapgen.pt",
        )

    def load_pretrained(self, load_path: Path) -> None:
        """加载预训练模型权重"""
        load_path = Path(load_path)
        checkpoint = torch.load(load_path, map_location="cpu")
        self.load_state_dict(checkpoint["model_state_dict"])
        print(f"✅ 已加载 HeightMapGen 预训练权重：{load_path}")


def create_height_mapgen(
    image_size: int = 64,
    in_channels: int = 1,
    out_channels: int = 1,
    text_embedding_dim: int = 768,
    hidden_size: int = 512,
    num_layers: int = 8,
    num_heads: int = 8,
    lambda_gradient: float = 0.5,
) -> HeightMapGen:
    """
    创建 HeightMapGen 的便捷函数

    Args:
        image_size: 生成图像尺寸
        in_channels: 输入通道数
        out_channels: 输出通道数
        text_embedding_dim: CLIP 文本嵌入维度
        hidden_size: DiT 隐藏层维度
        num_layers: Transformer 层数
        num_heads: 注意力头数
        lambda_gradient: 梯度损失权重

    Returns:
        HeightMapGen 实例
    """
    config = HeightMapGenConfig(
        image_size=image_size,
        in_channels=in_channels,
        out_channels=out_channels,
        text_embedding_dim=text_embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        lambda_gradient=lambda_gradient,
    )
    return HeightMapGen(config)
