"""
语义布局图生成器（LayoutMapGen）
从 LLM 输出的元素位置图生成精确的语义布局图
使用 DiT + BBDM 架构，配备布局一致性损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import numpy as np
from dataclasses import dataclass

from .bbddm_scheduler import BBDMScheduler, BBDMOutput
from .dit_model import DiTModel, create_dit_model


@dataclass
class LayoutMapGenConfig:
    """LayoutMapGen 配置"""

    # 模型配置
    image_size: int = 64  # 生成图像尺寸
    num_classes: int = 5  # 语义类别数（城镇、山地、水域、河流、平原）
    hidden_size: int = 512  # DiT 隐藏层维度
    num_layers: int = 8  # Transformer 层数
    num_heads: int = 8  # 注意力头数

    # BBDM 配置
    num_train_timesteps: int = 1000
    sigma_min: float = 1e-4
    sigma_max: float = 1.0

    # 文本条件
    text_embedding_dim: int = 768  # CLIP 文本嵌入维度

    # 损失函数权重
    lambda_layout_consistency: float = 1.0  # 布局一致性损失权重

    # 训练配置
    learning_rate: float = 1e-4
    gradient_clip: float = 1.0

    # 推理配置
    num_inference_timesteps: int = 50  # 推理步数


class LayoutConsistencyLoss(nn.Module):
    """
    布局一致性损失（Layout Consistency Loss）

    计算预测布局与 LLM 规划的元素位置图之间的 Soft-IoU 得分，
    强制生成的布局与初始位置对齐
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(
        self,
        predicted_layout: torch.Tensor,
        element_locations: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算布局一致性损失

        Args:
            predicted_layout: 预测的语义布局图，shape: (batch, num_classes, H, W)
                             每个通道为一个类别的概率图
            element_locations: LLM 输出的元素位置图，shape: (batch, num_classes, H, W)
                              二值掩码（框内/折线上为 1，其余为 0）

        Returns:
            布局一致性损失值（标量）
        """
        # 对预测布局进行 softmax 得到概率分布
        pred_probs = F.softmax(predicted_layout, dim=1)

        # 计算每个类别的 IoU
        iou_sum = 0.0
        for c in range(self.num_classes):
            pred_mask = pred_probs[:, c : c + 1, :, :]  # (batch, 1, H, W)
            gt_mask = element_locations[:, c : c + 1, :, :]  # (batch, 1, H, W)

            # 计算交并比
            intersection = (pred_mask * gt_mask).sum(dim=(2, 3))  # (batch,)
            union = pred_mask.sum(dim=(2, 3)) + gt_mask.sum(dim=(2, 3)) - intersection

            # Soft-IoU（避免除零）
            iou = intersection / (union + 1e-6)  # (batch,)
            iou_sum = iou_sum + iou

        # 平均 IoU
        avg_iou = iou_sum / self.num_classes

        # 损失 = 1 - IoU
        loss = 1.0 - avg_iou.mean()

        return loss


class LayoutMapGen(nn.Module):
    """
    语义布局图生成器

    输入：元素位置图 M_e + CLIP 文本嵌入
    输出：语义布局图 M_l（多通道语义分割图）
    """

    def __init__(self, config: Optional[LayoutMapGenConfig] = None):
        super().__init__()
        self.config = config or LayoutMapGenConfig()

        # 初始化 DiT 模型
        self.dit = create_dit_model(
            image_size=self.config.image_size,
            in_channels=self.config.num_classes,  # 输入：元素位置图的多通道表示
            out_channels=self.config.num_classes,  # 输出：预测噪声
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

        # 布局一致性损失
        self.layout_consistency_loss = LayoutConsistencyLoss(self.config.num_classes)

        # 优化器（在 train_step 中初始化）
        self.optimizer: Optional[torch.optim.Optimizer] = None

    def forward(
        self,
        element_locations: torch.Tensor,
        text_embeddings: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播：预测噪声

        Args:
            element_locations: 元素位置图，shape: (batch, num_classes, H, W)
            text_embeddings: CLIP 文本嵌入，shape: (batch, text_len, text_dim)
            t: 时间步，shape: (batch_size,) 或 None（训练时随机采样）

        Returns:
            预测的噪声，shape: (batch, num_classes, H, W)
        """
        batch_size = element_locations.shape[0]
        device = element_locations.device

        # 训练时随机采样时间步
        if t is None:
            t = self.scheduler.sample_timestep(batch_size, device)

        # 通过 DiT 预测噪声
        noise_pred = self.dit(element_locations, t, text_embeddings)

        return noise_pred

    def compute_loss(
        self,
        element_locations: torch.Tensor,
        target_layout: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        计算训练损失

        Args:
            element_locations: 元素位置图，shape: (batch, num_classes, H, W)
            target_layout: 真实语义布局图，shape: (batch, num_classes, H, W)
            text_embeddings: CLIP 文本嵌入，shape: (batch, text_len, text_dim)

        Returns:
            包含各损失项的字典
        """
        batch_size = element_locations.shape[0]
        device = element_locations.device

        # 采样时间步
        t = self.scheduler.sample_timestep(batch_size, device)

        # 前向扩散：添加噪声
        diffusion_output = self.scheduler.add_noise(
            source_image=element_locations,
            target_image=target_layout,
            t=t,
        )

        # 预测噪声
        noise_pred = self.forward(diffusion_output.x_t, text_embeddings, t)

        # 基础 BBDM 损失（MSE）
        loss_bbddm = F.mse_loss(noise_pred, diffusion_output.noise)

        # 预测布局（通过去噪得到）
        t_normalized = t / self.scheduler.num_train_timesteps
        alpha = (1 - t_normalized).view(-1, 1, 1, 1)
        beta = t_normalized.view(-1, 1, 1, 1)
        predicted_layout = (
            diffusion_output.x_t - alpha * element_locations - noise_pred
        ) / (beta + 1e-6)

        # 布局一致性损失
        loss_lc = self.layout_consistency_loss(predicted_layout, element_locations)

        # 总损失
        loss_total = loss_bbddm + self.config.lambda_layout_consistency * loss_lc

        return {
            "loss_total": loss_total,
            "loss_bbddm": loss_bbddm,
            "loss_layout_consistency": loss_lc,
        }

    def train_step(
        self,
        element_locations: torch.Tensor,
        target_layout: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> Dict[str, float]:
        """
        执行一步训练

        Args:
            element_locations: 元素位置图，shape: (batch, num_classes, H, W)
            target_layout: 真实语义布局图，shape: (batch, num_classes, H, W)
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
        losses = self.compute_loss(element_locations, target_layout, text_embeddings)

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
        element_locations: torch.Tensor,
        text_embeddings: torch.Tensor,
        num_timesteps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        推理：生成语义布局图

        Args:
            element_locations: 元素位置图，shape: (batch, num_classes, H, W)
            text_embeddings: CLIP 文本嵌入，shape: (batch, text_len, text_dim)
            num_timesteps: 推理步数（使用默认配置如果为 None）

        Returns:
            生成的语义布局图，shape: (batch, num_classes, H, W)
        """
        self.eval()

        batch_size = element_locations.shape[0]
        device = element_locations.device
        num_timesteps = num_timesteps or self.config.num_inference_timesteps

        # 初始化：将元素位置图作为扩散初始状态
        x_t = element_locations.clone()

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
                source_image=element_locations,
                x_t=x_t,
                t=t_batch,
            )

        # 输出最终的语义布局图
        layout_map = x_t

        return layout_map

    def save_pretrained(self, save_dir: Path) -> None:
        """保存模型权重"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "config": self.config,
            },
            save_dir / "layout_mapgen.pt",
        )

    def load_pretrained(self, load_path: Path) -> None:
        """加载预训练模型权重"""
        load_path = Path(load_path)
        checkpoint = torch.load(load_path, map_location="cpu")
        self.load_state_dict(checkpoint["model_state_dict"])
        print(f"✅ 已加载 LayoutMapGen 预训练权重：{load_path}")


def create_layout_mapgen(
    image_size: int = 64,
    num_classes: int = 5,
    text_embedding_dim: int = 768,
    hidden_size: int = 512,
    num_layers: int = 8,
    num_heads: int = 8,
    lambda_layout_consistency: float = 1.0,
) -> LayoutMapGen:
    """
    创建 LayoutMapGen 的便捷函数

    Args:
        image_size: 生成图像尺寸
        num_classes: 语义类别数
        text_embedding_dim: CLIP 文本嵌入维度
        hidden_size: DiT 隐藏层维度
        num_layers: Transformer 层数
        num_heads: 注意力头数
        lambda_layout_consistency: 布局一致性损失权重

    Returns:
        LayoutMapGen 实例
    """
    config = LayoutMapGenConfig(
        image_size=image_size,
        num_classes=num_classes,
        text_embedding_dim=text_embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        lambda_layout_consistency=lambda_layout_consistency,
    )
    return LayoutMapGen(config)
