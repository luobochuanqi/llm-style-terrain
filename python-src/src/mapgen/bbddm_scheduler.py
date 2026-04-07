"""
布朗桥扩散模型（BBDM）调度器
实现布朗桥扩散过程，替代传统扩散模型的线性扩散
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class BBDMOutput:
    """BBDM 调度器输出"""

    x_t: torch.Tensor
    noise: torch.Tensor
    t: torch.Tensor


class BBDMScheduler:
    """
    布朗桥扩散模型调度器

    核心思想：扩散初始状态为源图像（而非随机噪声），
    扩散终态为目标图像，中间状态被约束在两者之间的布朗桥路径上
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        sigma_min: float = 1e-4,
        sigma_max: float = 1.0,
    ):
        """
        初始化 BBDM 调度器

        Args:
            num_train_timesteps: 训练时间步总数
            sigma_min: 最小噪声标准差
            sigma_max: 最大噪声标准差
        """
        self.num_train_timesteps = num_train_timesteps
        self.sigma_min = torch.tensor(sigma_min, dtype=torch.float32)
        self.sigma_max = torch.tensor(sigma_max, dtype=torch.float32)

        # 线性插值系数
        self.register_timesteps()

    def register_timesteps(self) -> None:
        """注册时间步数组"""
        self.timesteps = torch.arange(
            self.num_train_timesteps - 1, -1, -1, dtype=torch.float32
        )

    def get_sigmas(self, t: torch.Tensor) -> torch.Tensor:
        """
        获取时间步 t 对应的噪声标准差

        Args:
            t: 时间步张量 (batch_size,)

        Returns:
            噪声标准差张量
        """
        # 归一化时间到 [0, 1]
        t_normalized = t / self.num_train_timesteps

        # 对数线性插值
        log_sigmas = (
            torch.log(self.sigma_min) * (1 - t_normalized)
            + torch.log(self.sigma_max) * t_normalized
        )
        return torch.exp(log_sigmas)

    def add_noise(
        self,
        source_image: torch.Tensor,
        target_image: torch.Tensor,
        t: torch.Tensor,
    ) -> BBDMOutput:
        """
        前向扩散：添加噪声到目标图像

        布朗桥扩散公式：
        x_t = (1 - t/T) * source + (t/T) * target + gamma
        gamma ~ N(0, sigma_t * I)

        Args:
            source_image: 源图像（扩散初始状态），shape: (batch, channel, H, W)
            target_image: 目标图像（扩散终态），shape: (batch, channel, H, W)
            t: 时间步，shape: (batch_size,)

        Returns:
            BBDMOutput 包含 x_t 和添加的噪声
        """
        # 归一化时间系数
        t_normalized = t / self.num_train_timesteps

        # 调整形状以支持广播
        shape = source_image.shape
        t_norm = t_normalized.view(-1, 1, 1, 1).expand(shape)

        # 布朗桥路径插值
        x_t = (1 - t_norm) * source_image + t_norm * target_image

        # 添加高斯噪声
        sigma_t = self.get_sigmas(t)
        noise = torch.randn_like(x_t) * sigma_t.view(-1, 1, 1, 1)
        x_t = x_t + noise

        return BBDMOutput(x_t=x_t, noise=noise, t=t)

    def sample_timestep(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        随机采样时间步用于训练

        Args:
            batch_size: 批量大小
            device: 设备

        Returns:
            随机时间步张量
        """
        t = torch.randint(
            0,
            self.num_train_timesteps,
            (batch_size,),
            device=device,
            dtype=torch.float32,
        )
        return t

    def get_denoising_coefficients(
        self,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取去噪系数，用于推理阶段的逐步去噪

        Args:
            t: 当前时间步

        Returns:
            (alpha, beta) 系数，用于计算去噪后的结果
        """
        t_normalized = t / self.num_train_timesteps
        alpha = 1 - t_normalized
        beta = t_normalized
        return alpha, beta

    def step_denoise(
        self,
        model_output: torch.Tensor,
        source_image: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        推理阶段：执行一步去噪

        Args:
            model_output: DiT 预测的噪声，shape: (batch, channel, H, W)
            source_image: 源图像（扩散初始状态）
            x_t: 当前带噪状态
            t: 当前时间步

        Returns:
            去噪后的状态 x_{t-1}
        """
        # 获取系数
        t_normalized = t / self.num_train_timesteps
        alpha, beta = self.get_denoising_coefficients(t)

        # 调整形状
        shape = x_t.shape
        alpha = alpha.view(-1, 1, 1, 1).expand(shape)
        beta = beta.view(-1, 1, 1, 1).expand(shape)

        # 预测目标图像
        predicted_target = (x_t - alpha * source_image - model_output) / beta

        # 计算下一步状态（使用较小的时间步）
        t_prev = torch.maximum(t - 1, torch.zeros_like(t))
        alpha_prev, beta_prev = self.get_denoising_coefficients(t_prev)

        x_prev = alpha_prev * source_image + beta_prev * predicted_target

        # 添加少量噪声以保持随机性（仅在 t > 0 时）
        sigma_t = self.get_sigmas(t_prev)
        noise = (
            torch.randn_like(x_prev)
            * sigma_t.view(-1, 1, 1, 1)
            * (t_prev > 0).float().view(-1, 1, 1, 1)
        )
        x_prev = x_prev + noise

        return x_prev
