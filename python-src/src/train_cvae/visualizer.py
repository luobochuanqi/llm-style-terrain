"""
可视化模块
实现实时训练曲线和样本生成
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from .model import cVAE
from .config import TrainingConfig


class Visualizer:
    """训练可视化器"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.fig = None
        self.axes = None

    def update_training_plot(
        self,
        history: Dict[str, List[float]],
        epoch: int,
    ):
        """更新实时训练曲线图"""
        if len(history["train_loss"]) < 2:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Train/Val Loss
        ax = axes[0, 0]
        epochs = range(1, len(history["train_loss"]) + 1)
        ax.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
        ax.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training & Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Recon & KL Loss
        ax = axes[0, 1]
        ax.plot(epochs, history["recon_loss"], "g-", label="Recon Loss", linewidth=2)
        ax.plot(epochs, history["kl_loss"], "m-", label="KL Loss", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Reconstruction & KL Divergence")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Beta
        ax = axes[1, 0]
        ax.plot(epochs, history["beta"], "c-", label="Beta", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Beta")
        ax.set_title(f"Beta (KL Weight) - Current: {history['beta'][-1]:.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Learning Rate
        ax = axes[1, 1]
        ax.plot(epochs, history["lr"], "orange", label="Learning Rate", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("LR")
        ax.set_title(f"Learning Rate - Current: {history['lr'][-1]:.2e}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存
        save_path = self.config.output_dir / "plots" / "training_curves.png"
        plt.savefig(save_path, dpi=150)
        plt.close()

        # 更新主图
        self.fig = fig
        self.axes = axes

    def save_final_plot(self, history: Dict[str, List[float]]):
        """保存最终训练曲线图"""
        self.update_training_plot(history, len(history["train_loss"]))

        # 保存高分辨率版本
        if self.fig is not None:
            self.fig.savefig(
                self.config.output_dir / "plots" / "training_curves_final.png",
                dpi=300,
                bbox_inches="tight",
            )

    @torch.no_grad()
    def save_epoch_samples(
        self,
        model: cVAE,
        dataloader,
        epoch: int,
        num_samples: int = 4,  # 减少默认数量
    ):
        """保存当前 epoch 的生成样本"""
        model.eval()

        # 获取一个 batch
        batch = next(iter(dataloader))
        actual_batch_size = batch[0].size(0)
        num_samples = min(num_samples, actual_batch_size)

        conditions = batch[0][:num_samples].to(self.config.device)
        heightmaps = batch[1][:num_samples].to(self.config.device)
        filenames = batch[2][:num_samples]

        # 重建
        recon, _ = model(heightmaps, conditions)

        # 生成插值样本
        has_interpolation = False
        if len(conditions) >= 2 and num_samples >= 2:
            # 在第一个和最后一个样本之间插值
            alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
            interpolations_list = []

            cond_start = conditions[0:1]
            cond_end = conditions[-1:]

            for alpha in alpha_values:
                cond_interp = (1 - alpha) * cond_start + alpha * cond_end
                z = torch.randn(1, model.latent_dim, device=self.config.device)
                interp = model.generate(cond_interp, z)
                interpolations_list.append(interp)

            interpolations = torch.cat(interpolations_list, dim=0)
            has_interpolation = True

        # 可视化
        fig, axes = plt.subplots(3, num_samples, figsize=(3 * num_samples, 9))

        # 第一行：原始图像
        for i in range(num_samples):
            ax = axes[0, i] if num_samples > 1 else axes[0]
            img = heightmaps[i, 0].cpu().numpy()
            ax.imshow(img, cmap="gray")
            ax.set_title(f"Original\n{filenames[i][:15]}")
            ax.axis("off")

        # 第二行：重建图像
        for i in range(num_samples):
            ax = axes[1, i] if num_samples > 1 else axes[1]
            img = recon[i, 0].cpu().numpy()
            ax.imshow(img, cmap="gray")
            ax.set_title(f"Reconstructed")
            ax.axis("off")

        # 第三行：插值样本
        if has_interpolation:
            for i in range(min(len(interpolations), num_samples)):
                ax = axes[2, i] if num_samples > 1 else axes[2]
                img = interpolations[i, 0].cpu().numpy()
                alpha = alpha_values[i]
                ax.imshow(img, cmap="gray")
                ax.set_title(f"Interpolation\nα={alpha:.2f}")
                ax.axis("off")
        else:
            # 如果没有插值，显示重建误差
            for i in range(num_samples):
                ax = axes[2, i] if num_samples > 1 else axes[2]
                error = torch.abs(recon[i, 0] - heightmaps[i, 0]).cpu().numpy()
                ax.imshow(error, cmap="hot")
                ax.set_title(f"Recon Error")
                ax.axis("off")

        plt.tight_layout()

        # 保存
        save_path = (
            self.config.output_dir / "samples" / f"epoch_{epoch + 1:03d}_samples.png"
        )
        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"✅ 样本已保存：{save_path}")

    def save_interpolation_grid(
        self,
        model: cVAE,
        cond_start: torch.Tensor,
        cond_end: torch.Tensor,
        num_steps: int = 10,
    ):
        """保存风格插值网格"""
        model.eval()

        # 插值条件向量
        interpolations_list = []

        for i in range(num_steps):
            alpha = i / (num_steps - 1)
            cond_interp = (1 - alpha) * cond_start + alpha * cond_end

            # 采样 z
            z = torch.randn(1, model.latent_dim, device=cond_start.device)
            img = model.generate(cond_interp, z)
            interpolations_list.append(img)

        # 合并为 tensor
        interpolations = torch.cat(interpolations_list, dim=0)

        # 网格可视化
        fig, axes = plt.subplots(1, num_steps, figsize=(3 * num_steps, 3))

        for i, ax in enumerate(axes):
            img = interpolations[i, 0].detach().cpu().numpy()
            ax.imshow(img, cmap="gray")
            alpha = i / (num_steps - 1)
            ax.set_title(f"α={alpha:.2f}")
            ax.axis("off")

        plt.tight_layout()

        save_path = self.config.output_dir / "samples" / "interpolation_grid.png"
        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"✅ 插值网格已保存：{save_path}")

    def plot_condition_distribution(
        self,
        conditions: torch.Tensor,
        terrain_types: List[str],
    ):
        """绘制条件向量分布图"""
        conditions_np = conditions.cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # S_score
        ax = axes[0]
        for terrain_type in ["Danxia", "Kasite"]:
            mask = [t == terrain_type for t in terrain_types]
            ax.hist(
                conditions_np[mask, 0],
                alpha=0.5,
                bins=20,
                label=terrain_type,
            )
        ax.set_xlabel("S_score (normalized)")
        ax.set_ylabel("Count")
        ax.set_title("Sharpness Distribution")
        ax.legend()

        # R_score
        ax = axes[1]
        for terrain_type in ["Danxia", "Kasite"]:
            mask = [t == terrain_type for t in terrain_types]
            ax.hist(
                conditions_np[mask, 1],
                alpha=0.5,
                bins=20,
                label=terrain_type,
            )
        ax.set_xlabel("R_score (normalized)")
        ax.set_title("Ruggedness Distribution")
        ax.legend()

        # C_score
        ax = axes[2]
        for terrain_type in ["Danxia", "Kasite"]:
            mask = [t == terrain_type for t in terrain_types]
            ax.hist(
                conditions_np[mask, 2],
                alpha=0.5,
                bins=20,
                label=terrain_type,
            )
        ax.set_xlabel("C_score (normalized)")
        ax.set_title("Complexity Distribution")
        ax.legend()

        plt.tight_layout()

        save_path = self.config.output_dir / "plots" / "condition_distribution.png"
        ax.set_xlabel("C_score (normalized)")
        ax.set_title("Complexity Distribution")
        ax.legend()

        plt.tight_layout()

        save_path = self.config.output_dir / "plots" / "condition_distribution.png"
        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"✅ 条件分布图已保存：{save_path}")
