"""
可视化模块
实现实时训练曲线和样本生成
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from .model import UNetcVAE
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

        ax = axes[0, 0]
        epochs = range(1, len(history["train_loss"]) + 1)
        ax.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
        ax.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training & Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        val_loss_clean = [
            x
            for x in history["val_loss"]
            if x < np.percentile(history["val_loss"], 95) * 10
        ]
        if len(val_loss_clean) > 0:
            max_loss = max(max(history["train_loss"]), max(val_loss_clean))
            ax.set_ylim(0, max_loss * 1.1)

        ax = axes[0, 1]
        ax.plot(epochs, history["recon_loss"], "g-", label="Recon Loss", linewidth=2)
        ax.plot(epochs, history["kl_loss"], "m-", label="KL Loss", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Reconstruction & KL Divergence")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(epochs, history["beta"], "c-", label="Beta", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Beta")
        ax.set_title(f"Beta (KL Weight) - Current: {history['beta'][-1]:.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.plot(epochs, history["lr"], "orange", label="Learning Rate", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("LR")
        ax.set_title(f"Learning Rate - Current: {history['lr'][-1]:.2e}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = self.config.output_dir / "plots" / "training_curves.png"
        plt.savefig(save_path, dpi=150)
        plt.close()

        self.fig = fig
        self.axes = axes

    def save_final_plot(self, history: Dict[str, List[float]]):
        """保存最终训练曲线图"""
        self.update_training_plot(history, len(history["train_loss"]))

        if self.fig is not None:
            self.fig.savefig(
                self.config.output_dir / "plots" / "training_curves_final.png",
                dpi=300,
                bbox_inches="tight",
            )

    @torch.no_grad()
    def save_epoch_samples(
        self,
        model: UNetcVAE,
        dataloader,
        epoch: int,
        num_samples: int = 4,
    ):
        """保存当前 epoch 的生成样本"""
        model.eval()

        batch = next(iter(dataloader))
        actual_batch_size = batch[0].size(0)
        num_samples = min(num_samples, actual_batch_size)

        conditions = batch[0][:num_samples].to(self.config.device)
        heightmaps = batch[1][:num_samples].to(self.config.device)
        filenames = batch[2][:num_samples]

        recon, _ = model(heightmaps, conditions)

        has_interpolation = False

        fig, axes = plt.subplots(3, num_samples, figsize=(3 * num_samples, 9))

        for i in range(num_samples):
            ax = axes[0, i] if num_samples > 1 else axes[0]
            img = heightmaps[i, 0].cpu().numpy()
            ax.imshow(img, cmap="gray")
            ax.set_title(f"Original\n{filenames[i][:15]}")
            ax.axis("off")

        for i in range(num_samples):
            ax = axes[1, i] if num_samples > 1 else axes[1]
            img = recon[i, 0].cpu().numpy()
            ax.imshow(img, cmap="gray")
            ax.set_title(f"Reconstructed")
            ax.axis("off")

        if has_interpolation:
            for i in range(num_samples):
                ax = axes[2, i] if num_samples > 1 else axes[2]
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                ax.set_title(f"Not Available")
                ax.axis("off")
        else:
            for i in range(num_samples):
                ax = axes[2, i] if num_samples > 1 else axes[2]
                error = torch.abs(recon[i, 0] - heightmaps[i, 0]).cpu().numpy()
                ax.imshow(error, cmap="hot")
                ax.set_title(f"Recon Error")
                ax.axis("off")

        plt.tight_layout()

        save_path = (
            self.config.output_dir / "samples" / f"epoch_{epoch + 1:03d}_samples.png"
        )
        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"✅ 样本已保存：{save_path}")

    def save_interpolation_grid(
        self,
        model: UNetcVAE,
        cond_start: torch.Tensor,
        cond_end: torch.Tensor,
        num_steps: int = 10,
    ):
        """保存风格插值网格"""
        model.eval()

        interpolations_list = []

        for i in range(num_steps):
            alpha = i / (num_steps - 1)
            cond_interp = (1 - alpha) * cond_start + alpha * cond_end

            z = torch.randn(1, model.latent_dim, device=cond_start.device)
            img = model.generate(cond_interp, z)
            interpolations_list.append(img)

        interpolations = torch.cat(interpolations_list, dim=0)

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
        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"✅ 条件分布图已保存：{save_path}")
