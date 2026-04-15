"""
训练循环模块
实现完整的训练流程，包括 beta warmup、早停、检查点保存
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict
import time

from .model import cVAE
from .config import TrainingConfig
from .visualizer import Visualizer


class Trainer:
    """cVAE 训练器

    管理完整的训练流程:
    - Beta warmup (逐渐增加 KL 权重)
    - 早停法 (patience=20)
    - 检查点保存
    - 实时可视化
    """

    def __init__(
        self,
        model: cVAE,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        visualizer: Optional[Visualizer] = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.visualizer = visualizer or Visualizer(config)

        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "recon_loss": [],
            "kl_loss": [],
            "beta": [],
            "lr": [],
        }

        # Beta warmup 调度
        self.beta_scheduler = BetaWarmupScheduler(
            max_beta=config.beta,
            warmup_epochs=config.beta_warmup_epochs,
        )

    def train_one_epoch(self) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()

        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, (conditions, heightmaps, _, _) in enumerate(self.train_loader):
            # 移动到设备
            conditions = conditions.to(self.config.device)
            heightmaps = heightmaps.to(self.config.device)

            # 前向传播
            recon, losses = self.model(heightmaps, conditions)

            # 反向传播
            self.model.optimizer.zero_grad()
            losses["loss_total"].backward()

            # 梯度裁剪
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip,
                )

            self.model.optimizer.step()

            # 累积损失
            total_loss += losses["loss_total"].item()
            total_recon += losses["loss_recon"].item()
            total_kl += losses["loss_kl"].item()

            # 记录日志
            if (batch_idx + 1) % self.config.log_every == 0:
                avg_loss = total_loss / (batch_idx + 1)
                avg_recon = total_recon / (batch_idx + 1)
                avg_kl = total_kl / (batch_idx + 1)

                print(
                    f"Epoch {self.current_epoch + 1} [{batch_idx + 1}/{num_batches}] "
                    f"Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f}"
                )

        # 计算平均损失
        avg_loss = total_loss / num_batches
        avg_recon = total_recon / num_batches
        avg_kl = total_kl / num_batches

        return {
            "train_loss": avg_loss,
            "recon_loss": avg_recon,
            "kl_loss": avg_kl,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证一个 epoch"""
        self.model.eval()

        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        num_batches = len(self.val_loader)

        for conditions, heightmaps, _, _ in self.val_loader:
            # 移动到设备
            conditions = conditions.to(self.config.device)
            heightmaps = heightmaps.to(self.config.device)

            # 前向传播
            recon, losses = self.model(heightmaps, conditions)

            # 累积损失
            total_loss += losses["loss_total"].item()
            total_recon += losses["loss_recon"].item()
            total_kl += losses["loss_kl"].item()

        # 计算平均损失
        avg_loss = total_loss / num_batches
        avg_recon = total_recon / num_batches
        avg_kl = total_kl / num_batches

        return {
            "val_loss": avg_loss,
            "val_recon": avg_recon,
            "val_kl": avg_kl,
        }

    def run_training(self):
        """执行完整训练流程"""
        print(f"\n{'=' * 60}")
        print(f"cVAE 地形风格迁移训练")
        print(f"{'=' * 60}")
        print(f"模式：{self.config.mode}")
        print(f"设备：{self.config.device}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Beta (max): {self.config.beta}")
        print(f"Beta warmup epochs: {self.config.beta_warmup_epochs}")
        print(f"早停 patience: {self.config.early_stop_patience}")
        print(f"{'=' * 60}\n")

        # 配置优化器
        self.model.configure_optimizers(
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # 训练循环
        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # 更新 beta
            current_beta = self.beta_scheduler.get_beta(epoch)
            self.model.beta = current_beta

            # 训练
            train_metrics = self.train_one_epoch()

            # 验证
            if (epoch + 1) % self.config.val_every == 0:
                val_metrics = self.validate()

                # 更新学习率
                current_lr = self.model.optimizer.param_groups[0]["lr"]

                # 记录历史
                self.training_history["train_loss"].append(train_metrics["train_loss"])
                self.training_history["val_loss"].append(val_metrics["val_loss"])
                self.training_history["recon_loss"].append(train_metrics["recon_loss"])
                self.training_history["kl_loss"].append(train_metrics["kl_loss"])
                self.training_history["beta"].append(current_beta)
                self.training_history["lr"].append(current_lr)

                # 打印摘要
                epoch_time = time.time() - epoch_start
                print(
                    f"\nEpoch {epoch + 1}/{self.config.num_epochs} ({epoch_time:.1f}s)"
                )
                print(
                    f"Train Loss: {train_metrics['train_loss']:.4f} | "
                    f"Val Loss: {val_metrics['val_loss']:.4f}"
                )
                print(
                    f"Recon: {train_metrics['recon_loss']:.4f} | "
                    f"KL: {train_metrics['kl_loss']:.4f} | "
                    f"Beta: {current_beta:.4f}"
                )

                # 检查早停
                if val_metrics["val_loss"] < self.best_val_loss - self.config.min_delta:
                    self.best_val_loss = val_metrics["val_loss"]
                    self.patience_counter = 0

                    # 保存最佳检查点
                    self.model.save_checkpoint(
                        self.config.output_dir / "checkpoints" / "model_best.pth",
                        epoch,
                    )
                else:
                    self.patience_counter += 1
                    print(
                        f"⚠️  早停计数：{self.patience_counter}/{self.config.early_stop_patience}"
                    )

                # 保存样本
                if (epoch + 1) % self.config.save_sample_every == 0:
                    self.visualizer.save_epoch_samples(
                        self.model,
                        self.val_loader,
                        epoch,
                    )

                # 更新实时绘图
                if (epoch + 1) % self.config.plot_update_every == 0:
                    self.visualizer.update_training_plot(
                        self.training_history,
                        epoch,
                    )

                # 早停检查
                if (
                    self.config.early_stop_patience is not None
                    and self.patience_counter >= self.config.early_stop_patience
                ):
                    print(f"\n✅ 触发早停 (epoch {epoch + 1})")
                    break

        # 训练完成
        total_time = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"✅ 训练完成!")
        print(f"总耗时：{total_time / 60:.1f} 分钟")
        print(f"最佳验证损失：{self.best_val_loss:.4f}")
        print(f"{'=' * 60}")

        # 保存最终检查点
        self.model.save_checkpoint(
            self.config.output_dir / "checkpoints" / "model_final.pth",
            self.current_epoch,
        )

        # 保存训练历史
        import pandas as pd

        history_df = pd.DataFrame(self.training_history)
        history_df.to_csv(
            self.config.output_dir / "logs" / "training_history.csv",
            index=False,
        )

        # 绘制最终训练曲线
        self.visualizer.save_final_plot(self.training_history)

        return self.training_history


class BetaWarmupScheduler:
    """Beta 热身调度器

    线性增加 beta 从 0 到 max_beta
    """

    def __init__(self, max_beta: float, warmup_epochs: int):
        self.max_beta = max_beta
        self.warmup_epochs = warmup_epochs

    def get_beta(self, epoch: int) -> float:
        """获取当前 epoch 的 beta 值"""
        if epoch >= self.warmup_epochs:
            return self.max_beta
        else:
            # 线性增加
            return self.max_beta * epoch / self.warmup_epochs
