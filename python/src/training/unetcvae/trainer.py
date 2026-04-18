"""
训练循环模块 - 支持 AMP 混合精度训练
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from pathlib import Path
from typing import Optional, Dict, List
import time

from .model import UNetcVAE
from .config import TrainingConfig
from .visualizer import Visualizer


class Trainer:
    """U-Net cVAE 训练器 (支持 AMP)"""

    def __init__(
        self,
        model: UNetcVAE,
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

        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "recon_loss": [],
            "kl_loss": [],
            "grad_loss": [],
            "beta": [],
            "lr": [],
        }

        self.beta_scheduler = BetaWarmupScheduler(
            max_beta=config.beta,
            warmup_epochs=config.beta_warmup_epochs,
        )

        self.use_amp = config.use_amp
        self.scaler: Optional[GradScaler] = None
        if self.use_amp:
            self.scaler = GradScaler()
            print(f"✅ 启用 AMP 混合精度训练")

        self.lr_scheduler = None

        self.sobel_x: Optional[torch.Tensor] = None
        self.sobel_y: Optional[torch.Tensor] = None
        if self.config.use_gradient_loss:
            self._init_gradient_loss()

    @torch.no_grad()
    def _get_skip_connections(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.model.encoder.get_skip_connections(x)

    def train_one_epoch(self) -> Dict[str, float]:
        self.model.train()

        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        total_grad = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, (conditions, heightmaps, _, _) in enumerate(self.train_loader):
            conditions = conditions.to(self.config.device)
            heightmaps = heightmaps.to(self.config.device)

            if self.use_amp:
                with autocast():
                    mu, logvar = self.model.encoder(heightmaps)
                    z = self.model.reparameterize(mu, logvar)
                    skip_connections = self._get_skip_connections(heightmaps)
                    recon = self.model.decode(z, conditions, skip_connections)

                    recon_loss = F.mse_loss(recon, heightmaps, reduction="sum")
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                    grad_loss = torch.tensor(0.0, device=self.config.device)
                    if self.config.use_gradient_loss:
                        grad_loss = self._compute_gradient_loss(recon, heightmaps)

                    loss_total = recon_loss + self.model.beta * kl_loss
                    if self.config.use_gradient_loss:
                        loss_total = (
                            loss_total + self.config.gradient_loss_weight * grad_loss
                        )

                    losses = {
                        "loss_total": loss_total,
                        "loss_recon": recon_loss,
                        "loss_kl": kl_loss,
                    }

                self.scaler.scale(losses["loss_total"]).backward()

                if self.config.gradient_clip > 0:
                    self.scaler.unscale_(self.model.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip,
                    )

                self.scaler.step(self.model.optimizer)
                self.scaler.update()
                self.model.optimizer.zero_grad()
            else:
                recon, losses = self.model(heightmaps, conditions)

                grad_loss = torch.tensor(0.0, device=self.config.device)
                if self.config.use_gradient_loss:
                    grad_loss = self._compute_gradient_loss(recon, heightmaps)
                    losses["loss_total"] = (
                        losses["loss_total"]
                        + self.config.gradient_loss_weight * grad_loss
                    )

                self.model.optimizer.zero_grad()
                losses["loss_total"].backward()

                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip,
                    )

                self.model.optimizer.step()

            total_loss += losses["loss_total"].item()
            total_recon += losses["loss_recon"].item()
            total_kl += losses["loss_kl"].item()
            total_grad += grad_loss.item()

            if (batch_idx + 1) % self.config.log_every == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(
                    f"Epoch {self.current_epoch + 1} [{batch_idx + 1}/{num_batches}] "
                    f"Loss: {avg_loss:.4f}"
                )

        return {
            "train_loss": total_loss / num_batches,
            "recon_loss": total_recon / num_batches,
            "kl_loss": total_kl / num_batches,
            "grad_loss": total_grad / num_batches,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()

        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        total_grad = 0.0
        num_batches = len(self.val_loader)

        for conditions, heightmaps, _, _ in self.val_loader:
            conditions = conditions.to(self.config.device)
            heightmaps = heightmaps.to(self.config.device)

            recon, losses = self.model(heightmaps, conditions)

            grad_loss = torch.tensor(0.0, device=self.config.device)
            if self.config.use_gradient_loss:
                grad_loss = self._compute_gradient_loss(recon, heightmaps)
                total_loss += grad_loss.item() * self.config.gradient_loss_weight

            total_loss += losses["loss_total"].item()
            total_recon += losses["loss_recon"].item()
            total_kl += losses["loss_kl"].item()
            total_grad += grad_loss.item()

        avg_loss = total_loss / num_batches
        avg_recon = total_recon / num_batches
        avg_kl = total_kl / num_batches
        avg_grad = total_grad / num_batches

        return {
            "val_loss": avg_loss,
            "val_recon": avg_recon,
            "val_kl": avg_kl,
            "val_grad": avg_grad,
        }

    def run_training(self):
        print(f"\n{'=' * 60}")
        print(f"Residual U-Net cVAE 地形生成训练")
        print(f"{'=' * 60}")
        print(f"模式：{self.config.mode}")
        print(f"设备：{self.config.device}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Beta (max): {self.config.beta}")
        print(f"Beta warmup epochs: {self.config.beta_warmup_epochs}")
        print(f"早停 patience: {self.config.early_stop_patience}")
        print(f"AMP: {self.config.use_amp}")
        print(f"{'=' * 60}\n")

        self.model.configure_optimizers(
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        if self.config.lr_scheduler_type == "cosine":
            self.lr_scheduler = CosineAnnealingLR(
                self.model.optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.lr_min,
            )
            print(
                f"✅ 使用 Cosine Annealing 学习率调度器 (T_max={self.config.num_epochs}, eta_min={self.config.lr_min})"
            )
        elif self.config.lr_scheduler_type == "plateau":
            self.lr_scheduler = ReduceLROnPlateau(
                self.model.optimizer,
                mode="min",
                factor=0.5,
                patience=20,
                min_lr=self.config.lr_min,
            )
            print(f"✅ 使用 ReduceLROnPlateau 学习率调度器")
        else:
            print(f"⚠️  未使用学习率调度器 (固定 LR={self.config.learning_rate})")

        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            current_beta = self.beta_scheduler.get_beta(epoch)
            self.model.beta = current_beta

            train_metrics = self.train_one_epoch()

            if (epoch + 1) % self.config.val_every == 0:
                val_metrics = self.validate()

                if self.lr_scheduler is not None:
                    if self.config.lr_scheduler_type == "plateau":
                        self.lr_scheduler.step(val_metrics["val_loss"])
                    else:
                        self.lr_scheduler.step()

                current_lr = self.model.optimizer.param_groups[0]["lr"]

                self.training_history["train_loss"].append(train_metrics["train_loss"])
                self.training_history["val_loss"].append(val_metrics["val_loss"])
                self.training_history["recon_loss"].append(train_metrics["recon_loss"])
                self.training_history["kl_loss"].append(train_metrics["kl_loss"])
                self.training_history["grad_loss"].append(
                    train_metrics.get("grad_loss", 0.0)
                )
                self.training_history["beta"].append(current_beta)
                self.training_history["lr"].append(current_lr)

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
                    f"Grad: {train_metrics.get('grad_loss', 0.0):.4f} | "
                    f"Beta: {current_beta:.4f}"
                )

                if val_metrics["val_loss"] < self.best_val_loss - self.config.min_delta:
                    self.best_val_loss = val_metrics["val_loss"]
                    self.patience_counter = 0

                    self.model.save_checkpoint(
                        str(self.config.output_dir / "checkpoints" / "model_best.pth"),
                        epoch,
                    )
                else:
                    self.patience_counter += 1
                    if self.config.early_stop_patience is not None:
                        print(
                            f"⚠️  早停计数：{self.patience_counter}/{self.config.early_stop_patience}"
                        )

                if (epoch + 1) % self.config.save_sample_every == 0:
                    self.visualizer.save_epoch_samples(
                        self.model,
                        self.val_loader,
                        epoch,
                    )

                if (epoch + 1) % self.config.plot_update_every == 0:
                    self.visualizer.update_training_plot(
                        self.training_history,
                        epoch,
                    )

                if (
                    self.config.early_stop_patience is not None
                    and self.patience_counter >= self.config.early_stop_patience
                ):
                    print(f"\n✅ 触发早停 (epoch {epoch + 1})")
                    break

        total_time = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"✅ 训练完成!")
        print(f"总耗时：{total_time / 60:.1f} 分钟")
        print(f"最佳验证损失：{self.best_val_loss:.4f}")
        print(f"{'=' * 60}")

        self.model.save_checkpoint(
            str(self.config.output_dir / "checkpoints" / "model_final.pth"),
            self.current_epoch,
        )

        import pandas as pd

        history_df = pd.DataFrame(self.training_history)
        history_df.to_csv(
            self.config.output_dir / "logs" / "training_history.csv",
            index=False,
        )

        self.visualizer.save_final_plot(self.training_history)

        return self.training_history

    def _init_gradient_loss(self):
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
            device=self.config.device,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32,
            device=self.config.device,
        ).view(1, 1, 3, 3)

        self.model.register_buffer("sobel_x", sobel_x)
        self.model.register_buffer("sobel_y", sobel_y)

    def _compute_gradient_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        sobel_x = self.model.get_buffer("sobel_x")
        sobel_y = self.model.get_buffer("sobel_y")

        pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
        target_grad_x = F.conv2d(target, sobel_x, padding=1)
        target_grad_y = F.conv2d(target, sobel_y, padding=1)

        grad_loss_x = F.mse_loss(pred_grad_x, target_grad_x)
        grad_loss_y = F.mse_loss(pred_grad_y, target_grad_y)

        return grad_loss_x + grad_loss_y


class BetaWarmupScheduler:
    """Beta 热身调度器"""

    def __init__(self, max_beta: float, warmup_epochs: int):
        self.max_beta = max_beta
        self.warmup_epochs = warmup_epochs

    def get_beta(self, epoch: int) -> float:
        if epoch >= self.warmup_epochs:
            return self.max_beta
        else:
            return self.max_beta * epoch / self.warmup_epochs
