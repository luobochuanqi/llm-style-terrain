#!/usr/bin/env python
"""
Residual U-Net cVAE 训练入口

支持三种模式:
- debug: 10 epochs (5-10 分钟)
- fast: 50 epochs (30-60 分钟)
- full: 200 epochs (2-4 小时)

使用方法:
    uv run python train_unetcvae.py --debug     # 调试模式
    uv run python train_unetcvae.py --fast      # 快速模式
    uv run python train_unetcvae.py --full      # 完整模式 (默认)
    uv run python train_unetcvae.py --no-amp    # 禁用 AMP
"""

import sys
import argparse
from pathlib import Path
import torch

# 添加父目录到路径以支持从 scripts/ 导入 src/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.unetcvae.config import TrainingConfig
from src.training.unetcvae.dataset import create_dataloader
from src.models.unetcvae.model import create_model
from src.training.unetcvae.trainer import Trainer
from src.training.unetcvae.visualizer import Visualizer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Residual U-Net cVAE 训练")

    parser.add_argument("--debug", action="store_true", help="调试模式：10 epochs")
    parser.add_argument("--fast", action="store_true", help="快速模式：50 epochs")
    parser.add_argument(
        "--full", action="store_true", help="完整模式：200 epochs (默认)"
    )
    parser.add_argument("--resume", action="store_true", help="从检查点恢复")
    parser.add_argument("--checkpoint", type=str, default=None, help="检查点路径")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="训练设备",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="启用 AMP 混合精度 (默认开启)",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="禁用 AMP 混合精度",
    )

    args = parser.parse_args()

    if args.debug:
        args.mode = "debug"
    elif args.fast:
        args.mode = "fast"
    else:
        args.mode = "full"

    return args


def main():
    """主入口"""
    args = parse_args()

    if args.mode == "debug":
        config = TrainingConfig.debug_mode()
    elif args.mode == "fast":
        config = TrainingConfig.fast_mode()
    else:
        config = TrainingConfig.full_mode()

    config.device = args.device
    config.use_amp = args.amp and not args.no_amp
    config.data_root = (
        Path(__file__).parent.parent / "data" / "training-dataset" / "preprocess"
    )

    print(f"\n{'=' * 60}")
    print(f"Residual U-Net cVAE 训练")
    print(f"{'=' * 60}")
    print(f"模式：{config.mode}")
    print(f"设备：{config.device}")
    print(f"AMP: {config.use_amp}")
    print(f"Epochs: {config.num_epochs}")
    print(f"{'=' * 60}\n")

    train_loader, train_dataset = create_dataloader(config, split="train", augment=True)
    val_loader, val_dataset = create_dataloader(config, split="val", augment=False)
    print(f"训练集：{len(train_dataset)} 个样本")
    print(f"验证集：{len(val_dataset)} 个样本")

    model = create_model(
        latent_dim=config.latent_dim,
        condition_dim=config.condition_dim,
        film_hidden_dim=config.film_hidden_dim,
        channels=config.unet_channels,
        beta=config.beta,
    )
    model = model.to(config.device)
    print(f"参数量：{sum(p.numel() for p in model.parameters()):,}")

    visualizer = Visualizer(config)

    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        visualizer=visualizer,
    )

    if args.resume and args.checkpoint:
        model.load_checkpoint(args.checkpoint, torch.device(config.device))

    try:
        history = trainer.run_training()
        print(f"\n{'=' * 60}")
        print(f"✅ 训练完成!")
        print(f"最佳验证损失：{trainer.best_val_loss:.4f}")
        print(f"{'=' * 60}")
    except KeyboardInterrupt:
        print(f"\n⚠️  用户中断训练")
        model.save_checkpoint(
            str(config.output_dir / "checkpoints" / "model_interrupted.pth"),
            trainer.current_epoch,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
