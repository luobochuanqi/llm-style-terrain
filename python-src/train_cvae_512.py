"""
512 分辨率 cVAE 地形风格迁移训练入口
"""

import argparse
import torch
from pathlib import Path
import sys

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from train_cvae.dataset import TerrainDataset, create_data_loaders
from train_cvae.model import create_model
from train_cvae.trainer import Trainer
from train_cvae.visualizer import Visualizer
from train_cvae.config_512 import TrainingConfig512


def main():
    parser = argparse.ArgumentParser(description="512 分辨率 cVAE 地形训练")
    parser.add_argument("--debug", action="store_true", help="调试模式 (10 epochs)")
    parser.add_argument("--fast", action="store_true", help="快速模式 (50 epochs)")
    parser.add_argument("--full", action="store_true", help="完整模式 (200 epochs)")
    parser.add_argument("--resume", type=str, help="从检查点恢复训练")

    args = parser.parse_args()

    # 选择配置模式
    if args.debug:
        config = TrainingConfig512.debug_mode()
    elif args.fast:
        config = TrainingConfig512.fast_mode()
    else:
        config = TrainingConfig512.full_mode()

    print(f"\n{'=' * 60}")
    print(f"cVAE 512 分辨率地形风格迁移训练")
    print(f"{'=' * 60}")
    print(f"模式：{config.mode}")
    print(f"分辨率：{config.image_size}x{config.image_size}")
    print(f"设备：{config.device}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Beta: {config.beta} (warmup: {config.beta_warmup_epochs} epochs)")
    print(f"输出目录：{config.output_dir}")
    print(f"{'=' * 60}\n")

    # 加载数据集
    print("加载数据集...")
    train_loader, val_loader = create_data_loaders(
        data_root=config.data_root.parent,  # 使用 preprocess 目录
        features_csv=config.features_csv,
        splits_dir=config.splits_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        image_size=config.image_size,
        log_transform_c=config.log_transform_c,
        augment_hflip=config.augment_hflip,
        augment_rotate_90=config.augment_rotate_90,
        augment_small_rotate=config.augment_small_rotate,
        use_weighted_sampling=config.use_weighted_sampling,
        danxia_weight=config.danxia_weight,
        kasite_weight=config.kasite_weight,
    )

    print(f"训练集：{len(train_loader.dataset)} 个样本")
    print(f"验证集：{len(val_loader.dataset)} 个样本")

    # 创建模型
    print("\n创建模型...")
    model = create_model(
        latent_dim=config.latent_dim,
        condition_dim=config.condition_dim,
        film_hidden_dim=config.film_hidden_dim,
        beta=config.beta,
        image_size=config.image_size,
    )

    # 移动到设备
    device = torch.device(config.device)
    model = model.to(device)

    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   总参数量：{total_params:,}")
    print(f"   可训练参数：{trainable_params:,}")

    # 创建可视化器
    visualizer = Visualizer(config)

    # 分析条件分布
    print("\n分析条件向量分布...")
    # 跳过绘图，直接开始训练
    # visualizer.plot_condition_distribution(...)

    # 创建训练器
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        visualizer=visualizer,
    )

    # 开始训练
    print("\n开始训练...")
    history = trainer.run_training()

    print(f"\n{'=' * 60}")
    print(f"✅ 训练完成!")
    print(f"最佳验证损失：{trainer.best_val_loss:.4f}")
    print(f"输出目录：{config.output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
