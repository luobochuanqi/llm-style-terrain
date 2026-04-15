#!/usr/bin/env python
"""
cVAE 地形风格迁移训练入口

支持三种模式:
- debug: 10 epochs (5-10 分钟，用于代码验证)
- fast: 50 epochs (30-60 分钟，用于快速验证)
- full: 200 epochs (2-4 小时，用于最终训练)

使用方法:
    python train_cvae.py --debug     # 调试模式
    python train_cvae.py --fast      # 快速模式
    python train_cvae.py --full      # 完整模式 (默认)
    python train_cvae.py --resume    # 从检查点恢复
    python train_cvae.py --interpolate  # 风格插值演示
"""

import sys
import argparse
from pathlib import Path
import torch

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.train_cvae.config import TrainingConfig
from src.train_cvae.dataset import create_dataloader
from src.train_cvae.model import create_model
from src.train_cvae.trainer import Trainer
from src.train_cvae.visualizer import Visualizer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="cVAE 地形风格迁移训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="调试模式：10 epochs (5-10 分钟)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="快速模式：50 epochs (30-60 分钟)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="完整模式：200 epochs (2-4 小时)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从检查点恢复训练",
    )
    parser.add_argument(
        "--interpolate",
        action="store_true",
        help="风格插值演示 (训练完成后)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="检查点路径 (用于 --resume 或 --interpolate)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="训练设备 (cuda 或 cpu)",
    )

    args = parser.parse_args()

    # 确定模式
    if args.debug:
        args.mode = "debug"
    elif args.fast:
        args.mode = "fast"
    elif args.full:
        args.mode = "full"
    else:
        args.mode = "full"  # 默认为完整模式

    return args


def run_interpolation_demo(config: TrainingConfig, checkpoint_path: str):
    """运行风格插值演示"""
    print(f"\n{'=' * 60}")
    print(f"cVAE 风格插值演示")
    print(f"{'=' * 60}")

    # 加载模型
    model = create_model(
        latent_dim=config.latent_dim,
        condition_dim=config.condition_dim,
        film_hidden_dim=config.film_hidden_dim,
        beta=config.beta,
    )

    device = torch.device(config.device)
    model = model.to(device)

    # 加载检查点 (移除 Sobel buffer)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"].copy()
    state_dict.pop("sobel_x", None)
    state_dict.pop("sobel_y", None)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 定义丹霞和喀斯特的典型条件向量 (已归一化的近似值)
    # 实际应该从测试集中统计
    cond_danxia = torch.tensor([[0.5, 0.8, 0.3]], device=device)  # 高 S, 高 R, 中 C
    cond_karst = torch.tensor([[-0.5, -0.5, -0.2]], device=device)  # 低 S, 低 R, 低 C

    # 创建可视化器
    visualizer = Visualizer(config)

    # 生成插值网格
    print("\n生成风格插值网格...")
    visualizer.save_interpolation_grid(
        model,
        cond_danxia,
        cond_karst,
        num_steps=10,
    )

    print(f"\n✅ 插值演示完成!")
    print(f"输出目录：{config.output_dir / 'samples'}")


def main():
    """主入口函数"""
    args = parse_args()

    # 处理插值演示
    if args.interpolate:
        if args.checkpoint is None:
            print("❌ 错误：--interpolate 需要指定 --checkpoint 参数")
            sys.exit(1)

        config = TrainingConfig.full_mode()
        run_interpolation_demo(config, args.checkpoint)
        return

    # 确定配置模式
    if args.mode == "debug":
        config = TrainingConfig.debug_mode()
    elif args.mode == "fast":
        config = TrainingConfig.fast_mode()
    else:
        config = TrainingConfig.full_mode()

    # 覆盖设备和数据路径
    config.device = args.device
    # 从当前脚本目录的父目录查找数据
    config.data_root = (
        Path(__file__).parent.parent / "data" / "training-dataset" / "preprocess"
    )

    print(f"\n{'=' * 60}")
    print(f"cVAE 地形风格迁移训练")
    print(f"{'=' * 60}")
    print(f"模式：{config.mode}")
    print(f"设备：{config.device}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Beta: {config.beta} (warmup: {config.beta_warmup_epochs} epochs)")
    print(f"输出目录：{config.output_dir}")
    print(f"{'=' * 60}\n")

    # 创建数据加载器
    print("加载数据集...")
    train_loader, train_dataset = create_dataloader(
        config,
        split="train",
        augment=True,
    )
    val_loader, val_dataset = create_dataloader(
        config,
        split="val",
        augment=False,
    )
    print(f"训练集：{len(train_dataset)} 个样本")
    print(f"验证集：{len(val_dataset)} 个样本")

    # 创建模型
    print("\n创建模型...")
    model = create_model(
        latent_dim=config.latent_dim,
        condition_dim=config.condition_dim,
        film_hidden_dim=config.film_hidden_dim,
        beta=config.beta,
    )
    model = model.to(config.device)
    print(f"参数量：{sum(p.numel() for p in model.parameters()):,}")

    # 创建可视化器
    visualizer = Visualizer(config)

    # 绘制条件分布
    print("\n分析条件向量分布...")
    all_conditions = []
    all_types = []
    for i in range(len(train_dataset)):
        conditions, _, _, types = train_dataset[i]
        all_conditions.append(conditions)
        all_types.append(types)
    if len(all_conditions) > 0:
        all_conditions = torch.stack(all_conditions)
        visualizer.plot_condition_distribution(all_conditions, all_types)

    # 创建训练器
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        visualizer=visualizer,
    )

    # 恢复训练
    if args.resume:
        checkpoint_path = config.output_dir / "checkpoints" / "model_best.pth"
        if checkpoint_path.exists():
            print(f"\n恢复检查点：{checkpoint_path}")
            start_epoch = model.load_checkpoint(str(checkpoint_path), config.device)
            trainer.current_epoch = start_epoch
        else:
            print(f"⚠️  未找到检查点，从头开始训练")

    # 开始训练
    print("\n开始训练...")
    try:
        history = trainer.run_training()

        print(f"\n{'=' * 60}")
        print(f"✅ 训练完成!")
        print(f"最佳验证损失：{trainer.best_val_loss:.4f}")
        print(f"输出目录：{config.output_dir}")
        print(f"{'=' * 60}")

    except KeyboardInterrupt:
        print(f"\n\n⚠️  用户中断训练")

        # 保存当前进度
        model.save_checkpoint(
            config.output_dir / "checkpoints" / "model_interrupted.pth",
            trainer.current_epoch,
        )
        print(f"✅ 已保存中断检查点")

        sys.exit(1)

    # 训练完成后运行插值演示
    print(f"\n{'=' * 60}")
    print(f"运行风格插值演示...")
    print(f"{'=' * 60}")

    checkpoint_path = config.output_dir / "checkpoints" / "model_best.pth"
    if checkpoint_path.exists():
        run_interpolation_demo(config, str(checkpoint_path))
    else:
        print(f"⚠️  未找到最佳检查点，跳过插值演示")


if __name__ == "__main__":
    main()
