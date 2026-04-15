#!/usr/bin/env python
"""
cVAE 地形生成推理入口

使用训练好的模型生成地形高度图
"""

import argparse
import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.inference import TerrainGenerator, StyleVector, BatchGenerator


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="cVAE 地形生成器 - 使用训练好的模型生成地形高度图",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成预设的几种地貌
  python generate.py --preset
  
  # 生成自定义风格的地形
  python generate.py --S 7.0 --R 6.5 --C 1.5 --output custom_terrain.png
  
  # 批量生成多个地形
  python generate.py --batch
  
  # 生成风格插值序列
  python generate.py --interpolate --S1 8.0 --R1 8.0 --C1 2.0 --S2 2.0 --R2 3.0 --C2 0.8
  
  # 创建 S-R 风格网格
  python generate.py --grid-sr --C 1.0
        """,
    )

    # 模型参数
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/cvae/checkpoints/model_best.pth",
        help="模型检查点路径 (默认：outputs/cvae/checkpoints/model_best.pth)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/inference",
        help="输出目录 (默认：outputs/inference)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        help="设备 (cuda 或 cpu，默认自动选择)",
    )

    # 生成模式
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--preset",
        action="store_true",
        help="生成预设的 5 种地貌类型（丹霞、喀斯特、沙漠、高山、平原）",
    )
    mode_group.add_argument(
        "--batch",
        action="store_true",
        help="批量生成多个自定义地形",
    )
    mode_group.add_argument(
        "--interpolate",
        action="store_true",
        help="生成两个风格之间的插值序列",
    )
    mode_group.add_argument(
        "--grid-sr",
        action="store_true",
        help="创建 S-R 风格网格",
    )

    # 自定义风格参数
    parser.add_argument(
        "--S", type=float, default=5.0, help="Sharpness (0-10, 默认：5.0)"
    )
    parser.add_argument(
        "--R", type=float, default=5.0, help="Ruggedness (0-10, 默认：5.0)"
    )
    parser.add_argument(
        "--C", type=float, default=1.0, help="Complexity (0-10, 默认：1.0)"
    )

    # 插值参数
    parser.add_argument("--S1", type=float, default=8.0, help="起始 S 值")
    parser.add_argument("--R1", type=float, default=8.0, help="起始 R 值")
    parser.add_argument("--C1", type=float, default=2.0, help="起始 C 值")
    parser.add_argument("--S2", type=float, default=2.0, help="结束 S 值")
    parser.add_argument("--R2", type=float, default=3.0, help="结束 R 值")
    parser.add_argument("--C2", type=float, default=0.8, help="结束 C 值")
    parser.add_argument("--steps", type=int, default=10, help="插值步数")

    # 网格参数
    parser.add_argument("--C-grid", type=float, default=1.0, help="网格固定 C 值")

    # 输出参数
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="单个输出文件名（用于单张生成）",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "raw"],
        default="png",
        help="输出格式 (png 或 raw)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子",
    )

    return parser.parse_args()


def main():
    """主入口"""
    args = parse_args()

    # 检查检查点文件
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ 模型检查点不存在：{checkpoint_path}")
        print(f"   请先运行训练：python train_cvae.py --full")
        sys.exit(1)

    print("=" * 70)
    print("cVAE 地形生成器")
    print("=" * 70)

    # 创建批量生成器
    batch_gen = BatchGenerator(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
    )

    # 根据模式执行
    if args.preset:
        # 模式 1：生成预设地貌
        print(f"\n生成预设的 5 种地貌类型...")
        batch_gen.generate_preset_landscapes(format=args.format)

    elif args.batch:
        # 模式 2：批量生成
        print(f"\n批量生成自定义地形...")

        style_configs = [
            {"name": "terrain_01", "S": 8.0, "R": 7.0, "C": 1.5, "seed": 42},
            {"name": "terrain_02", "S": 3.0, "R": 4.0, "C": 0.8, "seed": 123},
            {"name": "terrain_03", "S": 6.0, "R": 5.0, "C": 1.2, "seed": 456},
            {"name": "terrain_04", "S": 2.0, "R": 2.0, "C": 0.5, "seed": 789},
            {"name": "terrain_05", "S": 9.0, "R": 8.0, "C": 2.0, "seed": 101},
        ]

        batch_gen.generate_styles(style_configs, format=args.format)
        print(f"✅ 已生成 {len(style_configs)} 个地形")

    elif args.interpolate:
        # 模式 3：风格插值
        print(f"\n生成风格插值序列...")
        print(f"起始：S={args.S1}, R={args.R1}, C={args.C1}")
        print(f"结束：S={args.S2}, R={args.R2}, C={args.C2}")
        print(f"步数：{args.steps}")

        style_start = {
            "S": args.S1,
            "R": args.R1,
            "C": args.C1,
        }
        style_end = {
            "S": args.S2,
            "R": args.R2,
            "C": args.C2,
        }

        batch_gen.interpolate_and_save(
            style_start,
            style_end,
            num_steps=args.steps,
            format=args.format,
        )
        print(f"✅ 插值序列已保存")

    elif args.grid_sr:
        # 模式 4：S-R 网格
        print(f"\n创建 S-R 风格网格...")

        S_values = [2.0, 4.0, 6.0, 8.0]
        R_values = [2.0, 4.0, 6.0, 8.0]

        grid = batch_gen.create_style_grid(
            S_values=S_values,
            R_values=R_values,
            C_fixed=args.C_grid,
        )
        print(f"✅ S-R 风格网格已创建")

    else:
        # 模式 5：单张生成
        print(f"\n生成自定义地形...")
        print(f"风格：S={args.S}, R={args.R}, C={args.C}")

        generator = batch_gen.generator

        style = StyleVector(S=args.S, R=args.R, C=args.C)
        heightmap = generator.generate(style, seed=args.seed)

        if args.output:
            output_path = Path(args.output)
        else:
            output_path = (
                Path(args.output_dir)
                / f"terrain_S{args.S}_R{args.R}_C{args.C}.{args.format}"
            )

        generator.save_heightmap(heightmap, output_path, args.format)
        print(f"✅ 地形已生成")

    print(f"\n{'=' * 70}")
    print(f"输出目录：{Path(args.output_dir).absolute()}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
