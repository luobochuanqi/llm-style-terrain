#!/usr/bin/env python
"""
Residual U-Net cVAE 推理入口

用于加载训练好的模型进行地形生成：
- 从条件向量 (S/R/C) 生成地形
- 地形风格插值
- 批量生成

使用方法:
    # 从条件向量生成单个地形
    uv run python infer_unetcvae.py generate --checkpoint outputs/unetcvae/checkpoints/model_best.pth --condition 3.5 4.2 1.8 --output outputs/generated.png

    # 风格插值
    uv run python infer_unetcvae.py interpolate --checkpoint outputs/unetcvae/checkpoints/model_best.pth --start 2.0 3.0 1.0 --end 5.0 6.0 3.0 --steps 10 --output outputs/interpolation.png

    # 批量生成
    uv run python infer_unetcvae.py batch --checkpoint outputs/unetcvae/checkpoints/model_best.pth --conditions conditions.csv --output-dir outputs/batch/
"""

import sys
import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import pandas as pd
from typing import Tuple, Optional, List

sys.path.insert(0, str(Path(__file__).parent))

from src.train_unetcvae.model import UNetcVAE, create_model
from src.train_unetcvae.config import TrainingConfig


def load_model(checkpoint_path: str, device: torch.device) -> UNetcVAE:
    """加载训练好的模型"""
    print(f"📥 加载检查点：{checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = create_model(
        latent_dim=checkpoint.get("latent_dim", 128),
        condition_dim=3,
        film_hidden_dim=256,
        channels=(64, 128, 256, 512),
        beta=checkpoint.get("beta", 1.0),
    )

    state_dict = checkpoint["model_state_dict"]
    state_dict = {
        k: v for k, v in state_dict.items() if k not in ["sobel_x", "sobel_y"]
    }

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print(f"✅ 模型已加载 (epoch {checkpoint['epoch']})")
    return model


def parse_condition(condition_str: str) -> torch.Tensor:
    """解析条件字符串为张量"""
    values = [float(x.strip()) for x in condition_str.split()]
    if len(values) != 3:
        raise ValueError(f"条件必须包含 3 个值 (S R C)，当前：{len(values)} 个")
    return torch.tensor(values, dtype=torch.float32).unsqueeze(0)


def generate_from_condition(
    model: UNetcVAE,
    condition: torch.Tensor,
    device: torch.device,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """从条件向量生成地形"""
    model.eval()

    with torch.no_grad():
        output = model.generate(condition.to(device), seed=seed)

    return output


@torch.no_grad()
def interpolate_conditions(
    model: UNetcVAE,
    cond_start: torch.Tensor,
    cond_end: torch.Tensor,
    num_steps: int,
    device: torch.device,
    seed: Optional[int] = None,
) -> List[torch.Tensor]:
    """在两个条件向量之间插值生成"""
    model.eval()

    if seed is not None:
        torch.manual_seed(seed)

    outputs = []

    for i in range(num_steps):
        alpha = i / (num_steps - 1)
        cond_interp = (1 - alpha) * cond_start + alpha * cond_end
        cond_interp = cond_interp.to(device)

        output = model.generate(cond_interp)
        outputs.append(output)

    return outputs


def save_heightmap(
    heightmap: torch.Tensor,
    output_path: Path,
    stretch_contrast: bool = True,
    bit_depth: int = 8,
    gamma: float = 1.0,
    percentile: float = 98.0,
):
    """保存高度图为 PNG

    Args:
        heightmap: 高度图张量 (1, 1, 256, 256)，值范围 [0, 1]
        output_path: 输出路径
        stretch_contrast: 是否拉伸对比度到完整范围
        bit_depth: 输出位深 (8 或 16)
        gamma: 伽马校正值 (1.0=无校正，<1 让暗部更亮，>1 让亮部更暗)
        percentile: 百分位拉伸阈值 (默认 98%，避免极端峰值影响)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img = heightmap[0, 0].cpu().numpy()

    # 输出诊断信息
    orig_min, orig_max = img.min(), img.max()
    orig_mean = img.mean()
    print(
        f"📊 原始高度图统计：min={orig_min:.4f}, max={orig_max:.4f}, mean={orig_mean:.4f}"
    )

    if stretch_contrast and orig_max > orig_min:
        # 使用百分位拉伸，避免极端峰值
        p_low = np.percentile(img, 2)
        p_high = np.percentile(img, percentile)

        if p_high > p_low:
            img = (img - p_low) / (p_high - p_low)
            img = img.clip(0, 1)
            print(f"✅ 已拉伸对比度 (p2-p{percentile})")
        else:
            # 如果百分位范围太小，回退到全范围拉伸
            img = (img - orig_min) / (orig_max - orig_min)
            print(f"✅ 已拉伸对比度到 [0, 1] 范围 (全范围)")

    if gamma != 1.0:
        # 应用伽马校正
        img = np.power(img, 1.0 / gamma)
        print(f"✅ 已应用伽马校正：gamma={gamma}")

    if orig_mean < 0.1:
        print(f"ℹ️  模型输出均值偏低，建议使用 --gamma 0.5 让暗部更亮")

    # 转换为指定位深
    if bit_depth == 16:
        img_out = (img * 65535).clip(0, 65535).astype(np.uint16)
        Image.fromarray(img_out, mode="I;16").save(output_path)
        print(
            f"📊 保存为 16-bit PNG，像素范围：[{img_out.min()}, {img_out.max()}] / 65535"
        )
    else:
        img_out = (img * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img_out, mode="L").save(output_path)
        print(
            f"📊 保存为 8-bit PNG，像素范围：[{img_out.min()}, {img_out.max()}] / 255"
        )

    print(f"✅ 高度图已保存：{output_path}")


def save_heightmap_grid(
    heightmaps: List[torch.Tensor], output_path: Path, percentile: float = 98.0
):
    """保存高度图网格"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_images = len(heightmaps)
    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))

    grid = np.zeros((rows * 256, cols * 256), dtype=np.float64)

    # 先收集所有高度图
    all_hms = []
    for hm in heightmaps:
        img = hm[0, 0].cpu().numpy()
        all_hms.append(img)

    # 计算全局百分位范围
    all_pixels = np.concatenate([img.flatten() for img in all_hms])
    p_low = np.percentile(all_pixels, 2)
    p_high = np.percentile(all_pixels, percentile)
    p_range = p_high - p_low

    print(
        f"📊 网格高度图全局统计：p2={p_low:.4f}, p{percentile}={p_high:.4f}, range={p_range:.4f}"
    )

    # 填充网格并进行对比度拉伸
    for i, img in enumerate(all_hms):
        row = i // cols
        col = i % cols

        # 百分位拉伸
        if p_range > 1e-6:
            img_stretched = (img - p_low) / p_range
            img_stretched = img_stretched.clip(0, 1)
        else:
            img_stretched = np.zeros_like(img)

        # 转换为 16-bit
        img_out = (img_stretched * 65535).clip(0, 65535).astype(np.uint16)
        grid[row * 256 : (row + 1) * 256, col * 256 : (col + 1) * 256] = img_out

    Image.fromarray(grid, mode="I;16").save(output_path)
    print(f"✅ 高度图网格已保存：{output_path}")


def cmd_generate(args):
    """从条件向量生成单个地形"""
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)

    condition = parse_condition(args.condition)
    print(
        f"🎯 条件：S={condition[0, 0]:.1f}, R={condition[0, 1]:.1f}, C={condition[0, 2]:.1f}"
    )

    output = generate_from_condition(model, condition, device, args.seed)

    stretch = not args.no_stretch
    bit_depth = 16 if args.bit16 else 8
    save_heightmap(
        output, Path(args.output), stretch_contrast=stretch, bit_depth=bit_depth
    )


def cmd_interpolate(args):
    """风格插值"""
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)

    cond_start = parse_condition(args.start)
    cond_end = parse_condition(args.end)

    print(
        f"🎯 起始条件：S={cond_start[0, 0]:.1f}, R={cond_start[0, 1]:.1f}, C={cond_start[0, 2]:.1f}"
    )
    print(
        f"🎯 结束条件：S={cond_end[0, 0]:.1f}, R={cond_end[0, 1]:.1f}, C={cond_end[0, 2]:.1f}"
    )
    print(f"📊 插值步数：{args.steps}")

    outputs = interpolate_conditions(
        model, cond_start, cond_end, args.steps, device, args.seed
    )

    save_heightmap_grid(outputs, Path(args.output))


def cmd_batch(args):
    """批量生成"""
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)

    df = pd.read_csv(args.conditions)
    print(f"📊 批量生成：{len(df)} 个样本")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in df.iterrows():
        condition = torch.tensor([[row["S"], row["R"], row["C"]]], dtype=torch.float32)
        output = generate_from_condition(model, condition, device, args.seed)

        if "filename" in row:
            filename = Path(row["filename"]).stem
        else:
            filename = f"sample_{idx:04d}"

        output_path = output_dir / f"{filename}.png"
        save_heightmap(output, output_path)

    print(f"✅ 批量生成完成：{output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Residual U-Net cVAE 推理工具")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="推理设备",
    )

    subparsers = parser.add_subparsers(dest="command", help="命令")

    generate_parser = subparsers.add_parser("generate", help="从条件向量生成单个地形")
    generate_parser.add_argument("--checkpoint", required=True, help="模型检查点路径")
    generate_parser.add_argument("--condition", required=True, help="条件向量 'S R C'")
    generate_parser.add_argument("--output", required=True, help="输出文件路径")
    generate_parser.add_argument("--seed", type=int, default=None, help="随机种子")
    generate_parser.add_argument(
        "--no-stretch",
        action="store_true",
        help="不拉伸对比度 (默认自动拉伸)",
    )
    generate_parser.add_argument(
        "--16bit",
        action="store_true",
        dest="bit16",
        help="保存为 16-bit PNG (默认 8-bit)",
    )
    generate_parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="伽马校正值 (默认 1.0, <1 让暗部更亮，>1 让亮部更暗)",
    )
    generate_parser.set_defaults(func=cmd_generate)

    interpolate_parser = subparsers.add_parser("interpolate", help="风格插值")
    interpolate_parser.add_argument(
        "--checkpoint", required=True, help="模型检查点路径"
    )
    interpolate_parser.add_argument("--start", required=True, help="起始条件 'S R C'")
    interpolate_parser.add_argument("--end", required=True, help="结束条件 'S R C'")
    interpolate_parser.add_argument("--steps", type=int, default=10, help="插值步数")
    interpolate_parser.add_argument("--output", required=True, help="输出网格文件路径")
    interpolate_parser.add_argument("--seed", type=int, default=None, help="随机种子")
    interpolate_parser.set_defaults(func=cmd_interpolate)

    batch_parser = subparsers.add_parser("batch", help="批量生成")
    batch_parser.add_argument("--checkpoint", required=True, help="模型检查点路径")
    batch_parser.add_argument("--conditions", required=True, help="条件 CSV 文件")
    batch_parser.add_argument("--output-dir", required=True, help="输出目录")
    batch_parser.add_argument("--seed", type=int, default=None, help="随机种子")
    batch_parser.set_defaults(func=cmd_batch)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
