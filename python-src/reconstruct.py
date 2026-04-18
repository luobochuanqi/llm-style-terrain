#!/usr/bin/env python3
"""
使用训练好的 Residual U-Net cVAE 模型进行条件重构

给定一个输入高度图 + 条件向量 → 输出风格化版本
"""

import torch
import numpy as np
from pathlib import Path
import sys
from PIL import Image

sys.path.insert(0, str(Path.cwd()))

from src.train_unetcvae.model import create_model


def load_heightmap(path: str) -> torch.Tensor:
    """加载高度图 (支持 PNG 和 RAW)"""
    path = Path(path)

    if path.suffix == ".png":
        img = Image.open(path)
        img_np = np.array(img)
        original_dtype = img_np.dtype

        # 根据原始 dtype 归一化到 [0, 1]
        if original_dtype == np.uint16:
            img_array = img_np.astype(np.float32) / 65535.0
        elif original_dtype == np.uint8:
            img_array = img_np.astype(np.float32) / 255.0
        else:
            img_array = img_np.astype(np.float32)
            # 如果值大于 1，假设是 0-255 或 0-65535 范围
            if img_array.max() > 1.0:
                if img_array.max() > 256:
                    img_array = img_array / 65535.0
                else:
                    img_array = img_array / 255.0

        # 确保是单通道
        if len(img_array.shape) == 3:
            img_array = img_array.mean(axis=2)
    elif path.suffix == ".raw":
        img_array = (
            np.fromfile(path, dtype=np.uint8).reshape((256, 256)).astype(np.float32)
            / 255.0
        )
    else:
        raise ValueError(f"不支持的格式：{path.suffix}")

    # 转换为张量 (1, 1, 256, 256)
    heightmap = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)

    # 调整尺寸到 256x256
    if heightmap.shape[-1] != 256 or heightmap.shape[-2] != 256:
        heightmap = torch.nn.functional.interpolate(
            heightmap, size=(256, 256), mode="bilinear", align_corners=False
        )

    return heightmap


def main():
    # 配置
    input_path = "outputs/perlin/heightmap.png"  # 输入高度图
    output_path = "outputs/reconstruction.png"  # 输出重构图
    condition = torch.tensor([[4.0, 5.0, 1.5]])  # 条件向量 (S, R, C) = 丹霞地貌
    checkpoint_path = "outputs/unetcvae/checkpoints/model_best.pth"

    print("=" * 60)
    print("Residual U-Net cVAE 条件重构")
    print("=" * 60)

    # 加载模型
    print(f"\n📥 加载模型：{checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
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
    model.eval()
    print(f"✅ 模型已加载 (epoch {checkpoint['epoch']})")

    # 加载输入高度图
    print(f"\n📥 加载输入：{input_path}")
    heightmap = load_heightmap(input_path)
    print(
        f"✅ 输入高度图：min={heightmap.min():.4f}, max={heightmap.max():.4f}, mean={heightmap.mean():.4f}"
    )

    # 条件重构
    print(
        f"\n🎯 条件向量：S={condition[0, 0]:.1f}, R={condition[0, 1]:.1f}, C={condition[0, 2]:.1f}"
    )
    print("⏳ 正在重构...")

    with torch.no_grad():
        # Encoder 编码
        mu, logvar = model.encoder(heightmap)
        z = model.reparameterize(mu, logvar)

        # 获取 skip connections
        skip_connections = model.encoder.get_skip_connections(heightmap)

        # Decoder 解码 (带条件)
        recon = model.decode(z, condition, skip_connections)

    print(
        f"✅ 重构完成：min={recon.min():.4f}, max={recon.max():.4f}, mean={recon.mean():.4f}"
    )

    # 计算 MSE
    mse = torch.mean((recon - heightmap) ** 2)
    print(f"📊 与输入的 MSE: {mse:.6f}")

    # 保存结果
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 拉伸对比度并保存
    recon_img = recon[0, 0].cpu().numpy()
    recon_img = (recon_img - recon_img.min()) / (
        recon_img.max() - recon_img.min() + 1e-8
    )
    recon_img = (recon_img * 255).clip(0, 255).astype(np.uint8)

    Image.fromarray(recon_img, mode="L").save(output_path)
    print(f"\n✅ 重构图已保存：{output_path}")

    # 计算并显示梯度对比
    input_img = heightmap[0, 0].cpu().numpy()
    input_grad_x = np.abs(input_img[:, 1:] - input_img[:, :-1]).mean()
    input_grad_y = np.abs(input_img[1:, :] - input_img[:-1, :]).mean()
    recon_grad_x = (
        np.abs(recon_img[:, 1:].astype(float) - recon_img[:, :-1].astype(float)).mean()
        / 255
    )
    recon_grad_y = (
        np.abs(recon_img[1:, :].astype(float) - recon_img[:-1, :].astype(float)).mean()
        / 255
    )

    print(f"\n📊 梯度对比:")
    print(f"  输入：水平={input_grad_x:.4f}, 垂直={input_grad_y:.4f}")
    print(f"  重构：水平={recon_grad_x:.4f}, 垂直={recon_grad_y:.4f}")


if __name__ == "__main__":
    main()
