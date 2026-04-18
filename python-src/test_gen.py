#!/usr/bin/env python3
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd()))

from src.train_unetcvae.model import create_model
from PIL import Image

# 加载模型
checkpoint = torch.load(
    "outputs/unetcvae/checkpoints/model_best.pth", map_location="cpu"
)
model = create_model(
    latent_dim=checkpoint.get("latent_dim", 128),
    condition_dim=3,
    film_hidden_dim=256,
    channels=(64, 128, 256, 512),
    beta=checkpoint.get("beta", 1.0),
)

state_dict = checkpoint["model_state_dict"]
state_dict = {k: v for k, v in state_dict.items() if k not in ["sobel_x", "sobel_y"]}
model.load_state_dict(state_dict, strict=False)
model.eval()

# 加载一个真实训练样本
img = Image.open(
    "/home/luobo/mine/projects/llm-style-terrain/data/training-dataset/preprocess/normalized/train/danxia_train_001.png"
)
img_array = np.array(img).astype(np.float32) / 65535.0
heightmap = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
condition = torch.tensor([[4.0, 5.0, 1.5]])

print("=" * 60)
print("1. 重构测试 (使用真实图像的 skip connections)")
print("=" * 60)

with torch.no_grad():
    mu, logvar = model.encoder(heightmap)
    z = model.reparameterize(mu, logvar)
    skip_connections = model.encoder.get_skip_connections(heightmap)
    recon = model.decode(z, condition, skip_connections)

print(
    f"输入：min={heightmap.min():.4f}, max={heightmap.max():.4f}, mean={heightmap.mean():.4f}"
)
print(f"重构：min={recon.min():.4f}, max={recon.max():.4f}, mean={recon.mean():.4f}")
mse = torch.mean((recon - heightmap) ** 2)
print(f"MSE: {mse:.6f}")

print()
print("=" * 60)
print("2. 纯生成测试 (从随机 z，无真实 skip)")
print("=" * 60)

with torch.no_grad():
    z_random = torch.randn(1, 128)
    dummy_noise = torch.randn(1, 1, 256, 256) * 0.5
    skip_noise = model.encoder.get_skip_connections(dummy_noise)
    generated = model.decode(z_random, condition, skip_noise)

print(
    f"生成：min={generated.min():.4f}, max={generated.max():.4f}, mean={generated.mean():.4f}"
)

gen_img = generated[0, 0].cpu().numpy()
grad_x = np.abs(gen_img[:, 1:] - gen_img[:, :-1]).mean()
grad_y = np.abs(gen_img[1:, :] - gen_img[:-1, :]).mean()
print(f"生成图梯度：水平={grad_x:.4f}, 垂直={grad_y:.4f}")

gt_img = heightmap[0, 0].cpu().numpy()
gt_grad_x = np.abs(gt_img[:, 1:] - gt_img[:, :-1]).mean()
gt_grad_y = np.abs(gt_img[1:, :] - gt_img[:-1, :]).mean()
print(f"真实图梯度：水平={gt_grad_x:.4f}, 垂直={gt_grad_y:.4f}")

print()
print("=" * 60)
print("3. 分析")
print("=" * 60)
print(f"编码 z 统计：mean={mu.mean():.4f}, std={mu.std():.4f}")
print(f"随机 z 统计：mean={z_random.mean():.4f}, std={z_random.std():.4f}")

# 保存重构和生成图用于对比
from torchvision.utils import save_image

save_image(recon, "outputs/test_recon.png")
save_image(generated, "outputs/test_generated.png")
print("已保存重构图 outputs/test_recon.png 和生成图 outputs/test_generated.png")
