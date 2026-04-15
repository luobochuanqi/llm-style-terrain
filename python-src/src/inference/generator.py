"""
地形生成器
加载训练好的 cVAE 模型生成地形高度图
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Union, List
from dataclasses import dataclass
from PIL import Image
import numpy as np
import sys

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_cvae.model import cVAE


@dataclass
class StyleVector:
    """风格向量数据类

    Attributes:
        S: Sharpness (锐利度) 0-10
        R: Ruggedness (崎岖度) 0-10
        C: Complexity (复杂度) 0-10
    """

    S: float
    R: float
    C: float

    def __post_init__(self):
        """验证范围"""
        for name, val in [("S", self.S), ("R", self.R), ("C", self.C)]:
            if not 0 <= val <= 10:
                raise ValueError(f"{name} 值必须在 0-10 之间，当前为 {val}")

    def to_tensor(self, device: torch.device) -> torch.Tensor:
        """转换为 tensor"""
        return torch.tensor([[self.S, self.R, self.C]], device=device)

    @classmethod
    def danxia(cls) -> "StyleVector":
        """丹霞地貌风格"""
        return cls(S=7.0, R=6.5, C=1.5)

    @classmethod
    def karst(cls) -> "StyleVector":
        """喀斯特地貌风格"""
        return cls(S=2.0, R=3.0, C=0.8)

    @classmethod
    def desert(cls) -> "StyleVector":
        """沙漠地貌风格"""
        return cls(S=3.0, R=2.0, C=0.5)

    @classmethod
    def mountain(cls) -> "StyleVector":
        """高山地貌风格"""
        return cls(S=8.0, R=8.0, C=2.0)

    @classmethod
    def plains(cls) -> "StyleVector":
        """平原地貌风格"""
        return cls(S=1.0, R=1.5, C=0.3)


class TerrainGenerator:
    """地形生成器

    加载训练好的 cVAE 模型，根据风格向量生成地形高度图
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[str] = None,
    ):
        """
        Args:
            checkpoint_path: 模型检查点路径
            device: 设备 (cuda/cpu)，为 None 时自动选择
        """
        self.checkpoint_path = Path(checkpoint_path)

        # 自动选择设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 加载模型
        self.model = self._load_model()
        self.model.eval()

        print(f"✅ 地形生成器已初始化")
        print(f"   设备：{self.device}")
        print(f"   模型：{self.checkpoint_path}")

    def _load_model(self) -> cVAE:
        """加载模型"""
        from train_cvae.model import create_model

        model = create_model()

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"模型检查点不存在：{self.checkpoint_path}")

        # 加载检查点
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        # 移除 Sobel buffer (仅训练时使用，推理不需要)
        state_dict = checkpoint["model_state_dict"].copy()
        state_dict.pop("sobel_x", None)
        state_dict.pop("sobel_y", None)

        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)

        print(f"✅ 模型加载成功 (epoch {checkpoint.get('epoch', 'unknown')})")

        return model

    @torch.no_grad()
    def generate(
        self,
        style: Union[StyleVector, torch.Tensor],
        seed: Optional[int] = None,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        生成地形高度图

        Args:
            style: 风格向量或条件 tensor
            seed: 随机种子（可选）
            z: 隐向量（可选，为 None 时随机生成）

        Returns:
            高度图 tensor (1, 256, 256)，值范围 [0, 1]
        """
        # 设置随机种子
        if seed is not None:
            torch.manual_seed(seed)

        # 转换风格向量为 tensor
        if isinstance(style, StyleVector):
            condition = style.to_tensor(self.device)
        else:
            condition = style.to(self.device)

        # 生成或使用提供的隐向量
        if z is None:
            z = torch.randn(1, self.model.latent_dim, device=self.device)

        # 生成高度图
        heightmap = self.model.generate(condition, z)

        return heightmap

    @torch.no_grad()
    def generate_batch(
        self,
        styles: List[StyleVector],
        seeds: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        批量生成地形高度图

        Args:
            styles: 风格向量列表
            seeds: 随机种子列表（可选）

        Returns:
            高度图 batch (N, 256, 256)
        """
        heightmaps = []

        for i, style in enumerate(styles):
            seed = seeds[i] if seeds and i < len(seeds) else None
            h = self.generate(style, seed=seed)
            heightmaps.append(h)

        return torch.cat(heightmaps, dim=0)

    def save_heightmap(
        self,
        heightmap: torch.Tensor,
        output_path: Union[str, Path],
        format: str = "png",
    ) -> None:
        """
        保存高度图

        Args:
            heightmap: 高度图 tensor (1, H, W) 或 (1, 1, H, W) 或 (H, W)
            output_path: 输出路径
            format: 输出格式 (png/raw)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 确保是 CPU tensor
        heightmap = heightmap.cpu()

        # 处理不同的维度情况
        if heightmap.dim() == 4:
            # (1, 1, H, W) -> (H, W)
            heightmap = heightmap[0, 0]
        elif heightmap.dim() == 3:
            # (1, H, W) -> (H, W)
            heightmap = heightmap[0]
        # 如果已经是 (H, W)，保持不变

        h = heightmap.numpy()

        if format == "png":
            # 归一化到 0-255
            h_norm = (h * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(h_norm, mode="L")
            img.save(output_path)
            print(f"✅ 高度图已保存：{output_path}")

        elif format == "raw":
            # 保存为 16-bit raw
            h_uint16 = (h * 65535).clip(0, 65535).astype(np.uint16)
            h_uint16.tofile(output_path)
            print(f"✅ 高度图已保存 (raw): {output_path}")

        else:
            raise ValueError(f"不支持的格式：{format}")

    @torch.no_grad()
    def interpolate(
        self,
        style_start: StyleVector,
        style_end: StyleVector,
        num_steps: int = 10,
        seed: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """
        在两个风格之间插值生成一系列地形

        Args:
            style_start: 起始风格
            style_end: 结束风格
            num_steps: 插值步数
            seed: 随机种子（固定 z 以观察纯风格变化）

        Returns:
            高度图列表
        """
        if seed is not None:
            torch.manual_seed(seed)
            z = torch.randn(1, self.model.latent_dim, device=self.device)
        else:
            z = None

        heightmaps = []

        for i in range(num_steps):
            alpha = i / (num_steps - 1)

            # 线性插值风格向量
            S = (1 - alpha) * style_start.S + alpha * style_end.S
            R = (1 - alpha) * style_start.R + alpha * style_end.R
            C = (1 - alpha) * style_start.C + alpha * style_end.C

            style_interp = StyleVector(S=S, R=R, C=C)

            # 使用相同的 z 生成
            h = self.generate(style_interp, z=z)
            heightmaps.append(h)

        return heightmaps

    def create_comparison_grid(
        self,
        styles: List[StyleVector],
        labels: Optional[List[str]] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Image.Image:
        """
        创建对比网格图像

        Args:
            styles: 风格向量列表
            labels: 标签列表
            output_path: 输出路径（可选）

        Returns:
            PIL Image 对象
        """
        # 生成所有高度图
        heightmaps = []
        for style in styles:
            h = self.generate(style)
            heightmaps.append(h)

        # 转换为 numpy 数组
        images = []
        for h in heightmaps:
            h_np = (h[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            images.append(h_np)

        # 计算网格布局
        n = len(images)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))

        # 创建网格图像
        H, W = images[0].shape
        grid = np.zeros((rows * H, cols * W), dtype=np.uint8)

        for i, img in enumerate(images):
            row = i // cols
            col = i % cols
            grid[row * H : (row + 1) * H, col * W : (col + 1) * W] = img

        # 转换为 PIL Image
        grid_img = Image.fromarray(grid, mode="L")

        # 保存
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            grid_img.save(output_path)
            print(f"✅ 对比网格已保存：{output_path}")

        return grid_img
