"""
批量地形生成器
用于批量生成多个地形高度图
"""

import torch
from pathlib import Path
from typing import List, Optional
from PIL import Image
import numpy as np

from .generator import TerrainGenerator, StyleVector


class BatchGenerator:
    """批量地形生成器

    基于 TerrainGenerator，支持批量生成和保存
    """

    def __init__(
        self,
        checkpoint_path: str,
        output_dir: str = "outputs/inference",
        device: Optional[str] = None,
    ):
        """
        Args:
            checkpoint_path: 模型检查点路径
            output_dir: 输出目录
            device: 设备 (cuda/cpu)
        """
        self.generator = TerrainGenerator(checkpoint_path, device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_styles(
        self,
        style_configs: List[dict],
        format: str = "png",
    ) -> List[Path]:
        """
        根据配置列表生成多个地形

        Args:
            style_configs: 风格配置列表，每项包含：
                - name: 名称
                - S, R, C: 风格值 (0-10)
                - seed: 随机种子（可选）
            format: 输出格式

        Returns:
            输出文件路径列表
        """
        output_paths = []

        for config in style_configs:
            name = config.get("name", "unnamed")
            style = StyleVector(
                S=config.get("S", 5.0),
                R=config.get("R", 5.0),
                C=config.get("C", 5.0),
            )
            seed = config.get("seed")

            # 生成
            heightmap = self.generator.generate(style, seed=seed)

            # 保存
            output_path = self.output_dir / f"{name}.{format}"
            self.generator.save_heightmap(heightmap, output_path, format)
            output_paths.append(output_path)

        return output_paths

    def generate_preset_landscapes(
        self,
        format: str = "png",
    ) -> List[Path]:
        """
        生成预设的几种地貌类型

        Returns:
            输出文件路径列表
        """
        presets = [
            {"name": "danxia", "style": StyleVector.danxia()},
            {"name": "karst", "style": StyleVector.karst()},
            {"name": "desert", "style": StyleVector.desert()},
            {"name": "mountain", "style": StyleVector.mountain()},
            {"name": "plains", "style": StyleVector.plains()},
        ]

        output_paths = []

        for preset in presets:
            heightmap = self.generator.generate(preset["style"])
            output_path = self.output_dir / f"{preset['name']}.{format}"
            self.generator.save_heightmap(heightmap, output_path, format)
            output_paths.append(output_path)

        print(f"\n✅ 已生成 {len(output_paths)} 种预设地貌")

        return output_paths

    def create_style_grid(
        self,
        S_values: List[float],
        R_values: List[float],
        C_fixed: float = 1.0,
        output_path: Optional[str] = None,
    ) -> Image.Image:
        """
        创建 S-R 风格网格

        Args:
            S_values: S 值列表
            R_values: R 值列表
            C_fixed: 固定的 C 值
            output_path: 输出路径

        Returns:
            网格图像
        """
        styles = []
        labels = []

        for S in S_values:
            for R in R_values:
                styles.append(StyleVector(S=S, R=R, C=C_fixed))
                labels.append(f"S={S}, R={R}")

        # 生成对比网格
        if output_path is None:
            output_path = self.output_dir / "style_grid_SR.png"

        grid = self.generator.create_comparison_grid(
            styles,
            labels=labels,
            output_path=output_path,
        )

        return grid

    def interpolate_and_save(
        self,
        style_start: dict,
        style_end: dict,
        num_steps: int = 10,
        format: str = "png",
    ) -> List[Path]:
        """
        在两个风格之间插值并保存

        Args:
            style_start: 起始风格配置
            style_end: 结束风格配置
            num_steps: 插值步数
            format: 输出格式

        Returns:
            输出文件路径列表
        """
        start = StyleVector(**style_start)
        end = StyleVector(**style_end)

        heightmaps = self.generator.interpolate(
            start,
            end,
            num_steps=num_steps,
        )

        output_paths = []
        for i, h in enumerate(heightmaps):
            alpha = i / (num_steps - 1)
            output_path = self.output_dir / f"interp_{i:02d}_a{alpha:.2f}.{format}"
            self.generator.save_heightmap(h, output_path, format)
            output_paths.append(output_path)

        return output_paths
