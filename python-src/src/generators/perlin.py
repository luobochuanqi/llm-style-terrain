"""
Perlin 噪声高度图生成模块
"""

import numpy as np
from pathlib import Path
from typing import Optional
import noise

from ..config import GeneratorConfig


class PerlinHeightmapGenerator:
    """Perlin 噪声高度图生成器"""
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        """
        初始化生成器
        
        Args:
            config: 生成器配置，使用默认配置如果为 None
        """
        self.config = config or GeneratorConfig()
    
    def generate(self) -> np.ndarray:
        """
        生成 Perlin 噪声高度图
        
        Returns:
            生成的身高图数组
        """
        size = 2 ** self.config.n
        print(f"🚀 开始生成 {size}×{size} Perlin 噪声高度图...")
        
        # 1. 生成空数组
        heightmap = np.zeros((size, size), dtype=np.float32)
        
        # 2. 生成 Perlin 噪声（范围 -1 到 1）
        for y in range(size):
            for x in range(size):
                heightmap[y][x] = noise.pnoise2(
                    x / self.config.scale,
                    y / self.config.scale,
                    octaves=self.config.octaves,
                    persistence=self.config.persistence,
                    lacunarity=self.config.lacunarity,
                    repeatx=size,
                    repeaty=size,
                    base=self.config.seed,
                )
        
        # 3. 归一化到 [0, dtype_max]
        min_val, max_val = heightmap.min(), heightmap.max()
        dtype_max = np.iinfo(self.config.dtype).max
        normalized = (heightmap - min_val) / (max_val - min_val) * dtype_max
        heightmap_uint = normalized.astype(self.config.dtype)
        
        print(f"✅ Perlin 噪声高度图生成完成")
        return heightmap_uint
    
    def save(self, heightmap: np.ndarray, output_path: Path) -> None:
        """
        保存高度图到文件
        
        Args:
            heightmap: 高度图数组
            output_path: 输出文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        heightmap.tofile(output_path)
        print(f"✅ 高度图已保存至：{output_path}")
    
    def generate_and_save(self, output_path: Path) -> np.ndarray:
        """
        生成并保存高度图
        
        Args:
            output_path: 输出文件路径
            
        Returns:
            生成的身高图数组
        """
        heightmap = self.generate()
        self.save(heightmap, output_path)
        return heightmap


def generate_perlin_heightmap(config: Optional[GeneratorConfig] = None) -> np.ndarray:
    """
    便捷函数：生成 Perlin 噪声高度图
    
    Args:
        config: 生成器配置
        
    Returns:
        生成的身高图数组
    """
    generator = PerlinHeightmapGenerator(config)
    return generator.generate()
