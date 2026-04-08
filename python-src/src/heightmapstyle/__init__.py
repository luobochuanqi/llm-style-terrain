"""
HeightmapStyle 模型加载器
专门用于加载 dimentox/heightmapstyle 模型（基于 SD 1.x）
"""

from .model_loader import HeightmapStyleInference, load_heightmap_style

__all__ = ["HeightmapStyleInference", "load_heightmap_style"]
