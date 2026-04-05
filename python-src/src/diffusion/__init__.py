"""
Diffusion 模块初始化
"""

from .sdxl_inference import SDXLInference, refine_with_sdxl

__all__ = [
    "SDXLInference",
    "refine_with_sdxl",
]
