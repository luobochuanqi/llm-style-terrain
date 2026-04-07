"""
MapGen 模块包
包含语义布局图生成器（LayoutMapGen）和地形高度图生成器（HeightMapGen）
"""

from .bbddm_scheduler import BBDMScheduler, BBDMOutput
from .dit_model import DiTModel, DiTConfig, create_dit_model
from .layout_mapgen import (
    LayoutMapGen,
    LayoutMapGenConfig,
    LayoutConsistencyLoss,
    create_layout_mapgen,
)
from .height_mapgen import (
    HeightMapGen,
    HeightMapGenConfig,
    GradientLoss,
    create_height_mapgen,
)

__all__ = [
    # BBDM 调度器
    "BBDMScheduler",
    "BBDMOutput",
    # DiT 模型
    "DiTModel",
    "DiTConfig",
    "create_dit_model",
    # LayoutMapGen
    "LayoutMapGen",
    "LayoutMapGenConfig",
    "LayoutConsistencyLoss",
    "create_layout_mapgen",
    # HeightMapGen
    "HeightMapGen",
    "HeightMapGenConfig",
    "GradientLoss",
    "create_height_mapgen",
]
