"""
cVAE 地形风格迁移训练模块
基于条件变分自编码器，从 S/R/C 风格向量生成 256x256 地形高度图
"""

from .config import TrainingConfig
from .dataset import TerrainDataset, create_dataloader
from .model import cVAE, Encoder, FiLMDecoder, ConditionNormalizer, create_model
from .trainer import Trainer
from .visualizer import Visualizer

__all__ = [
    # 配置
    "TrainingConfig",
    # 数据集
    "TerrainDataset",
    "create_dataloader",
    # 模型
    "cVAE",
    "Encoder",
    "FiLMDecoder",
    "ConditionNormalizer",
    # 训练
    "Trainer",
    # 可视化
    "Visualizer",
]
