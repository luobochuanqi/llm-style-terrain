"""
cVAE 模型模块
基于条件变分自编码器，从 S/R/C 风格向量生成地形高度图
"""

from .model import cVAE, Encoder, FiLMDecoder, ConditionNormalizer, create_model

__all__ = [
    "cVAE",
    "Encoder",
    "FiLMDecoder",
    "ConditionNormalizer",
    "create_model",
]
