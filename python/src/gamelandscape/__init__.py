"""
GameLandscape 完整模型加载器
使用 gameLandscape_gameLandscapeHeightmap.safetensors (3.9GB) 生成游戏地形高度图
"""

from .model_loader import GameLandscapeInference, load_gamelandscape

__all__ = ["GameLandscapeInference", "load_gamelandscape"]
