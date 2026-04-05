"""
项目配置模块
统一管理所有配置参数
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import numpy as np


@dataclass
class GeneratorConfig:
    """噪声生成器配置"""
    n: int = 10  # 高度图尺寸为 2^n（如 n=10 → 1024x1024）
    dtype: type = np.uint8  # 数据类型
    scale: float = 300.0  # 噪声缩放（越大越平缓，建议 100-500）
    octaves: int = 6  # 噪声层数（越多细节越丰富，建议 4-8）
    persistence: float = 0.5  # 细节衰减率（0-1，越大细节权重越高）
    lacunarity: float = 2.0  # 细节频率倍数（建议 2.0）
    seed: int = 42  # 随机种子


@dataclass
class DiffusionConfig:
    """Diffusion 推理配置"""
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    torch_dtype: str = "float16"  # float16 或 float32
    use_safetensors: bool = True
    
    # 推理参数
    num_inference_steps: int = 25  # 去噪步数
    guidance_scale: float = 7.5  # 引导比例
    strength: float = 0.7  # 图生图强度（0-1，越大改变越多）
    
    # 提示词
    prompt: str = field(default_factory=lambda: """
        grayscale heightmap, Danxia landform terrain, only elevation information,
        no color, no texture details, smooth gradient from black to white,
        deep valleys in black, high peaks in white, no patterns, no edges,
        seamless tileable, topographic map, orthographic, high resolution, clean, no text
    """)
    
    negative_prompt: str = field(default_factory=lambda: """
        color, texture, noise, text, watermark, logo, signature, artifacts,
        dithering, messy lines, distorted, blurry, low contrast, uneven lighting,
        trees, buildings, people, clouds, sky, rivers, unnecessary details
    """)
    
    # 优化选项
    enable_cpu_offload: bool = True  # 启用 CPU 卸载节省显存


@dataclass
class OutputConfig:
    """输出配置"""
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    heightmap_filename: str = "heightmap_perlin.raw"
    diffusion_filename: str = "heightmap_diffusion.png"
    enable_preview: bool = False  # 是否生成后用 matplotlib 预览


@dataclass
class Config:
    """主配置类"""
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # 工作流控制
    workflow_steps: tuple = ("generate_noise", "diffusion_refine")  # 可执行步骤


# 全局配置实例
config = Config()
