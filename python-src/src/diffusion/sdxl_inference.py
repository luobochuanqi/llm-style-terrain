"""
Stable Diffusion XL 图生图推理模块
"""

import torch
from pathlib import Path
from typing import Optional, Union
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLPipeline

from ..config import DiffusionConfig


class SDXLInference:
    """SDXL 图生图推理器"""
    
    def __init__(self, config: Optional[DiffusionConfig] = None):
        """
        初始化推理器
        
        Args:
            config: 扩散模型配置
        """
        self.config = config or DiffusionConfig()
        self.pipe: Optional[StableDiffusionXLPipeline] = None
    
    def load_model(self) -> None:
        """加载 SDXL 模型并应用显存优化"""
        print("正在加载 SDXL 模型并应用显存优化策略...")
        
        # 解析 torch dtype
        torch_dtype = torch.float16 if self.config.torch_dtype == "float16" else torch.float32
        
        # 加载 Pipeline
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=torch_dtype,
            variant="fp16",
            use_safetensors=self.config.use_safetensors,
        )
        
        # 显存优化：启用 CPU 卸载
        if self.config.enable_cpu_offload:
            self.pipe.enable_model_cpu_offload()
            print("✅ 已启用 CPU 显存卸载优化")
        
        print("✅ SDXL 模型加载完成")
    
    def heightmap_to_image(self, heightmap: np.ndarray) -> Image.Image:
        """
        将高度图数组转换为 PIL Image
        
        Args:
            heightmap: 高度图数组 (H, W) 或 (H, W, 1)
            
        Returns:
            PIL Image 对象
        """
        # 确保是 2D 数组
        if heightmap.ndim == 3:
            heightmap = heightmap.squeeze()
        
        # 转换为 numpy 数组用于显示（归一化到 0-255）
        if heightmap.dtype == np.uint16:
            # 如果是 16-bit，缩放到 8-bit 用于显示
            display_map = (heightmap / 256).astype(np.uint8)
        else:
            display_map = heightmap
        
        # 转换为 RGB 图像（SDXL 需要）
        image = Image.fromarray(display_map).convert("RGB")
        return image
    
    def image_to_heightmap(self, image: Image.Image) -> np.ndarray:
        """
        将 PIL Image 转换回高度图数组
        
        Args:
            image: PIL Image 对象
            
        Returns:
            高度图数组 (uint8)
        """
        # 转为灰度图
        gray = image.convert("L")
        return np.array(gray, dtype=np.uint8)
    
    def refine_heightmap(
        self,
        heightmap: np.ndarray,
        output_path: Optional[Path] = None,
    ) -> np.ndarray:
        """
        使用图生图对高度图进行微调
        
        Args:
            heightmap: 输入的高度图数组
            output_path: 输出文件路径（可选）
            
        Returns:
            微调后的高度图数组
        """
        if self.pipe is None:
            self.load_model()
        
        print("开始执行图生图推理...")
        
        # 转换为 PIL Image
        init_image = self.heightmap_to_image(heightmap)
        
        # 创建随机数生成器
        generator = torch.Generator(device="cpu").manual_seed(42)
        
        # 执行图生图
        result_image = self.pipe(
            prompt=self.config.prompt.strip(),
            negative_prompt=self.config.negative_prompt.strip(),
            image=init_image,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            strength=self.config.strength,
            generator=generator,
        ).images[0]
        
        # 转换回高度图数组
        refined_heightmap = self.image_to_heightmap(result_image)
        
        print("✅ 图生图推理完成")
        
        # 保存结果
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result_image.save(output_path)
            print(f"✅ 结果已保存至：{output_path}")
        
        return refined_heightmap
    
    def unload_model(self) -> None:
        """卸载模型释放显存"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print("✅ 模型已卸载")


def refine_with_sdxl(
    heightmap: np.ndarray,
    config: Optional[DiffusionConfig] = None,
    output_path: Optional[Path] = None,
) -> np.ndarray:
    """
    便捷函数：使用 SDXL 对高度图进行微调
    
    Args:
        heightmap: 输入高度图
        config: 扩散模型配置
        output_path: 输出路径
        
    Returns:
        微调后的高度图
    """
    inferencer = SDXLInference(config)
    return inferencer.refine_heightmap(heightmap, output_path)
