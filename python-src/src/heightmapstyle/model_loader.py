"""
HeightmapStyle 模型加载和推理
dimentox/heightmapstyle 是基于 Stable Diffusion 1.x 的高度图专用模型
"""

import torch
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline


class HeightmapStyleInference:
    """HeightmapStyle 推理器"""

    def __init__(self, model_id: str = "dimentox/heightmapstyle"):
        """
        初始化推理器

        Args:
            model_id: HuggingFace 模型 ID
        """
        self.model_id = model_id
        self.pipe: Optional[StableDiffusionPipeline] = None

    def load_model(self) -> None:
        """加载 HeightmapStyle 模型"""
        print(f"正在加载 HeightmapStyle 模型：{self.model_id}")
        print("注意：这是基于 SD 1.x 的模型，不是 SDXL")

        # 加载 Pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )

        # 显存优化
        self.pipe.enable_model_cpu_offload()
        print("✅ 已启用 CPU 显存卸载")
        print("✅ HeightmapStyle 模型加载完成")

    def heightmap_to_image(self, heightmap: np.ndarray) -> Image.Image:
        """
        将高度图转换为 PIL Image

        Args:
            heightmap: 高度图数组 (H, W)

        Returns:
            PIL Image 对象（RGB）
        """
        if heightmap.ndim == 3:
            heightmap = heightmap.squeeze()

        if heightmap.dtype == np.uint16:
            display_map = (heightmap / 256).astype(np.uint8)
        else:
            display_map = heightmap

        image = Image.fromarray(display_map).convert("RGB")
        return image

    def image_to_heightmap(self, image: Image.Image) -> np.ndarray:
        """
        将 PIL Image 转换回高度图

        Args:
            image: PIL Image 对象

        Returns:
            高度图数组 (uint8)
        """
        gray = image.convert("L")
        return np.array(gray, dtype=np.uint8)

    def refine_heightmap(
        self,
        heightmap: np.ndarray,
        output_path: Optional[Path] = None,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        strength: float = 0.75,
    ) -> np.ndarray:
        """
        使用 HeightmapStyle 模型微调高度图

        Args:
            heightmap: 输入高度图
            output_path: 输出路径
            num_inference_steps: 推理步数
            guidance_scale: 引导比例
            strength: 图生图强度

        Returns:
            微调后的高度图
        """
        if self.pipe is None:
            self.load_model()

        print("开始执行 HeightmapStyle 推理...")

        # 转换为 PIL Image
        init_image = self.heightmap_to_image(heightmap)

        # 提示词（简化版，避免超过 77 tokens）
        prompt = """raw grayscale heightmap, elevation data, terrain structure"""

        negative_prompt = """color, texture, trees, buildings, lighting, shadows"""

        # 随机数生成器
        generator = torch.Generator(device="cpu").manual_seed(42)

        # 执行图生图
        result_image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            generator=generator,
        ).images[0]

        # 转换回高度图
        refined_heightmap = self.image_to_heightmap(result_image)

        print("✅ HeightmapStyle 推理完成")

        # 保存结果
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result_image.save(output_path)
            print(f"✅ 结果已保存：{output_path}")

        return refined_heightmap

    def unload_model(self) -> None:
        """卸载模型"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("✅ 模型已卸载")


def load_heightmap_style(
    heightmap: np.ndarray,
    model_id: str = "dimentox/heightmapstyle",
    output_path: Optional[Path] = None,
) -> np.ndarray:
    """
    便捷函数：使用 HeightmapStyle 模型微调高度图

    Args:
        heightmap: 输入高度图
        model_id: 模型 ID
        output_path: 输出路径

    Returns:
        微调后的高度图
    """
    inferencer = HeightmapStyleInference(model_id)
    return inferencer.refine_heightmap(heightmap, output_path)
