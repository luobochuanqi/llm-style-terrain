"""
Stable Diffusion XL + ControlNet 推理模块
使用 ControlNet 对高度图进行更精确的结构控制
"""

import torch
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
)

from ..config import ControlNetConfig


# 延迟导入 cv2（可选依赖）
def _get_cv2():
    """延迟导入 OpenCV"""
    try:
        import cv2

        return cv2
    except ImportError:
        raise ImportError(
            "OpenCV (cv2) is required for ControlNet. "
            "Install it with: pip install opencv-python"
        )


class SDXLControlNetInference:
    """SDXL + ControlNet 推理器"""

    def __init__(self, config: Optional[ControlNetConfig] = None):
        """
        初始化推理器

        Args:
            config: ControlNet 配置
        """
        self.config = config or ControlNetConfig()
        self.pipe: Optional[StableDiffusionXLControlNetPipeline] = None

    def load_model(self) -> None:
        """加载 SDXL + ControlNet 模型"""
        print("正在加载 ControlNet 模型...")

        # 解析 torch dtype
        torch_dtype = (
            torch.float16 if self.config.torch_dtype == "float16" else torch.float32
        )

        # 加载 ControlNet
        controlnet = ControlNetModel.from_pretrained(
            self.config.model_id,
            torch_dtype=torch_dtype,
            use_safetensors=self.config.use_safetensors,
            variant="fp16" if torch_dtype == torch.float16 else None,
        )

        # 加载 VAE（优化显存）
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch_dtype,
        )

        # 创建 Pipeline
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.config.base_model_id,
            controlnet=controlnet,
            vae=vae,
            torch_dtype=torch_dtype,
            use_safetensors=self.config.use_safetensors,
            variant="fp16" if torch_dtype == torch.float16 else None,
        )

        # 显存优化：启用 CPU 卸载
        if self.config.enable_cpu_offload:
            self.pipe.enable_model_cpu_offload()
            print("✅ 已启用 CPU 显存卸载")

        print("✅ ControlNet 模型加载完成")

    def heightmap_to_canny(self, heightmap: np.ndarray) -> Image.Image:
        """
        将高度图转换为 Canny 边缘图

        Args:
            heightmap: 高度图数组 (H, W), dtype: uint8

        Returns:
            Canny 边缘图（PIL Image, RGB）
        """
        cv2 = _get_cv2()

        # 确保是 2D 数组
        if heightmap.ndim == 3:
            heightmap = heightmap.squeeze()

        # 转换为 OpenCV 格式
        img_uint8 = heightmap.astype(np.uint8)

        # 计算 Canny 阈值
        low_threshold = int(self.config.canny_low_threshold * 255)
        high_threshold = int(self.config.canny_high_threshold * 255)

        # Canny 边缘检测
        edges = cv2.Canny(img_uint8, low_threshold, high_threshold)

        # 转为 RGB（ControlNet 需要 3 通道）
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        return Image.fromarray(edges_rgb)

    def heightmap_to_image(self, heightmap: np.ndarray) -> Image.Image:
        """
        将高度图数组转换为 PIL Image（用于 conditioning）

        Args:
            heightmap: 高度图数组 (H, W)

        Returns:
            PIL Image 对象（RGB）
        """
        # 确保是 2D 数组
        if heightmap.ndim == 3:
            heightmap = heightmap.squeeze()

        # 转换为 numpy 数组用于显示
        if heightmap.dtype == np.uint16:
            display_map = (heightmap / 256).astype(np.uint8)
        else:
            display_map = heightmap

        # 转换为 RGB 图像
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
        使用 ControlNet 对高度图进行微调

        Args:
            heightmap: 输入的高度图数组
            output_path: 输出文件路径（可选）

        Returns:
            微调后的高度图数组
        """
        if self.pipe is None:
            self.load_model()

        print("开始执行 ControlNet 推理...")

        # 生成 Canny 条件图
        canny_image = self.heightmap_to_canny(heightmap)

        # 创建随机数生成器
        generator = torch.Generator(device="cpu").manual_seed(42)

        # 执行推理
        result_image = self.pipe(
            prompt=self.config.prompt.strip(),
            negative_prompt=self.config.negative_prompt.strip(),
            image=canny_image,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            conditioning_scale=self.config.conditioning_scale,
            generator=generator,
        ).images[0]

        # 转换回高度图数组
        refined_heightmap = self.image_to_heightmap(result_image)

        print("✅ ControlNet 推理完成")

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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("✅ 模型已卸载")


def refine_with_controlnet(
    heightmap: np.ndarray,
    config: Optional[ControlNetConfig] = None,
    output_path: Optional[Path] = None,
) -> np.ndarray:
    """
    便捷函数：使用 ControlNet 对高度图进行微调

    Args:
        heightmap: 输入高度图
        config: ControlNet 配置
        output_path: 输出路径

    Returns:
        微调后的高度图
    """
    inferencer = SDXLControlNetInference(config)
    return inferencer.refine_heightmap(heightmap, output_path)
