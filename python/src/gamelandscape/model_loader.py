"""
GameLandscape 模型加载和推理
GameLandscapeHeightmap512_V1.0 是专门用于游戏地形高度图的 LoRA 模型
"""

import torch
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline


class GameLandscapeInference:
    """GameLandscape 推理器"""

    def __init__(
        self,
        model_path: str = "../../assets/gameLandscape_gameLandscapeHeightmap.safetensors",
    ):
        """
        初始化推理器

        Args:
            model_path: 完整模型文件路径（3.9GB checkpoint）
        """
        self.model_path = Path(__file__).parent / model_path
        self.pipe: Optional[StableDiffusionPipeline] = None

    def load_model(self) -> None:
        """加载 GameLandscape 完整模型"""
        print(f"正在加载 GameLandscape 完整模型：{self.model_path.name}")

        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在：{self.model_path}")

        try:
            # 直接从文件加载完整模型
            print("正在从 checkpoint 文件加载模型...")
            self.pipe = StableDiffusionPipeline.from_single_file(
                self.model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
            )

            # 3. 显存优化
            self.pipe.enable_model_cpu_offload()
            print("✅ 已启用 CPU 显存卸载")

        except Exception as e:
            print(f"❌ 加载失败：{e}")
            print("\n提示：确保模型文件是完整的 SD checkpoint")
            raise

        print("✅ GameLandscape 模型加载完成")

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

    def generate_heightmap(
        self,
        output_path: Optional[Path] = None,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.0,
        terrain_type: str = "Mountain",
        width: int = 512,
        height: int = 512,
    ) -> np.ndarray:
        """
        使用 GameLandscape LoRA 直接生成高度图（文生图）

        Args:
            output_path: 输出路径
            num_inference_steps: 推理步数
            guidance_scale: 引导比例
            terrain_type: 地形类型（Alpen, Hills, Mesa, Mountain, MountainFlow,
                         MountainWater, OceanIsland, Plain, River, RiverMountain,
                         SandyBeach, Volcano）
            width: 图像宽度
            height: 图像高度

        Returns:
            生成的高度图
        """
        if self.pipe is None:
            self.load_model()

        print("开始执行 GameLandscape 文生图推理...")
        print(f"地形类型：{terrain_type}")
        print(f"生成尺寸：{width}x{height}")

        # 提示词（使用 LoRA 推荐的地形关键词）
        prompt = f"""heightmap, {terrain_type}, natural terrain, 
        realistic elevation, game landscape, grayscale"""

        negative_prompt = """color, texture, buildings, roads, artificial structures,
        lighting, shadows, noise, low quality, watermark, text, rgb, colored"""

        # 随机数生成器
        generator = torch.Generator(device="cpu").manual_seed(42)

        # 执行文生图
        result_image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        ).images[0]

        # 转换为高度图
        heightmap = self.image_to_heightmap(result_image)

        print("✅ GameLandscape 文生图完成")

        # 保存结果
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result_image.save(output_path)
            print(f"✅ 结果已保存：{output_path}")

        return heightmap

    def unload_model(self) -> None:
        """卸载模型"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("✅ 模型已卸载")


def load_gamelandscape(
    model_path: str = "../../assets/gameLandscape_gameLandscapeHeightmap.safetensors",
    output_path: Optional[Path] = None,
    terrain_type: str = "Mountain",
) -> np.ndarray:
    """
    便捷函数：使用 GameLandscape 完整模型直接生成高度图（文生图）

    Args:
        model_path: 模型文件路径（完整 checkpoint）
        output_path: 输出路径
        terrain_type: 地形类型

    Returns:
        生成的高度图
    """
    inferencer = GameLandscapeInference(model_path)
    return inferencer.generate_heightmap(output_path, terrain_type=terrain_type)
