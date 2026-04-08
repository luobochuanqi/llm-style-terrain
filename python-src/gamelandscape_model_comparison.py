#!/usr/bin/env python
"""
GameLandscape 模型对比测试
对比三种生成方式：
1. 完整模型 - 文生图 (3.9GB)
2. 完整模型 - 图生图 (从噪声图生成)
3. LoRA 模型 - 文生图 (289MB + SD 1.5)
"""

import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

sys.path.insert(0, str(Path(__file__).parent))

from src.config import OutputConfig, GeneratorConfig
from src.generators.perlin import PerlinHeightmapGenerator

output_config = OutputConfig()
output_dir = output_config.gamelandscape_dir
output_dir.mkdir(parents=True, exist_ok=True)

# 模型文件
FULL_MODEL = (
    Path(__file__).parent / "assets/gameLandscape_gameLandscapeHeightmap.safetensors"
)
LORA_MODEL = Path(__file__).parent / "assets/GameLandscapeHeightmap512_V1.0.safetensors"
BASE_MODEL = "runwayml/stable-diffusion-v1-5"

# 测试参数
TEST_CONFIG = {
    "prompt": "heightmap, Mountain, natural terrain, realistic elevation, game landscape, grayscale",
    "negative_prompt": "color, texture, buildings, roads, artificial structures, lighting, shadows, noise, low quality, watermark, text, rgb, colored",
    "width": 512,
    "height": 512,
    "num_inference_steps": 25,
    "guidance_scale": 7.0,
    "seed": 42,
    "strength": 0.5,  # 图生图强度
}


def create_perlin_noise_image(
    width: int = 512, height: int = 512, seed: int = 42
) -> Image.Image:
    """使用 Perlin 噪声生成器创建噪声图作为图生图的输入"""
    print(f"正在生成 Perlin 噪声图 (seed={seed})...")

    # 使用 Perlin 噪声生成器 (n=9 表示 2^9=512)
    config = GeneratorConfig(n=9, scale=100, octaves=4, seed=seed)
    generator = PerlinHeightmapGenerator(config)
    heightmap = generator.generate()

    # 确保是 uint8 格式
    if heightmap.dtype == np.uint16:
        heightmap = (heightmap / 256).astype(np.uint8)

    # 转换为灰度图像
    image = Image.fromarray(heightmap).convert("L").convert("RGB")

    return image


def test_full_model_txt2img():
    """测试 1：完整模型 - 文生图（3.9GB）"""
    print("=" * 60)
    print("测试 1: 完整模型 - 文生图 (3.9GB)")
    print("=" * 60)

    if not FULL_MODEL.exists():
        print(f"❌ 模型文件不存在：{FULL_MODEL}")
        return

    print(f"模型文件：{FULL_MODEL.name}")
    print(f"文件大小：{FULL_MODEL.stat().st_size / 1024**3:.2f} GB")

    try:
        print("正在加载完整模型...")
        pipe = StableDiffusionPipeline.from_single_file(
            FULL_MODEL,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        pipe.enable_model_cpu_offload()

        generator = torch.Generator(device="cpu").manual_seed(TEST_CONFIG["seed"])

        print("正在执行文生图...")
        result = pipe(
            prompt=TEST_CONFIG["prompt"],
            negative_prompt=TEST_CONFIG["negative_prompt"],
            num_inference_steps=TEST_CONFIG["num_inference_steps"],
            guidance_scale=TEST_CONFIG["guidance_scale"],
            width=TEST_CONFIG["width"],
            height=TEST_CONFIG["height"],
            generator=generator,
        ).images[0]

        output_path = output_dir / "01_full_txt2img.png"
        result.save(output_path)
        print(f"✅ 完整模型文生图结果已保存：{output_path}")

        del pipe
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"❌ 测试失败：{e}")


def test_full_model_img2img():
    """测试 2：完整模型 - 图生图（从噪声图生成）"""
    print("\n" + "=" * 60)
    print("测试 2: 完整模型 - 图生图 (从噪声图)")
    print("=" * 60)

    if not FULL_MODEL.exists():
        print(f"❌ 模型文件不存在：{FULL_MODEL}")
        return

    print(f"模型文件：{FULL_MODEL.name}")
    print(f"输入：随机噪声图 (512x512)")
    print(f"强度：{TEST_CONFIG['strength']}")

    try:
        # 使用 Perlin 噪声生成器创建噪声图
        print("正在使用 Perlin 噪声生成器创建输入图...")
        noise_image = create_perlin_noise_image(
            TEST_CONFIG["width"], TEST_CONFIG["height"], TEST_CONFIG["seed"]
        )
        noise_path = output_dir / "input_perlin_noise.png"
        noise_image.save(noise_path)
        print(f"Perlin 噪声图已保存：{noise_path}")

        print("正在加载完整模型（图生图模式）...")
        pipe = StableDiffusionImg2ImgPipeline.from_single_file(
            FULL_MODEL,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        pipe.enable_model_cpu_offload()

        generator = torch.Generator(device="cpu").manual_seed(TEST_CONFIG["seed"])

        print("正在执行图生图...")
        result = pipe(
            prompt=TEST_CONFIG["prompt"],
            negative_prompt=TEST_CONFIG["negative_prompt"],
            image=noise_image,
            strength=TEST_CONFIG["strength"],
            num_inference_steps=TEST_CONFIG["num_inference_steps"],
            guidance_scale=TEST_CONFIG["guidance_scale"],
            generator=generator,
        ).images[0]

        output_path = output_dir / "02_full_img2img_from_noise.png"
        result.save(output_path)
        print(f"✅ 完整模型图生图结果已保存：{output_path}")

        del pipe
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"❌ 测试失败：{e}")


def test_lora_model_txt2img():
    """测试 3：LoRA 模型 - 文生图（289MB + SD 1.5）"""
    print("\n" + "=" * 60)
    print("测试 3: LoRA 模型 - 文生图 (289MB + SD 1.5)")
    print("=" * 60)

    if not LORA_MODEL.exists():
        print(f"❌ 模型文件不存在：{LORA_MODEL}")
        return

    print(f"LoRA 文件：{LORA_MODEL.name}")
    print(f"文件大小：{LORA_MODEL.stat().st_size / 1024**3:.2f} GB")
    print(f"基础模型：{BASE_MODEL}")

    try:
        print("正在加载基础 SD 1.5 模型...")
        pipe = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )

        print("正在加载 LoRA 权重...")
        pipe.load_lora_weights(
            LORA_MODEL.parent,
            weight_name=LORA_MODEL.name,
        )

        pipe.enable_model_cpu_offload()

        generator = torch.Generator(device="cpu").manual_seed(TEST_CONFIG["seed"])

        print("正在生成图像...")
        result = pipe(
            prompt=TEST_CONFIG["prompt"],
            negative_prompt=TEST_CONFIG["negative_prompt"],
            num_inference_steps=TEST_CONFIG["num_inference_steps"],
            guidance_scale=TEST_CONFIG["guidance_scale"],
            width=TEST_CONFIG["width"],
            height=TEST_CONFIG["height"],
            generator=generator,
        ).images[0]

        output_path = output_dir / "03_lora_txt2img.png"
        result.save(output_path)
        print(f"✅ LoRA 模型文生图结果已保存：{output_path}")

        del pipe
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"❌ 测试失败：{e}")


def main():
    print("\n" + "=" * 70)
    print("GameLandscape 模型对比测试 - 三向对比")
    print("=" * 70)
    print("\n测试项目:")
    print("1. 完整模型 - 文生图 (3.9GB)")
    print("2. 完整模型 - 图生图 (从 Perlin 噪声图，strength=0.8)")
    print("3. LoRA 模型 - 文生图 (289MB + SD 1.5)")
    print(f"\n输出目录：{output_dir.absolute()}")
    print("=" * 70)

    # 测试 1: 完整模型文生图
    test_full_model_txt2img()

    # 测试 2: 完整模型图生图（从 Perlin 噪声）
    test_full_model_img2img()

    # 测试 3: LoRA 模型文生图
    test_lora_model_txt2img()

    print("\n" + "=" * 70)
    print("✅ 对比测试完成！")
    print("=" * 70)
    print(f"\n结果对比:")
    print(f"  1. 完整模型文生图：outputs/gamelandscape/01_full_txt2img.png")
    print(f"  2. 完整模型图生图：outputs/gamelandscape/02_full_img2img_from_perlin.png")
    print(f"  3. LoRA 模型文生图：outputs/gamelandscape/03_lora_txt2img.png")
    print(f"  输入 Perlin 噪声图：outputs/gamelandscape/input_perlin_noise.png")
    print(f"\n请对比三种方式的生成效果，推荐选择效果最好的方式")
    print("\n💡 提示:")
    print("  - 方式 1: 最常用，直接从文本生成，随机性强")
    print("  - 方式 2: 从 Perlin 噪声图生成，可以通过噪声种子和参数控制结果")
    print("  - 方式 3: LoRA 版本，文件小但效果可能不如完整模型")


if __name__ == "__main__":
    main()
