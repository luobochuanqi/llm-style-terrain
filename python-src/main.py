"""
LLM Style Terrain 生成器
统一入口：Perlin 噪声生成 → SDXL 图生图微调
"""

import sys
from pathlib import Path
import numpy as np

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.generators import PerlinHeightmapGenerator
from src.diffusion import SDXLInference


def load_heightmap_from_raw(path: Path, size: int = 1024) -> np.ndarray:
    """从 raw 文件加载高度图"""
    return np.fromfile(path, dtype=np.uint8).reshape((size, size))


def run_workflow():
    """执行完整的工作流"""
    print("=" * 60)
    print("LLM Style Terrain 生成器")
    print("=" * 60)

    # 确保输出目录存在
    config.output.output_dir.mkdir(parents=True, exist_ok=True)

    heightmap_path = config.output.output_dir / config.output.heightmap_filename
    preview_path = heightmap_path.with_suffix(".png")
    diffusion_path = config.output.output_dir / config.output.diffusion_filename

    # 步骤 1: 生成 Perlin 噪声高度图（如果不存在则跳过）
    print("\n【步骤 1/2】生成 Perlin 噪声高度图")
    print("-" * 60)

    if heightmap_path.exists() and preview_path.exists():
        print(f"⏭️  检测到已存在的高度图，跳过生成")
        print(f"   - {heightmap_path}")
        print(f"   - {preview_path}")
        # 从 raw 文件加载
        size = 2**config.generator.n
        heightmap = load_heightmap_from_raw(heightmap_path, size)
    else:
        print("未检测到现有高度图，开始生成...")
        generator = PerlinHeightmapGenerator(config.generator)
        heightmap = generator.generate_and_save(heightmap_path)

    print(f"高度图尺寸：{heightmap.shape}")
    print(f"数据类型：{heightmap.dtype}")
    print(f"值范围：[{heightmap.min()}, {heightmap.max()}]")

    # 步骤 2: SDXL 图生图微调
    print("\n【步骤 2/2】SDXL 图生图微调")
    print("-" * 60)
    inferencer = SDXLInference(config.diffusion)
    refined_heightmap = inferencer.refine_heightmap(heightmap, diffusion_path)
    print(f"微调后高度图尺寸：{refined_heightmap.shape}")
    print(f"数据类型：{refined_heightmap.dtype}")
    print(f"值范围：[{refined_heightmap.min()}, {refined_heightmap.max()}]")

    # 完成
    print("\n" + "=" * 60)
    print("✅ 全部流程完成！")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  - 原始噪声高度图：{heightmap_path.absolute()}")
    print(f"  - SDXL 微调高度图：{diffusion_path.absolute()}")

    # 卸载模型释放显存
    inferencer.unload_model()

    return {
        "original_heightmap": heightmap,
        "refined_heightmap": refined_heightmap,
        "output_paths": {
            "original": heightmap_path,
            "refined": diffusion_path,
        },
    }


def main():
    """主入口函数"""
    try:
        result = run_workflow()
        return result
    except KeyboardInterrupt:
        print("\n⚠️  用户中断执行")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误：{e}")
        raise


if __name__ == "__main__":
    main()
