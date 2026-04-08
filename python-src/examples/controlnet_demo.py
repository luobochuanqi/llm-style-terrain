"""
ControlNet 高度图 refinement 示例
演示如何使用 ControlNet 对 Perlin 噪声高度图进行结构约束微调
"""

import sys
from pathlib import Path
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config, ControlNetConfig
from src.generators import PerlinHeightmapGenerator
from src.diffusion import SDXLControlNetInference


def run_controlnet_workflow():
    """执行 ControlNet 工作流"""
    print("=" * 60)
    print("ControlNet 高度图微调示例")
    print("=" * 60)

    # 配置
    config = Config()

    # 启用 ControlNet
    config.controlnet.enable = True
    config.controlnet.conditioning_scale = 0.5  # 中等控制强度
    config.controlnet.canny_low_threshold = 0.5
    config.controlnet.canny_high_threshold = 0.5
    config.controlnet.num_inference_steps = 25
    config.controlnet.guidance_scale = 5.0

    # 确保输出目录存在
    config.output.output_dir.mkdir(parents=True, exist_ok=True)

    # 步骤 1: 生成 Perlin 噪声高度图
    print("\n【步骤 1/3】生成 Perlin 噪声高度图")
    print("-" * 60)

    generator = PerlinHeightmapGenerator(config.generator)
    heightmap_path = config.output.output_dir / "heightmap_perlin.raw"
    heightmap = generator.generate_and_save(heightmap_path)

    print(f"高度图尺寸：{heightmap.shape}")
    print(f"数据类型：{heightmap.dtype}")
    print(f"值范围：[{heightmap.min()}, {heightmap.max()}]")

    # 步骤 2: 使用 ControlNet 微调
    print("\n【步骤 2/3】ControlNet 结构约束微调")
    print("-" * 60)

    inferencer = SDXLControlNetInference(config.controlnet)

    # 生成 Canny 边缘图（用于调试）
    canny_image = inferencer.heightmap_to_canny(heightmap)
    canny_path = config.output.output_dir / "canny_edges.png"
    canny_image.save(canny_path)
    print(f"✅ Canny 边缘图已保存：{canny_path}")

    # 执行 ControlNet 推理
    controlnet_path = config.output.output_dir / "heightmap_controlnet.png"
    refined_heightmap = inferencer.refine_heightmap(heightmap, controlnet_path)

    print(f"微调后高度图尺寸：{refined_heightmap.shape}")
    print(f"数据类型：{refined_heightmap.dtype}")
    print(f"值范围：[{refined_heightmap.min()}, {refined_heightmap.max()}]")

    # 卸载模型
    inferencer.unload_model()

    # 步骤 3: 对比结果
    print("\n【步骤 3/3】结果对比")
    print("-" * 60)

    print(f"原始高度图统计:")
    print(f"  - 平均值：{heightmap.mean():.2f}")
    print(f"  - 标准差：{heightmap.std():.2f}")

    print(f"\nControlNet 微调后统计:")
    print(f"  - 平均值：{refined_heightmap.mean():.2f}")
    print(f"  - 标准差：{refined_heightmap.std():.2f}")

    # 计算差异
    diff = np.abs(refined_heightmap.astype(int) - heightmap.astype(int))
    print(f"\n差异统计:")
    print(f"  - 平均差异：{diff.mean():.2f}")
    print(f"  - 最大差异：{diff.max()}")
    print(
        f"  - 相关系数：{np.corrcoef(heightmap.flatten(), refined_heightmap.flatten())[0, 1]:.4f}"
    )

    # 完成
    print("\n" + "=" * 60)
    print("✅ 全部流程完成！")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  - 原始高度图：{heightmap_path.absolute()}")
    print(f"  - Canny 边缘图：{canny_path.absolute()}")
    print(f"  - ControlNet 微调：{controlnet_path.absolute()}")

    return {
        "original_heightmap": heightmap,
        "refined_heightmap": refined_heightmap,
        "output_paths": {
            "original": heightmap_path,
            "canny": canny_path,
            "controlnet": controlnet_path,
        },
    }


def compare_controlnet_strength():
    """对比不同 ControlNet 强度的效果"""
    print("=" * 60)
    print("ControlNet 强度对比测试")
    print("=" * 60)

    # 生成小型测试高度图
    from src.config import GeneratorConfig

    test_config = GeneratorConfig(n=8, scale=100.0, seed=42)  # 256x256
    generator = PerlinHeightmapGenerator(test_config)
    heightmap = generator.generate()

    # 测试不同的 conditioning_scale
    strengths = [0.3, 0.5, 0.7]

    results = []
    for strength in strengths:
        print(f"\n测试 conditioning_scale = {strength}")
        print("-" * 60)

        config = ControlNetConfig(
            enable=True,
            conditioning_scale=strength,
            num_inference_steps=20,  # 减少步数加快测试
        )

        inferencer = SDXLControlNetInference(config)
        output_path = Path(f"outputs/controlnet_strength_{strength}.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        refined = inferencer.refine_heightmap(heightmap, output_path)
        inferencer.unload_model()

        # 计算与原始的相关性
        correlation = np.corrcoef(heightmap.flatten(), refined.flatten())[0, 1]
        results.append((strength, correlation, output_path))

        print(f"  相关系数：{correlation:.4f}")

    # 总结
    print("\n" + "=" * 60)
    print("强度对比结果:")
    print("-" * 60)
    for strength, corr, path in results:
        print(f"  strength={strength:.1f} → correlation={corr:.4f} → {path.name}")

    print("\n建议:")
    print("  - conditioning_scale=0.3: 弱控制，更多创作自由")
    print("  - conditioning_scale=0.5: 中等控制，推荐默认值")
    print("  - conditioning_scale=0.7: 强控制，几乎保持原结构")


def main():
    """主入口"""
    import argparse

    parser = argparse.ArgumentParser(description="ControlNet 高度图微调示例")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="运行强度对比测试",
    )
    args = parser.parse_args()

    try:
        if args.compare:
            compare_controlnet_strength()
        else:
            run_controlnet_workflow()
    except KeyboardInterrupt:
        print("\n⚠️  用户中断执行")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误：{e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
