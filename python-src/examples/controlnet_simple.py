"""
ControlNet 简单示例
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ControlNetConfig
from src.diffusion import SDXLControlNetInference


def test_config():
    """测试配置创建"""
    print("测试 ControlNet 配置...")
    
    config = ControlNetConfig(
        enable=True,
        conditioning_scale=0.5,
        canny_low_threshold=0.5,
        canny_high_threshold=0.5,
    )
    
    print(f"✅ 配置创建成功")
    print(f"   - model_id: {config.model_id}")
    print(f"   - conditioning_scale: {config.conditioning_scale}")
    print(f"   - canny thresholds: [{config.canny_low_threshold}, {config.canny_high_threshold}]")
    
    return config


def test_inferencer_creation():
    """测试推理器创建"""
    print("\n测试 ControlNet 推理器创建...")
    
    config = ControlNetConfig()
    inferencer = SDXLControlNetInference(config)
    
    print(f"✅ 推理器创建成功")
    print(f"   - config: {inferencer.config}")
    print(f"   - pipe: {inferencer.pipe}")
    
    return inferencer


def test_heightmap_to_canny():
    """测试高度图转 Canny 边缘（不加载模型）"""
    print("\n测试高度图 → Canny 边缘转换...")
    
    try:
        import cv2
    except ImportError:
        print("⚠️  OpenCV 未安装，跳过此测试")
        print("   安装：pip install opencv-python")
        return
    
    # 创建测试高度图
    size = 256
    heightmap = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        heightmap[i, :] = np.linspace(0, 255, size, dtype=np.uint8)
    
    config = ControlNetConfig()
    inferencer = SDXLControlNetInference(config)
    
    # 转换为 Canny
    canny_image = inferencer.heightmap_to_canny(heightmap)
    
    # 保存
    output_path = Path("outputs/test_canny.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canny_image.save(output_path)
    
    print(f"✅ Canny 边缘图已生成")
    print(f"   - 尺寸：{canny_image.size}")
    print(f"   - 模式：{canny_image.mode}")
    print(f"   - 保存位置：{output_path.absolute()}")


def main():
    """主函数"""
    print("=" * 60)
    print("ControlNet 简单示例")
    print("=" * 60)
    
    # 测试 1: 配置
    config = test_config()
    
    # 测试 2: 推理器创建
    inferencer = test_inferencer_creation()
    
    # 测试 3: 高度图转 Canny
    test_heightmap_to_canny()
    
    print("\n" + "=" * 60)
    print("✅ 所有基础测试完成！")
    print("=" * 60)
    print("\n下一步:")
    print("  1. 等待 uv sync 完成安装 opencv-python")
    print("  2. 运行完整示例：uv run python examples/controlnet_demo.py")
    print("\n注意:")
    print("  - 首次运行会下载 ControlNet 模型（约 2GB）")
    print("  - 需要 GPU 支持")
    print("  - 显存占用约 8-10GB")


if __name__ == "__main__":
    main()
