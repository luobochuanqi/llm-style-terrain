#!/usr/bin/env python
"""
HeightmapStyle 模型快速演示
运行：python heightmapstyle_demo.py
"""

import sys
from pathlib import Path

# 将 python-src 加入路径，保持 src 作为包
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import OutputConfig
from src.generators import PerlinHeightmapGenerator
from src.heightmapstyle import HeightmapStyleInference

# 获取输出配置
output_config = OutputConfig()

print("=" * 60)
print("HeightmapStyle 模型演示")
print("模型：dimentox/heightmapstyle")
print("=" * 60)

# 1. 生成 Perlin 噪声
print("\n生成 Perlin 噪声高度图...")
generator = PerlinHeightmapGenerator()
heightmap = generator.generate()
print(f"✅ 高度图尺寸：{heightmap.shape}")

# 2. 使用 HeightmapStyle 模型
print("\n加载 HeightmapStyle 模型...")
inferencer = HeightmapStyleInference()

# 3. 微调
print("\n执行高度图微调...")
output_config.heightmapstyle_dir.mkdir(parents=True, exist_ok=True)
output_path = output_config.heightmapstyle_dir / "demo_result.png"

refined = inferencer.refine_heightmap(
    heightmap,
    output_path=output_path,
)

print(f"\n✅ 完成！结果已保存：{output_path.absolute()}")
print(f"   微调后值范围：[{refined.min()}, {refined.max()}]")

# 卸载模型
inferencer.unload_model()
