#!/usr/bin/env python
"""
GameLandscape 模型快速演示
运行：python gamelandscape_demo.py
"""

import sys
from pathlib import Path

# 将 python-src 加入路径，保持 src 作为包
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import OutputConfig
from src.gamelandscape import GameLandscapeInference

# 获取输出配置
output_config = OutputConfig()

print("=" * 60)
print("GameLandscape 模型演示")
print("模型：GameLandscapeHeightmapGenerator V1.0 (完整模型)")
print("=" * 60)

# 1. 使用 GameLandscape 模型直接生成
print("\n加载 GameLandscape 模型...")
inferencer = GameLandscapeInference()

# 2. 文生图生成
print("\n执行文生图生成...")
output_config.gamelandscape_dir.mkdir(parents=True, exist_ok=True)
output_path = output_config.gamelandscape_dir / "demo_result.png"

# 可用地形类型：Alpen, Hills, Mesa, Mountain, MountainFlow,
# MountainWater, OceanIsland, Plain, River, RiverMountain,
# SandyBeach, Volcano
heightmap = inferencer.generate_heightmap(
    output_path=output_path,
    terrain_type="Mountain",  # 山地地形
    width=512,
    height=512,
)

print(f"\n✅ 完成！结果已保存：{output_path.absolute()}")
print(f"   生成尺寸：{heightmap.shape}")
print(f"   值范围：[{heightmap.min()}, {heightmap.max()}]")

# 卸载模型
inferencer.unload_model()
