# 输出目录结构调整

## 变更日期
2026-04-08

## 变更说明

将所有输出文件从 `outputs/` 根目录整理到分类子目录中，便于管理和清理。

## 新的目录结构

```
outputs/
├── perlin/              # Perlin 噪声生成
│   ├── heightmap.raw    # 原始高度图
│   └── heightmap.png    # 预览图
├── diffusion/           # SDXL 扩散微调
│   └── heightmap.png    # 微调后的高度图
├── controlnet/          # ControlNet 微调
│   ├── canny_edges.png  # Canny 边缘图
│   ├── heightmap.png    # 微调后的高度图
│   └── strength_test/   # 强度对比测试
├── heightmapstyle/      # HeightmapStyle 模型
│   └── demo_result.png
└── mapgen/             # MapGen 生成
    ├── layout.png       # 语义布局图
    └── height.png       # 地形高度图
```

## 修改的文件

1. **src/config.py**
   - 重构 `OutputConfig` 类
   - 添加各模块的专用输出目录配置

2. **main.py**
   - 使用 `outputs/perlin/` 和 `outputs/diffusion/`

3. **controlnet_demo.py**
   - 使用 `outputs/controlnet/`

4. **heightmapstyle_demo.py**
   - 已使用 `outputs/heightmapstyle/`（保持不变）

5. **mapgen_demo.py**
   - 从 `outputs/mapgen_demo/` 改为 `outputs/mapgen/`

## 迁移指南

### 保留现有文件

如果想保留现有的输出文件，可以手动移动：

```bash
cd python-src

# 移动 Perlin 噪声文件
mkdir -p outputs/perlin
mv outputs/heightmap_perlin.* outputs/perlin/

# 移动 SDXL 微调文件
mkdir -p outputs/diffusion
mv outputs/heightmap_diffusion.png outputs/diffusion/

# 移动 ControlNet 文件
mkdir -p outputs/controlnet
mv outputs/canny_edges.png outputs/controlnet/
mv outputs/heightmap_controlnet.png outputs/controlnet/
mv outputs/controlnet_strength_*.png outputs/controlnet/strength_test/

# 重命名 mapgen 目录
mv outputs/mapgen_demo outputs/mapgen
```

### 清理旧文件

或者直接删除旧文件，重新生成：

```bash
cd python-src
rm -rf outputs/*
```

## 优势

✅ **清晰的分类**: 每个模块有独立的输出目录
✅ **易于管理**: 可以快速定位和清理特定模块的输出
✅ **避免冲突**: 不同脚本的输出不会相互覆盖
✅ **便于测试**: 可以保留某些模块的输出，只清理其他模块

## 文档

详细的输出目录说明见：`outputs/README.md`
