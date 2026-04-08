# GameLandscape 模型快速开始

## 前提条件

确保完整模型文件已就位：
```bash
ls -la assets/gameLandscape_gameLandscapeHeightmap.safetensors
```

**文件大小应约为 3.9GB**

## 运行演示

```bash
cd python-src
python gamelandscape_demo.py
```

**首次运行**：直接从本地加载完整模型，无需下载其他文件。

## 输出位置

生成的文件会保存在：
```
outputs/gamelandscape/demo_result.png
```

## 自定义参数

编辑 `gamelandscape_demo.py` 中的推理参数：

```python
heightmap = inferencer.generate_heightmap(
    output_path=output_path,
    terrain_type="Mountain",  # 地形类型
    num_inference_steps=25,   # 推理步数（更多=更好但更慢）
    guidance_scale=7.0,       # 引导比例（7-9 推荐）
    width=512,
    height=512,
)
```

## 集成到工作流

```python
from src.gamelandscape import GameLandscapeInference

# 创建推理器（使用完整模型）
inferencer = GameLandscapeInference()

# 文生图：直接生成地形高度图
heightmap = inferencer.generate_heightmap(
    output_path="outputs/my_terrain.png",
    terrain_type="Mountain",  # 可选：Alpen, Hills, Mesa, Mountain, etc.
)
```

## 模型说明

本模块使用的是 **完整微调模型**（3.9GB），而非 LoRA 版本。

- ✅ **效果更好**：直接使用作者训练好的完整权重
- ✅ **无需基础模型**：不依赖 HuggingFace 下载
- ✅ **一致性高**：与 Civitai 展示效果一致
- ⚠️ **文件较大**：3.9GB（相比 LoRA 的~200MB）

## 可用地形类型

- `Alpen` - 阿尔卑斯山地
- `Hills` - 丘陵
- `Mesa` - 台地
- `Mountain` - 山地
- `MountainFlow` - 山地河流
- `MountainWater` - 山地水域
- `OceanIsland` - 海洋岛屿
- `Plain` - 平原
- `River` - 河流
- `RiverMountain` - 河谷山地
- `SandyBeach` - 沙滩
- `Volcano` - 火山
