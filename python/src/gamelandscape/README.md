# GameLandscape 完整模型

使用完整的 GameLandscape 微调模型生成游戏地形高度图。

## 模型信息

- **完整模型**: `gameLandscape_gameLandscapeHeightmap.safetensors`
- **位置**: `python-src/assets/`
- **文件大小**: 3.9GB
- **基础架构**: Stable Diffusion 1.5
- **类型**: 完整微调 Checkpoint
- **用途**: 游戏地形高度图生成（文生图）

## ⚠️ 重要说明

**本模块使用完整模型（3.9GB），而非 LoRA 版本（289MB）**

原因：
- ✅ 完整模型生成效果更好，与 Civitai 展示一致
- ✅ 无需下载基础模型
- ✅ 独立性更好，不受基础模型影响

## 快速开始

### 运行演示

```bash
cd python-src
python gamelandscape_demo.py
```

### 在代码中使用

```python
from src.gamelandscape import GameLandscapeInference

# 创建推理器（使用完整模型）
inferencer = GameLandscapeInference()

# 文生图：直接生成地形高度图
heightmap = inferencer.generate_heightmap(
    output_path="output.png",
    terrain_type="Mountain",      # 地形类型
    num_inference_steps=25,       # 推理步数
    guidance_scale=7.0,           # 引导比例
    width=512,
    height=512,
)
```

### 使用便捷函数

```python
from src.gamelandscape import load_gamelandscape

heightmap = load_gamelandscape(
    output_path="output.png",
    terrain_type="Mountain"
)
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_path` | `gameLandscape_gameLandscapeHeightmap.safetensors` | 完整模型文件路径 |
| `terrain_type` | "Mountain" | 地形类型关键词 |
| `num_inference_steps` | 25 | 推理步数（更多=更好但更慢） |
| `guidance_scale` | 7.0 | 引导比例（7-9 推荐） |
| `width` | 512 | 图像宽度 |
| `height` | 512 | 图像高度 |

## 可用的地形类型

使用以下关键词来控制生成的地形类型：

| 关键词 | 描述 |
|--------|------|
| **Alpen** | 阿尔卑斯山脉风格 |
| **Hills** | 丘陵地带 |
| **Mesa** | 台地/方山 |
| **Mountain** | 山脉 |
| **MountainFlow** | 山地水流 |
| **MountainWater** | 山水 |
| **OceanIsland** | 海洋岛屿 |
| **Plain** | 平原 |
| **River** | 河流 |
| **RiverMountain** | 山河 |
| **SandyBeach** | 沙滩 |
| **Volcano** | 火山 |

### 使用示例

```python
# 生成阿尔卑斯山地
inferencer.generate_heightmap(
    output_path="alpen_terrain.png",
    terrain_type="Alpen"
)

# 生成丘陵河流
inferencer.generate_heightmap(
    output_path="hills_river.png",
    terrain_type="Hills"
)

# 生成火山地形
inferencer.generate_heightmap(
    output_path="volcano.png",
    terrain_type="Volcano"
)

# 复杂地形组合
inferencer.generate_heightmap(
    output_path="complex.png",
    terrain_type="MountainRiverMountain"
)
```

## 模型加载流程

```
1. 从文件加载完整模型 (3.9GB checkpoint)
   ↓
2. 使用 from_single_file 直接加载
   ↓
3. 启用 CPU offload 优化显存
   ↓
4. 执行文生图推理
```

### 首次运行

首次运行时会从本地直接加载完整模型，无需从网络下载。

## 输出目录

运行演示后会生成：
- `outputs/gamelandscape/demo_result.png` - 生成的游戏地形高度图

## 完整模型 vs LoRA

| 特性 | 完整模型 | LoRA |
|------|---------|------|
| 文件大小 | 3.9GB | 289MB |
| 依赖 | 无 | 需要 SD 1.5 基础模型 |
| 灵活性 | 低 | 高（可换基础模型） |
| 加载速度 | 慢 | 快 |
| **生成质量** | ✅ **最佳** | ⚠️ 一般 |
| **一致性** | ✅ **与展示一致** | ⚠️ 受基础模型影响 |

## 显存优化

模型已自动启用 CPU offload，可以在 4GB 显存的 GPU 上运行。

如果显存仍然不足：

```python
# 降低分辨率
heightmap = inferencer.generate_heightmap(
    width=256,
    height=256,
)

# 减少推理步数
heightmap = inferencer.generate_heightmap(
    num_inference_steps=15,
)
```

## 注意事项

1. **模型文件**: 确保 `gameLandscape_gameLandscapeHeightmap.safetensors` 存在于 `assets/` 目录
2. **文件大小**: 应该约为 3.9GB
3. **显存要求**: 约 4-6GB（启用 CPU offload）
4. **输出格式**: 高度图是灰度 PNG，黑色=低谷，白色=高峰

## 故障排除

### 问题：模型文件不存在

**解决**: 
```bash
# 检查文件是否存在
ls -la assets/gameLandscape_gameLandscapeHeightmap.safetensors
```

### 问题：显存不足

**解决**:
```python
# 降低分辨率和步数
heightmap = inferencer.generate_heightmap(
    width=256,
    height=256,
    num_inference_steps=15,
)
```

### 问题：生成效果不理想

**解决**:
1. 调整 `terrain_type` 关键词
2. 增加 `guidance_scale` 到 7.5-9.0
3. 增加 `num_inference_steps` 到 30-50
4. 尝试不同的随机种子

## 提示词建议

### 基础提示词模板

```python
prompt = f"heightmap, {terrain_type}, natural terrain, realistic elevation, game landscape, grayscale"

negative_prompt = """color, texture, buildings, roads, artificial structures,
lighting, shadows, noise, low quality, watermark, text, rgb, colored"""
```

### 自定义提示词

```python
# 更详细的地形描述
prompt = """heightmap, Mountain, River, detailed terrain, 
natural erosion, realistic elevation, game ready"""

negative_prompt = """color, texture, vegetation, snow, 
artificial, urban, noise, blur"""
```

## 参考资源

- **Civitai**: Game Landscape Heightmap Generator
- **基础架构**: Stable Diffusion 1.5
- **推荐工具**: Blender, Unity, Unreal Engine 用于导入高度图
