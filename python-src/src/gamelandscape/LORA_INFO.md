# GameLandscape 完整模型说明

## 模型类型

**完整微调 Checkpoint (Full Fine-tuned Model)** - 独立的 SD 1.5 模型

## 文件信息

| 文件 | 大小 | 类型 | 用途 |
|------|------|------|------|
| `gameLandscape_gameLandscapeHeightmap.safetensors` | 3.9GB | 完整模型 | ✅ **推荐使用** |
| `GameLandscapeHeightmap512_V1.0.safetensors` | 289MB | LoRA 权重 | ⚠️ 备用（效果不如完整模型） |

## 为什么使用完整模型？

### LoRA vs 完整模型对比

| 特性 | LoRA (289MB) | 完整模型 (3.9GB) |
|------|--------------|------------------|
| **文件大小** | ✅ 小 | ❌ 大 |
| **生成质量** | ⚠️ 一般 | ✅ **最佳** |
| **依赖** | ❌ 需 SD 1.5 基础模型 | ✅ 独立运行 |
| **一致性** | ⚠️ 受基础模型影响 | ✅ **与展示一致** |
| **首次运行** | ❌ 需下载 4GB 基础模型 | ✅ 无需下载 |

### 问题诊断

如果你发现生成的地形**与 Civitai 展示的效果不符**，可能是因为：

1. ❌ 使用了 LoRA 版本 + 不同的基础模型
2. ✅ 解决方法：使用完整模型（3.9GB）

## 工作原理

```
┌─────────────────────────────────────┐
│  完整模型 Checkpoint (3.9GB)        │
│  gameLandscape_gameLandscape...    │
│  (包含完整的 UNet, VAE, TextEncoder)│
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  直接加载完整权重                    │
│  (无需基础模型)                      │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  游戏地形高度图生成                  │
│  (与 Civitai 展示效果一致)           │
└─────────────────────────────────────┘
```

## 使用方法

### 默认（推荐）- 使用完整模型

```python
from src.gamelandscape import GameLandscapeInference

# 自动使用完整模型
inferencer = GameLandscapeInference(
    model_path="../../assets/gameLandscape_gameLandscapeHeightmap.safetensors"
)

# 生成地形
heightmap = inferencer.generate_heightmap(
    output_path="outputs/gamelandscape/result.png",
    terrain_type="Mountain",
)
```

### 备用 - 使用 LoRA（不推荐）

```python
# 仅当磁盘空间不足时使用
inferencer = GameLandscapeInference(
    model_path="../../assets/GameLandscapeHeightmap512_V1.0.safetensors"
)
# 注意：需要自行修改 load_model 方法使用 load_lora_weights
```

## 模型结构

完整模型包含：
- ✅ **UNet** (686 个参数层) - 去噪网络
- ✅ **VAE** (248 个参数层) - 变分自编码器
- ✅ **Text Encoder** - 文本理解
- ✅ **完整权重** - 作者训练的所有参数

## 性能要求

| 项目 | 要求 |
|------|------|
| **显存** | 最低 4GB（启用 CPU offload） |
| **推荐显存** | 8GB+ |
| **磁盘空间** | 4GB（模型文件） |
| **首次运行** | 无需下载（模型已在本地） |

## 文件验证

```bash
# 检查完整模型文件
ls -lh assets/gameLandscape_gameLandscapeHeightmap.safetensors

# 应该显示约 3.9GB
-rw-r--r--  3.9G  gameLandscape_gameLandscapeHeightmap.safetensors
```

## 故障排除

### 问题 1: 生成效果不如预期

**症状**：生成的地形不像高度图，或有奇怪的色彩

**解决**：
1. 确认使用的是完整模型（3.9GB）
2. 检查提示词是否包含地形类型关键词
3. 调整 `guidance_scale` 到 7.0-9.0

### 问题 2: 显存不足

**症状**：CUDA out of memory

**解决**：
```python
# 代码已自动启用 CPU offload，如果仍不足：
# 1. 降低分辨率
heightmap = inferencer.generate_heightmap(
    width=256,  # 从 512 降低到 256
    height=256,
)

# 2. 减少推理步数
heightmap = inferencer.generate_heightmap(
    num_inference_steps=15,  # 从 25 降低到 15
)
```

### 问题 3: 模型加载失败

**症状**：`Error loading safetensors`

**解决**：
1. 确认文件完整性（重新下载如果损坏）
2. 检查路径是否正确
3. 确保 `diffusers` 和 `safetensors` 库已安装

## 与 HeightmapStyle 对比

| 特性 | GameLandscape | HeightmapStyle |
|------|---------------|----------------|
| **模型类型** | 完整 SD 1.5 | LoRA |
| **基础架构** | SD 1.5 | SD 1.x |
| **文件大小** | 3.9GB | ~200MB |
| **专长** | 游戏地形 | 通用高度图风格 |
| **文生图** | ✅ 支持 | ✅ 支持 |
| **图生图** | ✅ 支持 | ✅ 支持 |

## 参考资源

- **Civitai 页面**：Game Landscape Heightmap Generator
- **推荐提示词**：使用地形类型关键词（Mountain, River, etc.）
- **负面提示词**：color, texture, buildings, noise
