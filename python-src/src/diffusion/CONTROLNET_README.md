# ControlNet for SDXL Heightmap Refinement

使用 ControlNet 对高度图进行更精确的结构控制。

## 快速开始

### 1. 安装依赖

```bash
cd python-src
uv sync
```

### 2. 配置 ControlNet

在 `src/config.py` 中修改 `ControlNetConfig`：

```python
from src.config import Config, ControlNetConfig

config = Config()

# 启用 ControlNet
config.controlnet.enable = True

# 设置控制强度（0-1，建议 0.3-0.7）
config.controlnet.conditioning_scale = 0.5

# Canny 边缘检测阈值
config.controlnet.canny_low_threshold = 0.5
config.controlnet.canny_high_threshold = 0.5
```

### 3. 运行示例

```bash
# 完整工作流示例
uv run python examples/controlnet_demo.py

# 对比不同控制强度的效果
uv run python examples/controlnet_demo.py --compare
```

## 使用方法

### 方法 1: 使用配置类

```python
from src.config import ControlNetConfig
from src.diffusion import SDXLControlNetInference

# 创建配置
config = ControlNetConfig(
    enable=True,
    conditioning_scale=0.5,
    canny_low_threshold=0.5,
    canny_high_threshold=0.5,
)

# 创建推理器
inferencer = SDXLControlNetInference(config)

# 执行微调
refined_heightmap = inferencer.refine_heightmap(heightmap, output_path)

# 卸载模型
inferencer.unload_model()
```

### 方法 2: 使用便捷函数

```python
from src.diffusion import refine_with_controlnet

refined = refine_with_controlnet(
    heightmap,
    config=config,
    output_path="output.png"
)
```

## 配置参数说明

### 核心参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable` | bool | False | 是否启用 ControlNet |
| `model_id` | str | "diffusers/controlnet-canny-sdxl-1.0" | ControlNet 模型 |
| `conditioning_scale` | float | 0.5 | 控制强度（0-1） |

### Canny 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `control_type` | str | "canny" | ControlNet 类型 |
| `canny_low_threshold` | float | 0.5 | Canny 低阈值（0-1） |
| `canny_high_threshold` | float | 0.5 | Canny 高阈值（0-1） |

### 推理参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_inference_steps` | int | 25 | 去噪步数 |
| `guidance_scale` | float | 5.0 | 引导比例 |
| `enable_cpu_offload` | bool | True | CPU 卸载（节省显存） |

### 提示词

| 参数 | 类型 | 说明 |
|------|------|------|
| `prompt` | str | 正向提示词（高度图描述） |
| `negative_prompt` | str | 负向提示词（排除光影等效果） |

## Conditioning Scale 效果对比

| 值 | 效果 | 适用场景 |
|----|------|---------|
| 0.3 | 弱控制，SDXL 有更多创作自由 | 需要较大改动时 |
| 0.5 | 中等控制，平衡结构与细节 | **推荐默认值** |
| 0.7 | 强控制，几乎保持原结构 | 只添加细节时 |
| 0.9 | 极强控制，接近原图 | 几乎不改动 |

## Canny 阈值调优

```python
# 检测更多边缘（适合平缓地形）
config.canny_low_threshold = 0.3
config.canny_high_threshold = 0.5

# 检测主要边缘（适合陡峭地形）
config.canny_low_threshold = 0.6
config.canny_high_threshold = 0.7

# 默认（平衡）
config.canny_low_threshold = 0.5
config.canny_high_threshold = 0.5
```

## 工作流程

```
Perlin 噪声高度图
       ↓
Canny 边缘检测
       ↓
ControlNet 条件约束
       ↓
SDXL 图生图
       ↓
微调后的高度图
```

## 输出文件

运行示例后会生成：

1. `heightmap_perlin.raw` - 原始 Perlin 噪声高度图
2. `canny_edges.png` - Canny 边缘图（可视化）
3. `heightmap_controlnet.png` - ControlNet 微调后的高度图

## 与 SDXL img2img 的对比

| 特性 | SDXL img2img | ControlNet |
|------|--------------|------------|
| 结构保持 | 中等（依赖 strength） | **高**（边缘约束） |
| 细节添加 | 好 | **优秀** |
| 可控性 | 一般 | **精确** |
| 显存占用 | ~6-8GB | ~8-10GB |
| 推理速度 | 快 | 中等 |

## 显存优化

如果遇到显存不足：

```python
# 1. 启用 CPU 卸载（默认已启用）
config.enable_cpu_offload = True

# 2. 使用更小的图像尺寸
config.generator.n = 8  # 256x256 而非 1024x1024

# 3. 减少推理步数
config.num_inference_steps = 20

# 4. 使用 float16（默认）
config.torch_dtype = "float16"
```

## 故障排除

### 问题：ControlNet 效果不明显

**解决**：
- 增加 `conditioning_scale`（如 0.5 → 0.7）
- 调整 Canny 阈值以检测更多边缘
- 增加 `num_inference_steps`

### 问题：显存不足

**解决**：
- 确保 `enable_cpu_offload=True`
- 减小高度图尺寸
- 关闭其他 GPU 应用

### 问题：模型下载失败

ControlNet 模型会在首次运行时自动下载（约 2GB）。如果下载失败：

```bash
# 手动下载模型
huggingface-cli download diffusers/controlnet-canny-sdxl-1.0
```

## 扩展：使用其他 ControlNet 类型

目前支持 Canny，还可以添加：

```python
# Depth ControlNet（需要 depth map）
config.model_id = "diffusers/controlnet-depth-sdxl-1.0"
config.control_type = "depth"

# Pose ControlNet（不适用高度图）
# ...
```

## 参考资源

- [Diffusers ControlNet 文档](https://huggingface.co/docs/diffusers/api/pipelines/controlnet)
- [ControlNet 论文](https://arxiv.org/abs/2302.05543)
- [SDXL ControlNet 模型](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0)
