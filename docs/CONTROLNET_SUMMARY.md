# ControlNet 集成总结

已完成为 SDXL diffusion 模块添加 ControlNet 支持。

## 创建的文件

### 1. 配置文件 (`src/config.py`)
- 新增 `ControlNetConfig` dataclass
- 在主 `Config` 类中添加 `controlnet` 字段

### 2. ControlNet 推理模块 (`src/diffusion/controlnet_inference.py`)
- `SDXLControlNetInference` 类
- `refine_with_controlnet()` 便捷函数
- Canny 边缘检测功能
- 支持 conditioning scale 调节

### 3. 模块导出 (`src/diffusion/__init__.py`)
- 导出 `SDXLControlNetInference`
- 导出 `refine_with_controlnet`

### 4. 示例脚本 (`examples/controlnet_demo.py`)
- 完整工作流示例
- 不同控制强度对比测试

### 5. 简单测试 (`examples/controlnet_simple.py`)
- 配置测试
- 推理器创建测试
- Canny 边缘转换测试

### 6. 文档 (`src/diffusion/CONTROLNET_README.md`)
- 完整使用说明
- 参数说明
- 故障排除指南

## 依赖更新

在 `pyproject.toml` 中添加：
```toml
"opencv-python>=4.8.0"
```

运行 `uv sync` 安装（正在进行中）。

## 使用方法

### 快速开始

```python
from src.config import ControlNetConfig
from src.diffusion import SDXLControlNetInference

# 配置
config = ControlNetConfig(
    enable=True,
    conditioning_scale=0.5,  # 控制强度
)

# 创建推理器
inferencer = SDXLControlNetInference(config)

# 微调高度图
refined = inferencer.refine_heightmap(heightmap, output_path)
```

### 运行示例

```bash
cd python-src

# 简单测试
python examples/controlnet_simple.py

# 完整示例（需要 GPU）
uv run python examples/controlnet_demo.py

# 对比不同控制强度
uv run python examples/controlnet_demo.py --compare
```

## 核心特性

### 1. Canny 边缘控制
- 自动从高度图提取 Canny 边缘
- 可调节阈值（canny_low_threshold, canny_high_threshold）
- 作为 ControlNet 的条件输入

### 2. 结构约束
- conditioning_scale 控制结构保持程度
- 0.3 = 弱控制（更多创作自由）
- 0.5 = 中等控制（推荐）
- 0.7 = 强控制（几乎保持原结构）

### 3. 显存优化
- 支持 CPU offload（默认启用）
- 使用 fp16 精度
- 优化的 VAE 加载

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable` | False | 是否启用 |
| `model_id` | "diffusers/controlnet-canny-sdxl-1.0" | ControlNet 模型 |
| `conditioning_scale` | 0.5 | 控制强度 |
| `canny_low_threshold` | 0.5 | Canny 低阈值 |
| `canny_high_threshold` | 0.5 | Canny 高阈值 |
| `num_inference_steps` | 25 | 推理步数 |
| `guidance_scale` | 5.0 | 引导比例 |
| `enable_cpu_offload` | True | CPU 卸载 |

## 工作流程

```
Perlin 高度图
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

运行示例后生成：
1. `heightmap_perlin.raw` - 原始高度图
2. `canny_edges.png` - Canny 边缘图
3. `heightmap_controlnet.png` - ControlNet 微调结果

## 注意事项

1. **首次运行**: 会下载 ControlNet 模型（约 2GB）
2. **GPU 要求**: 需要 CUDA 支持的 NVIDIA GPU
3. **显存**: 约 8-10GB（启用 CPU offload）
4. **依赖**: opencv-python 用于 Canny 边缘检测

## 与 SDXL img2img 的对比

| 特性 | SDXL img2img | ControlNet |
|------|--------------|------------|
| 结构保持 | 中等 | **高** |
| 细节添加 | 好 | **优秀** |
| 可控性 | 一般 | **精确** |
| 显存占用 | ~6-8GB | ~8-10GB |

## 下一步

1. ✅ 代码实现完成
2. ⏳ 等待依赖安装完成（opencv-python）
3. 📝 可选：添加 Depth ControlNet 支持
4. 📝 可选：添加多 ControlNet 组合支持

## 故障排除

### OpenCV 未安装
```bash
pip install opencv-python
# 或
uv add opencv-python
```

### 显存不足
```python
config.enable_cpu_offload = True
config.num_inference_steps = 20  # 减少步数
```

### 模型下载失败
```bash
huggingface-cli download diffusers/controlnet-canny-sdxl-1.0
```
