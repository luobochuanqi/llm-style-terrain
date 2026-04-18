# HeightmapStyle 模型

专门的 `dimentox/heightmapstyle` 高度图模型加载器和演示。

## 模型信息

- **模型 ID**: `dimentox/heightmapstyle`
- **基础架构**: Stable Diffusion 1.x (不是 SDXL!)
- **用途**: 专门用于高度图生成的微调模型
- **HuggingFace**: https://huggingface.co/dimentox/heightmapstyle

## 快速开始

### 从项目根目录运行

```bash
# 运行演示
python heightmapstyle_demo.py
```

### 从 python-src 目录运行

```bash
cd python-src

# 运行演示
python -m heightmapstyle
```

## 使用方法

### 方法 1: 使用演示脚本

```bash
python heightmapstyle_demo.py
```

### 方法 2: 在 Python 代码中使用

```python
from heightmapstyle import HeightmapStyleInference

# 创建推理器
inferencer = HeightmapStyleInference()

# 微调高度图
refined = inferencer.refine_heightmap(
    heightmap,
    output_path="output.png",
    num_inference_steps=25,
    guidance_scale=7.5,
    strength=0.75,
)
```

### 方法 3: 使用便捷函数

```python
from heightmapstyle import load_heightmap_style

refined = load_heightmap_style(
    heightmap,
    model_id="dimentox/heightmapstyle",
    output_path="output.png"
)
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_inference_steps` | 25 | 推理步数 |
| `guidance_scale` | 7.5 | 引导比例（SD 1.x 推荐 7.5） |
| `strength` | 0.75 | 图生图强度（0-1） |

## 与 SDXL 的区别

| 特性 | SD 1.x (heightmapstyle) | SDXL |
|------|------------------------|------|
| 提示词长度 | 77 tokens | 77 + 224 tokens |
| 图像质量 | 好 | 更好 |
| 显存占用 | ~4-6GB | ~8-10GB |
| 推理速度 | 快 | 较慢 |
| 模型大小 | ~5GB | ~7GB |

## 注意事项

1. **提示词长度限制**: SD 1.x 只支持 77 tokens，提示词过长会被截断
2. **模型类型**: 这是 SD 1.x 模型，不能用 SDXL 的 pipeline 加载
3. **首次运行**: 会下载模型（约 5GB）
4. **显存优化**: 默认启用 CPU offload

## 输出文件

运行演示后会生成：
- `outputs/heightmapstyle/heightmap_styled.png` - 微调后的高度图

## 故障排除

### 问题：提示词被截断

**解决**: 缩短提示词到 77 tokens 以内

### 问题：显存不足

**解决**: 
```python
inferencer.pipe.enable_model_cpu_offload()
```

### 问题：模型下载失败

**解决**: 手动下载模型
```bash
huggingface-cli download dimentox/heightmapstyle
```

## 参考资源

- [HuggingFace 模型页面](https://huggingface.co/dimentox/heightmapstyle)
- [Stable Diffusion 文档](https://huggingface.co/docs/diffusers)
- [Diffusers GitHub](https://github.com/huggingface/diffusers)
