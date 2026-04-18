# main.py - 主工作流

## 功能描述

Perlin 噪声生成 → SDXL 图生图 refinement，生成 LLM-style 地形高度图。

## 基本用法

```bash
cd python
uv run python scripts/main.py
```

## 常用示例

### 执行完整工作流
```bash
uv run python scripts/main.py
```

执行后会：
1. 生成 Perlin 噪声高度图（1024x1024）
2. 使用 SDXL 图生图进行 refinement
3. 输出到 `outputs/perlin/` 和 `outputs/diffusion/`

## 参数说明

当前版本无命令行参数，配置在 `src/config.py` 中：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `GeneratorConfig.n` | 10 | 高度图尺寸 2^n（1024x1024） |
| `GeneratorConfig.scale` | 300 | 噪声缩放（越大越平缓） |
| `GeneratorConfig.octaves` | 6 | 噪声层数 |
| `DiffusionConfig.num_inference_steps` | 25 | SDXL 去噪步数 |
| `DiffusionConfig.strength` | 0.4 | 图生图强度（低值保留更多原结构） |
| `DiffusionConfig.enable_cpu_offload` | True | CPU 卸载节省显存 |

## 输出文件

| 文件 | 位置 | 说明 |
|------|------|------|
| Perlin 噪声 RAW | `outputs/perlin/heightmap.raw` | 原始噪声高度图 |
| Perlin 噪声 PNG | `outputs/perlin/heightmap.png` | 可视化预览 |
| SDXL refinement | `outputs/diffusion/heightmap.png` | SDXL refinement 后高度图 |

## 注意事项

- **首次运行会下载 SDXL 模型**（约 7GB）
- **需要 GPU 显存 ≥8GB**（启用 CPU offload）
- 运行时间：Perlin 生成 ~1 分钟，SDXL refinement ~5-10 分钟
