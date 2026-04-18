# infer_unetcvae.py - U-Net cVAE 推理

## 功能描述

加载训练好的 U-Net cVAE 模型进行地形生成，支持从条件向量生成、插值、批量生成。

## 基本用法

```bash
cd python
uv run python scripts/infer_unetcvae.py [命令] [参数]
```

## 常用示例

### 从条件向量生成单个地形
```bash
uv run python scripts/infer_unetcvae.py generate \
    --checkpoint outputs/unetcvae/checkpoints/model_best.pth \
    --condition "4.0 5.0 1.5" \
    --output outputs/generated/danxia.png
```

### 风格插值（生成 10 张过渡地形）
```bash
uv run python scripts/infer_unetcvae.py interpolate \
    --checkpoint outputs/unetcvae/checkpoints/model_best.pth \
    --start "2.0 3.0 1.0" \
    --end "5.0 6.0 2.5" \
    --steps 10 \
    --output outputs/interpolation/
```

### 批量生成（从 CSV 读取条件）
```bash
uv run python scripts/infer_unetcvae.py batch \
    --checkpoint outputs/unetcvae/checkpoints/model_best.pth \
    --conditions conditions.csv \
    --output-dir outputs/batch/
```

## 命令说明

### `generate` - 单张生成

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--checkpoint` | 模型检查点路径 | 必需 |
| `--condition` | 条件向量 "S R C" | 必需 |
| `--output` | 输出文件路径 | 必需 |
| `--device` | 设备 (cuda/cpu) | 自动选择 |

### `interpolate` - 风格插值

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--checkpoint` | 模型检查点路径 | 必需 |
| `--start` | 起始条件 "S R C" | 必需 |
| `--end` | 结束条件 "S R C" | 必需 |
| `--steps` | 插值步数 | 10 |
| `--output` | 输出目录 | 必需 |

### `batch` - 批量生成

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--checkpoint` | 模型检查点路径 | 必需 |
| `--conditions` | CSV 文件（每行：S,R,C,name） | 必需 |
| `--output-dir` | 输出目录 | 必需 |

## 条件向量参考

典型地貌的 S/R/C 值：

| 地貌类型 | S (Sharpness) | R (Ruggedness) | C (Complexity) |
|----------|---------------|----------------|----------------|
| 丹霞地貌 | 3.5-5.0 | 4.0-6.0 | 1.0-2.0 |
| 喀斯特地貌 | 2.0-4.0 | 4.0-6.5 | 1.5-3.0 |
| 沙漠地貌 | 3.0 | 2.0 | 0.5 |
| 高山地貌 | 8.0 | 8.0 | 2.0 |
| 平原地貌 | 1.0 | 1.5 | 0.3 |

## 输出文件

- **单张生成**: 指定路径的 PNG 文件
- **插值**: 输出目录中的序列帧 `frame_000.png`, `frame_001.png`, ...
- **批量**: 输出目录中以 name 命名的 PNG 文件

## 注意事项

- 模型检查点必须与训练配置匹配
- 生成时使用 GPU 可显著加速
- 固定 random seed 可获得可重复结果
