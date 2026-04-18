# generate.py - cVAE 推理

## 功能描述

使用训练好的 cVAE 模型生成地形高度图，支持预设地貌、自定义风格、插值序列和风格网格。

## 基本用法

```bash
cd python
uv run python scripts/generate.py [模式] [参数]
```

## 常用示例

### 生成预设的 5 种地貌
```bash
uv run python scripts/generate.py --preset
```

### 生成自定义风格地形
```bash
uv run python scripts/generate.py --S 7.0 --R 6.5 --C 1.5 --output custom.png
```

### 批量生成多个地形
```bash
uv run python scripts/generate.py --batch
```

### 风格插值序列
```bash
uv run python scripts/generate.py --interpolate \
    --S1 8.0 --R1 8.0 --C1 2.0 \
    --S2 2.0 --R2 3.0 --C2 0.8 \
    --steps 10
```

### 创建 S-R 风格网格
```bash
uv run python scripts/generate.py --grid-sr --C 1.0
```

## 参数说明

### 通用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--checkpoint` | 模型检查点路径 | `outputs/cvae/checkpoints/model_best.pth` |
| `--output-dir` | 输出目录 | `outputs/inference` |
| `--device` | 设备 (cuda/cpu) | 自动选择 |

### 风格参数

| 参数 | 说明 | 默认值 | 范围 |
|------|------|--------|------|
| `--S` | Sharpness（锐利度） | 5.0 | 0-10 |
| `--R` | Ruggedness（崎岖度） | 5.0 | 0-10 |
| `--C` | Complexity（复杂度） | 1.0 | 0-10 |

### 插值参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--S1, --R1, --C1` | 起始风格 | 8.0, 8.0, 2.0 |
| `--S2, --R2, --C2` | 结束风格 | 2.0, 3.0, 0.8 |
| `--steps` | 插值步数 | 10 |

### 网格参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--C-grid` | 网格固定 C 值 | 1.0 |

### 输出参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--output` | 单张输出文件名 | 自动生成 |
| `--format` | 输出格式 (png/raw) | png |
| `--seed` | 随机种子 | None |

## 输出文件

| 模式 | 输出文件 |
|------|----------|
| `--preset` | `outputs/inference/{地貌类型}.png` |
| 单张 | 指定路径或 `terrain_S{S}_R{R}_C{C}.png` |
| `--batch` | `outputs/inference/terrain_{01-05}.png` |
| `--interpolate` | `outputs/inference/interpolation_{00-09}.png` |
| `--grid-sr` | `outputs/inference/grid_SR.png` |

## 预设地貌类型

| 地貌 | S | R | C | 说明 |
|------|---|---|---|------|
| 丹霞 | 7.0 | 6.5 | 1.5 | 陡峭悬崖、平坦顶部 |
| 喀斯特 | 2.0 | 3.0 | 0.8 | 溶蚀地貌 |
| 沙漠 | 3.0 | 2.0 | 0.5 | 平缓沙丘 |
| 高山 | 8.0 | 8.0 | 2.0 | 高落差山脉 |
| 平原 | 1.0 | 1.5 | 0.3 | 平坦地形 |

## 注意事项

- 检查点文件必须存在
- 使用 `--seed` 可获得可重复结果
- 批量生成和插值会创建多个文件
