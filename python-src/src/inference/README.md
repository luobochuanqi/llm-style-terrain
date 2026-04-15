# cVAE 地形生成推理模块

使用训练好的 cVAE 模型生成地形高度图。

## 快速开始

### 1. 生成预设的 5 种地貌

```bash
cd python-src
uv run python generate.py --preset
```

生成文件：
- `outputs/inference/danxia.png` - 丹霞地貌
- `outputs/inference/karst.png` - 喀斯特地貌
- `outputs/inference/desert.png` - 沙漠地貌
- `outputs/inference/mountain.png` - 高山地貌
- `outputs/inference/plains.png` - 平原地貌

### 2. 生成自定义风格地形

```bash
uv run python generate.py --S 7.0 --R 6.5 --C 1.5 --output my_terrain.png
```

参数说明：
- `--S`: Sharpness (锐利度) 0-10
- `--R`: Ruggedness (崎岖度) 0-10
- `--C`: Complexity (复杂度) 0-10

### 3. 批量生成

```bash
uv run python generate.py --batch
```

生成 5 个不同风格的随机地形。

### 4. 风格插值

```bash
uv run python generate.py --interpolate --S1 8.0 --R1 8.0 --C1 2.0 --S2 2.0 --R2 3.0 --C2 0.8 --steps 10
```

在两个风格之间生成平滑过渡的 10 张地形图。

### 5. 创建 S-R 风格网格

```bash
uv run python generate.py --grid-sr --C 1.0
```

创建 4x4 的 S-R 风格组合网格。

## Python API 使用

### 基本用法

```python
from src.inference import TerrainGenerator, StyleVector

# 加载模型
generator = TerrainGenerator(
    checkpoint_path="outputs/cvae/checkpoints/model_best.pth",
    device="cuda",  # 或 "cpu"
)

# 使用预设风格
style = StyleVector.danxia()  # 丹霞地貌
heightmap = generator.generate(style)

# 保存
generator.save_heightmap(heightmap, "output.png")
```

### 自定义风格向量

```python
# 创建自定义风格
style = StyleVector(S=7.0, R=6.5, C=1.5)

# 生成（可指定随机种子）
heightmap = generator.generate(style, seed=42)
```

### 风格插值

```python
# 在两种风格之间插值
style_start = StyleVector(S=8.0, R=8.0, C=2.0)
style_end = StyleVector(S=2.0, R=3.0, C=0.8)

heightmaps = generator.interpolate(
    style_start,
    style_end,
    num_steps=10,
    seed=42,  # 固定种子，使用相同的 z
)

# 保存插值序列
for i, h in enumerate(heightmaps):
    generator.save_heightmap(h, f"interp_{i:02d}.png")
```

### 批量生成

```python
from src.inference import BatchGenerator

batch_gen = BatchGenerator(
    checkpoint_path="outputs/cvae/checkpoints/model_best.pth",
    output_dir="outputs/my_terrains",
)

# 批量生成自定义风格
configs = [
    {"name": "terrain_01", "S": 8.0, "R": 7.0, "C": 1.5, "seed": 42},
    {"name": "terrain_02", "S": 3.0, "R": 4.0, "C": 0.8, "seed": 123},
]

batch_gen.generate_styles(configs, format="png")
```

## 风格向量参考

### 预设风格

| 风格 | S (锐利度) | R (崎岖度) | C (复杂度) | 说明 |
|------|-----------|-----------|-----------|------|
| **丹霞** | 7.0 | 6.5 | 1.5 | 陡崖、块状山体 |
| **喀斯特** | 2.0 | 3.0 | 0.8 | 锥状峰林 |
| **沙漠** | 3.0 | 2.0 | 0.5 | 平缓沙丘 |
| **高山** | 8.0 | 8.0 | 2.0 | 陡峭山峰 |
| **平原** | 1.0 | 1.5 | 0.3 | 平坦地形 |

### 参数说明

- **S (Sharpness)**: 锐利度
  - 高值：陡峭、清晰的边缘
  - 低值：平缓、柔和的过渡
  
- **R (Ruggedness)**: 崎岖度
  - 高值：粗糙、不规则的地形
  - 低值：平滑、规则的地形
  
- **C (Complexity)**: 复杂度
  - 高值：丰富的细节和变化
  - 低值：简单的结构

## 输出格式

支持两种格式：

### PNG 格式（推荐）
- 8-bit 灰度图
- 值范围：0-255
- 可直接查看和编辑

```bash
uv run python generate.py --S 5.0 --R 5.0 --C 1.0 --format png
```

### RAW 格式
- 16-bit 二进制数据
- 值范围：0-65535
- 适合专业地形处理

```bash
uv run python generate.py --S 5.0 --R 5.0 --C 1.0 --format raw
```

## 命令行参数

```
用法：generate.py [选项]

必选参数:
  --checkpoint PATH     模型检查点路径 (默认：outputs/cvae/checkpoints/model_best.pth)
  --output-dir DIR      输出目录 (默认：outputs/inference)

生成模式:
  --preset              生成预设的 5 种地貌
  --batch               批量生成多个自定义地形
  --interpolate         生成风格插值序列
  --grid-sr             创建 S-R 风格网格

风格参数:
  --S FLOAT             Sharpness (0-10, 默认：5.0)
  --R FLOAT             Ruggedness (0-10, 默认：5.0)
  --C FLOAT             Complexity (0-10, 默认：1.0)

插值参数:
  --S1, --R1, --C1      起始风格值
  --S2, --R2, --C2      结束风格值
  --steps INT           插值步数 (默认：10)

输出参数:
  --output FILE         单个输出文件名
  --format [png|raw]    输出格式 (默认：png)
  --seed INT            随机种子
```

## 文件结构

```
python-src/
├── src/inference/
│   ├── __init__.py          # 包导出
│   ├── generator.py         # TerrainGenerator, StyleVector
│   └── batch_generator.py   # BatchGenerator
├── generate.py              # CLI 入口
└── outputs/inference/       # 输出目录
```

## 故障排查

### 问题：找不到模型检查点

解决：确保已运行训练
```bash
uv run python train_cvae.py --full
```

### 问题：显存不足

解决：使用 CPU 模式
```bash
uv run python generate.py --preset --device cpu
```

### 问题：生成的图像全灰

可能原因：
1. 模型未正确训练（检查训练损失）
2. 风格向量超出合理范围（确保在 0-10 之间）

## 性能参考

- **GPU (RTX 3060)**: 单张生成约 0.1 秒
- **CPU**: 单张生成约 1-2 秒
- **批量生成**: 5 张地形约 0.5 秒 (GPU)

## 下一步

生成的高度图可以用于：
1. 游戏地形制作
2. 3D 建模高度置换
3. 地理信息系统 (GIS) 分析
4. 艺术创作参考

祝生成愉快！🏔️🏜️🏞️
