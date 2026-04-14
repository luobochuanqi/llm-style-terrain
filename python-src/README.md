# LLM Style Terrain - Python 源码

Perlin 噪声高度图生成 + SDXL 图像到图像优化的地形生成器。

## 项目结构

```
python-src/
├── main.py                     # 主入口：Perlin 噪声 + SDXL 优化
├── test.py                     # 测试脚本 (Brownian 桥模拟)
├── mapgen_demo.py              # DiT 地图生成演示
├── heightmapstyle_demo.py      # 高度图风格迁移演示
├── gamelandscape_demo.py       # 游戏景观生成演示
├── controlnet_demo.py          # ControlNet 推理演示
├── src/
│   ├── config.py               # 配置数据类
│   ├── generators/             # Perlin 噪声生成
│   ├── diffusion/              # SDXL 推理
│   ├── mapgen/                 # DiT 地图生成 (实验性)
│   ├── heightmapstyle/         # 高度图风格迁移
│   ├── gamelandscape/          # 游戏景观生成
│   ├── dataprocess/            # 256×256 数据预处理 (三阶段管道)
│   ├── dataprocess_512/        # 512×512 数据预处理 (三阶段管道)
│   └── data/                   # 旧版数据模块 (保留兼容)
├── tools/
│   └── png2raw.py              # PNG 转 RAW 工具
└── outputs/                    # 输出文件 (gitignore)
```

## 入口脚本

### 1. main.py - 地形生成器主程序

生成 Perlin 噪声高度图并使用 SDXL 进行图像到图像优化。

```bash
uv run python main.py
```

**配置**：编辑 `src/config.py` 中的 `GeneratorConfig` 和 `DiffusionConfig`。

**输出**：
- `outputs/perlin/heightmap.raw` - Perlin 噪声原始高度图
- `outputs/diffusion/heightmap.png` - SDXL 优化后的高度图

---

### 2. 数据预处理管道 (256×256)

详见 `src/dataprocess/README.md`。

```bash
# 第一阶段：数据预处理 (256×256)
uv run python -m src.dataprocess.preprocess

# 第二阶段：特征提取
uv run python -m src.dataprocess.extract_features

# 第三阶段：可视化分析
uv run python -m src.dataprocess.visualize_features
```

**输出**：`../data/training-dataset/preprocess/`
- `normalized/` - 256×256 归一化图像
- `splits/` - 数据划分 CSV
- `features.csv` - 特征提取结果
- `visuals/` - 可视化图

---

### 3. 数据预处理管道 (512×512)

详见 `src/dataprocess_512/README.md`。

```bash
# 第一阶段：数据预处理 (512×512)
uv run python -m src.dataprocess_512.preprocess

# 第二阶段：特征提取
uv run python -m src.dataprocess_512.extract_features

# 第三阶段：可视化分析
uv run python -m src.dataprocess_512.visualize_features
```

**输出**：`../data/training-dataset/preprocess/`
- `normalized_512/` - 512×512 归一化图像
- `splits_512/` - 数据划分 CSV
- `features_512.csv` - 特征提取结果
- `visuals_512/` - 可视化图

---

### 4. mapgen_demo.py - DiT 地图生成

基于 Diffusion Transformer 的高度图生成实验。

```bash
uv run python mapgen_demo.py
```

---

### 5. heightmapstyle_demo.py - 风格迁移

高度图风格迁移演示。

```bash
uv run python heightmapstyle_demo.py
```

---

### 6. gamelandscape_demo.py - 游戏景观生成

游戏景观生成演示。

```bash
uv run python gamelandscape_demo.py
```

---

### 7. controlnet_demo.py - ControlNet 推理

ControlNet 条件生成演示。

```bash
uv run python controlnet_demo.py
```

---

### 8. tools/png2raw.py - 格式转换工具

将 PNG 高度图转换为 RAW 格式。

```bash
uv run python tools/png2raw.py input.png output.raw
```

---

## 依赖管理

使用 `uv` 进行依赖管理：

```bash
# 安装所有依赖
uv sync

# 添加新依赖
uv add package_name

# 开发依赖
uv add --dev package_name
```

## 配置

编辑 `src/config.py` 修改以下配置：

- `GeneratorConfig`: Perlin 噪声参数 (尺寸、缩放、octaves、种子)
- `DiffusionConfig`: SDXL 模型路径、推理步数、提示词
- `ControlNetConfig`: ControlNet 模型、conditioning scale、Canny 阈值
- `OutputConfig`: 输出目录和文件名

## 环境要求

- Python 3.11+
- GPU (用于 SDXL 推理，支持 CUDA)
- 显存建议：≥8GB (启用 CPU offload 可降低要求)

## 输出格式

- **Perlin 高度图**: 8-bit 灰度 RAW 文件，黑色=谷底，白色=山峰
- **SDXL 优化**: PNG 格式，保留高度信息的灰度图
- **归一化数据**: 256×256 或 512×512 浮点 PNG (0.0-1.0 范围)
