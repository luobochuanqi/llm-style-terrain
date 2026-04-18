# LLM Style Terrain - Python 源码

Perlin 噪声高度图生成 + SDXL 图像到图像优化 + **Residual U-Net cVAE 地形生成**。

## 项目结构

```
python-src/
├── main.py                     # 主入口：Perlin 噪声 + SDXL 优化
├── test.py                     # 测试脚本 (Brownian 桥模拟)
├── mapgen_demo.py              # DiT 地图生成演示
├── heightmapstyle_demo.py      # 高度图风格迁移演示
├── gamelandscape_demo.py       # 游戏景观生成演示
├── controlnet_demo.py          # ControlNet 推理演示
├── train_cvae.py               # cVAE 训练入口 (原始架构)
├── train_unetcvae.py           # Residual U-Net cVAE 训练入口
├── infer_unetcvae.py           # Residual U-Net cVAE 推理工具
├── src/
│   ├── config.py               # 配置数据类
│   ├── generators/             # Perlin 噪声生成
│   ├── diffusion/              # SDXL 推理
│   ├── mapgen/                 # DiT 地图生成 (实验性)
│   ├── heightmapstyle/         # 高度图风格迁移
│   ├── gamelandscape/          # 游戏景观生成
│   ├── dataprocess/            # 256×256 数据预处理 (三阶段管道)
│   ├── dataprocess_512/        # 512×512 数据预处理 (三阶段管道)
│   ├── data/                   # 旧版数据模块 (保留兼容)
│   ├── train_cvae/             # cVAE 训练模块 (原始架构)
│   └── train_unetcvae/         # Residual U-Net cVAE 训练模块
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

### 8. train_cvae.py - cVAE 训练 (原始架构)

训练原始 cVAE 模型用于地形风格生成。

```bash
# 调试模式 (10 epochs)
uv run python train_cvae.py --debug

# 快速模式 (50 epochs)
uv run python train_cvae.py --fast

# 完整模式 (200 epochs)
uv run python train_cvae.py --full
```

**输出**: `outputs/cvae/`

---

### 9. train_unetcvae.py - Residual U-Net cVAE 训练 ⭐

训练基于 Residual U-Net 架构的 cVAE 模型，具有更好的特征提取能力。

**架构特点**:
- Residual U-Net: 4 层下采样/上采样，ResNet Block 保证深层训练稳定
- FiLM 条件化：将 S/R/C 向量转换为 style code，通过 γ·h + β 调制 Decoder
- Skip Connections: Encoder 特征直接传递到 Decoder，保留空间细节
- AMP 混合精度：1.5-2x 速度提升，50% 显存降低

```bash
# 调试模式 (10 epochs, 验证代码)
uv run python train_unetcvae.py --debug

# 快速模式 (50 epochs, 快速验证)
uv run python train_unetcvae.py --fast

# 完整模式 (200 epochs, 最终训练)
uv run python train_unetcvae.py --full

# 禁用 AMP (排查问题)
uv run python train_unetcvae.py --full --no-amp
```

**输出**: `outputs/unetcvae/`
- `checkpoints/` - 模型检查点
- `samples/` - 训练过程中的生成样本
- `plots/` - 训练曲线图
- `logs/` - 训练日志 (CSV 格式)

**模型架构**:
```
Encoder: 256×256 → [4×ResBlock 下采样] → 16×16 → μ,logvar (128)
Decoder: z(128) + condition(3) → [FiLM-ResBlock + Upsample] → 256×256
```

---

### 10. infer_unetcvae.py - Residual U-Net cVAE 推理工具 ⭐

使用训练好的 U-Net cVAE 模型进行地形生成。

```bash
# 从条件向量生成单个地形
uv run python infer_unetcvae.py generate \
    --checkpoint outputs/unetcvae/checkpoints/model_best.pth \
    --condition "4.0 5.0 1.5" \
    --output outputs/generated/danxia.png

# 风格插值 (在两个地貌之间平滑过渡)
uv run python infer_unetcvae.py interpolate \
    --checkpoint outputs/unetcvae/checkpoints/model_best.pth \
    --start "2.0 3.0 1.0" \
    --end "5.0 6.0 2.5" \
    --steps 10 \
    --output outputs/interpolation/grid.png

# 批量生成 (从 CSV 文件)
uv run python infer_unetcvae.py batch \
    --checkpoint outputs/unetcvae/checkpoints/model_best.pth \
    --conditions example_conditions.csv \
    --output-dir outputs/batch/
```

**条件向量格式**: `"S R C"` (空格分隔)
- **S (Sharpness)**: 1-6，锐度/清晰度 (2.0=平缓，5.0=陡峭)
- **R (Ruggedness)**: 1-7，崎岖度/起伏 (3.0=平坦，6.0=崎岖)
- **C (Complexity)**: 1-4，复杂度/细节 (1.0=简单，3.0=复杂)

**地貌类型参考值**:
```
丹霞地貌：  S=3.5-5.0, R=4.0-6.0, C=1.0-2.0
喀斯特地貌：S=2.0-4.0, R=4.0-6.5, C=1.5-3.0
```

**输出格式**: 16-bit PNG 高度图 (256×256)
- 黑色 = 山谷 (0)
- 白色 = 山峰 (65535)

---

### 11. tools/png2raw.py - 格式转换工具

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

### 1. Perlin 噪声 + SDXL 配置

编辑 `src/config.py` 修改以下配置：

- `GeneratorConfig`: Perlin 噪声参数 (尺寸、缩放、octaves、种子)
- `DiffusionConfig`: SDXL 模型路径、推理步数、提示词
- `ControlNetConfig`: ControlNet 模型、conditioning scale、Canny 阈值
- `OutputConfig`: 输出目录和文件名

### 2. cVAE 训练配置

编辑 `src/train_cvae/config.py`:
- 训练模式 (debug/fast/full)
- Batch size、学习率、beta warmup
- 数据增强选项

### 3. Residual U-Net cVAE 训练配置

编辑 `src/train_unetcvae/config.py`:
- 训练模式 (debug/fast/full)
- U-Net 通道配置
- AMP 混合精度开关
- Gradient loss 权重

## 环境要求

- Python 3.11+
- GPU (用于 SDXL 推理和模型训练，支持 CUDA)
- 显存建议：
  - SDXL 推理：≥8GB (启用 CPU offload 可降低要求)
  - cVAE 训练：≥6GB
  - Residual U-Net cVAE 训练：≥8GB (启用 AMP 可降低至 4-5GB)

## 输出格式

- **Perlin 高度图**: 8-bit 灰度 RAW 文件，黑色=谷底，白色=山峰
- **SDXL 优化**: PNG 格式，保留高度信息的灰度图
- **归一化数据**: 256×256 或 512×512 浮点 PNG (0.0-1.0 范围)
- **cVAE 生成**: 16-bit PNG 高度图 (256×256)
- **Residual U-Net cVAE 生成**: 16-bit PNG 高度图 (256×256)
