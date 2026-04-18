# Python Scripts 索引

## 目录结构

```
python/
├── scripts/                 # 入口脚本目录
│   ├── README.md            # 本文件（索引导航）
│   ├── main.py              # 主工作流：Perlin + SDXL
│   ├── train_cvae.py        # cVAE 训练
│   ├── generate.py          # cVAE 推理
│   ├── train_unetcvae.py    # U-Net cVAE 训练
│   ├── infer_unetcvae.py    # U-Net cVAE 推理
│   └── ...
├── src/                     # 核心模块
│   ├── config.py            # main.py 专用配置
│   ├── models/              # 模型定义
│   │   ├── cvae/            # cVAE 模型
│   │   └── unetcvae/        # U-Net cVAE 模型
│   ├── training/            # 训练代码
│   │   ├── cvae/            # cVAE 训练
│   │   └── unetcvae/        # U-Net cVAE 训练
│   ├── dataprocess/         # 数据预处理（统一 256/512）
│   ├── generators/          # Perlin 噪声生成
│   ├── diffusion/           # SDXL 推理
│   └── inference/           # 推理工具
└── outputs/                 # 所有输出（gitignored）
```

## 快速开始

### 1. 安装依赖
```bash
cd python
uv sync
```

### 2. 主工作流（Perlin + SDXL）
```bash
uv run python scripts/main.py
```

### 3. cVAE 训练与推理
```bash
# 训练（debug/fast/full 模式）
uv run python scripts/train_unetcvae.py --fast

# 推理：从条件向量生成
uv run python scripts/infer_unetcvae.py generate \
    --checkpoint outputs/unetcvae/checkpoints/model_best.pth \
    --condition "4.0 5.0 1.5" \
    --output outputs/generated/danxia.png
```

### 4. 数据预处理
```bash
# 256x256
uv run python -m src.dataprocess.preprocess

# 512x512
uv run python -m src.dataprocess.preprocess --size 512
```

## 脚本分类

### 主工作流
| 脚本 | 功能 | 用法文档 |
|------|------|----------|
| `main.py` | Perlin 噪声 → SDXL  refinement | [main.md](main.md) |

### cVAE  Pipeline
| 脚本 | 功能 | 用法文档 |
|------|------|----------|
| `train_cvae.py` | cVAE 训练 | [train_cvae.md](train_cvae.md) |
| `generate.py` | cVAE 推理 | [generate.md](generate.md) |
| `train_unetcvae.py` | U-Net cVAE 训练 | [train_unetcvae.md](train_unetcvae.md) |
| `infer_unetcvae.py` | U-Net cVAE 推理 | [infer_unetcvae.md](infer_unetcvae.md) |

### Demo 脚本
| 脚本 | 功能 |
|------|------|
| `controlnet_demo.py` | ControlNet 高度图 refinement |
| `gamelandscape_demo.py` | GameScape 生成 |
| `heightmapstyle_demo.py` | 风格迁移 |
| `mapgen_demo.py` | MapGen 生成 |
| `test.py` | Brownian bridge demo |

## 关键概念

### 条件向量（S/R/C）
- **S (Sharpness)**: 锐利度 1-6（坡度）
- **R (Ruggedness)**: 崎岖度 1-7（粗糙度）
- **C (Complexity)**: 复杂度 1-4

典型地貌范围：
- **丹霞地貌**: S=3.5-5.0, R=4.0-6.0, C=1.0-2.0
- **喀斯特地貌**: S=2.0-4.0, R=4.0-6.5, C=1.5-3.0
- **沙漠地貌**: S=3.0, R=2.0, C=0.5
- **高山地貌**: S=8.0, R=8.0, C=2.0
- **平原地貌**: S=1.0, R=1.5, C=0.3

### 输出目录
所有输出保存在 `outputs/` 目录（已 gitignored）：
- `outputs/perlin/` - Perlin 噪声
- `outputs/diffusion/` - SDXL refinement
- `outputs/cvae/` - cVAE 训练输出
- `outputs/unetcvae/` - U-Net cVAE 训练输出
- `outputs/inference/` - 推理生成

## 环境要求

- **Python 3.11+**
- **GPU required** for SDXL/cVAE (CUDA)
- **VRAM**: ≥8GB for SDXL, ≥4-5GB for cVAE (with AMP)
- 使用 `uv` 进行依赖管理
