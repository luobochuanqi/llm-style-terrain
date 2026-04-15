# cVAE 512 分辨率训练指南

## 📋 概述

基于 Phase 1 优化配置（梯度损失 + Cosine LR + Beta=2.0），针对 512x512 高分辨率图像的训练配置。

## 🎯 关键调整

### 与 256 分辨率配置对比

| 参数 | 256 配置 | 512 配置 | 说明 |
|------|----------|----------|------|
| **分辨率** | 256x256 | **512x512** | 4 倍像素信息 |
| **Batch size** | 8 | **4** | 显存占用增加 4 倍 |
| **Latent dim** | 128 | **256** | 更大隐空间捕捉细节 |
| **输出目录** | outputs/cvae | **outputs/cvae_512** | 独立输出 |

### 保持不变的优化

- ✅ 梯度损失 (Sobel 边缘检测)
- ✅ Cosine Annealing 学习率 (1e-4 → 1e-6)
- ✅ Beta=2.0, warmup=150 epochs
- ✅ 禁用早停
- ✅ 加权采样 (Danxia 1.0 : Kasite 1.15)

## 🚀 快速开始

### 1. 数据集路径

```python
data_root = Path("../data/training-dataset/preprocess/normalized_512")
features_csv = Path("../data/training-dataset/preprocess/features_512.csv")
splits_dir = Path("../data/training-dataset/preprocess/splits_512")
```

### 2. 训练命令

```bash
cd python-src

# 调试模式（10 epochs, 验证代码）
uv run python train_cvae_512.py --debug

# 快速模式（50 epochs, 快速验证）
uv run python train_cvae_512.py --fast

# 完整训练（200 epochs, 最终结果）
uv run python train_cvae_512.py --full
```

### 3. 预期训练时间

| 模式 | Epochs | 预计时间 | 目的 |
|------|--------|----------|------|
| Debug | 10 | 5-10 分钟 | 代码验证 |
| Fast | 50 | 40-80 分钟 | 快速验证 |
| Full | 200 | 3-6 小时 | 最终训练 |

**注意**: 512 分辨率训练时间约为 256 分辨率的 1.5-2 倍（更大图像，更小 batch size）

## 📊 配置文件

### 核心配置 (`src/train_cvae/config_512.py`)

```python
@dataclass
class TrainingConfig512:
    # 数据
    image_size: int = 512
    batch_size: int = 4  # 减小以适应显存
    
    # 模型
    latent_dim: int = 256  # 增加到 256
    
    # Phase 1 优化
    beta: float = 2.0
    beta_warmup_epochs: int = 150
    use_gradient_loss: bool = True
    gradient_loss_weight: float = 0.2
    lr_scheduler_type: str = "cosine"
    lr_min: float = 1e-6
    
    # 训练
    num_epochs: int = 200
    early_stop_patience: None  # 禁用早停
```

## 📈 预期效果

### 优势
- ✅ **更多细节**: 512x512 保留更多地貌细节
- ✅ **更清晰边缘**: 高分辨率下梯度损失效果更明显
- ✅ **更好泛化**: 高分辨率训练提升模型容量

### 挑战
- ⚠️ **显存需求**: 需要 ~12-16GB VRAM (batch_size=4)
- ⚠️ **训练时间**: 增加 50-100%
- ⚠️ **数据质量**: 需要高质量的 512 训练数据

## 🔍 训练监控

### 关键指标
- **Val loss**: 预期 5000-7000 范围（比 256 略高，因为更多细节）
- **Grad loss**: 预期 0.015-0.025（边缘检测质量）
- **KL/Recon 比例**: 预期 12-18%（健康平衡）

### 输出文件
```
outputs/cvae_512/
├── checkpoints/
│   ├── model_best.pth      # 最佳验证损失
│   └── model_final.pth     # 最终 epoch 200
├── samples/
│   ├── epoch_XXX_samples.png
│   └── interpolation.png
├── plots/
│   ├── condition_distribution.png
│   └── training_curves.png
└── logs/
    └── training_history.csv
```

## 🎯 与 256 分辨率模型对比

### 预期改善
- **边缘清晰度**: +30-40% (512 + 梯度损失)
- **纹理细节**: +25-35% (更高分辨率)
- **整体质量**: 7.5/10 → **8.5-9/10**

### 推理使用
```bash
# 使用 512 模型生成
uv run python generate.py --checkpoint outputs/cvae_512/checkpoints/model_best.pth --size 512
```

## ⚠️ 注意事项

### 显存需求
- **推荐**: RTX 3090/4090 (24GB)
- **最低**: RTX 3060 (12GB, batch_size=2)
- **如果 OOM**: 减小 batch_size 到 2 或启用 gradient accumulation

### 数据质量
- 确保 512 数据是真实高分辨率，而非 256 插值
- 检查 features_512.csv 与图像匹配
- 验证 splits_512 划分合理性

## 🐛 故障排除

### 常见问题

**Q: CUDA OOM 错误**
```bash
# 解决方案 1: 减小 batch_size
config.batch_size = 2

# 解决方案 2: 启用 gradient accumulation (需要修改 trainer)
```

**Q: 训练损失不下降**
- 检查数据路径是否正确
- 验证 features_512.csv 格式
- 运行 --debug 模式排查

**Q: 加载 checkpoint 失败**
```python
# 移除 Sobel buffers
state_dict = checkpoint["model_state_dict"].copy()
state_dict.pop("sobel_x", None)
state_dict.pop("sobel_y", None)
model.load_state_dict(state_dict, strict=False)
```

## 📝 更新日志

- **2024-04-15**: 初始版本，基于 Phase 1 优化配置
- 配置：512x512, batch=4, latent=256
- 优化：梯度损失 + Cosine LR + Beta=2.0
- 数据集：normalized_512 + features_512 + splits_512

---

**建议**: 先用 --fast 模式验证配置正确，再运行 --full 完整训练！
