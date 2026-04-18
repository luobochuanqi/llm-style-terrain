# train_cvae.py - cVAE 训练

## 功能描述

训练标准 cVAE（带 FiLM 条件化的变分自编码器）模型，学习地形风格的隐空间表示。

## 基本用法

```bash
cd python
uv run python scripts/train_cvae.py [模式]
```

## 常用示例

### 调试模式（5-10 分钟）
```bash
uv run python scripts/train_cvae.py --debug
```

### 快速模式（30-60 分钟）
```bash
uv run python scripts/train_cvae.py --fast
```

### 完整模式（2-4 小时）
```bash
uv run python scripts/train_cvae.py --full
```

### 从检查点恢复训练
```bash
uv run python scripts/train_cvae.py --resume
```

### 风格插值演示
```bash
uv run python scripts/train_cvae.py --interpolate
```

## 参数说明

| 参数 | 说明 |
|------|------|
| `--debug` | 调试模式：10 epochs |
| `--fast` | 快速模式：50 epochs |
| `--full` | 完整模式：200 epochs（默认） |
| `--resume` | 从检查点恢复训练 |
| `--interpolate` | 训练完成后执行风格插值演示 |

## 训练配置

配置在 `src/training/cvae/config.py` 中：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `image_size` | 256 | 输入图像尺寸 |
| `latent_dim` | 128 | 隐空间维度 |
| `batch_size` | 16 | 批次大小 |
| `learning_rate` | 1e-4 | 学习率 |
| `beta` | 1.0 | KL 损失权重 |
| `warmup_epochs` | 10 | beta warmup 轮数 |

## 输出文件

| 文件/目录 | 位置 | 说明 |
|-----------|------|------|
| 检查点 | `outputs/cvae/checkpoints/` | 模型权重 |
| 训练日志 | `outputs/cvae/logs/` | TensorBoard 日志 |
| 可视化 | `outputs/cvae/visuals/` | 重建对比 |

最佳模型保存为 `model_best.pth`（验证损失最低）。

## 注意事项

- **需要 GPU 显存 ≥4GB**
- 训练数据需提前准备：`uv run python -m src.dataprocess.*`
- 使用 `--debug` 模式验证代码流程
