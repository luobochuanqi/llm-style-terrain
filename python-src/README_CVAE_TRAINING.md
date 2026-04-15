# cVAE 地形风格迁移训练

基于条件变分自编码器（cVAE）的地形高度图生成模型，从 S/R/C 风格向量生成 256×256 地形高度图。

## 功能特点

- ✅ **FiLM 条件化**：Feature-wise Linear Modulation，将 3 维风格向量融入解码过程
- ✅ **安全数据增强**：水平翻转 + 90°旋转（禁止垂直翻转以保护地形物理特性）
- ✅ **条件归一化**：C_score 对数变换处理严重偏态分布
- ✅ **加权采样**：处理类别不平衡（丹霞 vs 喀斯特）
- ✅ **Beta Warmup**：线性增加 KL 损失权重，防止过拟合
- ✅ **早停法**：自动监控验证损失，恢复最佳检查点
- ✅ **三种训练模式**：
  - `debug`: 10 epochs (5-10 分钟) - 代码验证
  - `fast`: 50 epochs (30-60 分钟) - 快速验证
  - `full`: 200 epochs (2-4 小时) - 最终训练

## 快速开始

### 调试模式（代码验证）

```bash
cd python-src
uv run python train_cvae.py --debug
```

### 快速训练（推荐用于测试）

```bash
cd python-src
uv run python train_cvae.py --fast
```

### 完整训练（生产使用）

```bash
cd python-src
uv run python train_cvae.py --full
```

### 从检查点恢复

```bash
cd python-src
uv run python train_cvae.py --resume --checkpoint outputs/cvae/checkpoints/model_best.pth
```

### 风格插值演示

```bash
cd python-src
uv run python train_cvae.py --interpolate --checkpoint outputs/cvae/checkpoints/model_best.pth
```

## 输出文件

训练完成后，输出文件位于 `outputs/cvae/` 目录：

```
outputs/cvae/
├── checkpoints/
│   ├── model_best.pth      # 最佳验证损失检查点
│   └── model_final.pth     # 最终检查点
├── samples/
│   ├── epoch_010_samples.png
│   ├── epoch_020_samples.png
│   └── interpolation_grid.png
├── plots/
│   ├── training_curves.png     # 训练曲线
│   └── condition_distribution.png  # 条件分布
└── logs/
    └── training_history.csv    # 训练历史
```

## 模型架构

### Encoder（编码器）
```
Input: (1, 256, 256)
→ Conv2d(1, 32, 4, 2, 1) + BN + ReLU → (32, 128, 128)
→ Conv2d(32, 64, 4, 2, 1) + BN + ReLU → (64, 64, 64)
→ Conv2d(64, 128, 4, 2, 1) + BN + ReLU → (128, 32, 32)
→ Conv2d(128, 256, 4, 2, 1) + BN + ReLU → (256, 16, 16)
→ Conv2d(256, 512, 4, 2, 1) + BN + ReLU → (512, 8, 8)
→ Flatten → Linear → mu (128), logvar (128)
```

### FiLM Decoder（条件化解码器）
```
Input: z (128) + condition (3)
→ condition → FC → 256D style code
→ z → FC → 512×8×8 → Reshape
→ ConvTranspose2d(512, 256, 4, 2, 1) + BN + ReLU + FiLM
→ ConvTranspose2d(256, 128, 4, 2, 1) + BN + ReLU + FiLM
→ ConvTranspose2d(128, 64, 4, 2, 1) + BN + ReLU + FiLM
→ ConvTranspose2d(64, 32, 4, 2, 1) + BN + ReLU + FiLM
→ ConvTranspose2d(32, 1, 4, 2, 1) + Sigmoid → (1, 256, 256)
```

**FiLM 调制**：`h = γ * h + β`，其中 γ, β 由 style code 生成

## 训练配置

关键配置参数（在 `src/train_cvae/config.py` 中）：

```python
@dataclass
class TrainingConfig:
    # 模式选择
    mode: str = "full"  # "debug" | "fast" | "full"
    
    # 模型
    latent_dim: int = 128
    condition_dim: int = 3  # S, R, C
    film_hidden_dim: int = 256
    beta: float = 2.0  # KL 损失权重
    
    # 训练
    num_epochs: int = 200
    learning_rate: float = 1e-4
    batch_size: int = 8
    beta_warmup_epochs: int = 50  # β从 0 线性增加到 2.0
    
    # 早停
    early_stop_patience: int = 20
    
    # 数据增强
    augment_hflip: bool = True
    augment_rotate_90: bool = True
    augment_vertical_flip: bool = False  # 禁止！
```

## 训练监控指标

| 指标 | 期望趋势 | 说明 |
|------|----------|------|
| **Reconstruction Loss** | 持续下降至 5000-7000 | MSE 重建损失（sum over batch） |
| **KL Divergence** | 先降后升，稳定在 200-400 | 隐空间正则化 |
| **Validation Loss** | 跟随训练损失下降 | 泛化能力指标 |
| **Beta** | 从 0 线性增加到 2.0 | KL 权重，前 50 epoch warmup |

## 常见问题

### Q: 训练出现 NaN 怎么办？
A: 检查学习率是否过高。debug 模式使用 1e-5 学习率，正常训练使用 1e-4。

### Q: 生成图像全灰怎么办？
A: 可能是条件向量未被有效利用。检查：
1. ConditionNormalizer 是否正确归一化
2. FiLM 层的 γ, β是否有效调制特征
3. 尝试增加 β值（KL 权重）

### Q: 验证损失不降反升？
A: 这是过拟合迹象。早停法会自动恢复最佳检查点。可以增加：
- beta 值（更强的正则化）
- early_stop_patience（更早停止）
- 数据增强强度

## 技术细节

### 为什么使用 FiLM 而不是简单拼接？

简单拼接 `[z; condition]`会导致：
- 128 维隐向量 `z` 淹没 3 维条件向量
- 模型学会忽略条件，仅从 `z` 重建

FiLM通过特征级调制：
- 条件向量直接影响每层解码器的通道分布
- 3 维条件→256 维 style code→调制 512/256/128/64/32 通道
- 条件影响强度提升 10-100 倍

### C_score 为什么需要对数变换？

`features.csv` 统计显示：
- 80% 样本 C_score < 2.0
- 少数 outlier 达到 10.0
- 严重右偏分布

对数变换 `log(C + 1e-5)` 使分布更接近高斯，便于归一化。

### 为什么禁止垂直翻转？

高度图中，白色=高山，黑色=山谷。
垂直翻转会：
- 山变成坑（地形反转）
- 破坏物理规律（水往高处流）
- 混淆模型学习

## 下一步

1. **训练完整模型**：`uv run python train_cvae.py --full`
2. **评估生成质量**：检查 `outputs/cvae/samples/` 中的插值网格
3. **推理使用**：加载 `model_best.pth` 进行生成
4. **扩展功能**：
   - 添加梯度损失使地形更清晰
   - 多任务学习（地貌分类）
   - 数据量增长后升级到扩散模型

## 相关论文

- [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871)
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
