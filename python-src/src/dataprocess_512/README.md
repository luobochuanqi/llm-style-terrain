# 512×512 数据预处理模块

本模块提供高度图数据集的完整预处理管道（512×512 分辨率），包含数据归一化、特征提取和可视化分析功能。

## 快速开始

```bash
cd python-src

# 第一阶段：数据预处理 (512×512 归一化 + 分层划分)
uv run python -m src.dataprocess_512.preprocess

# 第二阶段：特征提取 (S/R/C 三维度)
uv run python -m src.dataprocess_512.extract_features

# 第三阶段：可视化分析 (t-SNE/UMAP + 直方图)
uv run python -m src.dataprocess_512.visualize_features
```

## 脚本说明

### 1. preprocess.py - 数据预处理

**功能**：
- 读取 Danxia/Kasite 目录下的 16-bit PNG 高度图（512×512）
- 像素值归一化到 0.0-1.0 浮点数范围
- **保持原始 512×512 分辨率**（不下采样）
- 按地貌类型分层抽样，70/15/15 划分训练/验证/测试集

**输出**：
- `data/training-dataset/preprocess/normalized_512/{train,val,test}/` - 归一化图像
- `data/training-dataset/preprocess/splits_512/{train,val,test}.csv` - 划分文件列表

**文件命名**：`{type}_{split}_{id:03d}.png`

---

### 2. extract_features.py - 特征提取

**功能**：
- 计算三个特征维度：
  - **S (坡度)**: Sobel 算子梯度幅值的均值
  - **R (粗糙度)**: 9×9 窗口内标准差的中位数
  - **C (复杂度)**: Laplacian 算子响应的方差
- Min-Max 归一化到 0-10 分

**输出**：
- `data/training-dataset/preprocess/features_512.csv`

**CSV 格式**：
| 列名 | 说明 |
|------|------|
| `filename` | 文件相对路径 |
| `type` | 地貌类型 (Danxia/Kasite) |
| `S_raw`, `R_raw`, `C_raw` | 原始特征值 |
| `S_score`, `R_score`, `C_score` | 0-10 归一化打分 |

---

### 3. visualize_features.py - 可视化分析

**功能**：
- 读取 `features_512.csv` 中的特征数据
- 使用 t-SNE 将三维特征降维至 2D，绘制散点图
- 使用 UMAP 将三维特征降维至 2D，绘制散点图
- 绘制 S/R/C 三维特征的分布直方图
- Danxia 和 Kasite 用不同颜色区分

**输出**：
- `data/training-dataset/preprocess/visuals_512/tsne_plot.png`
- `data/training-dataset/preprocess/visuals_512/umap_plot.png`
- `data/training-dataset/preprocess/visuals_512/feature_distribution.png`

**判定标准**：
- ✅ 成功：两种地貌在降维空间中各自形成明显的簇
- ❌ 失败：点完全混杂，需增加特征维度

---

## 与 256×256 版本的区别

| 特性 | 256×256 版本 | 512×512 版本 |
|------|-------------|-------------|
| 模块路径 | `src.dataprocess` | `src.dataprocess_512` |
| 图像分辨率 | 256×256 | 512×512 |
| 输出目录 | `normalized/`, `splits/`, `features.csv`, `visuals/` | `normalized_512/`, `splits_512/`, `features_512.csv`, `visuals_512/` |
| 特征值范围 | 较小（下采样后细节丢失） | 较大（保留更多细节） |
| 计算速度 | 快 | 较慢 |
| 适用场景 | 快速验证、模型训练 | 高分辨率分析、精细特征提取 |

---

## 依赖

以下依赖已在 `pyproject.toml` 中配置：

```toml
"pandas>=2.0.0",
"scikit-learn>=1.3.0",
"umap-learn>=0.5.0",
"opencv-python>=4.8.0",
"pillow>=9.0.0",
"numpy>=1.24.0",
"matplotlib>=3.10.8",
```

运行 `uv sync` 安装所有依赖。

## 技术细节

### 特征计算公式

**S (坡度)**:
```python
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
S = np.mean(np.sqrt(sobelx**2 + sobely**2))
```

**R (粗糙度)**:
```python
kernel = np.ones((9, 9), np.float64) / 81.0
mean = cv2.filter2D(img, -1, kernel)
mean_sq = cv2.filter2D(img**2, -1, kernel)
variance = np.clip(mean_sq - mean**2, 0, None)
R = np.median(np.sqrt(variance))
```

**C (复杂度)**:
```python
lap = cv2.Laplacian(img, cv2.CV_64F)
C = np.var(lap)
```

### 分层抽样

使用 `sklearn.model_selection.train_test_split` 的 `stratify` 参数确保训练/验证/测试集中 Danxia 和 Kasite 的比例与原始数据集一致 (~54/46)。

随机种子设置为 42 以保证结果可复现。

## 输出结构

```
data/training-dataset/preprocess/
├── normalized_512/           # 512×512 归一化图像
│   ├── train/ (87 张)
│   ├── val/ (19 张)
│   └── test/ (19 张)
├── splits_512/               # 数据划分 CSV
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── features_512.csv          # 特征提取结果
└── visuals_512/              # 可视化分析结果
    ├── tsne_plot.png
    ├── umap_plot.png
    └── feature_distribution.png
```
