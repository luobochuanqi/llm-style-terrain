# HeightmapStyle 模型集成完成

## 文件结构

```
llm-style-terrain/
├── docs/
│   ├── HEIGHTMAPSTYLE_SETUP.md     # 集成说明
│   ├── CONTROLNET_SUMMARY.md       # ControlNet 总结
│   └── controlnet_run_20260408.md  # ControlNet 运行日志
├── python-src/
│   ├── heightmapstyle_demo.py      # HeightmapStyle 演示
│   ├── controlnet_demo.py          # ControlNet 演示
│   └── src/
│       ├── heightmapstyle/         # 专用模型文件夹
│       │   ├── __init__.py
│       │   ├── model_loader.py     # 模型加载器
│       │   └── README.md           # 详细文档
│       ├── config.py               # 已改回 SDXL 默认
│       └── ...
└── outputs/heightmapstyle/         # 输出目录
```

## 恢复的更改

✅ **SDXL 配置已恢复**
- `src/config.py` 中的 `model_id` 改回 `"stabilityai/stable-diffusion-xl-base-1.0"`
- 原有的 SDXL 工作流不受影响

## 新增功能

✅ **独立的 HeightmapStyle 模块**
- 位置：`python-src/src/heightmapstyle/`
- 专门加载 `dimentox/heightmapstyle` 模型（基于 SD 1.x）
- 不影响原有的 SDXL 代码

✅ **ControlNet 演示**
- 位置：`python-src/controlnet_demo.py`
- 展示 ControlNet 在高度图上的应用

✅ **演示入口**
- `python-src/heightmapstyle_demo.py`
- `python-src/controlnet_demo.py`

## 使用方法

### 运行 HeightmapStyle 演示

```bash
cd python-src
python heightmapstyle_demo.py
```

### 运行 ControlNet 演示

```bash
cd python-src
python controlnet_demo.py
```

### 在代码中使用

```python
# 从 python-src 目录导入
import sys
sys.path.insert(0, "src")

from heightmapstyle import HeightmapStyleInference

inferencer = HeightmapStyleInference()
refined = inferencer.refine_heightmap(heightmap, output_path="output.png")
```

## 模型信息

- **模型 ID**: `dimentox/heightmapstyle`
- **类型**: Stable Diffusion 1.x (不是 SDXL!)
- **用途**: 高度图专用微调模型
- **大小**: ~5GB
- **首次运行**: 会自动下载模型

## 注意事项

1. **提示词限制**: 最多 77 tokens（SD 1.x 的限制）
2. **显存占用**: 约 4-6GB（比 SDXL 低）
3. **输出格式**: PNG 图像（不是 RAW）

## 对比测试

你可以对比 SDXL 和 HeightmapStyle 的效果：

```bash
# 运行原有的 SDXL 工作流
cd python-src
python main.py

# 运行 HeightmapStyle 演示
cd python-src
python heightmapstyle_demo.py

# 运行 ControlNet 演示
cd python-src
python controlnet_demo.py
```

## 详细文档

- HeightmapStyle 使用：`python-src/src/heightmapstyle/README.md`
- ControlNet 使用：`python-src/src/diffusion/CONTROLNET_README.md`
