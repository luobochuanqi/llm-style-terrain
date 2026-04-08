# 高度图转换工具

PNG ↔ RAW 8 位高度图互转工具

## 快速开始

### PNG 转 RAW

```bash
cd python-src

# 基本用法
python tools.py png2raw input.png output.raw

# 使用详细工具（更多选项）
python tools/png2raw.py input.png -o output.raw
```

### RAW 转 PNG

```bash
cd python-src

# 自动推断尺寸
python tools.py raw2png input.raw --auto-size

# 指定尺寸
python tools.py raw2png input.raw --size 1024

# 指定输出文件
python tools.py raw2png input.raw -o output.png
```

### 查看文件信息

```bash
python tools.py info heightmap.raw
```

输出示例：
```
文件信息：outputs/heightmap_perlin.raw
------------------------------------------------------------
文件大小：1,048,576 字节
推断尺寸：1024×1024 (正方形)
数据类型：uint8 (8-bit)
最小值：0
最大值：255
平均值：117.80
```

## 详细用法

### png2raw 工具

```bash
python tools/png2raw.py --help
```

参数说明：
- `input`: 输入 PNG 文件路径（必需）
- `-o, --output`: 输出 RAW 文件路径（可选，默认自动生成）
- `-q, --quiet`: 安静模式，不打印详细信息
- `--output-dir`: 批量转换时的输出目录
- `--batch`: 批量转换多个文件

示例：
```bash
# 单个转换
python tools/png2raw.py heightmap.png -o outputs/heightmap.raw

# 批量转换
python tools/png2raw.py --batch *.png --output-dir outputs/
```

### 快捷命令

```bash
# 查看所有可用命令
python tools.py --help

# PNG → RAW
python tools.py png2raw input.png output.raw

# RAW → PNG（自动推断尺寸）
python tools.py raw2png input.raw --auto-size

# 查看文件信息
python tools.py info heightmap.raw
```

## 技术细节

### RAW 格式说明

- **数据类型**: uint8 (8-bit)
- **字节序**: 小端（little-endian）
- **布局**: 按行优先（row-major）排列
- **尺寸**: 正方形（如 1024×1024, 2048×2048）
- **文件大小**: 尺寸² 字节（1024×1024 = 1MB）

### PNG 格式说明

- **模式**: 灰度图（L 模式）
- **位深**: 8-bit
- **值范围**: 0（黑色/低谷）~ 255（白色/高峰）

### 转换保证

- ✅ **无损转换**: PNG → RAW → PNG 完全一致
- ✅ **自动推断**: RAW 转 PNG 时自动计算尺寸
- ✅ **批量处理**: 支持一次转换多个文件
- ✅ **跨平台**: RAW 格式兼容所有主流编程语言

## 使用场景

### 1. 地形生成工作流

```bash
# 1. 从 PNG 高度图开始（如手绘或软件导出）
python tools.py png2raw terrain.png terrain.raw

# 2. 使用 terrain.raw 进行后续处理
# 3. 将结果转回 PNG 预览
python tools.py raw2png result.raw --auto-size
```

### 2. 批量转换

```bash
# 转换整个目录
python tools/png2raw.py --batch heightmaps/*.png --output-dir raw_heightmaps/
```

### 3. 验证高度图

```bash
# 检查文件信息
python tools.py info heightmap.raw

# 生成预览图
python tools.py raw2png heightmap.raw --auto-size -o preview.png
```

## 与其他工具集成

### Unity / Unreal Engine

导出 RAW 格式高度图到游戏引擎：
```bash
python tools.py png2raw heightmap.png terrain.raw
```

### Blender

导入 RAW 高度图：
```python
# Blender Python 脚本
import numpy as np

# 读取 RAW
heightmap = np.fromfile("terrain.raw", dtype=np.uint8).reshape((1024, 1024))

# 归一化到 0-1
heightmap = heightmap / 255.0
```

### Python 项目

```python
from tools.png2raw import png_to_raw, raw_to_png

# PNG → RAW
heightmap = png_to_raw("input.png", "output.raw")

# RAW → PNG
heightmap = raw_to_png("input.raw", "output.png", size=1024)
```

## 故障排除

### 问题：RAW 文件尺寸不对

确保输入是正方形且尺寸是 2 的幂：
- 512×512 = 262,144 字节
- 1024×1024 = 1,048,576 字节
- 2048×2048 = 4,194,304 字节

### 问题：转换后高度图异常

检查输入 PNG 是否为灰度图。彩色 PNG 会自动转换，但可能导致意外的灰度值。

建议：在转换前使用图像编辑软件（如 Photoshop、GIMP）将 PNG 转为纯灰度图。

### 问题：显存不足

处理大型高度图（如 4096×4096）时，SDXL 推理可能显存不足。

解决方案：
1. 在 `src/config.py` 中启用 `enable_cpu_offload=True`
2. 使用较小的高度图尺寸
