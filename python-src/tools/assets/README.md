# 高度图转换工具测试资源

## 测试文件

- `heightmap-i.png`: 16 位灰度 PNG 高度图（输入）
- `heightmap-o.raw`: 8 位 RAW 高度图（输出）
- `heightmap-recovered.png`: 从 RAW 恢复的 PNG（验证用）
- `heightmap-o2.raw`: 使用 `-o` 选项生成的 RAW（验证用）

## 使用方法

### 从项目根目录

```bash
cd python-src

# PNG → RAW (位置参数)
uv run python tools.py png2raw tools/assets/heightmap-i.png tools/assets/heightmap-o.raw

# PNG → RAW (-o 选项)
uv run python tools.py png2raw tools/assets/heightmap-i.png -o tools/assets/heightmap-o.raw

# RAW → PNG
uv run python tools.py raw2png tools/assets/heightmap-o.raw tools/assets/heightmap-recovered.png

# 查看文件信息
uv run python tools.py info tools/assets/heightmap-o.raw
```

### 直接使用工具脚本

```bash
# PNG → RAW
uv run python tools/png2raw.py tools/assets/heightmap-i.png tools/assets/heightmap-o.raw

# RAW → PNG (自动推断尺寸)
uv run python tools/png2raw.py tools/assets/heightmap-o.raw --auto-size
```

## 验证转换

```bash
# 验证 PNG → RAW → PNG 无损转换
uv run python -c "
import numpy as np
from PIL import Image

# 读取并比较
original = np.array(Image.open('tools/assets/heightmap-i.png'))
if original.dtype == np.uint16:
    original = (original / 256).astype(np.uint8)
raw = np.fromfile('tools/assets/heightmap-o.raw', dtype=np.uint8).reshape((244, 244))
recovered = np.array(Image.open('tools/assets/heightmap-recovered.png'))

assert np.array_equal(original, raw), 'PNG → RAW 转换失败'
assert np.array_equal(raw, recovered), 'RAW → PNG 转换失败'
print('✅ 验证通过：转换完全无损')
"
```
