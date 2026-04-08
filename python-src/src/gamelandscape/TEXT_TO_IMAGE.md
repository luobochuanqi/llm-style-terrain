# GameLandscape 文生图模式

## 更新说明

已将 **图生图（img2img）** 改为 **文生图（txt2img）** 模式。

### 主要变化

#### 之前（图生图）
```python
# 需要先生成 Perlin 噪声
heightmap = generator.generate()

# 然后微调
refined = inferencer.refine_heightmap(heightmap, output_path)
```

#### 现在（文生图）
```python
# 直接从文本生成
heightmap = inferencer.generate_heightmap(
    output_path=output_path,
    terrain_type="Mountain",
    width=512,
    height=512,
)
```

## 使用方式

### 1. 运行演示脚本

```bash
cd python-src
python gamelandscape_demo.py
```

### 2. 在代码中使用

```python
from src.gamelandscape import GameLandscapeInference

inferencer = GameLandscapeInference()

# 生成山地地形
heightmap = inferencer.generate_heightmap(
    output_path="outputs/mountain.png",
    terrain_type="Mountain",
    width=512,
    height=512,
)
```

### 3. 使用便捷函数

```python
from src.gamelandscape import load_gamelandscape

heightmap = load_gamelandscape(
    output_path="outputs/river.png",
    terrain_type="River",
)
```

## 可用地形类型

- **Alpen** - 阿尔卑斯山脉
- **Hills** - 丘陵
- **Mesa** - 台地/方山
- **Mountain** - 山脉（默认）
- **MountainFlow** - 山地水流
- **MountainWater** - 山水
- **OceanIsland** - 海洋岛屿
- **Plain** - 平原
- **River** - 河流
- **RiverMountain** - 山河
- **SandyBeach** - 沙滩
- **Volcano** - 火山

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `terrain_type` | "Mountain" | 地形类型 |
| `width` | 512 | 图像宽度 |
| `height` | 512 | 图像高度 |
| `num_inference_steps` | 20 | 推理步数 |
| `guidance_scale` | 7.0 | 引导比例 |

## 优势

### 文生图 vs 图生图

| 特性 | 文生图 | 图生图 |
|------|--------|--------|
| 输入 | 文本提示词 | 输入图像 + 文本 |
| 创意性 | ⭐⭐⭐⭐⭐ 完全自由 | ⭐⭐⭐ 受输入限制 |
| 可控性 | ⭐⭐⭐⭐ 通过提示词 | ⭐⭐⭐⭐⭐ 通过输入图 |
| 速度 | ⭐⭐⭐⭐ 直接生成 | ⭐⭐⭐ 需要预处理 |
| 适用场景 | 从零创作 | 基于现有地形优化 |

## 工作流程

```
文本提示词（terrain_type）
    ↓
SD 1.5 + GameLandscape LoRA
    ↓
文生图推理
    ↓
512×512 高度图
```

## 示例代码

### 生成不同类型的地形

```python
from src.gamelandscape import GameLandscapeInference
from pathlib import Path

inferencer = GameLandscapeInference()
output_dir = Path("outputs/gamelandscape")
output_dir.mkdir(parents=True, exist_ok=True)

# 生成不同地形
terrains = ["Mountain", "Hills", "River", "Volcano", "OceanIsland"]

for terrain in terrains:
    print(f"生成 {terrain}...")
    heightmap = inferencer.generate_heightmap(
        output_path=output_dir / f"{terrain.lower()}.png",
        terrain_type=terrain,
    )
    print(f"✅ {terrain} 完成")
```

## 输出

生成的文件保存在：
```
outputs/gamelandscape/demo_result.png
```

## 注意事项

1. **首次运行**: 会下载 SD 1.5 基础模型（约 4GB）
2. **显存要求**: 约 4-6GB
3. **生成尺寸**: 默认 512×512，可以调整但可能影响质量
4. **随机性**: 每次生成结果不同（固定 seed=42 可复现）
