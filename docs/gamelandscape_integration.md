# GameLandscape 模型集成

## 集成日期
2026-04-08

## 模型信息

- **名称**: GameLandscapeHeightmap512_V1.0
- **格式**: Safetensors
- **用途**: 游戏地形高度图生成
- **尺寸**: 512×512

## 文件结构

```
python-src/
├── gamelandscape_demo.py         # 演示脚本
└── src/
    └── gamelandscape/            # 模型模块
        ├── __init__.py
        ├── model_loader.py       # 加载器
        ├── README.md             # 详细文档
        └── QUICKSTART.md         # 快速开始
```

## 与 HeightmapStyle 一致的架构

参照 heightmapstyle 模块的设计：

| 组件 | HeightmapStyle | GameLandscape |
|------|---------------|---------------|
| 演示脚本 | heightmapstyle_demo.py | gamelandscape_demo.py |
| 模块目录 | src/heightmapstyle/ | src/gamelandscape/ |
| 加载器 | model_loader.py | model_loader.py |
| 输出目录 | outputs/heightmapstyle/ | outputs/gamelandscape/ |
| 模型来源 | HuggingFace | 本地文件 |

## 使用方法

### 快速测试

```bash
cd python-src
python gamelandscape_demo.py
```

### 在代码中使用

```python
from src.gamelandscape import GameLandscapeInference

inferencer = GameLandscapeInference()
refined = inferencer.refine_heightmap(heightmap, "output.png")
```

## 输出配置

已更新 `OutputConfig` 添加：
```python
gamelandscape_dir: Path = Path("outputs/gamelandscape")
```

## 模型加载逻辑

模型加载器支持：
1. **Safetensors 格式**: 使用 `safetensors.torch.load_file`
2. **StableDiffusionPipeline**: 从单文件加载
3. **CPU Offload**: 自动启用显存优化

## 注意事项

1. **模型文件位置**: 
   - 应放置在 `python-src/assets/GameLandscapeHeightmap512_V1.0.safetensors`
   
2. **模型架构**:
   - 假设为 SD 1.x 架构
   - 如不兼容需调整 `model_loader.py` 中的加载逻辑

3. **显存要求**:
   - 约 4-6GB（启用 CPU offload）

## 文档

- **详细文档**: `python-src/src/gamelandscape/README.md`
- **快速开始**: `python-src/src/gamelandscape/QUICKSTART.md`
- **输出说明**: `python-src/outputs/README.md`

## 下一步

1. ✅ 创建模块结构
2. ✅ 实现加载器
3. ✅ 创建演示脚本
4. ✅ 配置输出目录
5. ⏳ 测试模型加载
6. ⏳ 验证输出质量
