# ControlNet 运行记录

## 运行时间
2026-04-08

## 运行命令
```bash
cd python-src
uv run python examples/controlnet_demo.py
```

## 运行输出

```
============================================================
ControlNet 高度图微调示例
============================================================

【步骤 1/3】生成 Perlin 噪声高度图
------------------------------------------------------------
🚀 开始生成 1024×1024 Perlin 噪声高度图...
✅ Perlin 噪声高度图生成完成
✅ 高度图已保存至：outputs/heightmap_perlin.raw
✅ 预览图已保存至：outputs/heightmap_perlin.png
高度图尺寸：(1024, 1024)
数据类型：uint8
值范围：[0, 255]

【步骤 2/3】ControlNet 结构约束微调
------------------------------------------------------------
✅ Canny 边缘图已保存：outputs/canny_edges.png
正在加载 ControlNet 模型...
✅ 已启用 CPU 显存卸载
✅ ControlNet 模型加载完成
开始执行 ControlNet 推理...
100%|████████████████████████████| 25/25 [19:01<00:00, 45.67s/it]
✅ ControlNet 推理完成
✅ 结果已保存至：outputs/heightmap_controlnet.png
微调后高度图尺寸：(1024, 1024)
数据类型：uint8
值范围：[45, 180]
✅ 模型已卸载

【步骤 3/3】结果对比
------------------------------------------------------------
原始高度图统计:
  - 平均值：117.80
  - 标准差：43.30

ControlNet 微调后统计:
  - 平均值：126.75
  - 标准差：26.11

差异统计:
  - 平均差异：40.72
  - 最大差异：187
  - 相关系数：-0.0024

============================================================
✅ 全部流程完成！
============================================================

输出文件:
  - 原始高度图：outputs/heightmap_perlin.raw
  - Canny 边缘图：outputs/canny_edges.png
  - ControlNet 微调：outputs/heightmap_controlnet.png
```

## 警告信息

```
UserWarning: The `local_dir_use_symlinks` argument is deprecated and ignored
Token indices sequence length is longer than the specified maximum sequence length 
for this model (88 > 77). Running this sequence through the model will result in 
indexing errors
The following part of your input was truncated because CLIP can only handle 
sequences up to 77 tokens: [', technical terrain data, gis elevation map, dem data, seamless']
```
