"""
MapGen 演示入口
演示两个 MapGen 模块的串联执行：元素位置图 → 语义布局图 → 高度图
"""

import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.mapgen import (
    create_layout_mapgen,
    create_height_mapgen,
    LayoutMapGen,
    HeightMapGen,
)


def create_dummy_element_locations(
    batch_size: int = 1,
    image_size: int = 64,
    num_classes: int = 5,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    创建模拟的元素位置图（LLM 输出）

    语义类别：
    0: 城镇 (urban)
    1: 山地 (mountain)
    2: 水域 (water)
    3: 河流 (river)
    4: 平原 (plains)

    Args:
        batch_size: 批量大小
        image_size: 图像尺寸
        num_classes: 类别数
        device: 设备

    Returns:
        元素位置图，shape: (batch, num_classes, H, W)
    """
    # 初始化为 zeros
    element_locations = torch.zeros(
        (batch_size, num_classes, image_size, image_size), device=device
    )

    # 模拟一些简单的布局（用矩形框表示元素位置）
    # 山地：左上区域
    element_locations[:, 1, 0 : image_size // 2, 0 : image_size // 2] = 1.0

    # 水域：右下区域
    element_locations[:, 2, image_size // 2 :, image_size // 2 :] = 1.0

    # 平原：右上区域
    element_locations[:, 4, 0 : image_size // 2, image_size // 2 :] = 1.0

    # 城镇：左下区域的一个小块
    element_locations[
        :, 0, image_size // 2 : 3 * image_size // 4, 0 : image_size // 4
    ] = 1.0

    # 河流：从左上到右下的对角线
    for i in range(image_size):
        j = i
        element_locations[:, 3, i, j] = 1.0

    return element_locations


def create_dummy_text_embeddings(
    batch_size: int = 1,
    text_len: int = 77,
    text_dim: int = 768,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    创建模拟的 CLIP 文本嵌入

    Args:
        batch_size: 批量大小
        text_len: 文本长度
        text_dim: 文本嵌入维度

    Returns:
        文本嵌入，shape: (batch, text_len, text_dim)
    """
    # 使用随机向量模拟（实际应用中应该是 CLIP 编码的真实文本）
    return torch.randn((batch_size, text_len, text_dim), device=device)


def visualize_layout(layout: torch.Tensor) -> np.ndarray:
    """
    将语义布局图可视化为 RGB 图像

    Args:
        layout: 语义布局图，shape: (batch, num_classes, H, W)

    Returns:
        RGB 图像数组，shape: (H, W, 3), dtype: uint8
    """
    # 取第一个样本并转移到 CPU
    layout_single = layout[0].cpu()  # (num_classes, H, W)

    # softmax 得到概率分布
    probs = torch.softmax(layout_single, dim=0)  # (num_classes, H, W)

    # 取最大概率的类别
    class_map = torch.argmax(probs, dim=0).cpu()  # (H, W)

    # 定义类别颜色
    colors = {
        0: [200, 200, 200],  # 城镇：灰色
        1: [139, 90, 43],  # 山地：棕色
        2: [70, 130, 180],  # 水域：蓝色
        3: [100, 149, 237],  # 河流：浅蓝色
        4: [34, 139, 34],  # 平原：绿色
    }

    # 转换为 RGB 图像
    H, W = class_map.shape
    rgb_image = np.zeros((H, W, 3), dtype=np.uint8)

    for class_idx, color in colors.items():
        mask = class_map == class_idx
        rgb_image[mask] = color

    return rgb_image


def visualize_height(height: torch.Tensor) -> np.ndarray:
    """
    将高度图可视化为灰度图像

    Args:
        height: 高度图，shape: (batch, 1, H, W)

    Returns:
        灰度图像数组，shape: (H, W), dtype: uint8
    """
    # 取第一个样本并转移到 CPU
    height_single = height[0, 0].cpu()  # (H, W)

    # 归一化到 [0, 1]
    height_min = height_single.min()
    height_max = height_single.max()
    height_norm = (height_single - height_min) / (height_max - height_min + 1e-6)

    # 转换为 0-255
    height_uint8 = (height_norm * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)

    return height_uint8


def run_mapgen_demo():
    """执行 MapGen 演示流程"""
    print("=" * 60)
    print("MapGen 演示：语义布局生成 → 高度图生成")
    print("=" * 60)

    # 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备：{device}")

    image_size = 64
    num_classes = 5
    batch_size = 1

    # 步骤 1: 创建模拟输入
    print("\n【步骤 1/4】创建模拟输入数据")
    print("-" * 60)
    element_locations = create_dummy_element_locations(
        batch_size=batch_size,
        image_size=image_size,
        num_classes=num_classes,
        device=device,
    )
    text_embeddings = create_dummy_text_embeddings(device=device)

    print(f"元素位置图形状：{element_locations.shape}")
    print(f"文本嵌入形状：{text_embeddings.shape}")
    print(f"  - 山地区域：左上角")
    print(f"  - 水域区域：右下角")
    print(f"  - 平原区域：右上角")
    print(f"  - 城镇区域：左下角小块")
    print(f"  - 河流：对角线")

    # 步骤 2: 使用简单规则生成语义布局图（而非随机模型）
    print("\n【步骤 2/4】生成语义布局图")
    print("-" * 60)

    # 直接对元素位置图进行平滑处理模拟布局生成
    layout = element_locations.clone()

    # 简单的高斯平滑让边界更自然
    layout_smooth = torch.zeros_like(layout)
    kernel_size = 5
    padding = kernel_size // 2
    for b in range(batch_size):
        for c in range(num_classes):
            # 使用简单的平均池化模拟平滑
            smoothed = torch.nn.functional.avg_pool2d(
                torch.nn.functional.pad(
                    layout[b : b + 1, c : c + 1],
                    (padding, padding, padding, padding),
                    mode="reflect",
                ),
                kernel_size=kernel_size,
                stride=1,
            )
            layout_smooth[b, c] = smoothed[0]

    # 归一化使每个位置的概率和为 1
    layout_smooth = layout_smooth.clamp(min=0)
    prob_sum = layout_smooth.sum(dim=1, keepdim=True).clamp(min=1e-6)
    layout = layout_smooth / prob_sum * 5  # 乘以类别数保持数值范围

    print(f"语义布局图形状：{layout.shape}")
    print("✅ 使用平滑滤波模拟布局生成（随机模型无法产生有效结果）")

    # 可视化并保存布局图
    layout_image = visualize_layout(layout)
    layout_pil = Image.fromarray(layout_image)

    # 步骤 3: 根据语义布局图生成高度图（基于规则）
    print("\n【步骤 3/4】生成地形高度图")
    print("-" * 60)

    # 定义每个类别的基础高度
    # 山地 (1): 高海拔，水域 (2): 低海拔，平原 (4): 中海拔，城镇 (0): 中低，河流 (3): 低
    height_map = torch.zeros((batch_size, 1, image_size, image_size), device=device)

    # 使用布局概率加权高度
    probs = torch.softmax(layout, dim=1)
    height_values = torch.tensor([0.3, 0.9, 0.1, 0.15, 0.4], device=device).view(
        1, -1, 1, 1
    )
    height_map = (probs * height_values).sum(dim=1, keepdim=True)

    # 添加一些噪声模拟自然地形
    noise = torch.randn_like(height_map) * 0.05
    height_map = height_map + noise
    height_map = height_map.clamp(0, 1)

    print(f"高度图形状：{height_map.shape}")
    print("✅ 基于语义布局生成高度图（山地=高，水域=低）")

    # 可视化并保存高度图
    height_image = visualize_height(height_map)
    height_pil = Image.fromarray(height_image, mode="L")

    # 步骤 4: 保存结果
    print("\n【步骤 4/4】保存结果")
    print("-" * 60)
    output_dir = Path("outputs/mapgen")
    output_dir.mkdir(parents=True, exist_ok=True)

    layout_path = output_dir / "layout.png"
    height_path = output_dir / "height.png"

    layout_pil.save(layout_path)
    height_pil.save(height_path)

    print(f"✅ 语义布局图已保存：{layout_path.absolute()}")
    print(f"✅ 高度图已保存：{height_path.absolute()}")

    # 完成
    print("\n" + "=" * 60)
    print("✅ MapGen 演示完成！")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  - 语义布局图：{layout_path.absolute()}")
    print(f"  - 地形高度图：{height_path.absolute()}")
    print(f"\n说明：演示使用规则基方法模拟 MapGen 流程。")
    print(f"实际训练后的模型会自动学习从布局到高度的映射。")

    return {
        "element_locations": element_locations,
        "layout": layout,
        "height": height_map,
        "output_paths": {
            "layout": layout_path,
            "height": height_path,
        },
    }


def main():
    """主入口函数"""
    try:
        result = run_mapgen_demo()
        return result
    except KeyboardInterrupt:
        print("\n⚠️  用户中断执行")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误：{e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
