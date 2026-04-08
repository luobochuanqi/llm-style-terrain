"""
PNG 转 RAW 8 位高度图工具
将 PNG 图像转换为 8-bit 二进制 RAW 格式高度图
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image


def png_to_raw(
    input_path: Path, output_path: Path, verbose: bool = True, size: int = None
) -> np.ndarray:
    """
    将 PNG 图像转换为 8-bit RAW 高度图

    Args:
        input_path: 输入 PNG 文件路径
        output_path: 输出 RAW 文件路径
        verbose: 是否打印详细信息
        size: 输出尺寸（可选，会自动缩放）

    Returns:
        高度图数组 (uint8)
    """
    if verbose:
        print(f"正在转换：{input_path} → {output_path}")

    # 1. 读取 PNG 图像
    image = Image.open(input_path)

    # 2. 转为 numpy 数组并处理位深
    img_array = np.array(image)

    # 处理 16 位图像：缩放到 8 位
    if img_array.dtype == np.uint16:
        if verbose:
            print(f"  - 检测到 16 位图像，缩放到 8 位")
        heightmap = (img_array / 256).astype(np.uint8)
    elif img_array.dtype == np.uint8:
        # 8 位图像，直接使用
        # 如果是 RGB/RGBA，先转灰度
        if len(img_array.shape) == 3:
            if verbose:
                print(f"  - 将图像从 {image.mode} 转换为灰度图")
            gray_image = image.convert("L")
            heightmap = np.array(gray_image, dtype=np.uint8)
        else:
            heightmap = img_array
    else:
        # 其他类型，转为 uint8
        if image.mode != "L":
            if verbose:
                print(f"  - 将图像从 {image.mode} 转换为灰度图")
            gray_image = image.convert("L")
        else:
            gray_image = image
        heightmap = np.array(gray_image, dtype=np.uint8)

    # 3. 如果指定了 size，调整图像尺寸
    if size is not None:
        if verbose:
            print(f"  - 调整尺寸到：{size}×{size}")
        heightmap_pil = Image.fromarray(heightmap).resize(
            (size, size), Image.Resampling.LANCZOS
        )
        heightmap = np.array(heightmap_pil, dtype=np.uint8)

    # 4. 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 5. 保存为 RAW 格式
    heightmap.tofile(output_path)

    if verbose:
        print(f"✅ 转换完成")
        print(f"  - 尺寸：{heightmap.shape[1]}×{heightmap.shape[0]}")
        print(f"  - 数据类型：{heightmap.dtype}")
        print(f"  - 值范围：[{heightmap.min()}, {heightmap.max()}]")
        print(f"  - 输出文件：{output_path.absolute()}")

    return heightmap


def raw_to_png(
    input_path: Path, output_path: Path, size: int = None, verbose: bool = True
) -> np.ndarray:
    """
    将 RAW 8 位高度图转换回 PNG 预览图

    Args:
        input_path: 输入 RAW 文件路径
        output_path: 输出 PNG 文件路径
        size: 图像尺寸（边长），如果为 None 则自动计算
        verbose: 是否打印详细信息

    Returns:
        高度图数组 (uint8)
    """
    if verbose:
        print(f"正在转换：{input_path} → {output_path}")

    # 1. 读取 RAW 文件
    raw_data = np.fromfile(input_path, dtype=np.uint8)

    # 2. 重塑为 2D 数组
    if size is None:
        # 自动计算尺寸（假设是正方形）
        size = int(np.sqrt(len(raw_data)))
        if size * size != len(raw_data):
            raise ValueError(f"无法自动推断尺寸：RAW 文件大小为 {len(raw_data)} 字节")

    heightmap = raw_data.reshape((size, size))

    # 3. 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 4. 保存为 PNG
    image = Image.fromarray(heightmap).convert("L")
    image.save(output_path)

    if verbose:
        print(f"✅ 转换完成")
        print(f"  - 尺寸：{size}×{size}")
        print(f"  - 数据类型：{heightmap.dtype}")
        print(f"  - 值范围：[{heightmap.min()}, {heightmap.max()}]")
        print(f"  - 输出文件：{output_path.absolute()}")

    return heightmap


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="PNG 与 RAW 8 位高度图互相转换工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # PNG → RAW
  python tools/png2raw.py input.png output.raw
  python tools/png2raw.py heightmap.png -o outputs/heightmap.raw
  
  # RAW → PNG
  python tools/png2raw.py input.raw output.png --size 1024
  python tools/png2raw.py heightmap.raw --auto-size

  # 批量转换当前目录所有 PNG
  python tools/png2raw.py *.png --output-dir outputs/
        """,
    )

    parser.add_argument("input", type=Path, help="输入文件路径（PNG 或 RAW）")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help="输出文件路径（可选，默认根据输入文件名自动生成）",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        dest="output_opt",
        default=None,
        help="输出文件路径（可选，默认根据输入文件名自动生成）",
    )
    parser.add_argument(
        "--size", type=int, default=None, help="RAW → PNG 时的图像尺寸（边长，像素）"
    )
    parser.add_argument(
        "--auto-size",
        action="store_true",
        help="RAW → PNG 时自动计算尺寸（假设是正方形）",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="批量转换时的输出目录"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="安静模式，不打印详细信息"
    )
    parser.add_argument(
        "--batch",
        type=Path,
        nargs="+",
        help="批量转换多个文件（配合 --output-dir 使用）",
    )

    args = parser.parse_args()

    # 批量转换模式
    if args.batch:
        if not args.output_dir:
            print("错误：批量转换需要指定 --output-dir")
            sys.exit(1)

        args.output_dir.mkdir(parents=True, exist_ok=True)

        for input_file in args.batch:
            output_file = (
                args.output_dir
                / input_file.with_suffix(
                    ".raw"
                    if input_file.suffix.lower() in [".png", ".jpg", ".jpeg"]
                    else ".png"
                ).name
            )
            try:
                if input_file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    png_to_raw(input_file, output_file, verbose=not args.quiet)
                else:
                    size = (
                        args.size
                        if args.size
                        else (
                            int(np.sqrt(np.fromfile(input_file, dtype=np.uint8).size))
                            if args.auto_size
                            else None
                        )
                    )
                    raw_to_png(
                        input_file, output_file, size=size, verbose=not args.quiet
                    )
            except Exception as e:
                print(f"⚠️  转换失败：{input_file} - {e}")

        return

    # 单文件转换模式
    input_path: Path = args.input

    # 确定输出路径（支持位置参数和 -o 选项）
    if args.output:
        output_path = args.output
    elif args.output_opt:
        output_path = args.output_opt
    else:
        # 自动生成输出文件名
        if args.output_dir:
            output_dir = args.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = input_path.parent

        if input_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            # PNG → RAW
            output_path = output_dir / f"{input_path.stem}.raw"
        else:
            # RAW → PNG
            output_path = output_dir / f"{input_path.stem}.png"

    # 执行转换
    try:
        if input_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            # PNG → RAW
            png_to_raw(input_path, output_path, verbose=not args.quiet, size=args.size)
        else:
            # RAW → PNG
            size = (
                args.size
                if args.size
                else (
                    int(np.sqrt(np.fromfile(input_path, dtype=np.uint8).size))
                    if args.auto_size
                    else None
                )
            )
            if size is None and not args.auto_size:
                print("错误：RAW → PNG 需要指定 --size 或使用 --auto-size")
                sys.exit(1)
            raw_to_png(input_path, output_path, size=size, verbose=not args.quiet)

    except Exception as e:
        print(f"❌ 转换失败：{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
