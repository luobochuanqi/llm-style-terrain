#!/usr/bin/env python
"""
高度图工具快捷入口
提供常用的高度图转换和管理工具
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_help():
    """打印帮助信息"""
    help_text = """
高度图工具箱

用法:
  python tools.py <命令> [参数]

可用命令:
  png2raw    将 PNG 转换为 RAW 8 位高度图
  raw2png    将 RAW 8 位高度图转换为 PNG 预览图
  info       显示高度图文件信息

示例:
  # PNG → RAW
  python tools.py png2raw input.png output.raw
  
  # RAW → PNG (自动推断尺寸)
  python tools.py raw2png input.raw --auto-size
  
  # RAW → PNG (指定尺寸)
  python tools.py raw2png input.raw --size 1024
  
  # 查看文件信息
  python tools.py info heightmap.raw
  
  # 查看工具帮助
  python tools.py png2raw --help
"""
    print(help_text)


def show_file_info(file_path: str):
    """显示高度图文件信息"""
    import numpy as np
    from PIL import Image

    path = Path(file_path)

    if not path.exists():
        print(f"错误：文件不存在 - {file_path}")
        return

    print(f"\n文件信息：{path.absolute()}")
    print("-" * 60)
    print(f"文件大小：{path.stat().st_size:,} 字节")

    if path.suffix.lower() == ".raw":
        # 尝试推断尺寸
        size_bytes = path.stat().st_size
        size = int(np.sqrt(size_bytes))
        if size * size == size_bytes:
            print(f"推断尺寸：{size}×{size} (正方形)")
            print(f"数据类型：uint8 (8-bit)")

            # 读取并显示统计信息
            data = np.fromfile(path, dtype=np.uint8).reshape((size, size))
            print(f"最小值：{data.min()}")
            print(f"最大值：{data.max()}")
            print(f"平均值：{data.mean():.2f}")
        else:
            print(f"⚠️  无法推断尺寸（{size_bytes} 不是完全平方数）")

    elif path.suffix.lower() == ".png":
        img = Image.open(path)
        print(f"图像尺寸：{img.width}×{img.height}")
        print(f"模式：{img.mode}")
        print(f"格式：{img.format}")


def main():
    if len(sys.argv) < 2:
        print_help()
        return

    command = sys.argv[1]

    if command in ["-h", "--help", "help"]:
        print_help()
        return

    elif command == "png2raw":
        # 调用 png2raw 工具
        from tools.png2raw import main as png2raw_main

        sys.argv = [sys.argv[0]] + sys.argv[
            2:
        ]  # 移除 'png2raw' 参数，保留脚本名和后续参数
        png2raw_main()

    elif command == "raw2png":
        # 调用 png2raw 工具的 raw2png 模式
        from tools.png2raw import main as png2raw_main

        sys.argv = [sys.argv[0]] + sys.argv[2:]  # 移除 'raw2png' 参数

        # 添加 --auto-size 标志（如果用户没有指定 size）
        if "--size" not in sys.argv and "--auto-size" not in sys.argv:
            sys.argv.append("--auto-size")

        png2raw_main()

    elif command == "info":
        if len(sys.argv) < 3:
            print("用法：python tools.py info <文件路径>")
            return
        show_file_info(sys.argv[2])

    else:
        print(f"未知命令：{command}")
        print("\n使用 'python tools.py --help' 查看可用命令")


if __name__ == "__main__":
    main()
