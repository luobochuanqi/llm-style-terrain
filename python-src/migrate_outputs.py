#!/usr/bin/env python
"""
输出文件迁移脚本
将旧的输出文件移动到新结构的目录中
"""

from pathlib import Path
import shutil


def migrate_outputs():
    """迁移输出文件到新结构"""
    print("=" * 60)
    print("输出文件迁移工具")
    print("=" * 60)

    outputs_dir = Path("outputs")

    if not outputs_dir.exists():
        print("❌ outputs 目录不存在")
        return

    # 定义迁移规则
    migrations = [
        # (源文件，目标文件)
        ("heightmap_perlin.raw", "perlin/heightmap.raw"),
        ("heightmap_perlin.png", "perlin/heightmap.png"),
        ("heightmap_diffusion.png", "diffusion/heightmap.png"),
        ("canny_edges.png", "controlnet/canny_edges.png"),
        ("heightmap_controlnet.png", "controlnet/heightmap.png"),
    ]

    # 迁移规则
    for src_name, dst_name in migrations:
        src = outputs_dir / src_name
        if src.exists():
            dst = outputs_dir / dst_name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            print(f"✅ {src_name} → {dst_name}")
        else:
            print(f"⏭️  {src_name} 不存在，跳过")

    # 迁移控制强度测试文件
    strength_files = list(outputs_dir.glob("controlnet_strength_*.png"))
    if strength_files:
        strength_dir = outputs_dir / "controlnet" / "strength_test"
        strength_dir.mkdir(parents=True, exist_ok=True)
        for f in strength_files:
            shutil.move(str(f), str(strength_dir / f.name))
            print(f"✅ {f.name} → controlnet/strength_test/")

    # 重命名 mapgen_demo 目录
    old_mapgen_dir = outputs_dir / "mapgen_demo"
    new_mapgen_dir = outputs_dir / "mapgen"
    if old_mapgen_dir.exists() and not new_mapgen_dir.exists():
        old_mapgen_dir.rename(new_mapgen_dir)
        print("✅ mapgen_demo/ → mapgen/")

    # 清理旧的测试文件
    test_files = [
        "perlin_test.png",
        "perlin_test.raw",
        "test_converted.raw",
        "test_original.png",
        "test_recovered.png",
        "test_quick.raw",
        "heightmap_perlin_check.png",
    ]

    print("\n删除旧的测试文件...")
    for f in test_files:
        test_file = outputs_dir / f
        if test_file.exists():
            test_file.unlink()
            print(f"✅ 删除 {f}")

    print("\n" + "=" * 60)
    print("迁移完成！")
    print("=" * 60)

    # 显示新的目录结构
    print("\n新的目录结构:")
    for subdir in ["perlin", "diffusion", "controlnet", "heightmapstyle", "mapgen"]:
        subpath = outputs_dir / subdir
        if subpath.exists():
            files = list(subpath.glob("*"))
            print(f"\n{subdir}/ ({len(files)} 个文件)")
            for f in files[:5]:  # 只显示前 5 个
                print(f"  - {f.name}")
            if len(files) > 5:
                print(f"  ... 还有 {len(files) - 5} 个文件")


if __name__ == "__main__":
    try:
        migrate_outputs()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 错误：{e}")
        import traceback

        traceback.print_exc()
