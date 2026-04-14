#!/usr/bin/env python3
"""
512x512 分辨率：数据预处理脚本

功能:
1. 遍历 Danxia/Kasite 目录，读取所有 PNG (保持 512x512)
2. 像素值归一化到 0.0-1.0 浮点数 (除以 65535)
3. 按 70/15/15 分层抽样划分训练/验证/测试集
4. 文件命名：{type}_{split}_{id:03d}.png

用法:
    cd python-src
    uv run python -m src.dataprocess_512.preprocess
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


def load_and_normalize(image_path: Path) -> np.ndarray:
    """读取 16-bit PNG 并归一化到 0.0-1.0 浮点数"""
    img = Image.open(image_path)
    img_array = np.array(img, dtype=np.float32)
    img_normalized = img_array / 65535.0
    return img_normalized


def main():
    base_dir = (
        Path(__file__).parent.parent.parent.parent
        / "data"
        / "training-dataset"
        / "preprocess"
    )
    danxia_dir = base_dir / "Danxia"
    kasite_dir = base_dir / "Kasite"

    output_base = base_dir / "normalized_512"
    splits_dir = base_dir / "splits_512"

    output_base.mkdir(exist_ok=True)
    (output_base / "train").mkdir(exist_ok=True)
    (output_base / "val").mkdir(exist_ok=True)
    (output_base / "test").mkdir(exist_ok=True)
    splits_dir.mkdir(exist_ok=True)

    data = []

    print("正在加载 Danxia 图片 (512x512)...")
    for img_path in sorted(danxia_dir.glob("*.png")):
        img_normalized = load_and_normalize(img_path)
        data.append(
            {"original_path": str(img_path), "type": "Danxia", "image": img_normalized}
        )

    print("正在加载 Kasite 图片 (512x512)...")
    for img_path in sorted(kasite_dir.glob("*.png")):
        img_normalized = load_and_normalize(img_path)
        data.append(
            {"original_path": str(img_path), "type": "Kasite", "image": img_normalized}
        )

    print(f"已加载图片总数：{len(data)} (512x512)")

    types = [item["type"] for item in data]
    indices = list(range(len(data)))

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, stratify=types, random_state=42
    )

    temp_types = [types[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=temp_types, random_state=42
    )

    print(f"训练集：{len(train_idx)}, 验证集：{len(val_idx)}, 测试集：{len(test_idx)}")

    split_info = {"train": train_idx, "val": val_idx, "test": test_idx}

    split_files = []

    for split_name, idx_list in split_info.items():
        output_dir = output_base / split_name
        csv_rows = []

        type_counters = {"Danxia": 1, "Kasite": 1}

        for idx in idx_list:
            item = data[idx]
            img_type = item["type"]
            img_id = type_counters[img_type]
            type_counters[img_type] += 1

            new_filename = f"{img_type.lower()}_{split_name}_{img_id:03d}.png"
            output_path = output_dir / new_filename

            img_pil = Image.fromarray((item["image"] * 65535).astype(np.uint16))
            img_pil.save(output_path)

            relative_path = f"normalized_512/{split_name}/{new_filename}"
            csv_rows.append(
                {
                    "filename": new_filename,
                    "path": relative_path,
                    "type": img_type,
                    "original_path": item["original_path"],
                }
            )

        df = pd.DataFrame(csv_rows)
        csv_path = splits_dir / f"{split_name}.csv"
        df.to_csv(csv_path, index=False)
        split_files.append(str(csv_path))

        print(f"已保存 {split_name}: {len(csv_rows)} 张图片 -> {csv_path}")

    print("\n✅ 512x512 预处理完成！")
    print(f"输出目录：{output_base}")
    print(f"划分文件：{', '.join(split_files)}")


if __name__ == "__main__":
    main()
