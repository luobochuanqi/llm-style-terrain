#!/usr/bin/env python3
"""
512x512 分辨率：特征提取脚本

功能:
1. 读取所有 512x512 归一化后的高度图
2. 计算三个特征维度:
   - S (坡度): Sobel 算子梯度幅值的均值
   - R (粗糙度): 9x9 窗口内标准差的中位数
   - C (复杂度): Laplacian 算子响应的方差
3. Min-Max 归一化到 0-10 分
4. 输出 features_512.csv

用法:
    cd python-src
    uv run python -m src.dataprocess_512.extract_features
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def compute_features(img: np.ndarray) -> tuple[float, float, float]:
    """计算 S/R/C 三个特征维度

    Args:
        img: 归一化后的图像数组 (0.0-1.0), float32 单通道

    Returns:
        (S, R, C) 原始特征值
    """
    img = img.astype(np.float64)

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    S = np.mean(np.sqrt(sobelx**2 + sobely**2))

    kernel = np.ones((9, 9), np.float64) / 81.0
    mean = cv2.filter2D(img, -1, kernel)
    mean_sq = cv2.filter2D(img**2, -1, kernel)
    variance = np.clip(mean_sq - mean**2, 0, None)
    R = np.median(np.sqrt(variance))

    lap = cv2.Laplacian(img, cv2.CV_64F)
    C = np.var(lap)

    return float(S), float(R), float(C)


def minmax_scale(values: list[float], min_val: float, max_val: float) -> list[float]:
    """Min-Max 归一化到 0-10 分"""
    if max_val - min_val < 1e-8:
        return [5.0] * len(values)
    return [(v - min_val) / (max_val - min_val) * 10.0 for v in values]


def main():
    base_dir = (
        Path(__file__).parent.parent.parent.parent
        / "data"
        / "training-dataset"
        / "preprocess"
    )
    normalized_dir = base_dir / "normalized_512"

    all_images = []

    for split in ["train", "val", "test"]:
        split_dir = normalized_dir / split
        if not split_dir.exists():
            continue
        for img_path in sorted(split_dir.glob("*.png")):
            all_images.append(
                {
                    "path": str(img_path),
                    "filename": f"normalized_512/{split}/{img_path.name}",
                    "type": "Danxia" if "danxia" in img_path.name.lower() else "Kasite",
                }
            )

    print(f"共找到 {len(all_images)} 张图片 (512x512)")

    features = []
    print("正在计算特征...")

    for item in all_images:
        img = cv2.imread(str(item["path"]), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"⚠️ 无法读取：{item['path']}")
            continue

        if img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        elif img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        S, R, C = compute_features(img)

        features.append(
            {
                "filename": item["filename"],
                "type": item["type"],
                "S_raw": S,
                "R_raw": R,
                "C_raw": C,
            }
        )

    df = pd.DataFrame(features)

    print("正在归一化打分...")

    S_score = minmax_scale(df["S_raw"].tolist(), df["S_raw"].min(), df["S_raw"].max())
    R_score = minmax_scale(df["R_raw"].tolist(), df["R_raw"].min(), df["R_raw"].max())
    C_score = minmax_scale(df["C_raw"].tolist(), df["C_raw"].min(), df["C_raw"].max())

    df["S_score"] = S_score
    df["R_score"] = R_score
    df["C_score"] = C_score

    output_path = base_dir / "features_512.csv"
    df.to_csv(output_path, index=False)

    print(f"\n✅ 512x512 特征提取完成！")
    print(f"输出文件：{output_path}")
    print(f"总图片数：{len(df)}")
    print(f"\n特征统计:")
    print(f"  S (坡度):   {df['S_raw'].min():.4f} - {df['S_raw'].max():.4f}")
    print(f"  R (粗糙度): {df['R_raw'].min():.4f} - {df['R_raw'].max():.4f}")
    print(f"  C (复杂度): {df['C_raw'].min():.4f} - {df['C_raw'].max():.4f}")


if __name__ == "__main__":
    main()
