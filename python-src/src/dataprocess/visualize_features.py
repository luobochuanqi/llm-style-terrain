#!/usr/bin/env python3
"""
第三阶段：可视化分析脚本

功能:
1. 读取 features.csv
2. 使用 t-SNE 和 UMAP 将三维特征降维至 2D
3. 绘制散点图，Danxia/Kasite 用不同颜色标注
4. 添加 S/R/C 三维特征的分布直方图
5. 保存可视化结果

用法:
    cd python-src
    python -m src.data.visualize_features
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠️ umap-learn 未安装，将跳过 UMAP 可视化")


def plot_tsne(df: pd.DataFrame, output_path: Path) -> None:
    """t-SNE 降维可视化"""
    features = df[["S_raw", "R_raw", "C_raw"]].values
    features_scaled = StandardScaler().fit_transform(features)

    tsne = TSNE(n_components=2, random_state=42, perplexity=10)
    features_2d = tsne.fit_transform(features_scaled)

    danxia_idx = df["type"] == "Danxia"
    kasite_idx = df["type"] == "Kasite"

    plt.figure(figsize=(10, 8))
    plt.scatter(
        features_2d[danxia_idx, 0],
        features_2d[danxia_idx, 1],
        c="#FF6B6B",
        alpha=0.7,
        label="Danxia",
        s=80,
        edgecolors="black",
        linewidth=0.5,
    )
    plt.scatter(
        features_2d[kasite_idx, 0],
        features_2d[kasite_idx, 1],
        c="#4ECDC4",
        alpha=0.7,
        label="Kasite",
        s=80,
        edgecolors="black",
        linewidth=0.5,
    )

    plt.xlabel("t-SNE 1", fontsize=12)
    plt.ylabel("t-SNE 2", fontsize=12)
    plt.title("t-SNE Visualization (S/R/C Features)", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✅ t-SNE 图已保存：{output_path}")


def plot_umap(df: pd.DataFrame, output_path: Path) -> None:
    """UMAP 降维可视化"""
    if not UMAP_AVAILABLE:
        print("⏭️ 跳过 UMAP 可视化")
        return

    features = df[["S_raw", "R_raw", "C_raw"]].values
    features_scaled = StandardScaler().fit_transform(features)

    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
    features_2d = reducer.fit_transform(features_scaled)

    danxia_idx = df["type"] == "Danxia"
    kasite_idx = df["type"] == "Kasite"

    plt.figure(figsize=(10, 8))
    plt.scatter(
        features_2d[danxia_idx, 0],
        features_2d[danxia_idx, 1],
        c="#FF6B6B",
        alpha=0.7,
        label="Danxia",
        s=80,
        edgecolors="black",
        linewidth=0.5,
    )
    plt.scatter(
        features_2d[kasite_idx, 0],
        features_2d[kasite_idx, 1],
        c="#4ECDC4",
        alpha=0.7,
        label="Kasite",
        s=80,
        edgecolors="black",
        linewidth=0.5,
    )

    plt.xlabel("UMAP 1", fontsize=12)
    plt.ylabel("UMAP 2", fontsize=12)
    plt.title("UMAP Visualization (S/R/C Features)", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✅ UMAP 图已保存：{output_path}")


def plot_feature_distributions(df: pd.DataFrame, output_path: Path) -> None:
    """绘制 S/R/C 三维特征的分布直方图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    danxia = df[df["type"] == "Danxia"]
    kasite = df[df["type"] == "Kasite"]

    axes[0].hist(
        danxia["S_raw"],
        bins=20,
        alpha=0.7,
        color="#FF6B6B",
        label="Danxia",
        edgecolor="black",
    )
    axes[0].hist(
        kasite["S_raw"],
        bins=20,
        alpha=0.7,
        color="#4ECDC4",
        label="Kasite",
        edgecolor="black",
    )
    axes[0].set_xlabel("S (Slope)", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title("S (Slope) Distribution", fontsize=13)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(
        danxia["R_raw"],
        bins=20,
        alpha=0.7,
        color="#FF6B6B",
        label="Danxia",
        edgecolor="black",
    )
    axes[1].hist(
        kasite["R_raw"],
        bins=20,
        alpha=0.7,
        color="#4ECDC4",
        label="Kasite",
        edgecolor="black",
    )
    axes[1].set_xlabel("R (Roughness)", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title("R (Roughness) Distribution", fontsize=13)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].hist(
        danxia["C_raw"],
        bins=20,
        alpha=0.7,
        color="#FF6B6B",
        label="Danxia",
        edgecolor="black",
    )
    axes[2].hist(
        kasite["C_raw"],
        bins=20,
        alpha=0.7,
        color="#4ECDC4",
        label="Kasite",
        edgecolor="black",
    )
    axes[2].set_xlabel("C (Complexity)", fontsize=12)
    axes[2].set_ylabel("Frequency", fontsize=12)
    axes[2].set_title("C (Complexity) Distribution", fontsize=13)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("S/R/C Feature Distribution", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✅ 特征分布图已保存：{output_path}")


def main():
    base_dir = (
        Path(__file__).parent.parent.parent.parent
        / "data"
        / "training-dataset"
        / "preprocess"
    )
    features_path = base_dir / "features.csv"
    visuals_dir = base_dir / "visuals"

    if not features_path.exists():
        print(f"❌ 特征文件不存在：{features_path}")
        print("请先运行 extract_features.py")
        return

    visuals_dir.mkdir(exist_ok=True)

    print("正在读取 features.csv...")
    df = pd.read_csv(features_path)
    print(f"共读取 {len(df)} 条记录")

    print("\n正在生成可视化...")

    plot_tsne(df, visuals_dir / "tsne_plot.png")
    plot_umap(df, visuals_dir / "umap_plot.png")
    plot_feature_distributions(df, visuals_dir / "feature_distribution.png")

    print(f"\n✅ 可视化完成！输出目录：{visuals_dir}")


if __name__ == "__main__":
    main()
