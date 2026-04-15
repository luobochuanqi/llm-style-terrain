"""
地形数据集模块
支持安全的数据增强和加权采样
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List, Dict
import random


class SafeAugmentation:
    """地形高度图的安全增强变换

    ⚠️ 禁止垂直翻转（山会变成坑）
    ✅ 允许：水平翻转、90° 旋转、小角度旋转 (±5°)
    """

    def __init__(
        self,
        hflip: bool = True,
        rotate_90: bool = True,
        vertical_flip: bool = False,
        small_rotate: bool = True,  # 新增：小角度旋转
    ):
        self.hflip = hflip
        self.rotate_90 = rotate_90
        self.vertical_flip = vertical_flip
        self.small_rotate = small_rotate

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """应用随机增强

        Args:
            image: 高度图张量 (1, H, W)

        Returns:
            增强后的高度图
        """
        # 水平翻转
        if self.hflip and random.random() > 0.5:
            image = torch.flip(image, dims=[2])

        # 90° 旋转 (0, 90, 180, 270 度)
        if self.rotate_90:
            k = random.randint(0, 3)
            image = torch.rot90(image, k=k, dims=[1, 2])

        # 新增：小角度旋转 (±5°)，增加数据多样性
        if self.small_rotate and random.random() > 0.5:
            angle = random.uniform(-5, 5)
            image = self._rotate_small(image, angle)

        # ⚠️ 禁止垂直翻转
        # if self.vertical_flip and random.random() > 0.5:
        #     image = torch.flip(image, dims=[1])

        return image

    def _rotate_small(self, image: torch.Tensor, angle: float) -> torch.Tensor:
        """小角度旋转 (±5°)，使用纯 torch 实现

        Args:
            image: 高度图张量 (1, H, W)
            angle: 旋转角度 (度)

        Returns:
            旋转后的高度图
        """
        import math

        H, W = image.shape[1], image.shape[2]
        center = (W - 1) / 2.0, (H - 1) / 2.0

        # 旋转矩阵
        angle_rad = math.radians(angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # 创建网格
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        x = x.float().to(image.device)
        y = y.float().to(image.device)

        # 旋转向量
        x_rot = (x - center[0]) * cos_a - (y - center[1]) * sin_a + center[0]
        y_rot = (x - center[0]) * sin_a + (y - center[1]) * cos_a + center[1]

        # 归一化到 [-1, 1]
        x_rot = 2.0 * x_rot / (W - 1) - 1.0
        y_rot = 2.0 * y_rot / (H - 1) - 1.0

        # 网格采样
        grid = torch.stack([x_rot, y_rot], dim=-1).unsqueeze(0)  # (1, H, W, 2)
        rotated = torch.nn.functional.grid_sample(
            image.unsqueeze(0),  # (1, 1, H, W)
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(0)  # (1, H, W)

        return rotated

    def __repr__(self):
        return (
            f"SafeAugmentation(hflip={self.hflip}, "
            f"rotate_90={self.rotate_90}, "
            f"small_rotate={self.small_rotate})"
        )


class ConditionNormalizer:
    """条件向量归一化器

    使用训练集统计量进行归一化，对 C_score 应用对数变换
    统计量来自 features.csv 的训练集部分
    """

    # 从训练集计算的统计量 (S_score, R_score, C_score)
    MEAN = torch.tensor([2.56, 4.04, 1.05])
    STD = torch.tensor([2.5, 2.2, 2.0])

    def __init__(self, log_transform_c: bool = True):
        self.log_transform_c = log_transform_c

    def __call__(self, conditions: torch.Tensor) -> torch.Tensor:
        """归一化条件向量

        Args:
            conditions: 原始条件向量 (3,) 或 (batch, 3)

        Returns:
            归一化后的条件向量
        """
        # 确保是 2D
        if conditions.dim() == 1:
            conditions = conditions.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        conditions = conditions.clone()

        # 对 C_score 应用对数变换
        if self.log_transform_c:
            conditions[:, 2] = torch.log(conditions[:, 2] + 1e-5)
            # 重新计算对数变换后的统计量
            mean = self.MEAN.clone()
            mean[2] = torch.log(torch.tensor(1.05) + 1e-5)
            std = self.STD.clone()
            std[2] = 1.5  # 近似值
        else:
            mean = self.MEAN
            std = self.STD

        # 标准化
        conditions = (conditions - mean.to(conditions.device)) / (
            std.to(conditions.device) + 1e-6
        )

        if squeeze:
            conditions = conditions.squeeze(0)

        return conditions


class TerrainDataset(Dataset):
    """PyTorch 地形数据集

    从 features.csv 和 split files 加载数据
    """

    def __init__(
        self,
        data_root: Path,
        split: str = "train",
        image_size: int = 256,
        augment: bool = True,
        log_transform_c: bool = True,
        features_csv: Optional[Path] = None,
        splits_dir: Optional[Path] = None,
    ):
        """
        Args:
            data_root: 数据根目录
            split: 数据集划分 ("train", "val", "test")
            image_size: 图像尺寸
            augment: 是否应用数据增强
            log_transform_c: 是否对 C_score 应用对数变换
        """
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.augment = augment and (split == "train")

        # 支持自定义 splits 目录
        if splits_dir is not None:
            split_csv = Path(splits_dir) / f"{split}.csv"
        else:
            split_csv = self.data_root / "splits" / f"{split}.csv"

        if not split_csv.exists():
            raise FileNotFoundError(f"Split file not found: {split_csv}")

        self.split_df = pd.read_csv(split_csv)

        # 支持自定义 features 文件
        if features_csv is not None:
            features_csv_path = Path(features_csv)
        else:
            features_csv_path = self.data_root / "features.csv"
            if not features_csv_path.exists():
                features_csv_path = (
                    self.data_root.parent / "preprocess" / "features.csv"
                )

        if not features_csv_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_csv_path}")

        features_df = pd.read_csv(features_csv_path)

        # splits 的 path 列已经包含完整路径 (normalized/train/xxx.png)
        # features 的 filename 列也是完整路径
        # 直接匹配 splits.path 和 features.filename
        self.data_df = self.split_df.merge(
            features_df, left_on="path", right_on="filename"
        )

        # 验证图像路径
        print(f"✅ 加载 {split} 数据集：{len(self.data_df)} 个样本")
        if len(self.data_df) == 0:
            raise ValueError(f"数据集为空，请检查 {split_csv}")

        # 条件归一化由模型内部的 ConditionNormalizer 处理

        # 数据增强
        if self.augment:
            self.transform = SafeAugmentation(
                hflip=True,
                rotate_90=True,
                vertical_flip=False,
                small_rotate=True,  # 启用小角度旋转
            )
        else:
            self.transform = None

        print(f"✅ 加载 {split} 数据集：{len(self.data_df)} 个样本")

    def __len__(self) -> int:
        return len(self.data_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        """获取单个样本

        Returns:
            condition: 归一化后的条件向量 (3,)
            heightmap: 高度图张量 (1, H, W)
            filename: 文件名
            terrain_type: 地貌类型 ("Danxia" or "Kasite")
        """
        row = self.data_df.iloc[idx]

        # 加载图像
        # 使用绝对路径，避免工作目录问题
        img_path_str = str(row["path"])

        # 尝试多个可能的路径
        possible_paths = [
            # 项目根目录/data/...
            Path(
                "/home/luobo/mine/projects/llm-style-terrain/data/training-dataset/preprocess"
            )
            / img_path_str,
            # 相对路径
            Path("../data/training-dataset/preprocess") / img_path_str,
        ]

        image = None
        for img_path in possible_paths:
            if img_path.exists():
                image = Image.open(img_path)
                break

        if image is None:
            raise FileNotFoundError(
                f"无法找到图像：{img_path_str}, 尝试的路径：{possible_paths}"
            )

        # 转换为张量 (16-bit PNG → float32 [0, 1])
        img_array = np.array(image)
        if img_array.dtype == np.uint16:
            img_array = img_array.astype(np.float32) / 65535.0
        elif img_array.dtype == np.uint8:
            img_array = img_array.astype(np.float32) / 255.0
        else:
            img_array = img_array.astype(np.float32)

        # 确保是单通道
        if len(img_array.shape) == 3:
            img_array = img_array.mean(axis=2)

        heightmap = torch.from_numpy(img_array).unsqueeze(0).float()

        # 调整尺寸
        if heightmap.shape[-1] != self.image_size:
            heightmap = torch.nn.functional.interpolate(
                heightmap.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        # 应用增强
        if self.transform is not None:
            heightmap = self.transform(heightmap)

        # 条件向量 (不归一化，由模型内部的 ConditionNormalizer 处理)
        conditions = torch.tensor(
            [row["S_score"], row["R_score"], row["C_score"]],
            dtype=torch.float32,
        )

        # 地貌类型 (使用 type_x 或 type，因为合并后可能有两个 type 列)
        type_col = "type_x" if "type_x" in self.data_df.columns else "type"
        terrain_type = row[type_col]

        # filename 也有两个：filename_x (from splits) and filename_y (from features)
        filename_col = (
            "filename_x" if "filename_x" in self.data_df.columns else "filename"
        )

        return conditions, heightmap, row[filename_col], terrain_type

    def get_terrain_types(self) -> Dict[str, int]:
        """获取地貌类型到索引的映射"""
        types = self.data_df["type"].unique()
        return {t: i for i, t in enumerate(types)}

    def get_weights(self) -> List[float]:
        """计算每个样本的采样权重（用于处理类别不平衡）"""
        weights = []
        # 合并后有两个 type 列：type_x (from splits) and type_y (from features)
        type_col = "type_x" if "type_x" in self.data_df.columns else "type"
        for _, row in self.data_df.iterrows():
            terrain_type = row[type_col]
            if terrain_type == "Danxia":
                weights.append(1.0)
            elif terrain_type == "Kasite":
                weights.append(1.15)  # 67/58 ≈ 1.15
            else:
                weights.append(1.0)
        return weights


def create_dataloader(
    config,
    split: str = "train",
    augment: bool = True,
    data_root: Optional[Path] = None,
    features_csv: Optional[Path] = None,
    splits_dir: Optional[Path] = None,
):
    """创建 DataLoader

    Args:
        config: 训练配置
        split: 数据集划分
        augment: 是否应用增强
        data_root: 自定义数据根目录（可选）
        features_csv: 自定义 features 文件（可选）
        splits_dir: 自定义 splits 目录（可选）

    Returns:
        dataloader: DataLoader
        dataset: TerrainDataset
    """
    # 使用自定义路径或配置中的默认路径
    dataset = TerrainDataset(
        data_root=data_root or config.data_root,
        split=split,
        image_size=config.image_size,
        augment=augment,
        log_transform_c=config.log_transform_c,
    )

    # 支持自定义 features 和 splits 路径
    if features_csv is not None or splits_dir is not None:
        # 重新加载数据集，使用自定义路径
        dataset_custom = TerrainDataset(
            data_root=data_root or config.data_root,
            split=split,
            image_size=config.image_size,
            augment=augment,
            log_transform_c=config.log_transform_c,
            features_csv=features_csv,
            splits_dir=splits_dir,
        )
        return dataset_custom

    # 训练集使用加权采样
    if split == "train" and config.use_weighted_sampling:
        weights = dataset.get_weights()
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )
        shuffle = False  # 使用 sampler 时不需要 shuffle
    else:
        sampler = None
        shuffle = split == "train"

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
        persistent_workers=(config.num_workers > 0),  # 保持 worker 进程活跃
        prefetch_factor=2 if config.num_workers > 0 else None,  # 预取因子
    )

    return dataloader, dataset


def create_data_loaders(
    data_root: Path,
    features_csv: Path,
    splits_dir: Path,
    batch_size: int = 8,
    num_workers: int = 2,
    image_size: int = 256,
    log_transform_c: bool = True,
    augment_hflip: bool = True,
    augment_rotate_90: bool = True,
    augment_small_rotate: bool = True,
    use_weighted_sampling: bool = True,
    danxia_weight: float = 1.0,
    kasite_weight: float = 1.15,
):
    """创建 train 和 val 数据加载器

    Args:
        data_root: 数据根目录（normalized_512 等）
        features_csv: features 文件路径
        splits_dir: splits 目录
        batch_size: batch size
        num_workers: 数据加载 worker 数
        image_size: 图像尺寸
        log_transform_c: 是否对 C_score 做 log 变换
        augment_hflip: 水平翻转
        augment_rotate_90: 90 度旋转
        augment_small_rotate: 小角度旋转
        use_weighted_sampling: 是否使用加权采样
        danxia_weight: 丹霞权重
        kasite_weight: 喀斯特权重

    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from typing import Optional

    # 创建临时配置对象（简化版）
    class SimpleConfig:
        def __init__(self):
            self.data_root = data_root
            self.image_size = image_size
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.log_transform_c = log_transform_c
            self.use_weighted_sampling = use_weighted_sampling
            self.danxia_weight = danxia_weight
            self.kasite_weight = kasite_weight

    config = SimpleConfig()

    # 创建训练集
    train_dataset = TerrainDataset(
        data_root=data_root,
        split="train",
        image_size=image_size,
        augment=True,
        log_transform_c=log_transform_c,
        features_csv=features_csv,
        splits_dir=splits_dir,
    )

    # 创建验证集
    val_dataset = TerrainDataset(
        data_root=data_root,
        split="val",
        image_size=image_size,
        augment=False,
        log_transform_c=log_transform_c,
        features_csv=features_csv,
        splits_dir=splits_dir,
    )

    # 加权采样 - 从 dataframe 读取，不触发 __getitem__
    if use_weighted_sampling:
        weights = []
        for idx in range(len(train_dataset.data_df)):
            row = train_dataset.data_df.iloc[idx]
            terrain_type = row.get("type", "")
            filename = row.get("filename", "")
            if "danxia" in terrain_type.lower() or "danxia" in filename.lower():
                weights.append(danxia_weight)
            else:
                weights.append(kasite_weight)

        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )
        train_shuffle = False
    else:
        sampler = None
        train_shuffle = True

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader
