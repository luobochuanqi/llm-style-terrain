"""
数据预处理配置
支持 256x256 和 512x512 两种分辨率
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class DataProcessConfig:
    """数据预处理配置"""

    # 分辨率：256 或 512
    image_size: int = 256

    # 数据集基础路径
    dataset_base: Path = field(
        default_factory=lambda: (
            Path(__file__).parent.parent.parent.parent.parent
            / "data"
            / "training-dataset"
            / "preprocess"
        )
    )

    # 输入目录
    danxia_dir: Path = field(default_factory=lambda: Path("Danxia"))
    kasite_dir: Path = field(default_factory=lambda: Path("Kasite"))

    # 输出目录命名
    output_dir_name: str = field(init=False)
    splits_dir_name: str = field(init=False)

    def __post_init__(self):
        if self.image_size == 256:
            self.output_dir_name = "normalized"
            self.splits_dir_name = "splits"
        elif self.image_size == 512:
            self.output_dir_name = "normalized_512"
            self.splits_dir_name = "splits_512"
        else:
            raise ValueError(f"image_size 必须为 256 或 512，当前为 {self.image_size}")

    @property
    def output_base(self) -> Path:
        """输出基础目录"""
        return self.dataset_base / self.output_dir_name

    @property
    def splits_base(self) -> Path:
        """划分文件目录"""
        return self.dataset_base / self.splits_dir_name

    def get_features_path(self) -> Path:
        """特征文件路径"""
        if self.image_size == 256:
            return self.dataset_base / "features.csv"
        else:
            return self.dataset_base / "features_512.csv"


# 全局配置实例（默认 256x256）
config_256 = DataProcessConfig(image_size=256)
config_512 = DataProcessConfig(image_size=512)
