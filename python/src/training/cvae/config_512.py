"""
512 分辨率训练配置
基于 Phase 1 优化配置，针对 512x512 图像调整
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


@dataclass
class TrainingConfig512:
    """512 分辨率训练配置类

    针对 512x512 图像的 cVAE 训练配置
    基于 Phase 1 优化配置 (梯度损失 + Cosine LR + Beta=2.0)
    """

    # ============ 模式选择 ============
    mode: Literal["debug", "fast", "full"] = "full"

    # Data
    data_root: Path = field(
        default_factory=lambda: Path("../data/training-dataset/preprocess/normalized_512")
    )
    features_csv: Path = field(
        default_factory=lambda: Path("../data/training-dataset/preprocess/features_512.csv")
    )
    splits_dir: Path = field(
        default_factory=lambda: Path("../data/training-dataset/preprocess/splits_512")
    )
    image_size: int = 512  # 512x512 分辨率
    batch_size: int = 4  # 512 分辨率显存占用更大，减小 batch size
    num_workers: int = 2

    # Data preprocessing
    log_transform_c: bool = True

    # Data augmentation
    augment_hflip: bool = True
    augment_rotate_90: bool = True
    augment_small_rotate: bool = True
    augment_vertical_flip: bool = False

    # Model
    latent_dim: int = 256  # 512 分辨率需要更大的隐空间
    condition_dim: int = 3
    film_hidden_dim: int = 256
    beta: float = 2.0
    beta_warmup_epochs: int = 150

    # Training
    num_epochs: int = 200
    learning_rate: float = 1e-4
    lr_min: float = 1e-6
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    device: str = "cuda"

    # Early stopping
    early_stop_patience: Optional[int] = None
    min_delta: float = 1e-3

    # Weighted sampling
    use_weighted_sampling: bool = True
    danxia_weight: float = 1.0
    kasite_weight: float = 1.15

    # Gradient loss
    use_gradient_loss: bool = True
    gradient_loss_weight: float = 0.2

    # Logging & visualization
    log_every: int = 10
    val_every: int = 1
    save_sample_every: int = 10
    plot_update_every: int = 1

    # Output
    output_dir: Path = field(default_factory=lambda: Path("outputs/cvae_512"))
    seed: int = 42

    # ============ 预设模式 ============

    @classmethod
    def fast_mode(cls) -> "TrainingConfig512":
        """快速模式：50 epochs"""
        config = cls()
        config.mode = "fast"
        config.num_epochs = 50
        config.beta_warmup_epochs = 25
        config.early_stop_patience = None
        config.save_sample_every = 5
        return config

    @classmethod
    def full_mode(cls) -> "TrainingConfig512":
        """完整模式：200 epochs"""
        config = cls()
        config.mode = "full"
        config.num_epochs = 200
        config.beta_warmup_epochs = 150
        config.early_stop_patience = None
        return config

    @classmethod
    def debug_mode(cls) -> "TrainingConfig512":
        """调试模式：10 epochs"""
        config = cls()
        config.mode = "debug"
        config.num_epochs = 10
        config.beta = 0.001
        config.beta_warmup_epochs = 0
        config.batch_size = 2
        config.num_workers = 0
        config.learning_rate = 1e-5
        config.early_stop_patience = None
        config.lr_scheduler_type = "none"
        return config

    def __post_init__(self):
        """创建输出目录"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "samples").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "logs").mkdir(parents=True, exist_ok=True)
