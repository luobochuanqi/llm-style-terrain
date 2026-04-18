"""
训练配置模块
统一管理所有训练超参数和预设模式
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


@dataclass
class TrainingConfig:
    """训练配置类

    支持三种预设模式:
    - debug: 10 epochs, 用于代码验证 (5-10 分钟)
    - fast: 50 epochs, 用于快速验证 (30-60 分钟)
    - full: 200 epochs, 用于最终训练 (2-4 小时)
    """

    # ============ 模式选择 ============
    mode: Literal["debug", "fast", "full"] = "full"

    # Data
    data_root: Path = field(
        default_factory=lambda: Path("../data/training-dataset/preprocess")
    )
    image_size: int = 256
    batch_size: int = 8
    num_workers: int = 2  # 使用 2 个 worker 进程，平衡速度和稳定性

    # Data preprocessing
    log_transform_c: bool = True  # 对 C_score 应用 log(C + 1e-5)

    # Data augmentation (仅安全变换!)
    augment_hflip: bool = True  # 水平翻转
    augment_rotate_90: bool = True  # 90° 旋转
    augment_small_rotate: bool = True  # 小角度旋转 (±5°)，新增
    augment_vertical_flip: bool = False  # 禁止！山会变成坑

    # Model
    latent_dim: int = 128
    condition_dim: int = 3
    film_hidden_dim: int = 256  # FiLM style code 维度
    beta: float = 2.0  # KL 损失权重 (从 3.0 降低到 2.0，避免过早正则化)
    beta_warmup_epochs: int = (
        150  # β从 0 增加到 2.0 的 epoch 数 (延长以防止 KL 散度过高)
    )

    # Training
    num_epochs: int = 200
    learning_rate: float = 1e-4
    lr_min: float = 1e-6  # Cosine Annealing 最小学习率
    lr_scheduler_type: str = "cosine"  # "cosine" | "plateau" | "none"
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    device: str = "cuda"

    # Early stopping
    early_stop_patience: Optional[int] = None  # 禁用早停，让训练完成 200 epochs
    min_delta: float = 1e-3

    # Weighted sampling
    use_weighted_sampling: bool = True
    danxia_weight: float = 1.0
    kasite_weight: float = 1.15  # 67/58 ≈ 1.15

    # Gradient loss (Sobel edge enhancement)
    use_gradient_loss: bool = True
    gradient_loss_weight: float = 0.2  # λ=0.2, if unstable reduce to 0.1

    # Logging & visualization
    log_every: int = 10
    val_every: int = 1
    save_sample_every: int = 10
    plot_update_every: int = 1

    # Output
    output_dir: Path = field(default_factory=lambda: Path("outputs/cvae"))
    seed: int = 42

    # ============ 预设模式工厂方法 ============

    @classmethod
    def fast_mode(cls) -> "TrainingConfig":
        """快速模式：50 epochs, 用于快速验证"""
        config = cls()
        config.mode = "fast"
        config.num_epochs = 50
        config.beta = 2.0
        config.beta_warmup_epochs = 25
        config.early_stop_patience = None
        config.save_sample_every = 5
        config.use_gradient_loss = True
        config.lr_scheduler_type = "cosine"
        return config

    @classmethod
    def full_mode(cls) -> "TrainingConfig":
        """完整模式：200 epochs, 用于最终训练"""
        config = cls()
        config.mode = "full"
        config.num_epochs = 200
        config.beta = 2.0
        config.beta_warmup_epochs = 150  # 延长 warmup
        config.early_stop_patience = None  # 禁用早停
        config.use_gradient_loss = True
        config.lr_scheduler_type = "cosine"
        return config

    @classmethod
    def debug_mode(cls) -> "TrainingConfig":
        """调试模式：10 epochs, 用于代码验证"""
        config = cls()
        config.mode = "debug"
        config.num_epochs = 10
        config.beta = 0.001
        config.beta_warmup_epochs = 0
        config.batch_size = 2  # 更小的 batch size
        config.num_workers = 0
        config.learning_rate = 1e-5  # 更低的学习率防止 NaN
        config.early_stop_patience = None  # 禁用早停
        config.use_gradient_loss = True
        config.lr_scheduler_type = "none"  # 调试模式不使用调度器
        return config

    def __post_init__(self):
        """初始化后处理：创建输出目录"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "samples").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "logs").mkdir(parents=True, exist_ok=True)
