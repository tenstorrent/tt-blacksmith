# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


class NetConfig(BaseModel):
    depth: int = 4
    width: int = 8
    samples: int = 8


class ModelConfig(BaseModel):
    deg: int = 2
    num_freqs: int = 10
    coarse: NetConfig = NetConfig()
    fine: NetConfig = NetConfig()
    coord_scope: Optional[float] = None
    sigma_init: float = 30.0
    sigma_default: float = -20.0
    weight_threshold: float = 1e-4
    uniform_ratio: float = 0.01
    beta: float = 0.1
    warmup_step: int = 0
    in_channels_dir: int = 32
    in_channels_xyz: int = 63


class DataLoadingConfig(BaseModel):
    dataset_name: str = "Tenstorrent/tt-nerf-p150-white"
    img_wh: List[int] = Field(default=[400, 400])
    batch_size: int = 1024


class TrainingConfig(BaseModel):
    use_forge: bool = False
    device: str = "cpu"
    val_only: bool = False
    epochs: int = 16
    loss: str = "mse"
    optimizer: str = "radam"
    lr: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    lr_scheduler: Optional[str] = None
    lr_scheduler_kwargs: Optional[dict] = None
    warmup_multiplier: float = 1.0
    warmup_epochs: int = 0
    ckpt_path: Optional[str] = None
    log_every: int = 5
    log_dir: str = "./logs"
    log_on_wandb: bool = False
    cache_voxels_fine: bool = False
    resume: bool = False
    render: bool = False


class CheckpointConfig(BaseModel):
    save_dir: str = "./checkpoints"
    render_dir: str = "./renders"
    save_every: int = 500
    keep_last: int = 3


class NerfConfig(BaseModel):
    project_name: str = "nerf"
    experiment_name: str = "nerf-training"
    tags: List[str] = Field(default=["nerf"])
    model: ModelConfig = ModelConfig()
    data_loading: DataLoadingConfig = DataLoadingConfig()
    training: TrainingConfig = TrainingConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
