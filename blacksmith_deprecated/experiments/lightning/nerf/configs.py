# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional

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
    input_dir: str = "./data/nerf_synthetic/lego"
    img_wh: List[int] = Field(default=[800, 800])
    batch_size: int = 1024


class TrainingConfig(BaseModel):
    use_forge: bool = False
    device: str = "cpu"
    val_only: bool = False
    epochs: int = 16
    loss: str = "mse"
    optimizer: str = "radam"
    optimizer_kwargs: Optional[dict] = None
    lr_scheduler: Optional[str] = None
    lr_scheduler_kwargs: Optional[dict] = None
    warmup_multiplier: float = 1.0
    warmup_epochs: int = 0
    ckpt_path: Optional[str] = None
    log_every: int = 5
    log_dir: str = "./logs"


class NerfConfig(BaseModel):
    experiment_name: str = "nerf-training"
    tags: List[str] = Field(default=["nerf"])
    dataset_id: str = "nerf"
    model: ModelConfig = ModelConfig()
    data_loading: DataLoadingConfig = DataLoadingConfig()
    training: TrainingConfig = TrainingConfig()


def load_config(path: str) -> NerfConfig:
    import yaml

    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return NerfConfig(**config)
