# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import jax.numpy as jnp
from pydantic import BaseModel, Field

_DTYPE_MAP = {
    "bfloat16": jnp.bfloat16,
    "float32": jnp.float32,
    "float16": jnp.float16,
}


class TrainingConfig(BaseModel):
    # Dataset settings
    dataset_id: str = Field(default="sst2")

    # Model settings
    model_name: str = Field(default="Qwen/Qwen3-0.6B")
    max_length: int = Field(default=128, gt=0)
    dtype: str = Field(default="bfloat16")
    mask_max_position_embeddings: Optional[int] = Field(default=None)

    @property
    def jax_dtype(self):
        return _DTYPE_MAP[self.dtype]

    # Training hyperparameters
    learning_rate: float = Field(default=2e-4, gt=0)
    warmup_steps: int = Field(default=0, ge=0)
    end_learning_rate: float = Field(default=0.0, ge=0)
    batch_size: int = Field(default=4, gt=0)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    num_epochs: int = Field(default=1, gt=0)
    val_steps_freq: Optional[int] = Field(default=None, ge=1)
    max_val_batches: Optional[int] = Field(default=None, ge=1)
    ignored_label_index: int = Field(default=-100)

    # LoRA settings
    lora_rank: int = Field(default=16, ge=1)
    lora_pattern: str = Field(default=r".*(q_proj|v_proj).*")

    # Logging settings
    steps_freq: int = Field(default=10, ge=1)
    log_level: str = Field(default="INFO")
    use_wandb: bool = Field(default=True)
    print_examples: bool = Field(default=False)
    wandb_project: str = Field(default="Qwen-TT-EasyDel-LoRA-Training")
    wandb_run_name: str = Field(default="qwen3-0.6b-sst2-tt-easydel")
    wandb_tags: list[str] = Field(default_factory=lambda: ["easydel", "qwen", "lora"])
    wandb_watch_mode: str = Field(default="all")
    wandb_log_freq: int = Field(default=1000)
    model_to_wandb: bool = Field(default=False)

    # Reproducibility settings
    seed: int = Field(default=42)

    # Device settings
    use_tt: bool = Field(default=True)
    num_devices: int = Field(default=1, ge=1)
