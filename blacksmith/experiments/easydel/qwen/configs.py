# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    # Dataset settings
    dataset_id: str = Field(default="wikitext")
    dataset_configuration: str = Field(default="wikitext-2-raw-v1")

    # Model settings
    model_name: str = Field(default="Qwen/Qwen3-0.6B")
    max_length: int = Field(default=128, gt=0)
    dtype: str = Field(default="jnp.bfloat16")
    max_position_embeddings: Optional[int] = Field(default=None)

    # Training hyperparameters
    learning_rate: float = Field(default=2e-4, gt=0)
    batch_size: int = Field(default=4, gt=0)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    num_epochs: int = Field(default=1, gt=0)
    val_steps_freq: Optional[int] = Field(default=None, ge=1)
    max_val_batches: Optional[int] = Field(default=None, ge=1)

    # LoRA settings
    lora_rank: int = Field(default=16, ge=1)
    lora_pattern: str = Field(default=r".*(q_proj|v_proj).*")

    # Logging settings
    steps_freq: int = Field(default=10, ge=1)
    log_level: str = Field(default="INFO")
    model_to_wandb: bool = Field(default=True)
    wandb_project: str = Field(default="Qwen-TT-EasyDel-LoRA-Training")
    wandb_run_name: str = Field(default="qwen3-0.6b-wikitext-tt-easydel")

    # Reproducibility settings
    seed: int = Field(default=42)

    # Device settings
    use_tt: bool = Field(default=True)
    num_devices: int = Field(default=1, ge=1)
