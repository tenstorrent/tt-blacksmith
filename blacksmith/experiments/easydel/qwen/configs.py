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
    dataset_id: str = Field(default="wikitext")
    dataset_configuration: str = Field(default="wikitext-2-raw-v1")
    text_column: str = Field(default="text")

    # Model settings
    model_name: str = Field(default="Qwen/Qwen3-0.6B")
    max_length: int = Field(default=128, gt=0)
    dtype: str = Field(default="bfloat16")
    mask_max_position_embeddings: Optional[int] = Field(default=None)

    @property
    def jax_dtype(self):
        key = self.dtype.removeprefix("jnp.")
        if key not in _DTYPE_MAP:
            raise ValueError(f"Unsupported dtype '{self.dtype}'. Use one of: {list(_DTYPE_MAP)}")
        return _DTYPE_MAP[key]

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
    print_examples: bool = Field(default=False)
    wandb_project: str = Field(default="Qwen-TT-EasyDel-LoRA-Training")
    wandb_run_name: str = Field(default="qwen3-0.6b-wikitext-tt-easydel")

    # Reproducibility settings
    seed: int = Field(default=42)

    # Device settings
    use_tt: bool = Field(default=True)
    num_devices: int = Field(default=1, ge=1)
