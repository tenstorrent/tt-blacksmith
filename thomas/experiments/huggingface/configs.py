# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    # Dataset settings
    dataset_id: str = Field(default="stanfordnlp/sst2")

    # Model settings
    model_name: str = Field(default="meta-llama/Llama-3.2-1B")
    max_length: int = Field(default=128, gt=0)
    dtype: str = Field(default="torch.bfloat16")

    # Training hyperparameters
    learning_rate: float = Field(default=2e-5, gt=0)
    batch_size: int = Field(default=32, gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    gradient_checkpointing: bool = Field(default=False)
    num_epochs: int = Field(default=1, gt=0)
    optim: str = Field(default="adamw_torch")

    # LoRA setup
    lora_r: int = Field(default=4, gt=0)
    lora_alpha: int = Field(default=8, gt=0)
    lora_dropout: float = Field(default=0.1, ge=0, le=1)
    lora_bias: str = Field(default="none")
    lora_target_modules: str = Field(default="all-linear")

    # Other settings
    seed: int = Field(default=23)
    output_dir: str = Field(default="experiments/results/llama32-1b-bs32-ft32-ml128-r4-a8-adamw_torch")
    wandb_project: str = Field(default="llama-finetuning")
    logging_steps: int = Field(default=10, gt=0)
    save_total_limit: int = Field(default=3, gt=0)
