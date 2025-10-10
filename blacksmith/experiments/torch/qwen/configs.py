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
    lora_target_modules: list[str] = Field(default_factory=lambda: ["all-linear"])
    lora_task_type: str = Field(default="CAUSAL_LM")

    # Other settings
    seed: int = Field(default=23)
    output_dir: str = Field(default="experiments/results/llama32-1b")
    report_to: str = Field(default="wandb")
    wandb_project: str = Field(default="llama-finetuning")
    wandb_run_name: str = Field(default="llama-finetuning_pure_torch")
    wandb_watch_mode: str = Field(default="all")
    wandb_log_freq: int = Field(default=1000)
    model_to_wandb: bool = Field(default=False)
    save_strategy: str = Field(default="epoch")
    logging_strategy: str = Field(default="steps")
    logging_steps: int = Field(default=10, gt=0)
    eval_frequency: int = Field(default=1000, gt=0)
    save_total_limit: int = Field(default=3, gt=0)
    do_train: bool = Field(default=True)
    do_eval: bool = Field(default=True)
    use_tt: bool = Field(default=True)
