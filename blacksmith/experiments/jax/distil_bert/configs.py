# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel, Field


class ExperimentConfig(BaseModel):
    # Dataset settings
    dataset_id: str = Field(default="glue/sst2")
    tokenizer_name: str = Field(default="bert-base-uncased")
    max_length: int = Field(default=128, gt=0)

    # Model settings
    teacher_model: str = Field(default="textattack/bert-base-uncased-SST-2")
    student_model: str = Field(default="distilbert-base-uncased")
    dtype: str = Field(default="jax.bfloat16")

    # Training hyperparameters
    learning_rate: float = Field(default=1e-5, gt=0)
    batch_size: int = Field(default=32, gt=0)
    num_epochs: int = Field(default=3, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    warmup_ratio: float = Field(default=0.06, ge=0, le=1.0)
    optimizer: str = Field(default="adamw")
    seed: int = Field(default=42)
    resume_from_checkpoint: bool = Field(default=False)

    # Loss settings
    temperature: float = Field(default=2.0, gt=0)
    alpha_kl: float = Field(default=0.45, ge=0)
    alpha_ce: float = Field(default=1.0, ge=0)
    alpha_cos: float = Field(default=0.1, ge=0)

    # Logging
    use_wandb: bool = Field(default=True)
    experiment_name: str = Field(default="Flax DistilBERT on SST-2")
    project_name: str = Field(default="bert-distillation")
    job_name: str = Field(default="distillation")
    log_every: int = Field(default=50, gt=0)
    log_val_every: int = Field(default=100, gt=0)
    do_checkpoint: bool = Field(default=True)
    checkpoint_every: int = Field(default=250, gt=0)
    keep_top_k_checkpoints: int = Field(default=2)
    output_dir: str = Field(default="blacksmith/experiments/jax/distil_bert")
