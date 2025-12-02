# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional
from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    # Dataset settings
    dataset_id: str = Field(default="sst2")

    # Model settings
    model_name: str = Field(default="meta-llama/Llama-3.2-1B")
    max_length: int = Field(default=128, gt=0)
    dtype: str = Field(default="torch.bfloat16")

    # Training hyperparameters
    training_type: str = Field(default="lora")  # [lora, adapters]
    learning_rate: float = Field(default=2e-5, gt=0)
    batch_size: int = Field(default=32, gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    gradient_checkpointing: bool = Field(default=False)
    weight_decay: float = Field(default=0.0, ge=0)
    num_epochs: int = Field(default=1, gt=0)
    optim: str = Field(default="adamw_torch")
    ignored_index: int = Field(default=-100)

    # Logging settings
    log_level: str = Field(default="INFO")
    use_wandb: bool = Field(default=True)
    wandb_project: str = Field(default="llama-finetuning")
    wandb_run_name: str = Field(default="tt-llama-test")
    wandb_tags: list[str] = Field(default_factory=lambda: ["test"])
    wandb_watch_mode: str = Field(default="all")
    wandb_log_freq: int = Field(default=1000)
    model_to_wandb: bool = Field(default=False)
    steps_freq: int = Field(default=25)
    epoch_freq: int = Field(default=1)

    # Checkpoint settings
    resume_from_checkpoint: bool = Field(default=False)
    resume_option: str = Field(default="last")  # [last, best, path]
    checkpoint_path: str = Field(default="")  # path to checkpoint if resume_option is "path"
    checkpoint_metric: str = Field(default="eval/loss")
    checkpoint_metric_mode: str = Field(default="min")  # [min, max]
    keep_last_n: int = Field(default=3, ge=0)
    keep_best_n: int = Field(default=3, ge=0)
    save_strategy: str = Field(default="epoch")
    project_dir: str = Field(default="blacksmith/experiments/torch/llama")
    save_optim: bool = Field(default=False)
    storage_backend: str = Field(default="local")
    sync_to_storage: bool = Field(default=False)
    load_from_storage: bool = Field(default=False)
    remote_path: str = Field(default="")

    # Reproducibility settings
    seed: int = Field(default=23)
    deterministic: bool = Field(default=False)

    # LoRA setup
    lora_r: int = Field(default=4, ge=0)
    lora_alpha: int = Field(default=8, gt=0)
    lora_target_modules: list[str] = Field(default_factory=lambda: ["all-linear"])
    lora_task_type: str = Field(default="CAUSAL_LM")

    # Adapter setup
    adapter_bottleneck_dim: int = Field(default=24, ge=0)
    adapter_non_linearity: str = Field(default="torch.nn.GELU")  # [torch.nn.ReLU, torch.nn.GELU, torch.nn.SiLU]
    adapter_layers: list[int] = Field(default_factory=lambda: [])  # [0, 1] for first and second adapter

    # Device settings
    parallelism_strategy: str = Field(default="single")  # [single, data_parallel, tensor_parallel]
    mesh_shape: str = Field(default="8,1")  # Used if parallelism_strategy != single
    tp_sharding_specs: dict[str, list[Optional[int]]] = Field(default_factory=dict)  # Used for model tp sharding

    # Other settings
    output_dir: str = Field(default="experiments/results/llama32-1b")
    logging_steps: int = Field(default=10, gt=0)
    do_train: bool = Field(default=True)
    print_examples: bool = Field(default=False)
    framework: str = Field(default="pytorch")
    use_tt: bool = Field(default=True)
