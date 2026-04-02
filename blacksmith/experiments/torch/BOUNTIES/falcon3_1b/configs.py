# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    # Dataset settings
    dataset_id: str = Field(default="wikitext")

    # Model settings
    model_name: str = Field(default="tiiuae/Falcon3-1B-Base")
    max_length: int = Field(default=128, gt=0)
    dtype: str = Field(default="torch.bfloat16")
    ignored_index: int = Field(default=-100)

    # Training hyperparameters
    training_type: str = Field(default="lora")  # [lora, adapters]
    learning_rate: float = Field(default=5e-5, gt=0)
    batch_size: int = Field(default=4, gt=0)
    gradient_checkpointing: bool = Field(default=False)
    num_epochs: int = Field(default=3, gt=0)
    optim: str = Field(default="adamw_torch")

    # Logging settings
    log_level: str = Field(default="INFO")
    use_wandb: bool = Field(default=True)
    wandb_project: str = Field(default="falcon3-finetuning")
    wandb_run_name: str = Field(default="tt-falcon3-wikitext")
    wandb_tags: list[str] = Field(default_factory=lambda: ["falcon3", "lora", "wikitext"])
    wandb_watch_mode: str = Field(default="all")
    wandb_log_freq: int = Field(default=100)
    model_to_wandb: bool = Field(default=False)
    steps_freq: int = Field(default=10)
    epoch_freq: int = Field(default=1)
    val_steps_freq: int = Field(default=50)
    print_examples: bool = Field(default=True)

    # Checkpoint settings
    resume_from_checkpoint: bool = Field(default=False)
    resume_option: str = Field(default="last")  # [last, best, path]
    checkpoint_path: str = Field(default="")  # path to checkpoint if resume_option is "path"
    checkpoint_metric: str = Field(default="eval/loss")
    checkpoint_metric_mode: str = Field(default="min")  # [min, max]
    keep_last_n: int = Field(default=3, ge=0)
    keep_best_n: int = Field(default=3, ge=0)
    save_strategy: str = Field(default="epoch")
    project_dir: str = Field(default="blacksmith/experiments/torch/BOUNTIES/falcon3_1b")
    save_optim: bool = Field(default=False)
    storage_backend: str = Field(default="local")
    sync_to_storage: bool = Field(default=False)
    load_from_storage: bool = Field(default=False)
    remote_path: str = Field(default="")

    # Reproducibility settings
    seed: int = Field(default=42)
    deterministic: bool = Field(default=False)

    # Device settings
    mesh_shape: Optional[list[int]] = Field(default=None)  # Use None for single device, [x,y] for 2D mesh.
    mesh_axis_names: Optional[list[str]] = Field(default=None)  # Use None for single device.
    # Model sharding patterns (regex pattern based - matches module names).
    # Format: List of tuples (regex_pattern, sharding_spec_tuple).
    model_sharding_patterns: Optional[List[Tuple[str, Tuple[Optional[str], ...]]]] = Field(default=None)
    input_sharding_dim: Optional[str] = Field(
        default=None
    )  # If defined, we will shard inputs along this mesh axis dimension.

    # LoRA setup - optimized for better learning
    lora_r: int = Field(default=32, gt=0)
    lora_alpha: int = Field(default=64, gt=0)
    lora_target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    lora_task_type: str = Field(default="CAUSAL_LM")

    # Embedding settings - Falcon3 is trained on limited languages (EN, FR, PT, ES).
    # Wikitext contains tokens from other languages that the frozen embedding layer
    # cannot represent well. Unfreezing embeddings allows the model to adapt its
    # token representations during fine-tuning, improving loss convergence.
    unfreeze_embeddings: bool = Field(default=True)

    # Other settings
    framework: str = Field(default="pytorch")
    use_tt: bool = Field(default=True)
