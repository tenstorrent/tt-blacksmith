# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Configuration for DPO (Direct Preference Optimization) training.

Based on the paper: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
https://arxiv.org/pdf/2305.18290

DPO is a training objective (loss function), orthogonal to PEFT methods like LoRA/adapters.
Use `training_model_type` to specify the parameter-efficient fine-tuning approach.
"""
from typing import List, Optional, Tuple

from pydantic import Field

from blacksmith.tools.templates.configs import TrainingConfig as BaseTrainingConfig
from blacksmith.tools.test_config import TestConfig


class DPOTrainingConfig(BaseTrainingConfig):
    """
    Configuration for DPO training.

    training_model_type: fine-tuning approach for the policy model -
        "lora" or "adapters" (any other value falls back to full fine-tuning).
        DPO itself is the training objective and is always applied by this experiment.
    """

    # Training type - DPO objective
    training_model_type: str = Field(default="lora")

    # Dataset settings
    dataset_id: str = Field(default="math_preference_dpo")

    # Model settings
    model_name: str = Field(default="google/gemma-1.1-2b-it")
    max_length: int = Field(default=128, gt=0)  # Reduced for memory efficiency
    dtype: str = Field(default="torch.bfloat16")
    ignored_index: int = Field(default=-100)

    # DPO-specific hyperparameters
    dpo_beta: float = Field(default=0.2, gt=0, description="DPO temperature parameter (higher = more conservative)")
    dpo_label_smoothing: float = Field(default=0.0, ge=0, le=0.5, description="Label smoothing for DPO loss")

    # Reference model settings
    # Standard DPO requires π_ref to be an SFT model trained on chosen responses.
    # If sft_checkpoint_path is empty, π_ref is initialized from the base model (less ideal but works).
    sft_checkpoint_path: str = Field(
        default="", description="Path to SFT checkpoint for reference model. If empty, uses base model."
    )

    # Training hyperparameters
    learning_rate: float = Field(default=1e-5, gt=0)  # Lower LR typical for DPO
    batch_size: int = Field(default=1, gt=0)  # Small batch size for memory efficiency
    gradient_accumulation_steps: int = Field(default=8, gt=0)  # Effective batch size = 8
    gradient_checkpointing: bool = Field(default=False)
    weight_decay: float = Field(default=0.0, ge=0)
    num_epochs: int = Field(default=2, gt=0)
    max_steps: int = Field(default=-1)  # -1 means use num_epochs
    optim: str = Field(default="adamw_torch")
    warmup_steps: int = Field(default=100, ge=0)

    # Logging settings
    log_level: str = Field(default="INFO")
    use_wandb: bool = Field(default=True)
    wandb_project: str = Field(default="gemma11-dpo")
    wandb_run_name: str = Field(default="tt-gemma11-dpo-math")
    wandb_tags: list[str] = Field(default_factory=lambda: ["gemma11", "dpo", "math"])
    wandb_watch_mode: str = Field(default="all")
    wandb_log_freq: int = Field(default=1000)
    model_to_wandb: bool = Field(default=False)
    steps_freq: int = Field(default=10)
    epoch_freq: int = Field(default=1)
    val_steps_freq: int = Field(default=32)
    print_examples: bool = Field(default=False)

    # Checkpoint settings
    resume_from_checkpoint: bool = Field(default=False)
    resume_option: str = Field(default="last")
    checkpoint_path: str = Field(default="")
    checkpoint_metric: str = Field(default="val/accuracy")
    checkpoint_metric_mode: str = Field(default="max")
    keep_last_n: int = Field(default=3, ge=0)
    keep_best_n: int = Field(default=3, ge=0)
    save_strategy: str = Field(default="step")
    save_steps: int = Field(default=20)
    project_dir: str = Field(default="blacksmith/experiments/torch/gemma11/dpo")
    save_optim: bool = Field(default=False)
    storage_backend: str = Field(default="local")
    sync_to_storage: bool = Field(default=False)
    load_from_storage: bool = Field(default=False)
    remote_path: str = Field(default="")

    # Reproducibility settings
    seed: int = Field(default=23)
    deterministic: bool = Field(default=False)

    # Device settings (mesh configuration for parallelism)
    mesh_shape: Optional[list[int]] = Field(default=None, description="Mesh shape for SPMD parallelism, e.g. [8, 1]")
    mesh_axis_names: Optional[list[str]] = Field(
        default=None, description="Axis names for mesh, e.g. ['data', 'model']"
    )
    # Data parallelism: mesh axis along which input batches are sharded (must be in mesh_axis_names or None).
    input_sharding_dim: Optional[str] = Field(default=None)
    # Tensor parallelism: list of (regex_pattern, sharding_spec_tuple) matched against module names.
    model_sharding_patterns: Optional[List[Tuple[str, Tuple[Optional[str], ...]]]] = Field(default=None)

    # LoRA setup (used when peft_method="lora")
    lora_r: int = Field(default=16, gt=0)
    lora_alpha: int = Field(default=32, gt=0)
    lora_target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    lora_task_type: str = Field(default="CAUSAL_LM")
    lora_dropout: float = Field(default=0.0, ge=0, le=1)

    # Adapter setup (used when peft_method="adapters")
    adapter_bottleneck_dim: int = Field(default=32, ge=0)
    adapter_non_linearity: str = Field(default="torch.nn.GELU")
    adapter_layers: list[int] = Field(default_factory=lambda: [])

    # Other settings
    framework: str = Field(default="pytorch")
    use_tt: bool = Field(default=True)
    optimization_level: int = Field(default=0, ge=0, le=2)
    do_validation: bool = Field(default=True)

    # Testing utils (used to limit training duration during CI runs).
    test_config: Optional[TestConfig] = Field(default=None)
