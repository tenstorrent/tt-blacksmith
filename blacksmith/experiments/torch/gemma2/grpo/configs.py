# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Configuration for GRPO (Group Relative Policy Optimization) training.

Based on the paper: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning
in Open Language Models" https://arxiv.org/pdf/2402.03300

GRPO is a training objective (loss function), orthogonal to PEFT methods like
LoRA/adapters. Use `training_model_type` to specify the parameter-efficient
fine-tuning approach for the policy model.
"""
from typing import Optional

from pydantic import BaseModel, Field

from blacksmith.tools.test_config import TestConfig


class GRPOTrainingConfig(BaseModel):
    """Configuration for GRPO training on GSM8K."""

    # Training type - PEFT approach for the policy model.
    training_model_type: str = Field(default="lora")  # [lora, adapters, full]

    # Dataset settings
    dataset_id: str = Field(default="gsm8k")

    # Model settings
    model_name: str = Field(default="google/gemma-2-2b-it")
    dtype: str = Field(default="torch.bfloat16")
    ignored_index: int = Field(default=-100)

    # GRPO / generation hyperparameters
    num_generations: int = Field(default=4, gt=1, description="G: completions sampled per prompt (>=2 for advantage std)")
    max_prompt_length: int = Field(default=256, gt=0)
    max_completion_length: int = Field(default=200, gt=0)
    temperature: float = Field(default=0.5, ge=0.0, description="Sampling temperature (0 = greedy)")
    top_k: int = Field(default=0, ge=0, description="Top-k sampling cutoff (0 = disabled)")
    grpo_beta: float = Field(default=0.005, ge=0.0, description="KL penalty coefficient")
    grpo_epsilon: float = Field(default=0.2, gt=0.0, description="PPO clip range (used when mu > 1)")
    num_grpo_iterations: int = Field(default=1, gt=0, description="mu: gradient updates per generated batch")
    advantage_eps: float = Field(default=1e-4, gt=0.0, description="Stabilizer in advantage normalization")
    format_reward_weight: float = Field(default=1.0, ge=0.0)
    correct_reward_weight: float = Field(default=2.0, ge=0.0)

    # Training hyperparameters
    learning_rate: float = Field(default=1e-5, gt=0)
    # batch_size is the number of prompts per step (P). Generation/training batch is P * num_generations.
    batch_size: int = Field(default=1, gt=0)
    gradient_accumulation_steps: int = Field(default=4, gt=0)
    gradient_checkpointing: bool = Field(default=False)
    weight_decay: float = Field(default=0.1, ge=0)
    num_epochs: int = Field(default=1, gt=0)
    max_steps: int = Field(default=-1)  # -1 means use num_epochs
    optim: str = Field(default="adamw_torch")
    warmup_steps: int = Field(default=10, ge=0)
    max_grad_norm: float = Field(default=0.1, gt=0)

    # Logging settings
    log_level: str = Field(default="INFO")
    use_wandb: bool = Field(default=True)
    wandb_project: str = Field(default="gemma2-grpo")
    wandb_run_name: str = Field(default="tt-gemma2-grpo-gsm8k")
    wandb_tags: list[str] = Field(default_factory=lambda: ["gemma2", "grpo", "gsm8k", "lora"])
    wandb_watch_mode: str = Field(default="all")
    wandb_log_freq: int = Field(default=1000)
    model_to_wandb: bool = Field(default=False)
    steps_freq: int = Field(default=1)
    epoch_freq: int = Field(default=1)
    val_steps_freq: int = Field(default=50)
    print_examples: bool = Field(default=True, description="Log a sample completion each logging step")

    # Checkpoint settings
    resume_from_checkpoint: bool = Field(default=False)
    resume_option: str = Field(default="last")  # [last, best, path]
    checkpoint_path: str = Field(default="")
    checkpoint_metric: str = Field(default="train/reward_mean")
    checkpoint_metric_mode: str = Field(default="max")  # [min, max]
    keep_last_n: int = Field(default=3, ge=0)
    keep_best_n: int = Field(default=1, ge=0)
    save_strategy: str = Field(default="step")
    save_steps: int = Field(default=50)
    project_dir: str = Field(default="blacksmith/experiments/torch/gemma2/grpo")
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

    # LoRA setup (blog uses r == alpha == 64, no dropout, all projections)
    lora_r: int = Field(default=64, gt=0)
    lora_alpha: int = Field(default=64, gt=0)
    lora_target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    lora_task_type: str = Field(default="CAUSAL_LM")
    lora_dropout: float = Field(default=0.0, ge=0, le=1)

    # Adapter setup (used when training_model_type == "adapters")
    adapter_bottleneck_dim: int = Field(default=32, ge=0)
    adapter_non_linearity: str = Field(default="torch.nn.GELU")
    adapter_layers: list[int] = Field(default_factory=lambda: [])

    # Other settings
    framework: str = Field(default="pytorch")
    use_tt: bool = Field(default=True)
    do_validation: bool = Field(default=False)
    val_max_batches: int = Field(default=8, gt=0, description="Batches used per GRPO validation pass")

    # Testing utils (used to limit training duration during CI runs).
    test_config: Optional[TestConfig] = Field(default=None)
