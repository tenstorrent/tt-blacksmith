# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional, Tuple

from pydantic import Field

from blacksmith.tools.templates.configs import TrainingConfig as BaseTrainingConfig
from blacksmith.tools.test_config import TestConfig


class TrainingConfig(BaseTrainingConfig):
    # Dataset settings
    dataset_id: str = Field(default="sst2")

    # Model settings
    model_name: str = Field(default="openai/gpt-oss-20b")

    # Training hyperparameters
    training_model_type: str = Field(default="lora")
    batch_size: int = Field(default=2, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    ignored_index: int = Field(default=-100)
    clip_grad_norm: bool = Field(default=True)

    # Logging settings
    wandb_project: str = Field(default="gpt-oss-finetuning")
    wandb_run_name: str = Field(default="tt-gpt-oss-test")

    # Checkpoint settings
    project_dir: str = Field(default="blacksmith/experiments/torch/gpt_oss")

    # LoRA setup
    lora_r: int = Field(default=16, ge=0)
    lora_alpha: int = Field(default=32, gt=0)
    lora_target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    lora_task_type: str = Field(default="CAUSAL_LM")

    # Device settings
    mesh_shape: Optional[list[int]] = Field(default=None)
    mesh_axis_names: Optional[list[str]] = Field(default=None)

    input_sharding_dim: Optional[str] = Field(default=None)
    # Model sharding patterns (regex pattern based - matches module names, shards .weight).
    model_sharding_patterns: Optional[List[Tuple[str, Tuple[Optional[str], ...]]]] = Field(default=None)
    # Parameter sharding patterns (regex pattern based - matches parameter names directly).
    param_sharding_patterns: Optional[List[Tuple[str, Tuple[Optional[str], ...]]]] = Field(default=None)

    # Other settings
    output_dir: str = Field(default="experiments/results/gpt_oss_20b")
    logging_steps: int = Field(default=10, gt=0)
    do_train: bool = Field(default=True)
    print_examples: bool = Field(default=False)
    test_config: Optional[TestConfig] = Field(default=None)
