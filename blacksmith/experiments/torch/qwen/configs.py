# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional, Tuple

from pydantic import Field

from blacksmith.tools.templates.configs import TrainingConfig as BaseTrainingConfig
from blacksmith.tools.test_config import TestConfig


class TrainingConfig(BaseTrainingConfig):
    # Dataset settings
    dataset_id: str = Field(default="text2sql")

    # Model settings
    model_name: str = Field(default="Qwen/Qwen2.5-0.5B")

    # Training hyperparameters
    training_model_type: str = Field(default="lora")  # [lora, adapters]
    print_examples: bool = Field(default=False)
    ignored_index: int = Field(default=-100)

    # Logging settings
    wandb_project: str = Field(default="qwen-finetuning")
    wandb_run_name: str = Field(default="tt-qwen-test")

    # Checkpoint settings
    project_dir: str = Field(default="blacksmith/experiments/torch/qwen")

    # LoRA setup
    lora_r: int = Field(default=4, ge=0)
    lora_alpha: int = Field(default=8, gt=0)
    lora_target_modules: list[str] = Field(default_factory=lambda: ["all-linear"])
    lora_task_type: str = Field(default="CAUSAL_LM")

    # Device settings
    mesh_shape: Optional[List[int]] = Field(default=None)  # Note that currently only 2D meshes are supported.
    mesh_axis_names: Optional[List[str]] = Field(default=None)  # e.g., ["data", "model"]
    input_sharding_dim: Optional[str] = Field(
        default=None
    )  # If defined, we will shard inputs along this mesh axis dimension.
    # Tensor parallelism sharding patterns (regex pattern based - matches module names).
    # Format: List of tuples (regex_pattern, sharding_spec_tuple).
    model_sharding_patterns: Optional[List[Tuple[str, Tuple[Optional[str], ...]]]] = Field(default=None)

    # Other settings
    test_config: Optional[TestConfig] = Field(default=None)
