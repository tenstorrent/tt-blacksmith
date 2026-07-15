# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
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
    model_name: str = Field(default="google/gemma-3-1b-it")
    ignored_index: int = Field(default=-100)

    # Training hyperparameters
    training_model_type: str = Field(default="lora")  # [lora, adapters]

    # Logging settings
    wandb_project: str = Field(default="gemma-finetuning")
    wandb_run_name: str = Field(default="tt-gemma-test")
    steps_freq: int = Field(default=10)
    val_steps_freq: int = Field(default=50)
    print_examples: bool = Field(default=True)

    # Checkpoint settings
    project_dir: str = Field(default="blacksmith/experiments/torch/gemma")

    # Device settings
    mesh_shape: Optional[list[int]] = Field(default=None)  # Use None for single device, [x,y] for 2D mesh.
    mesh_axis_names: Optional[list[str]] = Field(
        default=None
    )  # Use None for single device, ["data", "model"] for 2D mesh.
    # Model sharding patterns (regex pattern based - matches module names).
    # Format: List of tuples (regex_pattern, sharding_spec_tuple).
    model_sharding_patterns: Optional[List[Tuple[str, Tuple[Optional[str], ...]]]] = Field(default=None)
    input_sharding_dim: Optional[str] = Field(
        default=None
    )  # If defined, we will shard inputs along this mesh axis dimension.

    # LoRA setup
    lora_r: int = Field(default=4, gt=0)
    lora_alpha: int = Field(default=8, gt=0)
    lora_target_modules: list[str] = Field(default_factory=lambda: ["all-linear"])
    lora_task_type: str = Field(default="CAUSAL_LM")

    # Other settings
    test_config: Optional[TestConfig] = Field(default=None)
