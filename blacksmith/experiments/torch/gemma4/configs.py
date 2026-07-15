# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional, Tuple

from pydantic import Field

from blacksmith.tools.templates.configs import TrainingConfig as BaseTrainingConfig
from blacksmith.tools.test_config import TestConfig


class TrainingConfig(BaseTrainingConfig):
    # Dataset settings
    dataset_id: str = Field(default="wizardlm_evol")

    # Model settings
    model_name: str = Field(default="google/gemma-4-E2B-it")
    max_length: int = Field(default=1024, gt=0)

    # Training hyperparameters
    training_type: str = Field(default="lora")  # [lora, adapters]
    batch_size: int = Field(default=1, gt=0)
    gradient_accumulation_steps: int = Field(default=8, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    num_epochs: int = Field(default=3, gt=0)
    ignored_index: int = Field(default=-100)

    # Logging settings
    wandb_project: str = Field(default="gemma4-e2b-multichip-lora-quietbox")
    wandb_run_name: str = Field(default="tt-gemma4-e2b-wizardlm-evol")

    # Checkpoint settings
    project_dir: str = Field(default="blacksmith/experiments/torch/gemma4")

    # LoRA setup
    lora_r: int = Field(default=32, ge=0)
    lora_alpha: int = Field(default=64, gt=0)
    lora_target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    lora_task_type: str = Field(default="CAUSAL_LM")

    # Device settings
    mesh_shape: Optional[list[int]] = Field(default=None)  # Use None for single device, [x,y] for 2D mesh.
    mesh_axis_names: Optional[list[str]] = Field(
        default=None
    )  # Use None for single device, ["data", "model"] for 2D mesh.
    input_sharding_dim: Optional[str] = Field(
        default=None
    )  # If defined, we will shard inputs along this mesh axis dimension.

    # Model sharding patterns (regex pattern based - matches module names).
    # Format: List of tuples (regex_pattern, sharding_spec_tuple).
    model_sharding_patterns: Optional[List[Tuple[str, Tuple[Optional[str], ...]]]] = Field(default=None)

    # Other settings
    output_dir: str = Field(default="experiments/results/gemma4_e2b")
    logging_steps: int = Field(default=10, gt=0)
    do_train: bool = Field(default=True)
    print_examples: bool = Field(default=False)
    test_config: Optional[TestConfig] = Field(default=None)
