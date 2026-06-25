# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from blacksmith.tools.test_config import TestConfig


class TrainerConfig(BaseModel):
    """
    Configuration for the :class:`~blacksmith.tools.trainer.trainer.Trainer`.

    Root of the trainer-config hierarchy. Holds the strategy-agnostic fields the
    generic training machinery consumes (the training loop, optimizer and device
    manager). Concrete per-strategy configs (e.g. ``LoraLLMConfig``) extend this
    with the fields their trainer needs.
    """

    # Dataset settings
    dataset_id: str = Field(default="dataset_id")

    # Model settings
    model_name: str = Field(default="model_name")
    dtype: str = Field(default="dtype")

    # Training hyperparameters
    learning_rate: float = Field(default=2e-5, gt=0)
    batch_size: int = Field(default=8, gt=0)
    num_epochs: int = Field(default=1, gt=0)
    optim: str = Field(default="adamw_torch")
    weight_decay: float = Field(default=0.0, ge=0)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    gradient_checkpointing: bool = Field(default=False)
    training_type: str = Field(default="lora")

    # Validation settings
    val_steps_freq: int = Field(default=25, ge=1)

    # Reproducibility settings
    framework: str = Field(default="pytorch")  # Dispatch key for ReproducibilityManager.
    seed: int = Field(default=23)
    deterministic: bool = Field(default=False)

    # Device / sharding settings
    use_tt: bool = Field(default=True)
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
