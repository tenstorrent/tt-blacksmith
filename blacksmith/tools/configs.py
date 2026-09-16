# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Base training config shared by every tt-crank experiment.

Experiment-specific configs subclass `TrainingConfig` and add their own fields;
see `blacksmith/experiments/torch/llama/configs.py`.
"""
from typing import Optional

import torch
from pydantic import BaseModel, ConfigDict, Field


class TestConfig(BaseModel):
    """Limits applied when an experiment runs under pytest."""

    model_config = ConfigDict(extra="forbid")

    max_steps_per_epoch: Optional[int] = Field(
        default=None,
        description="Maximum number of batches to process per epoch.",
    )


class MeshConfig(BaseModel):
    """Multi-chip layout, expressed the way tt-crank models it.

    tt-crank exposes the chips as one logical `tt` device backed by a
    `MeshDevice`, and parallelism is expressed with torch DTensor over a
    `DeviceMesh` (`torch.tt.init_device_mesh`). That replaces tt-xla's SPMD
    `xs.Mesh` + `mark_sharding` annotations.

    `shape` is `[rows, cols]` (or `[n]` for a 1-D mesh); `axis_names` names each
    axis so `data_axis` / `tensor_axis` below can refer to them.
    """

    model_config = ConfigDict(extra="forbid")

    shape: list[int] = Field(min_length=1, max_length=2)
    axis_names: list[str] = Field(min_length=1, max_length=2)

    # Batch is sharded along this axis (data parallelism). None disables DP.
    data_axis: Optional[str] = Field(default=None)
    # Megatron column/row sharding is applied along this axis. None disables TP.
    tensor_axis: Optional[str] = Field(default=None)

    def model_post_init(self, _context) -> None:
        if len(self.shape) != len(self.axis_names):
            raise ValueError(f"mesh shape {self.shape} and axis_names {self.axis_names} must have the same length")
        for field, axis in (("data_axis", self.data_axis), ("tensor_axis", self.tensor_axis)):
            if axis is not None and axis not in self.axis_names:
                raise ValueError(f"mesh.{field}={axis!r} is not one of axis_names {self.axis_names}")

    def axis_index(self, name: str) -> int:
        return self.axis_names.index(name)

    def axis_size(self, name: str) -> int:
        return self.shape[self.axis_index(name)]


class TrainingConfig(BaseModel):
    # Dataset settings
    dataset_id: str = Field(default="sst2")

    # Model settings
    model_name: str = Field(default="path/to/model")
    max_length: int = Field(default=128, gt=0)
    dtype: str = Field(default="torch.bfloat16")

    # Training hyperparameters
    learning_rate: float = Field(default=2e-5, gt=0)
    batch_size: int = Field(default=32, gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    num_epochs: int = Field(default=1, gt=0)
    ignored_index: int = Field(default=-100)

    # Logging settings
    log_level: str = Field(default="INFO")
    use_wandb: bool = Field(default=False)
    wandb_project: str = Field(default="model-finetuning")
    wandb_run_name: str = Field(default="tt-model-test")
    wandb_tags: list[str] = Field(default_factory=lambda: ["test"])
    steps_freq: int = Field(default=25)
    val_steps_freq: int = Field(default=25)
    measure_e2e_time: bool = Field(default=False)

    # Checkpoint settings
    save_strategy: str = Field(default="none")  # [none, step, epoch]
    output_dir: str = Field(default="results")
    keep_last_n: int = Field(default=2, ge=0)

    # Reproducibility settings
    seed: int = Field(default=23)

    # Device settings. None => single chip.
    mesh: Optional[MeshConfig] = Field(default=None)

    # tt-crank compile options; forwarded verbatim to
    # torch.compile(backend="tt", options=...). See tt_crank.torch CompileOption.
    optimization_level: int = Field(default=0, ge=0, le=2)
    enable_const_eval: bool = Field(default=True)
    enable_trace: bool = Field(default=False)
    math_fidelity: str = Field(default="HiFi4")  # [LoFi, HiFi2, HiFi3, HiFi4]
    fp32_dest_acc_en: bool = Field(default=True)
    experimental_weight_dtype: Optional[str] = Field(default=None)  # [BfpBf8, BfpBf4]

    # Other settings
    use_tt: bool = Field(default=True)
    print_examples: bool = Field(default=False)
    test_config: Optional[TestConfig] = Field(default=None)

    def torch_dtype(self) -> torch.dtype:
        dtypes = {
            "torch.bfloat16": torch.bfloat16,
            "torch.float32": torch.float32,
        }
        try:
            return dtypes[self.dtype]
        except KeyError as e:
            raise ValueError(f"Unsupported dtype {self.dtype!r}; expected one of {sorted(dtypes)}") from e
