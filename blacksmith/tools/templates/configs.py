# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Base training config shared by every experiment.

Same surface as the tt-xla tree (`blacksmith_xla/tools/templates/configs.py`)
so an experiment config ports over unchanged, plus the tt-crank device and
compile knobs (`mesh`, `math_fidelity`, `fp32_dest_acc_en`, ...).

Fields that only meant something under tt-xla are kept so old YAMLs still
validate; they are marked below and ignored by the tt-crank tooling.
"""
from enum import Enum
from typing import Optional

import torch
from pydantic import BaseModel, ConfigDict, Field

from blacksmith.tools.test_config import TestConfig


class Framework(Enum):
    PYTORCH = "pytorch"
    JAX = "jax"
    EASYDEL = "easydel"


class MeshConfig(BaseModel):
    """Multi-chip layout, expressed the way tt-crank models it.

    tt-crank exposes the chips as one logical `tt` device backed by a
    `MeshDevice`, and parallelism is expressed with torch DTensor over a
    `DeviceMesh` (`torch.tt.init_device_mesh`). This replaces tt-xla's
    `mesh_shape` / `mesh_axis_names` / `input_sharding_dim` /
    `model_sharding_patterns` quartet: the sharding rules live in
    `DeviceManager`, the YAML only names the axes.

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
    dataset_id: str = Field(default="path/to/dataset")

    # Model settings
    model_name: str = Field(default="path/to/model")
    max_length: int = Field(default=128, gt=0)
    dtype: str = Field(default="torch.bfloat16")

    # Mixed precision settings.
    # `weight_dtype_overrides` is tt-xla only (per-tensor JSON overrides via tt_torch); tt-crank has
    # no per-tensor override, `get_model` raises if it is set under use_tt.
    weight_dtype_overrides: Optional[str] = Field(default=None)  # JSON path (relative to the yaml if not absolute)
    # Compiler-level default weight dtype. tt-crank: BfpDtype name ("BfpBf8" | "BfpBf4").
    experimental_weight_dtype: Optional[str] = Field(default=None)

    # Training hyperparameters
    learning_rate: float = Field(default=2e-5, gt=0)
    batch_size: int = Field(default=32, gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    gradient_checkpointing: bool = Field(default=False)
    num_epochs: int = Field(default=1, gt=0)
    optim: str = Field(default="adamw_torch")

    # Logging settings
    log_level: str = Field(default="INFO")
    use_wandb: bool = Field(default=True)
    wandb_project: str = Field(default="model-finetuning")
    wandb_run_name: str = Field(default="tt-model-test")
    wandb_tags: list[str] = Field(default_factory=lambda: ["test"])
    wandb_watch_mode: str = Field(default="all")
    wandb_log_freq: int = Field(default=1000)
    model_to_wandb: bool = Field(default=False)
    steps_freq: int = Field(default=25)
    val_steps_freq: int = Field(default=25)
    epoch_freq: int = Field(default=1)
    measure_e2e_time: bool = Field(default=False)

    # Checkpoint settings
    resume_from_checkpoint: bool = Field(default=False)
    resume_option: str = Field(default="last")  # [last, best, path]
    checkpoint_path: str = Field(default="")  # path to checkpoint if resume_option is "path"
    checkpoint_metric: str = Field(default="eval/loss")
    checkpoint_metric_mode: str = Field(default="min")  # [min, max]
    keep_last_n: int = Field(default=3, ge=0)
    keep_best_n: int = Field(default=3, ge=0)
    save_strategy: str = Field(default="epoch")  # [none, step, epoch]
    project_dir: str = Field(default="blacksmith/experiments/torch/model")
    save_optim: bool = Field(default=False)
    storage_backend: str = Field(default="local")
    sync_to_storage: bool = Field(default=False)
    load_from_storage: bool = Field(default=False)
    remote_path: str = Field(default="")

    # Reproducibility settings
    seed: int = Field(default=23)
    deterministic: bool = Field(default=False)

    # Embedding settings
    unfreeze_embeddings: bool = Field(default=False)

    # Prompt formatting for instruction datasets.
    prompt_format: str = Field(default="default")  # [default, chat]
    chat_system_prompt: Optional[str] = Field(default=None)  # only used when prompt_format="chat"

    # Device settings. None => single chip. See MeshConfig.
    mesh: Optional[MeshConfig] = Field(default=None)

    # tt-crank compile options; forwarded to torch.compile(backend="tt", options=...)
    # through `DeviceManager.compile_options()`. See tt_crank.torch CompileOption.
    optimization_level: int = Field(default=0, ge=0, le=2)
    enable_const_eval: bool = Field(default=True)
    enable_trace: bool = Field(default=False)
    trace_region_size: int = Field(default=1000000000, gt=0)  # tt-xla only: DRAM region size (bytes) for runtime trace
    math_fidelity: str = Field(default="HiFi4")  # [LoFi, HiFi2, HiFi3, HiFi4]
    fp32_dest_acc_en: bool = Field(default=True)

    # Other settings
    framework: str = Field(default="pytorch")
    use_tt: bool = Field(default=True)
    print_examples: bool = Field(default=False)
    test_config: Optional[TestConfig] = Field(default=None)

    def torch_dtype(self) -> torch.dtype:
        # Broader than TrainerConfig: some experiment YAMLs still use float16.
        dtypes = {
            "torch.bfloat16": torch.bfloat16,
            "torch.float32": torch.float32,
            "torch.float16": torch.float16,
        }
        try:
            return dtypes[self.dtype]
        except KeyError as e:
            raise ValueError(f"Unsupported dtype {self.dtype!r}; expected one of {sorted(dtypes)}") from e
