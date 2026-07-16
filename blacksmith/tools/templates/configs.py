# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from enum import Enum
from typing import Optional

import torch
from pydantic import BaseModel, Field


class Framework(Enum):
    PYTORCH = "pytorch"
    JAX = "jax"
    EASYDEL = "easydel"


class TrainingConfig(BaseModel):
    # Dataset settings
    dataset_id: str = Field(default="path/to/dataset")

    # Model settings
    model_name: str = Field(default="path/to/model")
    max_length: int = Field(default=128, gt=0)
    dtype: str = Field(default="torch.bfloat16")

    # Mixed precision settings (tt-xla backend only). See tt-xla/docs/src/mixed_precision.md.
    weight_dtype_overrides: Optional[str] = Field(default=None)  # JSON path (relative to the yaml if not absolute)
    experimental_weight_dtype: Optional[str] = Field(
        default=None
    )  # compiler-level default: "bfp_bf8" | "bfp_bf4" | "bf16"

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
    save_strategy: str = Field(default="epoch")
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

    # Other settings
    framework: str = Field(default="pytorch")
    use_tt: bool = Field(default=True)
    enable_trace: bool = Field(default=False)
    trace_region_size: int = Field(default=1000000000, gt=0)  # DRAM region size (bytes) for runtime trace
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> e7f3d8121 (Address comments)
    enable_const_eval: bool = Field(default=True)

=======
    enable_const_eval: bool = Field(default=False)
<<<<<<< HEAD
    
>>>>>>> 9cc221d45 (Disable const eval for llama 3.1 8b single chip)
=======

>>>>>>> cb15b8aae (pre-commit)
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
