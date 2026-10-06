# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

from pydantic import Field, field_validator

from blacksmith.tools.trainer.configs import TrainerConfig

# Storage dtype of every matmul weight the compiler lowers. bf16 is the default, so it maps
# to "leave the option unset".
WEIGHT_DTYPE_OPTIONS = ("bfp_bf8", "bfp_bf4", "bf16")


class CrankTrainerConfig(TrainerConfig):
    # Model settings
    max_length: int = Field(gt=0)

    # LoRA setup; read only when `training_model_type` is "lora".
    lora_r: int = Field(default=4, gt=0)
    lora_alpha: int = Field(default=8, gt=0)
    lora_target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_task_type: str = Field(default="CAUSAL_LM")

    # Mixed precision: compiler-level storage dtype of every matmul weight.
    experimental_weight_dtype: Optional[str] = Field(default=None)  # "bfp_bf8" | "bfp_bf4" | "bf16"

    # AdamW hyperparameters (`learning_rate` and `weight_decay` are inherited). The script
    # always runs fused `torch.optim.AdamW`, so `optim` defaults to it and must name it if set.
    optim: str = Field(default="adamw_torch")
    adam_beta1: float = Field(default=0.9, ge=0, lt=1)
    adam_beta2: float = Field(default=0.999, ge=0, lt=1)
    adam_eps: float = Field(default=1e-8, gt=0)

    max_steps: Optional[int] = Field(default=None, gt=0)

    ignored_index: int = Field(default=-100)

    # tt-crank compile options.
    enable_trace: bool = Field(default=False)

    # tt-mlir perf metrics: exact FLOPs per compiled graph, reported as HFU next to MFU.
    # `perf_metrics_file` is a base name; tt-mlir writes `<perf_metrics_file>.json`.
    perf_metrics_enabled: bool = Field(default=True)
    perf_metrics_file: str = Field(default="perf_metrics")

    # IR dump (TTIR + TTNN) of every graph compiled during the first step, written to
    # `<artifacts_dir>/<artifacts_name>_<UTC timestamp>/`. None disables the dump.
    artifacts_name: Optional[str] = Field(default=None)
    artifacts_dir: str = Field(default=".data/artifacts")

    # Reporting.
    print_examples: bool = Field(default=False)
    measure_e2e_time: bool = Field(default=False)

    @field_validator("optim")
    @classmethod
    def check_optim(cls, value: str) -> str:
        if value != "adamw_torch":
            raise ValueError(f"tt-crank experiments only support optim='adamw_torch', got {value!r}")
        return value

    @field_validator("experimental_weight_dtype")
    @classmethod
    def check_weight_dtype(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value not in WEIGHT_DTYPE_OPTIONS:
            raise ValueError(f"Unsupported experimental_weight_dtype {value!r}; expected one of {WEIGHT_DTYPE_OPTIONS}")
        return value
