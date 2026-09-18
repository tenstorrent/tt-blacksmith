# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

from pydantic import Field

from blacksmith.experiments.torch.llama.configs import TrainingConfig


class CrankTrainingConfig(TrainingConfig):
    """Llama fine-tuning through tt-crank, the `torch.compile` "tt" backend on tt-mlir.

    Extends the Llama config with the knobs the tt-crank experiments expose. The inherited
    `experimental_weight_dtype`, `enable_const_eval` and `enable_trace` are forwarded to the
    compiler as options.
    """

    # AdamW hyperparameters (`learning_rate` and `weight_decay` are inherited).
    adam_beta1: float = Field(default=0.9, ge=0, lt=1)
    adam_beta2: float = Field(default=0.999, ge=0, lt=1)
    adam_eps: float = Field(default=1e-8, gt=0)

    # Run length. `max_steps` caps the optimizer steps across epochs (None runs `num_epochs`
    # in full).
    max_steps: Optional[int] = Field(default=None, gt=0)

    # tt-crank compile options.
    opt_level: int = Field(default=1, ge=0)

    # tt-mlir perf metrics: exact FLOPs per compiled graph, reported as HFU next to MFU.
    # `perf_metrics_file` is a base name; tt-mlir writes `<perf_metrics_file>.json`.
    perf_metrics_enabled: bool = Field(default=True)
    perf_metrics_file: str = Field(default="perf_metrics")

    # IR dump (TTIR + TTNN) of every graph compiled during the first step, written to
    # `<artifacts_dir>/<artifacts_name>_<UTC timestamp>/`. None disables the dump.
    artifacts_name: Optional[str] = Field(default=None)
    artifacts_dir: str = Field(default=".data/artifacts")

    # Defaults that differ from the tt-xla experiment.
    project_dir: str = Field(default="blacksmith/experiments/torch/llama/crank/lora")
    enable_const_eval: bool = Field(default=False)
