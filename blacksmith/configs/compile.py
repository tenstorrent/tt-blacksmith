# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

from pydantic import BaseModel, Field


class CompileConfig(BaseModel):
    """
    tt-crank compile options, passed to every `torch.compile(..., backend="tt")` call
    through `DeviceManager.compile_options()`. Nested as `TrainerConfig.compile` so a
    training that does not go through the compiler simply leaves the block out.
    """

    optimization_level: int = Field(default=1, ge=0, le=2)
    enable_const_eval: bool = Field(default=False)
    fp32_dest_acc_en: bool = Field(default=True)
    math_fidelity: str = Field(default="HiFi4")  # [LoFi, HiFi2, HiFi3, HiFi4]
    enable_trace: bool = Field(default=False)
    # Compiler-wide weight dtype, "bfp_bf8" | "bfp_bf4" | "bf16" (no override).
    experimental_weight_dtype: Optional[str] = Field(default=None)
