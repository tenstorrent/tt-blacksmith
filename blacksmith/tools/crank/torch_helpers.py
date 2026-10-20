# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch.distributed.tensor import DTensor

from blacksmith.configs.compile import CompileConfig


def to_host(obj):
    """Move tensors in a (nested) container to CPU, gathering DTensor shards."""
    if isinstance(obj, DTensor):
        return obj.full_tensor().cpu()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: to_host(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_host(v) for v in obj)
    return obj


def loss_to_float(loss: torch.Tensor | float) -> float:
    """Pull a loss to host as a float, reducing a `Partial` DTensor first."""
    if not isinstance(loss, torch.Tensor):
        return float(loss)
    if isinstance(loss, DTensor):
        loss = loss.full_tensor()
    return float(loss.detach().cpu())


def collate_fn_for_causal_lm(batch):
    """
    Collate function that pre-shifts labels for Causal LM.
    Shifts labels to exclude first token.
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    shifted_labels = labels[:, 1:].contiguous()

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": shifted_labels}


def tt_compile_options(config) -> dict:
    """tt-crank options for `torch.compile(fn, backend="tt", options=...)`, from `config.compile`."""
    from tt_crank.torch._compile import BfpDtype, CompileOption, MathFidelity

    compile_config = config.compile or CompileConfig()
    options = {
        CompileOption.FP32_DEST_ACC_EN: compile_config.fp32_dest_acc_en,
        CompileOption.MATH_FIDELITY: getattr(MathFidelity, compile_config.math_fidelity),
        CompileOption.MATH_APPROX_MODE: compile_config.math_approx_mode,
        CompileOption.OPT_LEVEL: compile_config.optimization_level,
        CompileOption.ENABLE_CONST_EVAL: compile_config.enable_const_eval,
        CompileOption.ENABLE_TRACE: compile_config.enable_trace,
    }
    weight_dtype = compile_config.experimental_weight_dtype
    if weight_dtype and weight_dtype != "bf16":
        bfp = {"bfp_bf8": "BfpBf8", "bfp_bf4": "BfpBf4"}.get(weight_dtype.lower(), weight_dtype)
        options[CompileOption.EXPERIMENTAL_WEIGHT_DTYPE] = getattr(BfpDtype, bfp)
    return options
