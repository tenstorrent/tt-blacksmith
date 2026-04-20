# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import torch.nn.functional as F


class RecomputeCheckpoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fn, *args):
        ctx.fn = fn
        ctx.tensor_mask = tuple(isinstance(a, torch.Tensor) for a in args)
        ctx.non_tensor_args = tuple(a for a, m in zip(args, ctx.tensor_mask) if not m)
        ctx.save_for_backward(*(a for a, m in zip(args, ctx.tensor_mask) if m))
        with torch.no_grad():
            return fn(*args)

    @staticmethod
    def backward(ctx, *grad_outputs):
        saved = [t.detach().requires_grad_(t.is_floating_point()) for t in ctx.saved_tensors]
        tensor_iter = iter(saved)
        non_tensor_iter = iter(ctx.non_tensor_args)
        reconstructed = tuple(next(tensor_iter) if m else next(non_tensor_iter) for m in ctx.tensor_mask)
        with torch.enable_grad():
            outputs = ctx.fn(*reconstructed)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        diff_inputs = [t for t in saved if t.requires_grad]
        diff_grads = torch.autograd.grad(outputs, diff_inputs, grad_outputs=grad_outputs)
        diff_iter = iter(diff_grads)
        saved_grads = [next(diff_iter) if t.requires_grad else None for t in saved]
        grad_iter = iter(saved_grads)
        arg_grads = tuple(next(grad_iter) if m else None for m in ctx.tensor_mask)
        return (None, *arg_grads)


def _wrap_with_recompute(module):
    original_forward = module.forward

    def recompute_forward(*args, **kwargs):
        keys = tuple(kwargs.keys())
        n_pos = len(args)

        def fn(*flat_args):
            pos = flat_args[:n_pos]
            kw = dict(zip(keys, flat_args[n_pos:]))
            return original_forward(*pos, **kw)

        return RecomputeCheckpoint.apply(fn, *args, *kwargs.values())

    module.forward = recompute_forward
    return module


def apply_recompute(module: nn.Module, targets: list[nn.Module]):
    for name, child in module.named_children():
        if type(child) in targets:
            setattr(module, name, _wrap_with_recompute(child))
        else:
            apply_recompute(child, targets)
    return module
