# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Workarounds shared by the torch experiments.

Same names as the tt-xla tree so scripts port over unchanged. tt-crank executes
eagerly, so the `sync` knobs below are accepted and ignored: there is no lazy
graph whose signature has to be kept stable and nothing to fence.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# Materialize AdamW state up front. Under tt-xla this kept the fused fwd+bwd+optimizer graph's
# input signature stable across step 0; under tt-crank it is a numerical no-op (Adam's own lazy
# init also starts the moments at zero) kept so ported scripts run unchanged. `sync` is ignored.
def materialize_adamw_state(optimizer: torch.optim.Optimizer, sync: bool = True) -> None:
    for group in optimizer.param_groups:
        for p in group["params"]:
            if not p.requires_grad:
                continue
            state = optimizer.state[p]
            state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
            state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)


# Pre-seed zero .grad tensors (never None) so the first micro-batch of a gradient-accumulation
# window accumulates (grad += g) like the rest. Harmless under eager execution. `sync` is ignored.
def materialize_grads(optimizer: torch.optim.Optimizer, sync: bool = True) -> None:
    for group in optimizer.param_groups:
        for p in group["params"]:
            if not p.requires_grad:
                continue
            p.grad = torch.zeros_like(p, memory_format=torch.preserve_format)


# After restoring an optimizer from a CPU checkpoint, move its state (crucially `step`) back onto
# the parameter device so AdamW does not mix CPU state with device params. The tt-xla version also
# forced `capturable=True` to keep the fused step graph stable; eager tt-crank has no such graph.
def restore_capturable_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    params = [p for group in optimizer.param_groups for p in group["params"]]
    if not params or params[0].device.type == "cpu":
        return
    device = params[0].device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor) and v.device != device:
                state[k] = v.to(device)


# Smallest probability fed to log(). bfloat16's smallest normal is ~1.2e-38, so
# 1e-9 is far inside range while still bounding log() at about -20.7.
_LOG_EPS = 1e-9


# Custom cross-entropy over one-hot targets, originally because of
# https://github.com/tenstorrent/tt-xla/issues/1993; still needed so the whole loss (and its
# backward) lowers into the compiled tt graph instead of falling back to eager.
def cross_entropy_loss(shift_logits, expected_output, labels_mask):
    # NOTE: `log(softmax(x))` rather than `log_softmax(x)` because tt-crank does not lower
    # `aten._log_softmax` (it lowers `aten._softmax`). The clamp is what keeps that decomposition
    # safe: softmax alone is computed stably on device, but feeding an underflowed-to-zero
    # probability into log() would produce -inf and poison the reduction. Fold this back into
    # `F.log_softmax` once tt-crank lowers it -- the fused op is both more accurate and one graph
    # node cheaper.
    probs = F.softmax(shift_logits, dim=-1)  # [batch, seq_len, vocab_size]
    log_probs = torch.log(torch.clamp(probs, min=_LOG_EPS))
    # Cross entropy: -sum(target * log_prob) over vocab dimension.
    ce_loss = -(expected_output * log_probs).sum(dim=-1, keepdim=True)  # [batch, seq_len, 1]

    # Apply mask to ignore padding tokens.
    labels_mask = labels_mask.unsqueeze(-1).to(ce_loss.dtype)  # [batch, seq_len, 1]
    ce_loss = ce_loss * labels_mask

    # Compute mean over ALL valid tokens (not per-sample average).
    # Sum over seq_len dimension first.
    ce_loss_summed = ce_loss.sum(dim=1, keepdim=True)  # [batch, 1, 1]
    num_valid_per_sample = labels_mask.sum(dim=1, keepdim=True)  # [batch, 1, 1]

    # Then sum over batch dimension.
    total_loss = ce_loss_summed.sum(dim=0, keepdim=True)  # [1, 1, 1]
    num_valid_total = num_valid_per_sample.sum(dim=0, keepdim=True)  # [1, 1, 1]

    # Divide total loss by total valid tokens (not average of averages).
    num_valid_total = torch.clamp(num_valid_total, min=1.0)  # Avoid division by zero.
    loss = total_loss / num_valid_total  # [1, 1, 1]
    return loss


# Used in conjunction with cross_entropy_loss.
def transform_labels(labels, ignored_index, vocab_size):
    labels_mask = labels != ignored_index
    labels = torch.where(labels_mask, labels, 0)
    expected_output = F.one_hot(labels, num_classes=vocab_size)

    return expected_output, labels_mask


class TTLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, device=None, dtype=None):
        super().__init__()
        # Handle cases where `normalized_shape` is an int.
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=device, dtype=dtype))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        # Determine which dimensions to compute the mean and variance over.
        dims = tuple(range(-len(self.normalized_shape), 0))

        # Compute mean and variance.
        mean = x.mean(dim=dims, keepdim=True)
        # We use `unbiased=False` to match PyTorch's native `nn.LayerNorm` implementation.
        var = x.var(dim=dims, unbiased=False, keepdim=True)

        # Normalize.
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Apply learnable affine parameters if specified.
        if self.elementwise_affine:
            x_norm = x_norm * self.weight + self.bias

        return x_norm


def replace_layernorm(module):
    """
    Recursively replaces all nn.LayerNorm modules in a PyTorch model
    with CustomLayerNorm, preserving their weights and biases.
    """
    for name, child in module.named_children():
        if isinstance(child, torch.nn.LayerNorm):
            # 1. Initialize the custom layer with the same configuration.
            custom_ln = TTLayerNorm(
                normalized_shape=child.normalized_shape,
                eps=child.eps,
                elementwise_affine=child.elementwise_affine,
                device=child.weight.device if child.weight is not None else None,
                dtype=child.weight.dtype if child.weight is not None else None,
            )

            # 2. Copy the learned parameters if `elementwise_affine` is True.
            if child.elementwise_affine:
                with torch.no_grad():
                    custom_ln.weight.copy_(child.weight)
                    custom_ln.bias.copy_(child.bias)
                    custom_ln.weight.requires_grad = False
                    custom_ln.bias.requires_grad = False

            # 3. Replace the original layer with the custom one.
            setattr(module, name, custom_ln)
        else:
            # Recursively apply to child modules (e.g., inside `nn.Sequential`).
            replace_layernorm(child)

    return module
