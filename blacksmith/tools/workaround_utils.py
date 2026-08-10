# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import torch.nn.functional as F


def _sync_xla(params) -> None:
    """Flush pending XLA work; a no-op on any other backend.

    `torch_xla` is imported lazily (the same way `models/torch/wan2_2/device.py` does it)
    because a module-scope import makes every importer of this module require the XLA
    runtime -- including CheckpointManager, which imports
    `restore_capturable_optimizer_state`. The pure-torch tt-kurbla backend executes
    eagerly and has nothing to flush.
    """
    if not params or params[0].device.type != "xla":
        return
    import torch_xla

    torch_xla.sync(wait=True)


# Materialize AdamW state so the fused fwd+bwd+optimizer XLA graph compiles only once.
# AdamW lazily allocates step/exp_avg/exp_avg_sq on the first step, which changes the graph's
# input signature after step 0 and forces a second compilation. Pre-initializing to zero is a
# numerical no-op (Adam's own lazy init also starts moments at zero) and keeps the signature stable.
# Pass sync=False to leave the zero-fills pending so a later sync fuses them with other
# pre-seeds (grads, loss) into a single one-time materialization graph.
def materialize_adamw_state(optimizer: torch.optim.Optimizer, sync: bool = True) -> None:
    for group in optimizer.param_groups:
        for p in group["params"]:
            if not p.requires_grad:
                continue
            state = optimizer.state[p]
            state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
            state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
    if sync:
        _sync_xla([p for group in optimizer.param_groups for p in group["params"]])


# Pre-seed zero .grad tensors (never None) so the first micro-batch of a gradient-
# accumulation window accumulates (grad += g) like the rest instead of assigning fresh
# grads, which would compile a second, distinct fwd+bwd graph. Paired with optimizer_step
# re-zeroing grads in place (set_to_none=False) so they stay tensors across windows.
def materialize_grads(optimizer: torch.optim.Optimizer, sync: bool = True) -> None:
    for group in optimizer.param_groups:
        for p in group["params"]:
            if not p.requires_grad:
                continue
            p.grad = torch.zeros_like(p, memory_format=torch.preserve_format)
    if sync:
        _sync_xla([p for group in optimizer.param_groups for p in group["params"]])


# After restoring an optimizer from a CPU checkpoint, re-enable capturable AdamW and move its
# state (crucially `step`) back onto the parameter device. Old checkpoints store capturable=False
# and CPU state; load_state_dict restores both, which makes AdamW compute the step-dependent bias
# correction host-side and bake it into the fused step graph as a constant (recompile as the step
# advances), and mismatches CPU state against device params. No-op off XLA.
def restore_capturable_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    params = [p for group in optimizer.param_groups for p in group["params"]]
    if not params or params[0].device.type != "xla":
        return
    device = params[0].device
    for group in optimizer.param_groups:
        group["capturable"] = True
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor) and v.device != device:
                state[k] = v.to(device)
    _sync_xla(params)


# Custom cross-entropy loss because of https://github.com/tenstorrent/tt-xla/issues/1993.
def cross_entropy_loss(shift_logits, expected_output, labels_mask):
    log_probs = F.log_softmax(shift_logits, dim=-1)  # [batch, seq_len, vocab_size]
    # Cross entropy: -sum(target * log_prob) over vocab dimension.
    ce_loss = -(expected_output * log_probs).sum(dim=-1, keepdim=True)  # [batch, seq_len, 1]

    # Apply mask to ignore padding tokens.
    labels_mask = labels_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
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
