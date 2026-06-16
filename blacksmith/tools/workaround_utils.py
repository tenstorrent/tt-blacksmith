# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import torch.nn.functional as F


# Custom cross-entropy loss because of https://github.com/tenstorrent/tt-xla/issues/1993.
def cross_entropy_loss(shift_logits, expected_output, labels_mask):
    # Flatten leading (batch/seq) dims so every vocab-sized tensor is 2D
    # [tokens, vocab]. On TT this keeps the loss (and its backward) on tensors
    # that tile normally; a singleton middle dim (e.g. [seq, 1, vocab]) would
    # tile-pad 32x and blow up DRAM. Mathematically identical: total CE over all
    # valid tokens divided by the valid-token count.
    vocab_size = shift_logits.shape[-1]
    logits = shift_logits.reshape(-1, vocab_size)  # [tokens, vocab]
    targets = expected_output.reshape(-1, vocab_size)  # [tokens, vocab]
    mask = labels_mask.reshape(-1, 1).to(logits.dtype)  # [tokens, 1]

    log_probs = F.log_softmax(logits, dim=-1)
    # Cross entropy: -sum(target * log_prob) over vocab dimension.
    ce_loss = -(targets * log_probs).sum(dim=-1, keepdim=True)  # [tokens, 1]
    ce_loss = ce_loss * mask

    total_loss = ce_loss.sum()
    num_valid_total = torch.clamp(mask.sum(), min=1.0)  # Avoid division by zero.
    loss = total_loss / num_valid_total
    return loss


# Used in conjunction with cross_entropy_loss.
def transform_labels(labels, ignored_index, vocab_size):
    labels_mask = labels != ignored_index
    labels = torch.where(labels_mask, labels, 0)
    expected_output = F.one_hot(labels, num_classes=vocab_size).to(torch.bfloat16)

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
