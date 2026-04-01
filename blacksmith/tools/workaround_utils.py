# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn.functional as F
import torch.nn as nn


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

# Custom LayerNorm for TT-Forge. Necessary because of Albert experiment getting inf loss.
class TTLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, device=None, dtype=None):
        super().__init__()
        # Handle cases where normalized_shape is an int
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=device, dtype=dtype))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape, device=device, dtype=dtype))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # Determine which dimensions to compute the mean and variance over
        dims = tuple(range(-len(self.normalized_shape), 0))
        
        # Compute mean and variance
        mean = x.mean(dim=dims, keepdim=True)
        # We use unbiased=False to match PyTorch's native nn.LayerNorm implementation
        var = x.var(dim=dims, unbiased=False, keepdim=True)
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply learnable affine parameters if specified
        if self.elementwise_affine:
            x_norm = x_norm * self.weight + self.bias
            
        return x_norm

if __name__ == "__main__":
    for _ in range(100):
        x = torch.randn(1, 100, 100)
        official_layer_norm = nn.LayerNorm(x.shape[-1], eps=1e-12, elementwise_affine=True)
        layer_norm = TTLayerNorm(x.shape[-1], eps=1e-12, elementwise_affine=True)
        official_output = official_layer_norm(x)
        output = layer_norm(x)

        assert torch.allclose(official_output, output, atol=1e-6, rtol=1e-6)
