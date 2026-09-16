# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Loss, label and batch helpers shared by the tt-crank torch experiments."""
import torch
import torch.nn.functional as F


def collate_fn_for_causal_lm(batch):
    """Pre-shift the labels for causal LM, and drop unused columns.

    Position t's logit predicts token t+1, so the targets are `labels[:, 1:]`
    and the training step pairs them with `logits[:, :-1]`. Shifting here rather
    than in the step keeps the two halves of the shift next to their shapes:
    the batch that reaches the model already has `labels` one shorter than
    `input_ids`.
    """
    return {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": batch["labels"][:, 1:].contiguous(),
    }


def transform_labels(labels: torch.Tensor, ignored_index: int, vocab_size: int):
    """Turn `[B, S]` label ids into a one-hot target plus a validity mask.

    Carried over from the tt-xla experiments: the loss below consumes one-hot
    targets rather than indices, which keeps it to ops tt-crank lowers.
    """
    labels_mask = labels != ignored_index
    labels = torch.where(labels_mask, labels, 0)
    expected_output = F.one_hot(labels, num_classes=vocab_size)
    return expected_output, labels_mask


# Smallest probability fed to log(). bfloat16's smallest normal is ~1.2e-38, so
# 1e-9 is far inside range while still bounding log() at about -20.7.
_LOG_EPS = 1e-9


def cross_entropy_loss(shift_logits, expected_output, labels_mask):
    """Token-mean cross entropy over one-hot targets.

    Written out of softmax/log/mul/sum rather than `F.cross_entropy` so the whole
    loss (and its backward) lowers into the compiled tt graph.

    NOTE: `log(softmax(x))` rather than `log_softmax(x)` because tt-crank does
    not lower `aten._log_softmax` (it lowers `aten._softmax`). The clamp is what
    keeps that decomposition safe: softmax alone is computed stably on device,
    but feeding an underflowed-to-zero probability into log() would produce -inf
    and poison the reduction. Fold this back into `F.log_softmax` once tt-crank
    lowers it -- the fused op is both more accurate and one graph node cheaper.
    """
    probs = F.softmax(shift_logits, dim=-1)  # [B, S, V]
    log_probs = torch.log(torch.clamp(probs, min=_LOG_EPS))
    ce_loss = -(expected_output * log_probs).sum(dim=-1, keepdim=True)  # [B, S, 1]

    labels_mask = labels_mask.unsqueeze(-1).to(ce_loss.dtype)  # [B, S, 1]
    ce_loss = ce_loss * labels_mask

    # Mean over all valid tokens, not a mean of per-sample means.
    total_loss = ce_loss.sum(dim=1, keepdim=True).sum(dim=0, keepdim=True)  # [1, 1, 1]
    num_valid = labels_mask.sum(dim=1, keepdim=True).sum(dim=0, keepdim=True)  # [1, 1, 1]
    return total_loss / torch.clamp(num_valid, min=1.0)


def show_examples(examples, tokenizer, logger, max_examples: int = 5):
    """Log a few decoded prediction/target pairs."""
    for i, example in enumerate(examples[:max_examples]):
        prompt = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
        target_ids = [t for t in example["expected_output"] if t >= 0]
        logger.info(
            f"--- example {i} ---\n"
            f"prompt: {prompt}\n"
            f"target: {tokenizer.decode(target_ids, skip_special_tokens=True)}\n"
            f"predicted: {tokenizer.decode(example['prediction'], skip_special_tokens=True)}"
        )


def loss_to_float(loss: torch.Tensor) -> float:
    """Pull a loss value to host as a plain float.

    Under data parallelism the loss reduces over the sharded batch dim, so it
    comes back as a `Partial` DTensor -- each chip holds a piece of the sum.
    `full_tensor()` fires the all-reduce that makes it the real value;
    `.cpu()` alone would silently report one chip's partial.
    """
    from torch.distributed.tensor import DTensor

    if isinstance(loss, DTensor):
        loss = loss.full_tensor()
    return float(loss.detach().cpu())
