# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional

import numpy as np

from blacksmith.tools.logging_manager import TrainingLogger

IGNORED_LABEL = -100


def show_predictions(
    collected: list[dict],
    tokenizer,
    *,
    num_tokens: int = 20,
    ignored_index: int = IGNORED_LABEL,
    training_logger: Optional[TrainingLogger] = None,
) -> None:
    """Print collected prediction examples (CPU-only, no forward pass).

    Generalised from the Qwen branch's ``_show_predictions``
    in ``train_steps.py``.

    Args:
        collected: List of dicts with keys ``input_ids``,
            ``labels``, ``predictions``, ``per_token_loss``
            (numpy arrays).
        tokenizer: HuggingFace tokenizer for decoding.
        num_tokens: Number of leading tokens to show per
            example.
        ignored_index: Label value treated as "don't care".
        training_logger: Optional :class:`TrainingLogger`;
            when *None* falls back to module-level logger.
    """
    log = training_logger.info if training_logger is not None else logging.getLogger(__name__).info

    for i, ex in enumerate(collected):
        input_ids = ex["input_ids"]
        labels = ex["labels"]
        predictions = ex["predictions"]
        per_token_loss = ex["per_token_loss"]

        shift_labels = labels[1:].astype(np.int32)
        target_ids = shift_labels[:num_tokens]
        pred_ids = predictions[:num_tokens]
        token_losses = per_token_loss[:num_tokens]

        tok_valid = target_ids != ignored_index
        valid_targets = target_ids[tok_valid]
        valid_preds = pred_ids[tok_valid]

        input_text = tokenizer.decode(
            input_ids.tolist(),
            skip_special_tokens=True,
        )[:200]
        target_text = tokenizer.decode(
            valid_targets.tolist(),
            skip_special_tokens=False,
        )
        pred_text = tokenizer.decode(
            valid_preds.tolist(),
            skip_special_tokens=False,
        )

        valid = shift_labels != ignored_index
        correct = int((predictions[valid] == shift_labels[valid]).sum())
        total = int(valid.sum())

        log(f"\n--- Example {i + 1} ---")
        log(f"  Input:        {input_text!r}")
        log(f"  Target IDs:   {target_ids.tolist()}")
        log(f"  Pred IDs:     {pred_ids.tolist()}")
        log(f"  Target text:  {target_text!r}")
        log(f"  Pred text:    {pred_text!r}")
        log(f"  Token losses: " f"{np.round(token_losses, 4).tolist()}")
        mean_loss = float(per_token_loss.mean())
        log(f"  Mean loss:    {mean_loss:.4f}")
        acc = correct / max(total, 1)
        log(f"  Accuracy:     {correct}/{total} = {acc:.3f}")
