# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Tensor helpers for the tt-crank scripts.

`to_host` / `loss_to_float` are tt-crank specific (DTensor gathering). The rest are copies
of the helpers the tt-crank ports use from `blacksmith.tools.torch_helpers`; they are
duplicated here so the tt-crank tree does not import tt-xla code.
"""
import torch
from torch.distributed.tensor import DTensor


def to_host(obj):
    """Recursively move tensors in a (nested) container to CPU, gathering DTensor shards.

    tt-crank tensors cannot be serialized in place (`torch.save` tries to rebind a
    `tt` storage to a CPU tensor), and a sharded DTensor has to be gathered first
    so a checkpoint is always the unsharded, host-side view.
    """
    if isinstance(obj, DTensor):
        return obj.full_tensor().cpu()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: to_host(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_host(v) for v in obj)
    return obj


def loss_to_float(loss: torch.Tensor) -> float:
    """Pull a loss value to host as a plain float.

    Under data parallelism the loss reduces over the sharded batch dim, so it
    comes back as a `Partial` DTensor -- each chip holds a piece of the sum.
    `full_tensor()` fires the all-reduce that makes it the real value;
    `.item()` alone would silently report one chip's partial.
    """
    if isinstance(loss, DTensor):
        loss = loss.full_tensor()
    return float(loss.detach().cpu())


def print_trainable_params(model):
    """Helper function for lora models to check number of trainable parameters."""
    total_params = sum([p.numel() for p in model.parameters()])
    trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])

    print(
        f"""
    {total_params} total params,
    {trainable_params}" trainable params,
    {(100.0 * trainable_params / total_params):.2f}% of all params are trainable.
    """
    )


def show_examples(examples, tokenizer, config, logger):

    for i, example in enumerate(examples):
        logger.info(f"\nExample {i + 1} (from batch {example['batch_num']}):")

        input_ids = example["input_ids"]
        expected = example["expected"]
        predicted = example["predicted"]

        valid_mask = expected != config.ignored_index
        if not valid_mask.any():
            logger.info(f"  No valid tokens (all {config.ignored_index})")
            continue

        valid_targets = expected[valid_mask]
        valid_preds = predicted[valid_mask]

        show_len = min(10, len(valid_targets))
        target_tokens = valid_targets[:show_len].tolist()
        pred_tokens = valid_preds[:show_len].tolist()

        logger.info(f"Target IDs:  {target_tokens}")
        logger.info(f"Pred IDs:    {pred_tokens}")

        try:
            target_text = tokenizer.decode(target_tokens, skip_special_tokens=False)
            pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=False)
            input_text = tokenizer.decode(input_ids, skip_special_tokens=True)
            logger.info(f"Input text:  '{input_text}'")
            logger.info(f"Target text: '{target_text}'")
            logger.info(f"Pred text:   '{pred_text}'")
        except Exception as e:
            logger.info(f"  (Could not decode text: {e})")

        correct = (valid_targets == valid_preds).float().mean()
        logger.info(f"Accuracy: {correct.item():.3f} ({(valid_targets == valid_preds).sum()}/{len(valid_targets)})")


def collect_examples(
    batch_size, collected_examples, max_examples, input_ids, expected_output, predictions, num_val_batches
):
    if len(collected_examples) < max_examples:
        import random

        input_ids = to_host(input_ids)
        expected_output = to_host(expected_output)
        predictions = to_host(predictions)

        sample_indices = random.sample(range(batch_size), min(batch_size, max_examples - len(collected_examples)))
        for idx in sample_indices:
            collected_examples.append(
                {
                    "input_ids": input_ids[idx],
                    "expected": expected_output[idx],
                    "predicted": predictions[idx],
                    "batch_num": num_val_batches,
                }
            )
    return collected_examples


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
