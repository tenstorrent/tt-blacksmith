# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch


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


def model_memory_size(model, input_dtype=torch.float32):
    total_params = 0
    total_grads = 0
    for param in model.parameters():
        param_size = param.numel()
        total_params += param_size

        if param.requires_grad:
            total_grads += param_size

    # Calculate buffer size (non-parameters that require memory)
    total_buffers = sum(buf.numel() for buf in model.buffers())

    # Size in bytes = (Number of elements) * (Size of each element in bytes)
    # We assume parameters and gradients are stored in the same type as input dtype
    element_size = torch.tensor(0, dtype=input_dtype).element_size()
    total_memory_bytes = (total_params + total_grads + total_buffers) * element_size

    # Convert bytes to gigabytes
    total_memory_gb = total_memory_bytes / 1e9

    print(f"Input dtype: {input_dtype}")
    print(f"Model size: {total_memory_gb:.2f} GB")
    print(f"Parameters: {total_params} | Gradients: {total_grads} | Buffers: {total_buffers}")

    return total_memory_gb


def log_mem(stage):
    allocated = torch.cuda.memory_allocated() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[{stage}] Allocated: {allocated:.2f} GB | Peak: {peak:.2f} GB")


def show_examples(examples, tokenizer, config, logger):

    for i, example in enumerate(examples):
        logger.info(f"\nExample {i+1} (from batch {example['batch_num']}):")

        # NOTE: Move example tensors to CPU, because tokenizer does not work with tensors on TT device
        input_ids = example["input_ids"].to("cpu")
        expected = example["expected"].to("cpu")
        predicted = example["predicted"].to("cpu")

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
