# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from transformers import StaticCache


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

        input_ids = input_ids.to("cpu")
        expected_output = expected_output.to("cpu")
        predictions = predictions.to("cpu")

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


def run_decode_example_from_batch(
    model,
    tokenizer,
    batch,
    ignored_index: int,
    device,
    logger,
    batch_index: int = 0,
    **decode_kwargs,
):
    """Recover (prompt_ids, target_ids) from a causal-LM batch and run decode.

    Assumes `labels` were pre-shifted by `collate_fn_for_causal_lm` (drops
    position 0) and that the dataset masked the prompt span with `ignored_index`.
    Then in the shifted labels the first non-ignored index is (prompt_len - 1),
    so prompt_end = first_valid + 1 = prompt_len recovers exactly the prompt
    range in `input_ids`. The non-ignored labels are the (shifted) response
    token IDs, i.e. the ground-truth completion.
    """
    labels_row = batch["labels"][batch_index].to("cpu")
    input_ids_row = batch["input_ids"][batch_index].to("cpu")
    valid_mask = labels_row != ignored_index
    if not valid_mask.any():
        raise ValueError("All labels are masked; cannot locate prompt boundary for decode example.")
    prompt_end = valid_mask.int().argmax().item() + 1
    prompt_ids = input_ids_row[:prompt_end]
    target_ids = labels_row[valid_mask]
    run_decode_example(
        model=model,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        target_ids=target_ids,
        device=device,
        logger=logger,
        **decode_kwargs,
    )


def run_decode_example(
    model,
    tokenizer,
    prompt_ids,
    device,
    logger,
    target_ids=None,
    max_prompt_length: int = 32,
    max_cache_length: int = 128,
    max_new_tokens: int = 64,
):
    """
    Run one autoregressive decode on a single prompt.
    The code is adapted from:
    https://github.com/tenstorrent/tt-xla/blob/main/examples/pytorch/llama.py
    """
    # Clamp generation length to whatever room the static cache has left after
    # the prefill window; otherwise cache_position would index past max_cache_len.
    room_in_cache = max_cache_length - max_prompt_length
    if room_in_cache < 1:
        raise ValueError(
            f"max_prompt_length ({max_prompt_length}) fills the entire cache "
            f"({max_cache_length}); no room to generate."
        )
    max_new_tokens = min(max_new_tokens, room_in_cache)

    logger.info("\n=== Decode Example ===")
    logger.info(f"Prompt: {tokenizer.decode(prompt_ids, skip_special_tokens=True)!r}")
    if target_ids is not None:
        logger.info(f"Target: {tokenizer.decode(target_ids, skip_special_tokens=True)!r}")
    input_args = construct_inputs_for_decode(
        prompt_ids=prompt_ids,
        pad_token_id=tokenizer.pad_token_id,
        model_config=model.model.config,
        max_prompt_length=max_prompt_length,
        max_cache_len=max_cache_length,
    )
    # Transfer inputs to device.
    for layer in input_args["past_key_values"].layers:
        layer.keys = layer.keys.to(device)
        layer.values = layer.values.to(device)
    input_args["input_ids"] = input_args["input_ids"].to(device)
    input_args["cache_position"] = input_args["cache_position"].to(device)
    input_args["attention_mask"] = input_args["attention_mask"].to(device)
    # Run generation loop.
    output_tokens = []
    with torch.no_grad():
        for step in range(max_new_tokens):
            if step == 0:
                logger.info("RUNNING PREFILL")
            output = model(**input_args)
            next_token_id = output.logits[:, -1].argmax(dim=-1).to("cpu")  # shape (1,)
            output_tokens.append(tokenizer.decode(next_token_id))
            if next_token_id.item() == tokenizer.eos_token_id:
                break
            # Advance inputs for next step.
            input_args["input_ids"] = next_token_id.unsqueeze(-1).to(device)
            host_cache_pos = input_args["cache_position"].to("cpu")
            next_pos = host_cache_pos[-1:] + 1
            input_args["cache_position"] = next_pos.to(device)
    logger.info(f"Generated: {''.join(output_tokens)!r}")


def construct_inputs_for_decode(
    prompt_ids,  # 1D LongTensor, len <= max_prompt_length
    pad_token_id: int,
    model_config,
    max_prompt_length: int,
    max_cache_len: int,
):
    """Build StaticCache inputs from a pre-tokenized prompt (batch size 1)."""
    prompt_ids = prompt_ids[-max_prompt_length:]
    L = prompt_ids.shape[0]
    input_ids = torch.full((1, max_prompt_length), pad_token_id, dtype=torch.long)
    input_ids[0, -L:] = prompt_ids
    # StaticCache must be built on CPU; transfer happens later.
    # See https://github.com/tenstorrent/tt-xla/issues/1645
    static_cache = StaticCache(
        config=model_config,
        max_batch_size=1,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    static_cache.early_initialization(
        batch_size=1,
        num_heads=model_config.num_key_value_heads,
        head_dim=head_dim,
        dtype=torch.bfloat16,
        device="cpu",
    )
    cache_position = torch.arange(0, max_prompt_length)
    # Mask is 0 over left-pad slots and 1 elsewhere; sized to max_cache_len.
    attention_mask = torch.ones((1, max_cache_len), dtype=torch.long)
    attention_mask[0, : max_prompt_length - L] = 0
    return {
        "input_ids": input_ids,
        "past_key_values": static_cache,
        "cache_position": cache_position,
        "use_cache": True,
        "attention_mask": attention_mask,
    }
