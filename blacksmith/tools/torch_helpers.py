# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch_xla
from transformers import StaticCache


def accumulate_metric_tensors(
    totals: dict[str, torch.Tensor | None],
    metrics: dict[str, torch.Tensor],
) -> None:
    """Accumulate detached metric tensors into ``totals`` in place."""
    for key, value in metrics.items():
        if key not in totals:
            continue
        detached = value.detach()
        if totals[key] is None:
            totals[key] = detached
        else:
            totals[key] = totals[key] + detached


def average_metric_tensors(totals: dict[str, torch.Tensor | None], count: int) -> dict[str, float]:
    """Average accumulated metric tensors over ``count`` and return them as Python floats."""
    return {key: (total / count).item() for key, total in totals.items() if total is not None}


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


def _sample_next_tokens(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    sample_rng_on_cpu: bool = False,
) -> torch.Tensor:
    """Pick the next token id per row from last-position logits.

    ``logits`` is (B, V) on the compute device. Greedy (``argmax``) when
    ``temperature <= 0``; otherwise temperature (+ optional top-k) sampling via the
    Gumbel-max trick.     By default Uniform noise is ``rand_like`` on the logits
    device. When ``sample_rng_on_cpu`` is True (test/CI), noise is drawn on CPU
    then moved to device — TT device RNG is not reliably seedable. Seeding is
    handled by ``ReproducibilityManager``. Temperature scale, top-k, Gumbel
    transform and ``argmax`` always stay on device.
    """
    if not temperature or temperature <= 0:
        return logits.argmax(dim=-1)

    logits = logits.float() / temperature
    if top_k and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[:, -1, None]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)

    if sample_rng_on_cpu:
        u = torch.rand(logits.shape, dtype=torch.float32).clamp_(1e-10, 1.0 - 1e-10)
        u = u.to(device=logits.device, dtype=logits.dtype)
    else:
        u = torch.rand_like(logits).clamp_(1e-10, 1.0 - 1e-10)
    gumbel = -torch.log(-torch.log(u))
    return torch.argmax(logits + gumbel, dim=-1)


def generate_completions(
    model,
    model_config,
    prompt_input_ids: torch.Tensor,  # (B, max_prompt_length), left-padded
    prompt_attention_mask: torch.Tensor,  # (B, max_prompt_length), 0 over left-pad
    pad_token_id: int,
    eos_token_id: int,
    max_prompt_length: int,
    max_completion_length: int,
    device,
    temperature: float = 0.0,
    top_k: int = 0,
    dtype=torch.bfloat16,
    use_tt: bool = False,
    sample_rng_on_cpu: bool = False,
):
    """Batched autoregressive generation over left-padded prompts, fully on device.

    The whole decode loop stays on the compute device: the model forward, token
    sampling (Gumbel-max, see ``_sample_next_tokens``), EOS bookkeeping and the
    StaticCache all run on device. Unlike a typical decode helper, no per-step
    logits are copied to host and no host-side sampling/early-stop is performed;
    only the final id / validity tensors are moved to CPU once (for the host-side
    reward computation). When ``use_tt`` is True, ``torch_xla.sync`` is invoked
    once per step to keep the lazy graph bounded without a host transfer.

    When ``sample_rng_on_cpu`` is True (intended for tests/CI), Uniform noise for
    Gumbel sampling is drawn on CPU then moved to device (TT RNG is not reliably
    seedable). Seeding is left to ``ReproducibilityManager``.

    Returns ``(completion_ids, completion_valid)``:
      - ``completion_ids`` (B, max_completion_length) LongTensor on CPU; positions
        after a row emits EOS are ``pad_token_id``.
      - ``completion_valid`` (B, max_completion_length) bool mask on CPU; True for
        real generated tokens up to and including EOS, False for trailing pad.

    Left padding aligns the generation frontier across rows (last real token sits
    at index ``max_prompt_length - 1`` for every row), so a single ``cache_position``
    advances all rows together. Adapted from the single-example decode helper /
    https://github.com/tenstorrent/tt-xla/blob/main/examples/pytorch/llama.py
    """
    batch_size = prompt_input_ids.shape[0]
    max_cache_len = max_prompt_length + max_completion_length

    # StaticCache must be built on CPU, then moved to device.
    # See https://github.com/tenstorrent/tt-xla/issues/1645
    static_cache = StaticCache(
        config=model_config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=dtype,
    )
    # Prefer the explicit head_dim (Gemma 2 sets head_dim != hidden_size / num_heads);
    # fall back to the derived value for models that omit it.
    head_dim = getattr(model_config, "head_dim", None) or (model_config.hidden_size // model_config.num_attention_heads)
    static_cache.early_initialization(
        batch_size=batch_size,
        num_heads=model_config.num_key_value_heads,
        head_dim=head_dim,
        dtype=dtype,
        device="cpu",
    )
    for layer in static_cache.layers:
        layer.keys = layer.keys.to(device)
        layer.values = layer.values.to(device)
        layer.device = device
        if isinstance(getattr(layer, "cumulative_length", None), torch.Tensor):
            layer.cumulative_length = layer.cumulative_length.to(device)

    # Padding mask sized to the whole cache: 0 over prompt left-pad slots, 1 elsewhere
    # (future positions are handled by the causal mask inside the model).
    attention_mask = torch.ones((batch_size, max_cache_len), dtype=torch.long)
    attention_mask[:, :max_prompt_length] = prompt_attention_mask
    attention_mask = attention_mask.to(device)

    input_ids = prompt_input_ids.to(device)
    cache_position = torch.arange(0, max_prompt_length, device=device)

    # All decode state lives on device (no host reads inside the loop).
    eos_id = torch.tensor(eos_token_id, device=device)
    pad_id = torch.tensor(pad_token_id, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    token_steps = []
    valid_steps = []

    with torch.no_grad():
        for _ in range(max_completion_length):
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=static_cache,
                cache_position=cache_position,
                use_cache=True,
            )
            next_logits = output.logits[:, -1, :]
            # A row's token this step is valid iff it had not already finished.
            valid_this = ~finished
            next_tokens = _sample_next_tokens(
                next_logits, temperature, top_k, sample_rng_on_cpu=sample_rng_on_cpu
            )
            # Force already-finished rows to pad so generation stays batched.
            next_tokens = torch.where(finished, pad_id.expand_as(next_tokens), next_tokens)
            finished = finished | (next_tokens == eos_id)

            token_steps.append(next_tokens)
            valid_steps.append(valid_this)

            input_ids = next_tokens.unsqueeze(-1)
            cache_position = cache_position[-1:] + 1
            if use_tt:
                torch_xla.sync(wait=True)

    # Single device -> host transfer at the very end.
    completion_ids = torch.stack(token_steps, dim=1).to("cpu")
    completion_valid = torch.stack(valid_steps, dim=1).to("cpu")
    return completion_ids, completion_valid

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
    # Prefer the explicit head_dim (Gemma 2 sets head_dim != hidden_size / num_heads);
    # fall back to the derived value for models that omit it.
    head_dim = getattr(model_config, "head_dim", None) or (model_config.hidden_size // model_config.num_attention_heads)
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
