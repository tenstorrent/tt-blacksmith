# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import random
from pathlib import Path
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from easydel import AutoEasyDeLModelForCausalLM
from flax import nnx
from jax.typing import DTypeLike
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from blacksmith.experiments.easydel.qwen.configs import TrainingConfig
from blacksmith.experiments.easydel.qwen.data_loading import load_sst2_batches
from blacksmith.experiments.easydel.qwen.train_steps import (
    create_eval_inspect_step_fn,
    create_eval_step_fn,
    create_train_step_fn,
    evaluate,
)
from blacksmith.tools.cli import generate_config, parse_cli_options
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.workaround_utils_jax import apply_gqa_workaround

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def _select_preferred_device(
    use_tt: bool = True,
) -> tuple[jax.Device, str]:
    """Select compute device: TT > GPU > CPU."""
    logger = logging.getLogger(__name__)
    cpu = jax.devices("cpu")[0]
    if not use_tt:
        try:
            gpu_devs = jax.devices("gpu")
            if gpu_devs:
                return gpu_devs[0], "gpu"
        except Exception:
            logger.info("No GPU devices available, falling back to CPU")
        return cpu, "cpu"
    try:
        tt_devs = jax.devices("tt")
    except Exception:
        logger.info("No TT devices available, falling back to CPU")
        tt_devs = []
    if tt_devs:
        return tt_devs[0], "tt"
    return cpu, "cpu"


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Load a HuggingFace tokenizer, ensuring a pad token exists."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_name: str,
    *,
    dtype: DTypeLike = jnp.bfloat16,
    mask_max_position_embeddings: Optional[int] = None,
) -> tuple[nnx.Module, PreTrainedTokenizerBase]:
    """Load a causal LM and its tokenizer via EasyDel + HuggingFace."""
    config_overrides = {}
    if mask_max_position_embeddings is not None:
        config_overrides["mask_max_position_embeddings"] = mask_max_position_embeddings
    kwargs = {"dtype": dtype}
    if config_overrides:
        kwargs["config_kwargs"] = config_overrides
    model = AutoEasyDeLModelForCausalLM.from_pretrained(
        model_name,
        **kwargs,
    )
    tokenizer = load_tokenizer(model_name)
    return model, tokenizer


def _load_and_prepare_batches(
    training_config: TrainingConfig,
) -> tuple[list[dict], list[dict]]:
    """Load SST-2 dataset and return train/val lists of batch dicts.

    Each batch dict has keys input_ids, labels, and attention_mask
    (all jnp.ndarray). Labels contain -100 at prompt positions so
    only response tokens contribute to the loss.
    """
    train_ids, train_labels, train_masks = load_sst2_batches(
        training_config,
        split="train",
    )
    val_ids, val_labels, val_masks = load_sst2_batches(
        training_config,
        split="validation",
    )

    def _to_batch(ids, labels, masks):
        return {
            "input_ids": np.asarray(ids, dtype=np.uint32),
            "labels": np.asarray(labels, dtype=np.int32),
            "attention_mask": np.asarray(masks, dtype=np.int32),
        }

    train_batches = [_to_batch(train_ids[i], train_labels[i], train_masks[i]) for i in range(len(train_ids))]
    val_batches = [_to_batch(val_ids[i], val_labels[i], val_masks[i]) for i in range(len(val_ids))]

    return train_batches, val_batches


def _training_loop(
    training_config: TrainingConfig,
    training_logger: TrainingLogger,
    jit_train_step: Callable,
    jit_eval_step: Callable,
    lora_params: nnx.State,
    frozen_state: nnx.State,
    opt_state: optax.OptState,
    train_batches: list[dict[str, np.ndarray]],
    val_batches: list[dict[str, np.ndarray]],
    vocab_size: int,
    *,
    jit_inspect_step: Optional[Callable] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> tuple[int, list[float]]:
    """Execute the training and validation loop.

    Must be called inside a with mesh: context.

    train_batches / val_batches are lists of dicts with keys
    input_ids, labels, attention_mask.

    Returns (global_step, step_losses).

    global_step counts micro-batches. When gradient_accumulation_steps > 1
    the underlying optimizer (wrapped in optax.MultiSteps) only updates
    weights every k micro-batches, but the step counter still increments
    per micro-batch.
    """
    global_step = 0
    steps_freq = training_config.steps_freq
    ignored = training_config.ignored_label_index
    running_losses: list[float] = []
    step_losses: list[float] = []

    inspect_kwargs = {}
    if jit_inspect_step is not None and tokenizer is not None:
        inspect_kwargs = {
            "jit_inspect_step": jit_inspect_step,
            "tokenizer": tokenizer,
        }

    if val_batches:
        val_loss = evaluate(
            jit_eval_step,
            lora_params,
            frozen_state,
            val_batches,
            **inspect_kwargs,
        )
        training_logger.info(f"  Initial validation loss: {val_loss:.4f}")
        training_logger.log_metrics({"val/loss": val_loss}, step=0)

    cpu = jax.devices("cpu")[0]
    rng = np.random.default_rng(training_config.seed)

    for epoch in range(training_config.num_epochs):
        epoch_losses: list[float] = []
        num_batches = len(train_batches)
        batch_order = rng.permutation(num_batches)
        training_logger.info(
            f"Epoch {epoch + 1}: shuffled {num_batches} training batches (seed={training_config.seed})"
        )

        for batch_idx in range(num_batches):
            batch = train_batches[batch_order[batch_idx]]
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            attention_mask = batch["attention_mask"]

            with jax.default_device(cpu):
                shift = labels[:, 1:].astype(jnp.int32)
                valid = shift != ignored
                safe = jnp.where(valid, shift, 0)
                label_mask = valid.astype(jnp.float32)
                one_hot = jax.nn.one_hot(
                    safe,
                    vocab_size,
                ).astype(jnp.float32)

            loss, lora_params, opt_state, grad_stats = jit_train_step(
                lora_params,
                frozen_state,
                opt_state,
                input_ids,
                one_hot,
                label_mask,
                attention_mask,
            )

            current_loss = float(loss)
            g_norm = float(grad_stats["grad_norm"])
            g_max = float(grad_stats["grad_max"])
            epoch_losses.append(current_loss)
            running_losses.append(current_loss)
            step_losses.append(current_loss)
            global_step += 1

            # Buffer per-step metrics with commit=False so train/* and val/*
            # land on the same W&B step; flush with commit=True at end of iteration.
            training_logger.log_metrics(
                {
                    "train/loss": current_loss,
                    "grad/global_norm": g_norm,
                    "grad/global_max": g_max,
                    "epoch": epoch + 1,
                    "batch": batch_idx + 1,
                },
                step=global_step,
                commit=False,
            )

            if len(running_losses) == steps_freq:
                avg = float(np.mean(running_losses))
                training_logger.log_metrics(
                    {"train/avg_window_loss": avg},
                    step=global_step,
                    commit=False,
                )
                training_logger.info(
                    f"Epoch {epoch + 1}, "
                    f"Batch {batch_idx + 1:3d}: "
                    f"Loss = {current_loss:.4f} | "
                    f"Avg {steps_freq} = {avg:.4f} | "
                    f"grad_norm = {g_norm:.4f}, "
                    f"grad_max = {g_max:.4f}"
                )
                running_losses = []
            else:
                training_logger.info(
                    f"Epoch {epoch + 1}, "
                    f"Batch {batch_idx + 1:3d}: "
                    f"Loss = {current_loss:.4f} "
                    f"({len(running_losses)}/{steps_freq}) | "
                    f"grad_norm = {g_norm:.4f}, "
                    f"grad_max = {g_max:.4f}"
                )

            if (
                training_config.val_steps_freq is not None
                and val_batches
                and global_step % training_config.val_steps_freq == 0
            ):
                val_loss = evaluate(
                    jit_eval_step,
                    lora_params,
                    frozen_state,
                    val_batches,
                    **inspect_kwargs,
                )
                training_logger.info(f"  [Step {global_step}] Validation loss: {val_loss:.4f}")
                training_logger.log_metrics(
                    {"val/loss": val_loss},
                    step=global_step,
                    commit=False,
                )

            # Flush all buffered metrics for this global_step in one commit.
            training_logger.log_metrics({}, step=global_step, commit=True)

        avg_epoch = float(np.mean(epoch_losses))
        training_logger.info(f"Epoch {epoch + 1} complete — avg loss: {avg_epoch:.4f}")

        if val_batches:
            # The last batch of the epoch already committed global_step;
            # bump by one so the end-of-epoch val lands on a fresh W&B step.
            global_step += 1
            val_loss = evaluate(
                jit_eval_step,
                lora_params,
                frozen_state,
                val_batches,
                **inspect_kwargs,
            )
            training_logger.info(f"  Epoch {epoch + 1} validation loss: {val_loss:.4f}")
            training_logger.log_metrics({"val/loss": val_loss}, step=global_step)

    return global_step, step_losses


def main(training_config: TrainingConfig) -> None:
    """Run full LoRA fine-tuning pipeline."""
    random.seed(training_config.seed)
    np.random.seed(training_config.seed)

    training_logger = TrainingLogger(training_config)

    cpu_device = jax.devices("cpu")[0]
    current_device, device_kind = _select_preferred_device(
        use_tt=training_config.use_tt,
    )
    jax.config.update("jax_default_device", current_device)

    if device_kind == "tt":
        apply_gqa_workaround()

    training_logger.info(
        f"Loading {training_config.model_name} model... Using device: {device_kind} -> {current_device}"
    )

    model, tokenizer = load_model(
        training_config.model_name,
        dtype=training_config.jax_dtype,
        mask_max_position_embeddings=(training_config.mask_max_position_embeddings),
    )

    num_devices = training_config.num_devices
    devices_for_mesh = tuple(
        jax.devices(device_kind)[:num_devices],
    )
    mesh = jax.make_mesh((num_devices,), ("X",), devices=devices_for_mesh)
    model.config.set_model_mesh(mesh)

    training_logger.log_model_info(
        {
            "num_hidden_layers": model.config.num_hidden_layers,
            "hidden_size": model.config.hidden_size,
            "intermediate_size": model.config.intermediate_size,
            "vocab_size": model.config.vocab_size,
            "max_position_embeddings": model.config.max_position_embeddings,
            "device": device_kind,
            "framework": "jax_easydel",
        }
    )

    train_batches, val_batches = _load_and_prepare_batches(
        training_config,
    )

    training_logger.info(
        f"Applying LoRA (rank={training_config.lora_rank}, pattern={training_config.lora_pattern!r})..."
    )
    with jax.default_device(cpu_device):
        model = model.apply_lora_to_layers(
            lora_rank=training_config.lora_rank,
            lora_pattern=training_config.lora_pattern,
            verbose=True,
        )

    graphdef, lora_params, frozen_state = nnx.split(
        model,
        nnx.LoRAParam,
        ...,
    )

    num_train_batches = len(train_batches)
    total_batches = num_train_batches * training_config.num_epochs
    accum = training_config.gradient_accumulation_steps
    total_opt_steps = total_batches // accum

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=training_config.learning_rate,
        warmup_steps=training_config.warmup_steps,
        decay_steps=total_opt_steps,
        end_value=training_config.end_learning_rate,
    )
    training_logger.info(
        f"  LR schedule: warmup {training_config.warmup_steps} "
        f"optimizer steps, cosine decay over {total_opt_steps} "
        f"optimizer steps "
        f"({training_config.learning_rate} -> {training_config.end_learning_rate})"
    )

    base_optimizer = optax.adamw(learning_rate=schedule)
    if accum > 1:
        optimizer = optax.MultiSteps(base_optimizer, every_k_schedule=accum)
        eff = training_config.batch_size * accum
        training_logger.info(f"  Gradient accumulation: {accum} steps -> Effective batch size {eff}")
    else:
        optimizer = base_optimizer
    opt_state = optimizer.init(lora_params)

    jit_train_step = create_train_step_fn(graphdef, optimizer)
    jit_eval_step = create_eval_step_fn(graphdef)
    jit_inspect_step = create_eval_inspect_step_fn(graphdef) if training_config.print_examples else None

    if training_config.max_val_batches is not None:
        orig = len(val_batches)
        val_batches = val_batches[: training_config.max_val_batches]
        training_logger.info(f"  Using {len(val_batches)} of {orig} validation batches")

    training_logger.info("Starting training on SST-2 dataset...")

    try:
        with mesh:
            global_step, step_losses = _training_loop(
                training_config,
                training_logger,
                jit_train_step,
                jit_eval_step,
                lora_params,
                frozen_state,
                opt_state,
                train_batches,
                val_batches,
                model.config.vocab_size,
                jit_inspect_step=jit_inspect_step,
                tokenizer=(tokenizer if training_config.print_examples else None),
            )

        training_logger.log_summary(
            {
                "total_steps": global_step,
                "final_loss": float(step_losses[-1]) if step_losses else float("nan"),
            }
        )
        training_logger.info("TRAINING COMPLETED")

    except Exception as e:
        training_logger.error(f"Error during training: {e}")
        raise

    finally:
        training_logger.finish()


if __name__ == "__main__":
    default_cfg = Path(__file__).parent / "single_chip" / "test_qwen3_0.6b_lora.yaml"
    args = parse_cli_options(default_config=default_cfg)
    training_config: TrainingConfig = generate_config(
        TrainingConfig,
        args.config,
        args.test_config,
    )

    if training_config.use_tt:
        os.environ.setdefault("PJRT_DEVICE", "TT")
        os.environ.setdefault("XLA_STABLEHLO_COMPILE", "1")

    main(training_config)
