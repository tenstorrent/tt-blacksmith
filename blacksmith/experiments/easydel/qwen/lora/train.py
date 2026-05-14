# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from blacksmith.datasets.jax.sst2.sst2_dataset import load_sst2_batches
from blacksmith.experiments.easydel.qwen.configs import TrainingConfig
from blacksmith.tools.cli import generate_config, parse_cli_options
from blacksmith.tools.jax.checkpoint_manager import JaxCheckpointManager
from blacksmith.tools.jax.device_manager import JaxDeviceManager
from blacksmith.tools.jax.easydel.helpers import (
    apply_lora,
    build_optimizer,
    load_easydel_causal_lm,
)
from blacksmith.tools.jax.easydel.partitioning import easydel_partition_specs_for_lora
from blacksmith.tools.jax.easydel.train_steps import (
    create_eval_inspect_step_fn,
    create_eval_step_fn,
    create_loss_and_grad_step_fn,
    evaluate,
)
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.reproducibility_manager import ReproducibilityManager


def validate(
    jit_eval_step,
    lora_params,
    frozen_state,
    validation_batches,
    logger,
    *,
    jit_inspect_step=None,
    tokenizer=None,
):
    """Run evaluation on validation batches and log the result."""
    logger.info("Starting validation...")
    validation_loss = evaluate(
        jit_eval_step,
        lora_params,
        frozen_state,
        validation_batches,
        jit_inspect_step=jit_inspect_step,
        tokenizer=tokenizer,
    )
    logger.info(f"Average validation loss: {validation_loss:.4f}")
    return validation_loss


def train(
    config: TrainingConfig,
    device_manager: JaxDeviceManager,
    logger: TrainingLogger,
    checkpoint_manager: JaxCheckpointManager,
):
    logger.info("Starting training...")

    model, tokenizer = load_easydel_causal_lm(
        config.model_name,
        device_manager,
        dtype=config.jax_dtype,
        mask_max_position_embeddings=config.mask_max_position_embeddings,
    )
    logger.info(f"Loaded {config.model_name} model.")
    logger.log_model_info(
        {
            "num_hidden_layers": model.config.num_hidden_layers,
            "hidden_size": model.config.hidden_size,
            "intermediate_size": model.config.intermediate_size,
            "vocab_size": model.config.vocab_size,
            "max_position_embeddings": model.config.max_position_embeddings,
            **device_manager.describe(),
            "framework": "jax_easydel",
        }
    )

    train_input_ids, train_labels, train_attention_masks = load_sst2_batches(config, split="train")
    validation_input_ids, validation_labels, validation_attention_masks = load_sst2_batches(config, split="validation")

    def _make_batch(input_ids, labels, attention_mask):
        return {
            "input_ids": np.asarray(input_ids, dtype=np.uint32),
            "labels": np.asarray(labels, dtype=np.int32),
            "attention_mask": np.asarray(attention_mask, dtype=np.int32),
        }

    train_batches = [
        _make_batch(train_input_ids[i], train_labels[i], train_attention_masks[i]) for i in range(len(train_input_ids))
    ]
    validation_batches = [
        _make_batch(validation_input_ids[i], validation_labels[i], validation_attention_masks[i])
        for i in range(len(validation_input_ids))
    ]
    logger.info(
        f"Loaded {config.dataset_id} dataset. "
        f"Train batches: {len(train_batches)}, Validation batches: {len(validation_batches)}"
    )

    # Apply LoRA.
    logger.info(f"Applying LoRA (rank={config.lora_rank}, pattern={config.lora_pattern!r})...")
    model = apply_lora(
        model,
        rank=config.lora_rank,
        pattern=config.lora_pattern,
        on_cpu=(device_manager.device_kind == "tt"),
    )

    graphdef, lora_params, frozen_state, _shardings = easydel_partition_specs_for_lora(model, device_manager.mesh)
    vocab_size = model.config.vocab_size

    # Build optimizer.
    num_train_batches = len(train_batches)
    total_train_batches = num_train_batches * config.num_epochs
    accumulation_steps = config.gradient_accumulation_steps
    total_optimizer_steps = total_train_batches // accumulation_steps

    optimizer, _learning_rate_schedule = build_optimizer(config, total_opt_steps=total_optimizer_steps)
    logger.info(
        f"  LR schedule: warmup {config.warmup_steps} optimizer steps, "
        f"cosine decay over {total_optimizer_steps} optimizer steps "
        f"({config.learning_rate} -> {config.end_learning_rate})"
    )
    if accumulation_steps > 1:
        effective_batch_size = config.batch_size * accumulation_steps
        logger.info(
            f"  Gradient accumulation: {accumulation_steps} steps -> Effective batch size {effective_batch_size}"
        )
    optimizer_state = optimizer.init(lora_params)

    # Compile JIT steps.
    jit_loss_and_grad = create_loss_and_grad_step_fn(graphdef)
    jit_eval_step = create_eval_step_fn(graphdef)
    jit_inspect_step = create_eval_inspect_step_fn(graphdef) if config.print_examples else None

    # Load checkpoint if needed.
    if config.resume_from_checkpoint:
        checkpoint = checkpoint_manager.load_checkpoint(
            params_target=lora_params,
            opt_state_target=optimizer_state,
        )
        if checkpoint is not None:
            lora_params = checkpoint["params"]
            optimizer_state = checkpoint.get("opt_state", optimizer_state)
            logger.info(f"Resumed from step {checkpoint['step']}, epoch {checkpoint['epoch']}")

    if config.max_val_batches is not None:
        original_validation_batch_count = len(validation_batches)
        validation_batches = validation_batches[: config.max_val_batches]
        logger.info(f"  Using {len(validation_batches)} of {original_validation_batch_count} validation batches")

    inspect_kwargs: dict = {}
    if jit_inspect_step is not None and tokenizer is not None:
        inspect_kwargs = {"jit_inspect_step": jit_inspect_step, "tokenizer": tokenizer}

    global_step = 0
    running_losses: list[float] = []
    step_losses: list[float] = []
    numpy_random_generator = np.random.default_rng(config.seed)

    try:
        # Initial validation.
        validation_loss = validate(
            jit_eval_step, lora_params, frozen_state, validation_batches, logger, **inspect_kwargs
        )
        logger.log_metrics({"val/loss": validation_loss}, commit=True, step=global_step)

        for epoch in range(config.num_epochs):
            num_batches = len(train_batches)
            shuffled_batch_order = numpy_random_generator.permutation(num_batches)
            epoch_losses: list[float] = []
            logger.info(f"Epoch {epoch + 1}: shuffled {num_batches} training batches (seed={config.seed})")

            for batch_index in range(num_batches):
                batch = train_batches[shuffled_batch_order[batch_index]]
                input_ids = batch["input_ids"]
                labels = batch["labels"]
                attention_mask = batch["attention_mask"]

                shifted_labels = labels[:, 1:].astype(jnp.int32)
                valid_mask = shifted_labels != config.ignored_label_index
                safe_labels = jnp.where(valid_mask, shifted_labels, 0)
                label_mask = valid_mask.astype(jnp.float32)
                one_hot_labels = jax.nn.one_hot(safe_labels, vocab_size).astype(jnp.float32)

                # Forward + backward.
                loss, gradients, gradient_stats = jit_loss_and_grad(
                    lora_params, frozen_state, input_ids, one_hot_labels, label_mask, attention_mask
                )

                # Optimizer step (CPU-offloaded when optimizer_on_cpu=True on TT).
                lora_params, optimizer_state = device_manager.optimizer_step(
                    optimizer, optimizer_state, lora_params, gradients
                )

                current_loss = float(loss)
                gradient_norm = float(gradient_stats["grad_norm"])
                gradient_max = float(gradient_stats["grad_max"])
                epoch_losses.append(current_loss)
                running_losses.append(current_loss)
                step_losses.append(current_loss)
                global_step += 1

                logger.log_metrics(
                    {
                        "train/loss": current_loss,
                        "grad/global_norm": gradient_norm,
                        "grad/global_max": gradient_max,
                        "epoch": epoch + 1,
                        "batch": batch_index + 1,
                    },
                    step=global_step,
                    commit=False,
                )

                if len(running_losses) == config.steps_freq:
                    average_window_loss = float(np.mean(running_losses))
                    logger.log_metrics({"train/avg_window_loss": average_window_loss}, step=global_step, commit=False)
                    logger.info(
                        f"Epoch {epoch + 1}, Batch {batch_index + 1:3d}: "
                        f"Loss = {current_loss:.4f} | Avg {config.steps_freq} = {average_window_loss:.4f} | "
                        f"grad_norm = {gradient_norm:.4f}, grad_max = {gradient_max:.4f}"
                    )
                    running_losses = []
                else:
                    logger.info(
                        f"Epoch {epoch + 1}, Batch {batch_index + 1:3d}: "
                        f"Loss = {current_loss:.4f} ({len(running_losses)}/{config.steps_freq}) | "
                        f"grad_norm = {gradient_norm:.4f}, grad_max = {gradient_max:.4f}"
                    )

                # Periodic validation.
                if (
                    config.val_steps_freq is not None
                    and validation_batches
                    and global_step % config.val_steps_freq == 0
                ):
                    validation_loss = validate(
                        jit_eval_step,
                        lora_params,
                        frozen_state,
                        validation_batches,
                        logger,
                        **inspect_kwargs,
                    )
                    logger.log_metrics({"val/loss": validation_loss}, step=global_step, commit=False)

                # Commit metrics to W&B.
                logger.log_metrics({}, step=global_step, commit=True)

                # Save step checkpoint.
                if checkpoint_manager.should_save_checkpoint(global_step):
                    checkpoint_manager.save_checkpoint(
                        step=global_step,
                        epoch=epoch,
                        params=lora_params,
                        opt_state=optimizer_state,
                        metrics={"train/loss": current_loss},
                    )

            average_epoch_loss = float(np.mean(epoch_losses))
            logger.info(f"Epoch {epoch + 1} complete - avg loss: {average_epoch_loss:.4f}")

            # End-of-epoch validation.
            end_of_epoch_metrics: dict = {}
            if validation_batches:
                global_step += 1
                validation_loss = validate(
                    jit_eval_step, lora_params, frozen_state, validation_batches, logger, **inspect_kwargs
                )
                logger.log_metrics({"val/loss": validation_loss}, step=global_step)
                end_of_epoch_metrics = {"val/loss": validation_loss}

            # Save epoch checkpoint.
            if checkpoint_manager.should_save_checkpoint(global_step, epoch):
                checkpoint_manager.save_checkpoint(
                    step=global_step,
                    epoch=epoch,
                    params=lora_params,
                    opt_state=optimizer_state,
                    metrics=end_of_epoch_metrics,
                )

        logger.log_summary(
            {
                "total_steps": global_step,
                "final_loss": float(step_losses[-1]) if step_losses else float("nan"),
            }
        )
        logger.info("TRAINING COMPLETED")

    except Exception as exception:
        traceback_string = traceback.format_exc()
        logger.error(f"Training failed with error: {exception}", traceback_string)
        raise
    finally:
        logger.finish()


if __name__ == "__main__":
    # Config setup.
    default_config_path = Path(__file__).parent / "single_chip" / "qwen3_0_6b_sst2.yaml"
    args = parse_cli_options(default_config=default_config_path)
    config: TrainingConfig = generate_config(TrainingConfig, args.config, args.test_config, args.test_checkpoint_path)

    # Reproducibility setup.
    ReproducibilityManager(config).setup()

    # Logger setup.
    logger = TrainingLogger(config, args.test_log_filename_prefix)

    # Device setup.
    device_manager = JaxDeviceManager(config)
    logger.info(f"Using device: {device_manager.device_kind}")

    # Checkpoint manager setup.
    checkpoint_manager = JaxCheckpointManager(config, logger)

    # Start training.
    with device_manager.mesh:
        train(config, device_manager, logger, checkpoint_manager)
