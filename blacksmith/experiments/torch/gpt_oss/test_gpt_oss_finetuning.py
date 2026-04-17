# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import traceback
from pathlib import Path

import torch
import torch_xla
from tqdm import tqdm

from blacksmith.datasets.torch.dataset_utils import get_dataset
from blacksmith.experiments.torch.gpt_oss.configs import TrainingConfig
from blacksmith.models.torch.gpt_oss.model_overrides import get_model
from blacksmith.tools.checkpoints_manager import CheckpointManager
from blacksmith.tools.cli import generate_config, parse_cli_options
from blacksmith.tools.device_manager import DeviceManager
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.reproducibility_manager import ReproducibilityManager
from blacksmith.tools.torch_helpers import (
    collate_fn_for_causal_lm,
    collect_examples,
    show_examples,
)
from blacksmith.tools.workaround_utils import cross_entropy_loss, transform_labels


def validate(model, val_data_loader, loss_fn, logger, device, config, tokenizer=None):
    logger.info("Starting validation...")
    total_val_loss = 0.0
    num_val_batches = 0
    collected_examples = []

    with torch.no_grad():
        for batch in tqdm(val_data_loader, desc="Validation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            # Expected output must be prepared on CPU first due to an OOM issue.
            # See https://github.com/tenstorrent/tt-blacksmith/issues/455.
            expected_output = batch["labels"]

            # Shard model if tensor parallelism is used.
            device_manager.shard_model(model)

            # Forward pass.
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Shift logits for causal LM: predict next token
            # logits[:, :-1] predicts tokens at positions 1:
            shift_logits = logits[:, :-1, :].contiguous()

            expected_output_one_hot, labels_mask = transform_labels(
                expected_output, config.ignored_index, model.model.config.vocab_size
            )

            if config.use_tt:
                loss = loss_fn(shift_logits, expected_output_one_hot, labels_mask)
            else:
                loss = loss_fn(
                    shift_logits,
                    expected_output_one_hot.to(device),
                    labels_mask.to(device),
                )

            # Predictions
            predictions = shift_logits.argmax(dim=-1)
            if config.use_tt:
                torch_xla.sync(wait=True)

            total_val_loss += loss.item()
            num_val_batches += 1

            if config.print_examples:
                collected_examples = collect_examples(
                    batch_size=expected_output.shape[0],
                    collected_examples=collected_examples,
                    max_examples=10,
                    input_ids=input_ids,
                    expected_output=expected_output,
                    predictions=predictions,
                    num_val_batches=num_val_batches,
                )

    if config.print_examples and tokenizer is not None:
        logger.info("Printing validation examples...")
        show_examples(collected_examples, tokenizer, config, logger)

    avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
    logger.info(f"Average validation loss: {avg_val_loss}")
    return avg_val_loss


# Training step extracted into a separate function to keep large vocab-sized
# tensors (e.g. logits) scoped locally. This ensures they do not propagate beyond
# the step via the computation graph, avoiding unnecessary and expensive
# CCLs in multi-chip setups.
# Issue itself should be investigated further.
def training_step_inner(batch, model, loss_fn, gradient_accumulation_steps):
    output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = output.logits
    shift_logits = logits[:, :-1, :].contiguous()
    loss = loss_fn(shift_logits, batch["expected_output"], batch["labels_mask"])
    # Scale loss by number of accumulation steps to get correct effective batch size.
    scaled_loss = loss / gradient_accumulation_steps
    scaled_loss.backward()
    return loss.detach()


def train(
    config: TrainingConfig,
    device_manager: DeviceManager,
    logger: TrainingLogger,
    checkpoint_manager: CheckpointManager,
):
    logger.info("Starting training...")

    # Load model.
    model = get_model(config, device_manager.device)
    logger.info(f"Loaded {config.model_name} model.")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)

    # Load checkpoint if needed.
    if config.resume_from_checkpoint:
        checkpoint_manager.load_checkpoint(model, optimizer)

    # Load dataset.
    train_dataset = get_dataset(config=config, split="train", collate_fn=collate_fn_for_causal_lm)
    train_dataloader = train_dataset.get_dataloader()
    logger.info(f"Loaded {config.dataset_id} dataset. Train dataset size: {len(train_dataloader)*config.batch_size}")

    eval_dataset = get_dataset(config=config, split="validation", collate_fn=collate_fn_for_causal_lm)
    eval_dataloader = eval_dataset.get_dataloader()
    logger.info(f"Loaded {config.dataset_id} dataset. Eval dataset size: {len(eval_dataloader)*config.batch_size}")

    tokenizer = train_dataset.tokenizer

    GRAD_CLAMP_VALUE = 10_000
    GRAD_CLIP_MAX_NORM = 1.0

    global_step = 0
    running_loss = 0.0

    try:
        # Initial validation
        model.eval()
        val_loss = validate(
            model,
            eval_dataloader,
            cross_entropy_loss,
            logger,
            device_manager.device,
            config,
            tokenizer,
        )

        logger.log_metrics({"val/loss": val_loss}, commit=True, step=global_step)
        model.train()

        for epoch in range(config.num_epochs):
            accumulation_step = 0

            for batch in tqdm(train_dataloader, desc="Training"):
                # Zero out gradients at the start of accumulation cycle
                if accumulation_step == 0:
                    optimizer.zero_grad()

                # TODO: Refactor when https://github.com/tenstorrent/tt-blacksmith/issues/327 is resolved.
                expected_output, labels_mask = transform_labels(
                    batch["labels"], config.ignored_index, model.model.config.vocab_size
                )
                batch = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "expected_output": expected_output,
                    "labels_mask": labels_mask,
                }
                # Shard batch if data parallelism is used.
                batch = device_manager.prepare_batch(batch)
                device_manager.shard_model(model)

                # Training step.
                loss_ = training_step_inner(batch, model, cross_entropy_loss, config.gradient_accumulation_steps)

                # Clamp gradient values.
                for p in trainable_params:
                    if p.grad is not None:
                        p.grad = p.grad.clamp(-GRAD_CLAMP_VALUE, GRAD_CLAMP_VALUE)

                # Clip gradient norms.
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=GRAD_CLIP_MAX_NORM)

                if config.use_tt:
                    torch_xla.sync(wait=True)

                running_loss += loss_.item()
                accumulation_step += 1

                logger.info(f"Current loss and step: {loss_.item()} {global_step}")

                # Only step the optimizer after accumulating gradients.
                if accumulation_step == config.gradient_accumulation_steps:
                    device_manager.optimizer_step(optimizer)

                    accumulation_step = 0
                    global_step += 1

                    if global_step % config.steps_freq == 0:
                        avg_loss = running_loss / (config.steps_freq * config.gradient_accumulation_steps)
                        logger.log_metrics({"train/loss": avg_loss}, commit=False, step=global_step)
                        running_loss = 0.0

                    # Validation
                    if global_step % config.val_steps_freq == 0:
                        model.eval()
                        val_loss = validate(
                            model,
                            eval_dataloader,
                            cross_entropy_loss,
                            logger,
                            device_manager.device,
                            config,
                            tokenizer,
                        )
                        logger.log_metrics({"val/loss": val_loss}, commit=False, step=global_step)
                        model.train()

                    # Commit metrics to W&B.
                    logger.log_metrics({}, commit=True, step=global_step)

                    # Save step checkpoint.
                    if checkpoint_manager.should_save_checkpoint(global_step):
                        checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)

            # Save epoch checkpoint.
            if checkpoint_manager.should_save_checkpoint(global_step, epoch):
                checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)

        # Save final model.
        final_model_path = checkpoint_manager.save_checkpoint(
            model, global_step, epoch, optimizer, checkpoint_name="final_model.pth"
        )
        logger.log_artifact(final_model_path, artifact_type="model", name="final_model.pth")

    except Exception as e:
        traceback_str = traceback.format_exc()
        logger.error(f"Training failed with error: {str(e)}", traceback_str)
        raise
    finally:
        logger.finish()


if __name__ == "__main__":
    # Config setup
    default_config = Path(__file__).parent / "lora" / "loudbox" / "test_gpt_oss_20b_finetuning.yaml"
    args = parse_cli_options(default_config=default_config)
    config: TrainingConfig = generate_config(TrainingConfig, args.config, args.test_config)

    # Reproducibility setup
    repro_manager = ReproducibilityManager(config)
    repro_manager.setup()

    # Logger setup.
    logger = TrainingLogger(config, args.test_log_filename_prefix)

    # Device setup
    device_manager = DeviceManager(config)
    logger.info(f"Using device: {device_manager.device}")

    # Use highest numerical precision for stable fine-tuning convergence.
    # fp32_dest_acc_en: accumulate partial results in FP32 to avoid precision loss.
    # math_fidelity hifi4: use all 4 mantissa phases for full precision multiplications.
    if config.use_tt:
        torch_xla.set_custom_compile_options({"fp32_dest_acc_en": True, "math_fidelity": "hifi4"})

    # Checkpoint manager setup
    checkpoint_manager = CheckpointManager(config, logger, device_manager.device)

    # Start training.
    train(config, device_manager, logger, checkpoint_manager)
