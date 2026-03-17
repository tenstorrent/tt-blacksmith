# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import traceback
from pathlib import Path

import torch
import torch_xla
from tqdm import tqdm

from blacksmith.datasets.torch.dataset_utils import get_dataset
from blacksmith.experiments.torch.BOUNTIES.falcon3_1b.configs import TrainingConfig
from blacksmith.models.torch.huggingface.hf_models import get_model
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


def validate(model, val_data_loader, loss_fn, device, config, logger, tokenizer=None):
    logger.info("\n=== Starting Validation ===")
    model.eval()
    total_val_loss = 0.0
    num_val_batches = 0
    collected_examples = []

    with torch.no_grad():
        for batch in tqdm(val_data_loader, desc="Validation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            # Keep labels on CPU to avoid TT device holding extra tensors (OOM).
            # See https://github.com/tenstorrent/tt-blacksmith/issues/455.
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            shift_logits = outputs.logits[:, :-1, :].contiguous()

            loss = loss_fn(
                shift_logits.view(-1, model.model.config.vocab_size),
                labels.view(-1).to(shift_logits.device),
            )

            predictions = shift_logits.argmax(dim=-1)

            if config.use_tt:
                torch_xla.sync(wait=True)

            total_val_loss += loss.item()
            num_val_batches += 1

            if config.print_examples:
                collected_examples = collect_examples(
                    batch_size=labels.shape[0],
                    collected_examples=collected_examples,
                    max_examples=10,
                    input_ids=input_ids,
                    expected_output=labels,
                    predictions=predictions,
                    num_val_batches=num_val_batches,
                )

    if config.print_examples and tokenizer is not None:
        logger.info("\n=== Validation Examples (Random samples) ===")
        show_examples(collected_examples, tokenizer, config, logger)

    avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
    logger.info(f"Average validation loss: {avg_val_loss}")
    return avg_val_loss


def train_step(model, batch, loss_fn, device_manager, config):
    batch = device_manager.prepare_batch(batch)
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

    shift_logits = outputs.logits[:, :-1, :].contiguous()
    loss = loss_fn(
        shift_logits.view(-1, model.model.config.vocab_size),
        batch["labels"].view(-1),
    )

    loss.backward()

    if config.use_tt:
        torch_xla.sync(wait=True)

    return loss.item()


def setup_training(config, device_manager, logger, checkpoint_manager):
    """Load model, datasets, and initialize training components."""
    model = get_model(config, device_manager.device)
    logger.info(f"Loaded {config.model_name} model.")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if config.resume_from_checkpoint:
        checkpoint_manager.load_checkpoint()

    train_dataset = get_dataset(config=config, split="train", collate_fn=collate_fn_for_causal_lm)
    train_dataloader = train_dataset.get_dataloader()
    logger.info(f"Loaded {config.dataset_id} dataset. Train dataset size: {len(train_dataloader)*config.batch_size}")

    eval_dataset = get_dataset(config=config, split="validation", collate_fn=collate_fn_for_causal_lm)
    eval_dataloader = eval_dataset.get_dataloader()
    logger.info(f"Loaded {config.dataset_id} dataset. Eval dataset size: {len(eval_dataloader)*config.batch_size}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=config.ignored_index)

    return (
        model,
        train_dataloader,
        eval_dataloader,
        train_dataset.tokenizer,
        optimizer,
        loss_fn,
    )


def train(
    config: TrainingConfig,
    device_manager: DeviceManager,
    logger: TrainingLogger,
    checkpoint_manager: CheckpointManager,
):
    """Main training loop for Falcon3-1B LoRA fine-tuning."""
    logger.info("Starting training...")

    model, train_dataloader, eval_dataloader, tokenizer, optimizer, loss_fn = setup_training(
        config, device_manager, logger, checkpoint_manager
    )

    global_step = 0
    running_loss = 0.0

    try:
        # Initial validation.
        model.eval()
        val_loss = validate(
            model,
            eval_dataloader,
            loss_fn,
            device_manager.device,
            config,
            logger,
            tokenizer,
        )
        logger.log_metrics({"val/loss": val_loss}, commit=True, step=global_step)
        model.train()

        for epoch in range(config.num_epochs):
            model.train()

            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs}"):
                global_step += 1
                optimizer.zero_grad()

                running_loss += train_step(model, batch, loss_fn, device_manager, config)

                # Update parameters.
                device_manager.optimizer_step(optimizer)

                if global_step % config.steps_freq == 0:
                    avg_loss = running_loss / config.steps_freq if global_step > 0 else running_loss
                    logger.log_metrics(
                        {"train/loss": avg_loss},
                        commit=False,
                        step=global_step,
                    )
                    running_loss = 0.0

                if global_step % config.val_steps_freq == 0:
                    model.eval()
                    val_loss = validate(
                        model,
                        eval_dataloader,
                        loss_fn,
                        device_manager.device,
                        config,
                        logger,
                        tokenizer,
                    )
                    logger.log_metrics({"val/loss": val_loss}, commit=False, step=global_step)
                    model.train()

                logger.log_metrics({}, commit=True, step=global_step)

                if checkpoint_manager.should_save_checkpoint(global_step):
                    checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)

            if checkpoint_manager.should_save_checkpoint(global_step, epoch):
                checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)

        final_model_path = checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)
        logger.log_artifact(final_model_path, artifact_type="model", name="final_model.pth")

    except Exception as e:
        traceback_str = traceback.format_exc()
        logger.error(f"Training failed with error: {str(e)}", traceback_str)
        raise
    finally:
        logger.finish()


if __name__ == "__main__":
    # Config setup
    default_config = Path(__file__).parent / "test_falcon3_finetuning.yaml"
    args = parse_cli_options(default_config=default_config)
    config: TrainingConfig = generate_config(TrainingConfig, args.config)

    # Reproducibility setup
    repro_manager = ReproducibilityManager(config)
    repro_manager.setup()

    # Logger setup
    logger = TrainingLogger(config)

    # Device setup
    device_manager = DeviceManager(config)
    logger.info(f"Using device: {device_manager.device}")

    # Checkpoint manager setup
    checkpoint_manager = CheckpointManager(config, logger, device_manager.device)

    # Start training
    train(config, device_manager, logger, checkpoint_manager)
