# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import traceback
from pathlib import Path

import torch
import torch_xla
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from blacksmith.datasets.torch.dataset_utils import get_dataset
from blacksmith.experiments.torch.qwen.configs import TrainingConfig
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


def validate(
    model: torch.nn.Module,
    val_data_loader: DataLoader,
    loss_fn: torch.nn.Module,
    logger: TrainingLogger,
    device_manager: DeviceManager,
    config: TrainingConfig,
    tokenizer: AutoTokenizer = None,
) -> float:
    logger.info("Starting validation...")

    collected_examples = []
    max_examples = 5
    total_val_loss = 0.0
    num_val_batches = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_data_loader, desc="Validation"):
            batch = device_manager.prepare_batch(batch)

            # Forward pass
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = outputs.logits

            # Shift logits for causal LM: predict next token
            # logits[:, :-1] predicts tokens at positions 1:
            shift_logits = logits[:, :-1, :].contiguous()

            # Loss
            loss = loss_fn(shift_logits.view(-1, model.model.config.vocab_size), batch["labels"].view(-1))
            total_val_loss += loss.item()

            # Predictions
            predictions = shift_logits.argmax(dim=-1)
            if config.use_tt:
                torch_xla.sync(wait=True)

            num_val_batches += 1

            if config.print_examples:
                collected_examples = collect_examples(
                    batch_size=batch["labels"].shape[0],
                    collected_examples=collected_examples,
                    max_examples=max_examples,
                    input_ids=batch["input_ids"],
                    expected_output=batch["labels"],
                    predictions=predictions,
                    num_val_batches=num_val_batches,
                )

    if config.print_examples and tokenizer is not None:
        logger.info("Printing validation examples...")
        show_examples(collected_examples, tokenizer, config, logger)

    avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
    logger.info(f"Average validation loss: {avg_val_loss}")

    return avg_val_loss


def train(
    config: TrainingConfig, device_manager: DeviceManager, logger: TrainingLogger, checkpoint_manager: CheckpointManager
):
    logger.info("Starting training...")

    # Load model
    model = get_model(config, device_manager.device)
    logger.info(f"Loaded {config.model_name} model.")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Init training components (optimizer, lr scheduler, etc.)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # Load checkpoint if needed
    if config.resume_from_checkpoint:
        checkpoint_manager.load_checkpoint(model, optimizer)

    # Load dataset
    train_dataset = get_dataset(config=config, split="train", collate_fn=collate_fn_for_causal_lm)
    train_dataloader = train_dataset.get_dataloader()
    logger.info(f"Loaded {config.dataset_id} dataset. Train dataset size: {len(train_dataloader)*config.batch_size}")

    eval_dataset = get_dataset(config=config, split="test", collate_fn=collate_fn_for_causal_lm)
    eval_dataloader = eval_dataset.get_dataloader()
    logger.info(f"Loaded {config.dataset_id} dataset. Eval dataset size: {len(eval_dataloader)*config.batch_size}")

    global_step = 0
    running_loss = 0.0
    try:
        for epoch in range(config.num_epochs):
            model.train()

            for batch in tqdm(train_dataloader):
                optimizer.zero_grad()

                batch = device_manager.prepare_batch(batch)

                # Forward pass
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

                # Compute loss
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                loss = loss_fn(shift_logits.view(-1, model.model.config.vocab_size), batch["labels"].view(-1))
                running_loss += loss.item()

                # Backward pass
                loss.backward()
                if config.use_tt:
                    torch_xla.sync(wait=True)

                # Update parameters
                optimizer.step()
                if config.use_tt:
                    torch_xla.sync(wait=True)

                global_step += 1
                if global_step % config.steps_freq == 0:
                    avg_loss = running_loss / config.steps_freq
                    logger.log_metrics({"train/loss": avg_loss}, commit=False, step=global_step)
                    running_loss = 0.0

                    # Do validation
                    if config.do_validation:
                        valid_loss = validate(
                            model,
                            eval_dataloader,
                            loss_fn,
                            logger,
                            device_manager.device,
                            config,
                            eval_dataset.tokenizer,
                        )
                        logger.log_metrics({"val/loss": valid_loss}, step=global_step)
                        model.train()

                    # Save step checkpoint
                    if checkpoint_manager.should_save_checkpoint(global_step):
                        checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)

            # Save epoch checkpoint
            if checkpoint_manager.should_save_checkpoint(global_step, epoch):
                checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)

        # Save final model
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
    default_config = Path(__file__).parent / "test_qwen_finetuning.yaml"
    args = parse_cli_options(default_config=default_config)
    config: TrainingConfig = generate_config(TrainingConfig, args.config)

    # Reproducibility setup
    repro_manager = ReproducibilityManager(config)
    repro_manager.setup()

    # Logger setup
    logger = TrainingLogger(config)

    # Checkpoint manager setup
    checkpoint_manager = CheckpointManager(config, logger)

    # Device setup
    device_manager = DeviceManager(config)
    logger.info(f"Using device: {device_manager.device}")

    # Start training
    train(config, device_manager, logger, checkpoint_manager)
