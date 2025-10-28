# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import torch
import traceback
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from torch.utils.data import DataLoader
from tqdm import tqdm

from blacksmith.datasets.torch.llama.sst_dataset import SSTDataset
from blacksmith.experiments.torch.llama.configs import TrainingConfig
from blacksmith.models.torch.huggingface.hf_models import get_model
from blacksmith.tools.cli import generate_config
from blacksmith.tools.torch_helpers import show_examples, collect_examples
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.checkpoints_manager import CheckpointManager
from blacksmith.tools.reproducibility_manager import ReproducibilityManager


def validate(model, val_data_loader, loss_fn, logger, device, config, tokenizer=None):
    logger.info("Starting validation...")
    total_val_loss = 0.0
    num_val_batches = 0
    collected_examples = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(val_data_loader, desc="Validation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            expected_output = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Shift logits for causal LM: predict next token
            # logits[:, :-1] predicts tokens at positions 1:
            shift_logits = logits[:, :-1, :].contiguous()

            # Loss
            loss = loss_fn(shift_logits.view(-1, model.model.config.vocab_size), expected_output.view(-1))
            total_val_loss += loss.item()

            # Predictions
            predictions = shift_logits.argmax(dim=-1)
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

    if config.print_examples:
        logger.info("Printing validation examples...")
        show_examples(collected_examples, tokenizer, config, logger)

    avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
    logger.info(f"Average validation loss: {avg_val_loss}")
    return avg_val_loss


def collate_fn_with_shifted_labels(batch):
    """
    Collate function that pre-shifts labels for causal LM.
    Shifts labels to exclude first token.
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])

    shifted_labels = labels[:, 1:].contiguous()

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": shifted_labels}


def train(config: TrainingConfig, device: torch.device, logger: TrainingLogger, checkpoint_manager: CheckpointManager):
    logger.info("Starting training...")

    # Load model
    model = get_model(config, device)
    logger.info(f"Loaded {config.model_name} model.")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Init training components (optimizer, lr scheduler, etc.)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # Load checkpoint if needed
    if config.resume_from_checkpoint:
        checkpoint_manager.load_checkpoint(model, optimizer)

    # Get dataset
    dataset = SSTDataset(config)
    tokenizer = dataset.tokenizer
    train_set, eval_set = dataset.load_tokenized_data()

    train_dataloader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn_with_shifted_labels
    )
    logger.info(f"Loaded {config.dataset_id} dataset. Train dataset size: {len(train_dataloader)}")
    eval_dataloader = DataLoader(
        eval_set, batch_size=config.batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn_with_shifted_labels
    )
    logger.info(f"Loaded {config.dataset_id} dataset. Eval dataset size: {len(eval_dataloader)}")

    global_step = 0
    running_loss = 0.0
    try:
        model.train()
        for epoch in range(config.num_epochs):

            for batch in tqdm(train_dataloader, desc="Training"):
                # Zero out gradients
                optimizer.zero_grad()

                # Get input ids and attention mask
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                # Get expected output
                expected_output = batch["labels"].to(device)

                # Forward pass
                output = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = output.logits

                # Shift logits for causal LM: predict next token
                # logits[:, :-1] predicts tokens at positions 1:
                shift_logits = logits[:, :-1, :].contiguous()

                loss = loss_fn(shift_logits.view(-1, model.model.config.vocab_size), expected_output.view(-1))

                running_loss += loss.item()

                # Backward pass
                loss.backward()

                # Optimizer step
                if config.use_tt:
                    xm.optimizer_step(optimizer)
                    torch_xla.sync(wait=True)
                else:
                    optimizer.step()

                global_step += 1
                if global_step % config.steps_freq == 0:
                    avg_loss = running_loss / config.steps_freq
                    logger.log_metrics({"train/loss": avg_loss}, step=global_step)
                    running_loss = 0.0

                    # Do validation
                    valid_loss = validate(model, eval_dataloader, loss_fn, logger, device, config, tokenizer)
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
    config_file_path = os.path.join(os.path.dirname(__file__), "test_llama_fine_tuning_pure_torch.yaml")
    config = generate_config(TrainingConfig, config_file_path)

    # Reproducibility setup
    repro_manager = ReproducibilityManager(config)
    repro_manager.setup()

    # Logger setup
    logger = TrainingLogger(config)

    # Checkpoint manager setup
    checkpoint_manager = CheckpointManager(config, logger)

    # Device setup
    if config.use_tt:
        xr.runtime.set_device_type("TT")
        device = xm.xla_device()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Start training
    train(config, device, logger, checkpoint_manager)
