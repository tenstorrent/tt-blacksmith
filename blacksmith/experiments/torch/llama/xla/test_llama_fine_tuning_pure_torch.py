# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import traceback

import torch
import torch_xla
import torch_xla.runtime as xr
from tqdm import tqdm

from blacksmith.datasets.torch.dataset_utils import get_dataset
from blacksmith.experiments.torch.llama.configs import TrainingConfig
from blacksmith.models.torch.huggingface.hf_models import get_model
from blacksmith.tools.cli import generate_config
from blacksmith.tools.torch_helpers import show_examples, collect_examples, collate_fn_for_causal_lm
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.checkpoints_manager import CheckpointManager
from blacksmith.tools.device_manager import DeviceManager, ParallelStrategy
from blacksmith.tools.reproducibility_manager import ReproducibilityManager
from blacksmith.tools.workaround_utils import cross_entropy_loss, transform_labels


def validate(model, val_data_loader, loss_fn, logger, device, config, tokenizer=None):
    logger.info("Starting validation...")
    total_val_loss = 0.0
    num_val_batches = 0
    collected_examples = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(val_data_loader, desc="Validation"):
            import time
            start_time = time.time()
            # Zero out gradients
            #optimizer.zero_grad()

            # print the shapes of batch tensors
            print(f"Validation batch tensor shapes: { {k: v.shape for k, v in batch.items()} }", flush=True)

            expected_output, labels_mask = transform_labels(
                    batch, config.ignored_index, model.model.config.vocab_size
            )
            batch = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "expected_output": expected_output,
                "labels_mask": labels_mask,
            }

            batch = device_manager.prepare_batch(batch)
            device_manager.shard_model(model)
            # Forward pass
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = outputs.logits

            # Shift logits for causal LM: predict next token
            # logits[:, :-1] predicts tokens at positions 1:
            shift_logits = logits[:, :-1, :].contiguous()

            loss = cross_entropy_loss(shift_logits, batch["expected_output"], batch["labels_mask"])

            # Predictions
            predictions = shift_logits.argmax(dim=-1)
            if config.use_tt:
                torch_xla.sync(wait=True)

            total_val_loss += loss.item()

            end_time = time.time()
            print(f"Validation Step time: {end_time - start_time} seconds", flush=True)

            num_val_batches += 1

            if config.print_examples:
                #print("Stampam examples...", flush=True)
                collected_examples = collect_examples(
                    batch_size=expected_output.shape[0],
                    collected_examples=collected_examples,
                    max_examples=10,
                    input_ids=batch["input_ids"],
                    expected_output=batch["expected_output"],
                    predictions=predictions,
                    num_val_batches=num_val_batches,
                )

    if config.print_examples and tokenizer is not None:
        logger.info("Printing validation examples...")
        show_examples(collected_examples, tokenizer, config, logger)

    avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
    logger.info(f"Average validation loss: {avg_val_loss}")
    return avg_val_loss

def training_step_inner(batch, model, loss_fn):
    output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = output.logits
    # Shift logits for causal LM: predict next token
    shift_logits = logits[:, :-1, :].contiguous()
    # Compute loss
    loss = cross_entropy_loss(shift_logits, batch["expected_output"], batch["labels_mask"])
    # Backward pass
    loss.backward()
    # Return detached loss - function scope cleans up logits, shift_logits, output
    return loss.detach()

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

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=config.ignored_index)

    # Load checkpoint if needed
    if config.resume_from_checkpoint:
        checkpoint_manager.load_checkpoint(model, optimizer)

    # Load dataset
    train_dataset = get_dataset(config=config, split="train", collate_fn=collate_fn_for_causal_lm)
    train_dataloader = train_dataset.get_dataloader()
    logger.info(f"Loaded {config.dataset_id} dataset. Train dataset size: {len(train_dataloader)*config.batch_size}")

    eval_dataset = get_dataset(config=config, split="validation", collate_fn=collate_fn_for_causal_lm)
    eval_dataloader = eval_dataset.get_dataloader()
    logger.info(f"Loaded {config.dataset_id} dataset. Eval dataset size: {len(eval_dataloader)*config.batch_size}")

    tokenizer = train_dataset.tokenizer

    global_step = 0
    running_loss = 0.0
    try:
        model.train()
        for epoch in range(config.num_epochs):

            for batch in tqdm(train_dataloader, desc="Training"):
                import time
                start_time = time.time()
                # Zero out gradients
                optimizer.zero_grad()

                # print the shapes of batch tensors
                print(f"Training batch tensor shapes: { {k: v.shape for k, v in batch.items()} }", flush=True)

                expected_output, labels_mask = transform_labels(
                        batch, config.ignored_index, model.model.config.vocab_size
                )
                batch = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "expected_output": expected_output,
                    "labels_mask": labels_mask,
                }

                batch = device_manager.prepare_batch(batch)
                device_manager.shard_model(model)

                loss_tensor = training_step_inner(batch, model, cross_entropy_loss)

                if config.use_tt:
                    torch_xla.sync(wait=True)

                # Optimizer step
                device_manager.optimizer_step(optimizer)

                running_loss += loss_tensor.item()
                end_time = time.time()
                print(f"Step {global_step}, Loss: {loss_tensor.item()}", flush=True)
                print(f"Step time: {end_time - start_time} seconds", flush=True)
                #if global_step > 30:
                #    exit(0)
                #global_step += 1
                #exit(0)
                xr.clear_computation_cache()
                if global_step % config.steps_freq == 0:
                    #continue
                    avg_loss = running_loss / config.steps_freq
                    logger.log_metrics({"train/loss": avg_loss}, commit=False, step=global_step)
                    running_loss = 0.0

                    # Do validation
                    valid_loss = validate(
                        model, eval_dataloader, cross_entropy_loss, logger, device_manager.device, config, tokenizer
                    )
                    logger.log_metrics({"val/loss": valid_loss}, step=global_step)
                    model.train()
                    xr.clear_computation_cache()
                    #exit(0)
                    # Save step checkpoint
                    if checkpoint_manager.should_save_checkpoint(global_step):
                        continue
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
    parser = argparse.ArgumentParser(description="LLaMA Fine-Tuning with PyTorch and XLA")
    parser.add_argument("--config", type=str, required=False, help="Path to the configuration YAML file.")
    args = parser.parse_args()
    if args.config:
        config_file_path = args.config
    else:
        config_file_path = os.path.join(os.path.dirname(__file__), "lora/test_lora.yaml")
    config = generate_config(TrainingConfig, config_file_path)

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
