# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import random
import numpy as np
import torch
import traceback
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from blacksmith.datasets.torch.llama.sst_dataset import SSTDataset
from blacksmith.experiments.torch.llama.configs import TrainingConfig
from blacksmith.models.torch.huggingface.hf_models import get_model, TextModelWrapper
from blacksmith.tools.cli import generate_config
from blacksmith.tools.torch_helpers import show_examples, collect_examples


def validate(model, val_data_loader, loss_fn, device, config, tokenizer=None):
    print(f"\n=== Starting Validation ===")
    total_val_loss = 0.0
    num_val_batches = 0
    collected_examples = []

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
        print(f"\n=== Validation Examples (Random samples) ===")
        show_examples(collected_examples, tokenizer, config)

    avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
    print(f"Average validation loss: {avg_val_loss}")
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


def train(config, device):
    # Get model
    model = get_model(config)
    model = model.to(device)

    # Initialize wandb
    run = wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=vars(config), save_code=True)
    run.watch(model, log=config.wandb_watch_mode, log_freq=config.wandb_log_freq)

    # Get dataset
    dataset = SSTDataset(config)
    tokenizer = dataset.tokenizer
    train_set, eval_set = dataset.load_tokenized_data()

    train_data_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn_with_shifted_labels
    )
    val_data_loader = DataLoader(
        eval_set, batch_size=config.batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn_with_shifted_labels
    )

    # Get optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Get loss function
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=config.ignored_index)

    # Training
    global_step = 0
    running_loss = 0.0

    try:
        model.train()
        for epoch in range(config.num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{config.num_epochs} ===")

            for batch in tqdm(train_data_loader, desc="Training"):
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

                print(f"Loss: {loss.item():.6f}")
                running_loss += loss.item()

                # Backward pass
                loss.backward()

                # Optimizer step on CPU
                optimizer.step()

                # Sync XLA device
                if config.use_tt:
                    torch_xla.sync(wait=True)

                global_step += 1

                if global_step % config.logging_steps == 0:
                    avg_loss = running_loss / config.logging_steps
                    run.log({"train/loss": avg_loss, "step": global_step})
                    running_loss = 0.0
                    model.eval()

                    # Validation phase
                    avg_val_loss = validate(model, val_data_loader, loss_fn, device, config, tokenizer)

                    run.log({"epoch": epoch + 1, "val/loss": avg_val_loss, "step": global_step})

                    model.train()
                    if config.save_strategy == "steps":
                        checkpoint_path = os.path.join(
                            config.output_dir, "checkpoints", f"checkpoint-{global_step}.pth"
                        )
                        torch.save(model.state_dict(), checkpoint_path)

            if config.save_strategy == "epoch":
                checkpoint_path = os.path.join(config.output_dir, "checkpoints", f"checkpoint-{epoch+1}.pth")
                torch.save(model.state_dict(), checkpoint_path)

        # Save final model
        final_model_path = os.path.join(config.output_dir, "checkpoints", "final_model.pth")
        torch.save(model.state_dict(), final_model_path)

        if config.model_to_wandb:
            artifact = wandb.Artifact("final_model", type="model")
            artifact.add_file(final_model_path)
            run.log_artifact(artifact)

    except Exception as e:
        error_msg = f"Training failed with error: {str(e)}"
        traceback_str = traceback.format_exc()
        print(error_msg)
        print(traceback_str)
        raise
    finally:
        wandb.finish()


if __name__ == "__main__":
    config_file_path = os.path.join(os.path.dirname(__file__), "test_llama_fine_tuning_pure_torch.yaml")
    config = generate_config(TrainingConfig, config_file_path)

    os.makedirs(os.path.join(config.output_dir, "checkpoints"), exist_ok=True)

    # Device setup
    if config.use_tt:
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr

        xr.runtime.set_device_type("TT")
        device = xm.xla_device()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train(config, device)
