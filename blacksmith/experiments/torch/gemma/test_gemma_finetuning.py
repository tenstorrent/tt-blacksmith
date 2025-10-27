# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import traceback

import torch
from torch.utils.data import DataLoader
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from tqdm import tqdm

from blacksmith.experiments.torch.gemma.configs import TrainingConfig
from blacksmith.datasets.torch.llama.sst_dataset import SSTDataset
from blacksmith.models.torch.huggingface.hf_models import get_model
from blacksmith.tools.cli import generate_config
from blacksmith.tools.reproducibility_manager import ReproducibilityManager
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.checkpoints_manager import CheckpointManager


def show_examples(examples, tokenizer):

    for i, example in enumerate(examples):
        if i > 10:
            break
        print(f"\nExample {i+1} (from batch {example['batch_num']}):")

        input_ids = example["input_ids"].to("cpu")
        expected = example["expected"].to("cpu")
        predicted = example["predicted"].to("cpu")

        valid_mask = expected != -100
        if not valid_mask.any():
            print(f"  No valid tokens (all -100)")
            continue

        valid_targets = expected[valid_mask]
        valid_preds = predicted[valid_mask]

        show_len = min(10, len(valid_targets))
        target_tokens = valid_targets[:show_len].tolist()
        pred_tokens = valid_preds[:show_len].tolist()

        print(f"Target IDs:  {target_tokens}")
        print(f"Pred IDs:    {pred_tokens}")

        try:
            target_text = tokenizer.decode(target_tokens, skip_special_tokens=False)
            pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=False)
            input_text = tokenizer.decode(input_ids, skip_special_tokens=True)
            print(f"Input text:  '{input_text}'")
            print(f"Target text: '{target_text}'")
            print(f"Pred text:   '{pred_text}'")
        except Exception as e:
            print(f"  (Could not decode text: {e})")

        correct = (valid_targets == valid_preds).float().mean()
        print(
            f"Accuracy: {correct.item():.3f} ({(valid_targets == valid_preds).sum()}/{len(valid_targets)})"
        )


def validate(model, val_data_loader, loss_fn, device, config, tokenizer=None):
    print(f"\n=== Starting Validation ===")
    model.eval()
    total_val_loss = 0.0
    num_val_batches = 0
    collected_examples = []
    max_examples = 10

    with torch.no_grad():
        for batch in tqdm(val_data_loader, desc="Validation"):
            if num_val_batches > 10:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            expected_output = batch["labels"].to(device)

            # Forward pass + loss
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            # move logits to cpu
            logits = logits.to("cpu")
            expected_output = expected_output.to("cpu")
            input_ids = input_ids.to("cpu")
            attention_mask = attention_mask.to("cpu")

            # Shift logits and labels for causal LM: predict next token
            # logits[:, :-1] predicts tokens at positions 1:, so compare with labels[:, 1:]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = expected_output[:, 1:].contiguous()

            loss = loss_fn(
                shift_logits.view(-1, model.model.config.vocab_size),
                shift_labels.view(-1),
            )
            total_val_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            shift_predictions = predictions[:, :-1].contiguous()
            num_val_batches += 1

            if len(collected_examples) < max_examples:
                batch_size = expected_output.shape[0]
                import random

                sample_indices = random.sample(
                    range(batch_size),
                    min(batch_size, max_examples - len(collected_examples)),
                )

                for idx in sample_indices:
                    collected_examples.append(
                        {
                            "input_ids": input_ids[idx],
                            "expected": shift_labels[idx],
                            "predicted": shift_predictions[idx],
                            "batch_num": num_val_batches,
                        }
                    )

    print(f"\n=== Validation Examples (Random samples) ===")
    show_examples(collected_examples, tokenizer)
    avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
    print(f"Average validation loss: {avg_val_loss}")
    return avg_val_loss


def train(
    config: TrainingConfig,
    device: torch.device,
    logger: TrainingLogger,
    checkpoint_manager: CheckpointManager,
):
    logger.info("Starting training...")

    # Load model
    model = get_model(config, device)
    logger.info(f"Loaded {config.model_name} model.")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    # Load checkpoint if needed
    if config.resume_from_checkpoint:
        checkpoint_manager.load_checkpoint()

    # Load dataset
    dataset = SSTDataset(config)
    tokenizer = dataset.tokenizer
    train_set, eval_set = dataset.load_tokenized_data()

    train_data_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True, drop_last=True
    )
    val_data_loader = DataLoader(
        eval_set, batch_size=config.batch_size, shuffle=False, drop_last=True
    )

    # Init training components (optimizer, lr scheduler, etc.)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    global_step = 0
    running_loss = 0.0
    try:
        for epoch in range(config.num_epochs):
            model.train()

            for batch in tqdm(train_data_loader):
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Forward pass
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                logits = outputs.logits

                # Shift logits and labels for causal LM: predict next token
                # logits[:, :-1] predicts tokens at positions 1:, so compare with labels[:, 1:]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                loss = loss_fn(
                    shift_logits.view(-1, model.model.config.vocab_size),
                    shift_labels.view(-1),
                )
                # loss = output.loss
                loss_cpu = loss.item()
                print(f"Loss: {loss_cpu:.6f} {loss.device}")

                running_loss += loss_cpu

                # Backward pass
                loss.backward()

                # Update parameters
                if config.use_tt:
                    xm.optimizer_step(optimizer)
                    torch_xla.sync(wait=True)
                else:
                    optimizer.step()

                if global_step % config.steps_freq == 0:
                    avg_loss = (
                        running_loss / config.steps_freq
                        if global_step > 0
                        else running_loss
                    )
                    logger.log_metrics({"train/loss": avg_loss}, step=global_step)
                    running_loss = 0.0
                
                # Validation phase
                if global_step % config.val_steps_freq == 0:
                    avg_val_loss = validate(
                        model, val_data_loader, loss_fn, device, config, tokenizer
                    )

                    logger.log_metrics(
                        {"epoch": epoch + 1, "val/loss": avg_val_loss},
                        step=global_step,
                    )

                if checkpoint_manager.should_save_checkpoint(global_step):
                    checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)

                global_step += 1

            if checkpoint_manager.should_save_checkpoint(global_step, epoch):
                checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)

        # Save final model
        final_model_path = checkpoint_manager.save_checkpoint(
            model, global_step, epoch, optimizer
        )
        logger.log_artifact(
            final_model_path, artifact_type="model", name="final_model.pth"
        )

    except Exception as e:
        traceback_str = traceback.format_exc()
        logger.error(f"Training failed with error: {str(e)}", traceback_str)
        raise
    finally:
        logger.finish()


if __name__ == "__main__":
    # Config setup
    config_file_path = os.path.join(
        os.path.dirname(__file__), "test_gemma_finetuning.yaml"
    )
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
