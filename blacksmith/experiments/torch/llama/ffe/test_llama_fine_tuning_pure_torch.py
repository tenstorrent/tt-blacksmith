# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import traceback
from pathlib import Path

import torch
import wandb
from tqdm import tqdm

from blacksmith.datasets.torch.dataset_utils import get_dataset
from blacksmith.experiments.torch.llama.configs import TrainingConfig
from blacksmith.experiments.torch.llama.ffe.utils import TextModelWrapper, get_model
from blacksmith.tools.cli import generate_config, parse_cli_options


def show_examples(examples, tokenizer, config):
    for i, example in enumerate(examples):
        print(f"\nExample {i+1} (from batch {example['batch_num']}):")

        input_ids = example["input_ids"]
        expected = example["expected"]
        predicted = example["predicted"]

        valid_mask = expected != config.ignored_index
        if not valid_mask.any():
            print(f"  No valid tokens (all {config.ignored_index})")
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
        print(f"Accuracy: {correct.item():.3f} ({(valid_targets == valid_preds).sum()}/{len(valid_targets)})")


def validate(model, val_data_loader, loss_fn, device, config, vocab_size, dtype, tokenizer=None):
    print(f"\n=== Starting Validation ===")
    total_val_loss = 0.0
    num_val_batches = 0
    collected_examples = []
    max_examples = 10

    with torch.no_grad():
        for batch in tqdm(val_data_loader, desc="Validation"):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            expected_output = batch["labels"]

            # Forward pass + loss
            if config.use_tt:
                inputs = [input_ids, attention_mask]
                logits = model(*inputs)[0]  # logits is [V, N]
                labels_for_loss = prepare_labels(expected_output, vocab_size, dtype)
                loss = loss_fn(logits, labels_for_loss)[0]
                predictions = logits.t().contiguous().argmax(dim=-1)  # [N]
                predictions = predictions.view(expected_output.shape)  # [B, T]
            else:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                expected_output = expected_output.to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = loss_fn(logits.view(-1, vocab_size), expected_output.view(-1))
                predictions = logits.argmax(dim=-1)  # [B, T]

            total_val_loss += loss.item()
            num_val_batches += 1

            if len(collected_examples) < max_examples:
                batch_size = expected_output.shape[0]
                import random

                sample_indices = random.sample(
                    range(batch_size), min(batch_size, max_examples - len(collected_examples))
                )

                for idx in sample_indices:
                    collected_examples.append(
                        {
                            "input_ids": input_ids[idx],
                            "expected": expected_output[idx],
                            "predicted": predictions[idx],
                            "batch_num": num_val_batches,
                        }
                    )

    print(f"\n=== Validation Examples (Random samples) ===")
    show_examples(collected_examples, tokenizer, config)
    avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
    print(f"Average validation loss: {avg_val_loss}")
    return avg_val_loss


def prepare_labels(expected_output: torch.Tensor, vocab_size: int, dtype: torch.dtype) -> torch.Tensor:
    expected_output_flat = expected_output.view(-1)  # [N]
    mask = expected_output_flat != -100
    N = expected_output_flat.numel()

    labels_rows = torch.zeros(N, vocab_size, dtype=dtype, device=expected_output_flat.device)
    if mask.any():
        labels_rows[mask] = torch.nn.functional.one_hot(expected_output_flat[mask], num_classes=vocab_size).to(dtype)

    labels_for_loss = labels_rows.transpose(0, 1).contiguous().to(dtype)
    return labels_for_loss


def train(config, model, tokenizer, train_data_loader, val_data_loader):
    run = wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=vars(config), save_code=True)
    run.watch(model, log=config.wandb_watch_mode, log_freq=config.wandb_log_freq)
    device = None

    if config.use_tt:
        import forge
        from forge._C import DataFormat
        from forge._C.runtime.experimental import DeviceSettings, configure_devices
        from forge.config import CompilerConfig

        compiler_cfg = CompilerConfig()
        if config.dtype == "torch.bfloat16":
            forge_dtype = DataFormat.Float16_b
            compiler_cfg.default_df_override = DataFormat.Float16_b
            dtype = torch.bfloat16
        elif config.dtype == "torch.float32":
            forge_dtype = DataFormat.Float32
            dtype = torch.float32
        else:
            raise ValueError(f"Invalid dtype: {config.dtype}")

        # Enable program cache on all devices
        settings = DeviceSettings()
        settings.enable_program_cache = True
        configure_devices(device_settings=settings)

        # Create a sample input for compilation
        input_prompt = "Hey how are you doing today?"
        inputs = tokenizer(
            input_prompt,
            return_tensors="pt",
            max_length=config.max_length,
            padding="max_length",
            truncation=True,
        )

        input_ids = inputs["input_ids"]
        input_ids = input_ids.repeat(config.batch_size, 1)
        attn_mask = inputs["attention_mask"]
        attn_mask = attn_mask.repeat(config.batch_size, 1)
        sample_inputs = [input_ids, attn_mask]

        framework_model = TextModelWrapper(model=model, text_embedding=model.model.model.embed_tokens)
        tt_optimizer = forge.optimizers.Adam(learning_rate=config.learning_rate)
        compiled_model = forge.compile(
            framework_model, sample_inputs, optimizer=tt_optimizer, training=True, compiler_cfg=compiler_cfg
        )
    else:
        torch_optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        device = torch.device("cuda")
        model.to(device)

    vocab_size = model.model.config.vocab_size

    if config.use_tt:
        # TODO: Remove this once softmax is fixed
        from blacksmith.experiments.torch.llama.ffe.loss import CrossEntropyLoss

        loss_tt = CrossEntropyLoss(name="cross_entropy_loss", dtype=forge_dtype)

        N_sample = config.batch_size * config.max_length
        loss_predictions_sample = torch.rand(vocab_size, N_sample, dtype=dtype).requires_grad_(True)
        loss_labels_sample = torch.zeros(vocab_size, N_sample, dtype=dtype)
        loss_inputs = [loss_predictions_sample, loss_labels_sample]
        loss_inputs = forge.tensor.to_forge_tensors(loss_inputs)

        tt_loss = forge.compile(
            loss_tt,
            sample_inputs=loss_inputs,
            attach_to=compiled_model,
            training=True,
        )
    else:
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=config.ignored_index)

    global_step = 0
    running_loss = 0.0
    try:
        for epoch in range(config.num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{config.num_epochs} ===")
            model.train()

            for batch in tqdm(train_data_loader, desc="Training"):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                expected_output = batch["labels"]

                if config.use_tt:
                    inputs = [input_ids, attention_mask]
                    # Forward pass
                    logits = compiled_model(*inputs)[0]
                    labels_for_loss = prepare_labels(expected_output, vocab_size, model.dtype)
                    # Loss
                    loss = tt_loss(logits, labels_for_loss)[0]
                    running_loss += loss.item()
                    print(f"Loss: {loss}")
                    # Backward pass
                    tt_loss.backward()
                    # Optimizer step
                    tt_optimizer.step()
                else:
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    expected_output = expected_output.to(device)
                    # Forward pass
                    outputs = model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    # Loss
                    loss = loss_fn(logits.view(-1, vocab_size), expected_output.view(-1))
                    running_loss += loss.item()
                    print(f"Loss: {loss}")
                    # Backward pass
                    loss.backward()
                    # Optimizer step
                    torch_optimizer.step()
                    torch_optimizer.zero_grad()

                global_step += 1

                # Log training loss at specified intervals
                if global_step % config.logging_steps == 0:
                    avg_loss = running_loss / config.logging_steps
                    run.log({"train/loss": avg_loss, "step": global_step})
                    running_loss = 0.0

                    # Validation phase
                    if config.use_tt:
                        avg_val_loss = validate(
                            compiled_model, val_data_loader, tt_loss, device, config, vocab_size, dtype, tokenizer
                        )
                    else:
                        model.eval()
                        avg_val_loss = validate(
                            model, val_data_loader, loss_fn, device, config, vocab_size, None, tokenizer
                        )
                    run.log({"epoch": epoch + 1, "val/loss": avg_val_loss, "step": global_step})

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
        run.alert(title="Training Failed", text=error_msg, level=wandb.AlertLevel.ERROR)
        run.log({"error": error_msg, "traceback": traceback_str})
        raise
    finally:
        wandb.finish()


if __name__ == "__main__":
    default_config = Path(__file__).parent / "test_llama_fine_tuning_pure_torch.yaml"
    args = parse_cli_options(default_config=default_config)
    config: TrainingConfig = generate_config(TrainingConfig, args.config)

    os.makedirs(os.path.join(config.output_dir, "checkpoints"), exist_ok=True)

    model = get_model(config)

    # Load dataset
    train_dataset = get_dataset(config=config, split="train")
    train_dataloader = train_dataset.get_dataloader()

    eval_dataset = get_dataset(config=config, split="validation")
    eval_dataloader = eval_dataset.get_dataloader()

    if config.do_train:
        train(config, model, train_dataset.tokenizer, train_dataloader, eval_dataloader)
