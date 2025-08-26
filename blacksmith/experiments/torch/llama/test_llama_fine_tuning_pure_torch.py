# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import traceback

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from blacksmith.datasets.torch.llama.sst_dataset import SSTDataset
from blacksmith.experiments.torch.llama.configs import TrainingConfig
from blacksmith.models.torch.huggingface.hf_models import get_model, TextModelWrapper
from blacksmith.tools.cli import generate_config


def validate(model, val_data_loader, loss_fn, device, config, vocab_size):
    total_val_loss = 0.0
    num_val_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_data_loader, desc="Validation"):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            expected_output = batch["labels"]

            # Forward pass
            if config.use_tt:
                inputs = [input_ids, attention_mask]
                logits = model(*inputs)[0]
            else:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

            # Calculate loss
            loss = loss_fn(logits.view(-1, vocab_size), expected_output.view(-1))
            total_val_loss += loss.item()
            num_val_batches += 1

    avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
    return avg_val_loss


def train(config, model, tokenizer, train_data_loader, val_data_loader):
    run = wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=vars(config), save_code=True)
    run.watch(model, log=config.wandb_watch_mode, log_freq=config.wandb_log_freq)
    device = None

    torch_optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    if config.use_tt:
        import forge
        from forge.config import CompilerConfig
        from forge._C import DataFormat
        from forge._C.runtime.experimental import configure_devices, DeviceSettings

        compiler_cfg = CompilerConfig()
        if config.dtype == "torch.bfloat16":
            compiler_cfg.default_df_override = DataFormat.Float16

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
        compiled_model = forge.compile(
            framework_model, sample_inputs, optimizer=torch_optimizer, training=True, compiler_cfg=compiler_cfg
        )
    else:
        device = torch.device("cuda")
        model.to(device)

    # Create a torch loss and leave on CPU
    # Can be changed when https://github.com/tenstorrent/tt-metal/issues/18997 resolved
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    global_step = 0
    running_loss = 0.0
    vocab_size = model.model.config.vocab_size
    try:
        for epoch in range(config.num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{config.num_epochs} ===")
            model.train()

            for batch in tqdm(train_data_loader, desc="Training"):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                expected_output = batch["labels"]

                # Forward pass
                if config.use_tt:
                    inputs = [input_ids, attention_mask]
                    logits = compiled_model(*inputs)[0]
                else:
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    expected_output = expected_output.to(device)
                    outputs = model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits

                # Calculate loss
                loss = loss_fn(logits.view(-1, vocab_size), expected_output.view(-1))
                running_loss += loss.item()

                # Backward pass
                loss.backward()

                if config.use_tt:
                    compiled_model.backward()

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
                        avg_val_loss = validate(compiled_model, val_data_loader, loss_fn, device, config, vocab_size)
                    else:
                        model.eval()
                        avg_val_loss = validate(model, val_data_loader, loss_fn, device, config, vocab_size)
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
    config_file_path = os.path.join(os.path.dirname(__file__), "test_llama_fine_tuning_pure_torch.yaml")
    config = generate_config(TrainingConfig, config_file_path)

    os.makedirs(os.path.join(config.output_dir, "checkpoints"), exist_ok=True)

    model = get_model(config)

    dataset = SSTDataset(config)
    train_set, eval_set = dataset.load_tokenized_data()
    train_data_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, drop_last=True)
    eval_data_loader = DataLoader(eval_set, batch_size=config.batch_size, shuffle=False, drop_last=True)

    if config.do_train:
        train(config, model, dataset.tokenizer, train_data_loader, eval_data_loader)
