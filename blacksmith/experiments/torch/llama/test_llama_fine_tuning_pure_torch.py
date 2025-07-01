# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import traceback
import re
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from blacksmith.datasets.torch.llama.sst_dataset import SSTDataset
from blacksmith.experiments.torch.llama.configs import TrainingConfig
from blacksmith.models.torch.huggingface.hf_models import get_model
from blacksmith.tools.cli import generate_config
from blacksmith.datasets.torch.llama.sst_utils import VALUE2LBL


def validate(model, val_data_loader, loss_fn, device, config):
    """Run validation and return average validation loss."""
    model.eval()
    total_val_loss = 0.0
    num_val_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_data_loader, desc="Validation", leave=False):
            input_ids = batch["input_ids"]
            expected_output = batch["labels"]
            
            input_ids = input_ids.to(device)
            attention_mask = batch["attention_mask"].to(device)
            expected_output = expected_output.to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Calculate loss
            loss = loss_fn(logits.view(-1, model.config.vocab_size), expected_output.view(-1))
            total_val_loss += loss.item()
            num_val_batches += 1
    
    avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
    return avg_val_loss


def train(config, model, train_data_loader, val_data_loader=None):
    run = wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=vars(config), save_code=True)
    run.watch(model, log=config.wandb_watch_mode, log_freq=config.wandb_log_freq)

    if config.use_tt:
        import forge

        tt_optimizer = forge.optimizers.AdamW()
        sample_inputs = [torch.randint(0, model.config.vocab_size, (config.batch_size, config.max_length))]
        compiled_model = forge.compile(model, sample_inputs, optimizer=tt_optimizer, training=True)
    else:
        torch_optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        device = torch.device("cuda")
        model.to(device)

    # Create a torch loss and leave on CPU
    # Can be changed when https://github.com/tenstorrent/tt-metal/issues/18997 resolved
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # Variables for tracking best validation loss
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    try:
        global_step = 0
        running_loss = 0.0
        log_every_n_steps = config.logging_steps

        for epoch in range(config.num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{config.num_epochs} ===")
            
            # Training phase
            model.train()
            epoch_train_loss = 0.0
            num_train_batches = 0

            for batch in tqdm(train_data_loader, desc="Training"):
                input_ids = batch["input_ids"]
                expected_output = batch["labels"]

                if config.use_tt:
                    logits = compiled_model(input_ids)[0]
                else:
                    input_ids = input_ids.to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    expected_output = expected_output.to(device)

                    # Forward pass
                    outputs = model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits

                # Calculate loss
                loss = loss_fn(logits.view(-1, model.config.vocab_size), expected_output.view(-1))
                running_loss += loss.item()
                epoch_train_loss += loss.item()
                num_train_batches += 1

                # Backward pass
                loss.backward()
                
                # Optimizer step
                if config.use_tt:
                    compiled_model.backward()
                    tt_optimizer.step()
                else:
                    torch_optimizer.step()
                    torch_optimizer.zero_grad()

                global_step += 1
                
                # Log training loss at specified intervals
                if global_step % log_every_n_steps == 0:
                    avg_loss = running_loss / log_every_n_steps
                    run.log({"train/loss": avg_loss, "step": global_step})
                    running_loss = 0.0

                    # Validation phase
                    print("Running validation...")
                    avg_val_loss = validate(model, val_data_loader, loss_fn, device, config)
                    
                    # Log validation loss
                    run.log({
                        "epoch": epoch + 1,
                        "train/epoch_loss": avg_loss,
                        "val/epoch_loss": avg_val_loss,
                        "step": global_step
                    })
                    
                    print(f"Epoch {epoch + 1} - Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                    
                    # Early stopping logic
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        epochs_without_improvement = 0

                        # Save best model
                        best_model_path = os.path.join(config.output_dir, "checkpoints", "best_model.pth")
                        torch.save(model.state_dict(), best_model_path)
                        print(f"New best validation loss: {best_val_loss:.4f} - Model saved")
                    else:
                        epochs_without_improvement += 1
                        print(f"No improvement for {epochs_without_improvement} epochs")
                        
                        if epochs_without_improvement >= 3:
                            print(f"Early stopping triggered after no improvement")
                            break

                    # testing
                    if avg_loss < 0.1:
                        break

            # Calculate average training loss for the epoch
            avg_train_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else 0.0

            # Save checkpoint at the end of each epoch
            if config.save_strategy == "epoch":
                checkpoint_path = os.path.join(config.output_dir, "checkpoints", f"checkpoint-{epoch+1}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                
                # Save additional training state for resuming
                training_state = {
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'best_val_loss': best_val_loss,
                    'optimizer_state_dict': torch_optimizer.state_dict() if not config.use_tt else None,
                    'model_state_dict': model.state_dict(),
                    'config': vars(config)
                }
                training_state_path = os.path.join(config.output_dir, "checkpoints", f"training_state-{epoch+1}.pth")
                torch.save(training_state, training_state_path)

                # Log to wandb
                # artifact = wandb.Artifact(f"checkpoint-{epoch+1}", type="model")
                # artifact.add_file(checkpoint_path)
                # run.log_artifact(artifact)

        # Save final model
        final_model_path = os.path.join(config.output_dir, "checkpoints", "final_model.pth")
        torch.save(model.state_dict(), final_model_path)

        # Final summary
        if avg_val_loss is not None:
            run.summary['final_train_loss'] = avg_train_loss
            run.summary['final_val_loss'] = avg_val_loss
            run.summary['best_val_loss'] = best_val_loss
            print(f"\nTraining Complete!")
            print(f"Final Train Loss: {avg_train_loss:.4f}")
            print(f"Final Val Loss: {avg_val_loss:.4f}")
            print(f"Best Val Loss: {best_val_loss:.4f}")
        else:
            run.summary['final_train_loss'] = avg_train_loss
            print(f"\nTraining Complete! Final Train Loss: {avg_train_loss:.4f}")

        # artifact = wandb.Artifact("final_model", type="model")
        # artifact.add_file(final_model_path)
        # run.log_artifact(artifact)

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

    # print(dataset.tokenizer.decode(train_set[0]["labels"], skip_special_tokens=True))
    # target_text = dataset.tokenizer.decode([token if token != -100 else dataset.tokenizer.pad_token_id for token in train_set[22]["labels"]])

    if config.do_train:
        train(config, model, train_data_loader, eval_data_loader)
