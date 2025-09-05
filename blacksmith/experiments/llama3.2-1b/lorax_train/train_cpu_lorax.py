#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Real LoRA training on Llama with TinyStories dataset.
Fun, engaging stories instead of boring Wikipedia!
Focuses on MLP layers only as originally requested.
NOW RUNNING ON CPU DEVICE! 🚀
"""
import os
import warnings

# Disable TT plugins for CPU-only training
os.environ['TORCH_XLA_DISABLE_DEVICE_PLUGINS'] = '1'

# Force CPU-only execution BEFORE importing JAX
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import optax
import numpy as np
from transformers import FlaxAutoModelForCausalLM, AutoTokenizer, AutoConfig

from datasets import load_dataset
import lorax
from lorax import LORA_FULL, LORA_FREEZE

# Import blacksmith dataset and config classes
from blacksmith.datasets.torch.llama.sst_dataset import SSTDataset
from blacksmith.experiments.torch.llama.configs import TrainingConfig

# Import wandb for experiment tracking
import wandb


MODEL_NAME = "Erland/Llama-3.2-1B-JAX"

# Wandb configuration
DEFAULT_EXPERIMENT_NAME = "Llama-CPU-LoRA-Training"
DEFAULT_RUN_NAME = "llama-3.2-1b-sst2-cpu-lorax"

def setup_wandb(training_config):
    """Setup wandb for experiment tracking."""
    wandb_run = wandb.init(
        project=DEFAULT_EXPERIMENT_NAME,
        name=DEFAULT_RUN_NAME,
        config={
            "model_name": training_config.model_name,
            "dataset_id": training_config.dataset_id,
            "max_length": training_config.max_length,
            "learning_rate": training_config.learning_rate,
            "batch_size": training_config.batch_size,
            "num_epochs": training_config.num_epochs,
            "lora_rank": 4,
            "lora_target_modules": ["mlp.gate_proj.kernel", "mlp.up_proj.kernel", "mlp.down_proj.kernel"],
            "device": "cpu",
            "framework": "jax_lorax"
        },
    )
    print(f"🔗 Started wandb run: {wandb_run.name}")
    return wandb_run

def log_to_wandb(data_dict, step=None):
    """Helper function to log data to wandb."""
    wandb.log(data_dict, step=step)




def create_batches(data, batch_size=8):
    """Create training batches."""
    num_batches = len(data) // batch_size
    batched_data = data[: num_batches * batch_size].reshape(num_batches, batch_size, -1)
    return batched_data



def main():
    print("🚀 Starting Llama 3.2-1B LoRA Training on SST Dataset")
    print("🎯 CPU Device Status:")
    print(f"   Available devices: {jax.devices()}")
    print(f"   Default backend: {jax.default_backend()}")
    cpu_device = jax.devices("cpu")[0]

    # Quick device test
    test_array = jnp.array([1.0, 2.0, 3.0])
    print(f"   Test computation device: {test_array.device}")
    print()

    print("🤖 Loading Llama 3.2-1B model...")
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.use_cache = False
    config.num_hidden_layers = 16
    config.dtype = jnp.bfloat16

    model = FlaxAutoModelForCausalLM.from_pretrained(MODEL_NAME, config=config, from_pt=False, dtype=jnp.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create training configuration for SST dataset
    training_config = TrainingConfig(
        model_name=MODEL_NAME,
        dataset_id="stanfordnlp/sst2",
        max_length=128,
        learning_rate=1e-4,
        batch_size=32,
        num_epochs=5
    )
    
    # Setup wandb for experiment tracking
    wandb_run = setup_wandb(training_config)

    print("📚 Loading SST dataset...")
    dataset_loader = SSTDataset(training_config)
    train_dataset, validation_dataset = dataset_loader.load_tokenized_data()
    
    print(f"📊 Dataset sizes:")
    print(f"   Training: {len(train_dataset)} samples")
    print(f"   Validation: {len(validation_dataset)} samples")
    print("🔄 Converting training data to JAX arrays...")
    
    # Extract input_ids, attention_mask, and labels from the dataset
    train_input_ids = []
    train_attention_mask = []
    train_labels = []
    
    for item in train_dataset:
        train_input_ids.append(np.array(item['input_ids']))
        train_attention_mask.append(np.array(item['attention_mask']))
        train_labels.append(np.array(item['labels']))
    
    # Convert to JAX arrays
    train_input_ids = jnp.array(train_input_ids)
    train_attention_mask = jnp.array(train_attention_mask)
    train_labels = jnp.array(train_labels)
    
    print(f"✅ Data shapes: input_ids={train_input_ids.shape}, attention_mask={train_attention_mask.shape}, labels={train_labels.shape}")
    
    # Create batches for input_ids, attention_mask, and labels
    input_id_batches = create_batches(train_input_ids, batch_size=training_config.batch_size)
    attention_mask_batches = create_batches(train_attention_mask, batch_size=training_config.batch_size)
    label_batches = create_batches(train_labels, batch_size=training_config.batch_size)
    print(f"📦 Created {len(input_id_batches)} training batches")

    # LoRA spec: ONLY MLP layers (as originally requested!)
    def decision_fn(path, param):
        path_str = ".".join(str(k.key) if hasattr(k, "key") else str(k) for k in path)
        # Llama MLP params use gate/up/down proj, no bias
        if ".mlp." in path_str and (
            ".gate_proj.kernel" in path_str or ".up_proj.kernel" in path_str or ".down_proj.kernel" in path_str
        ):
            rank = 4
            print(f"✅ Applying LoRA rank {rank} to: {path_str}")
            return rank
        else:
            return LORA_FREEZE

    # Create LoRA spec and parameters
    print("⚙️ Setting up LoRA configuration...")
    lora_spec = lorax.simple_spec(model.params, decision_fn=decision_fn, tune_vectors=False)
    lora_params = lorax.init_lora(model.params, lora_spec, jax.random.PRNGKey(42))

    # Split into trainable and frozen pytrees
    print("🔄 Splitting parameters into trainable and frozen pytrees...")
    trainable_params, frozen_params = lorax.split_trainable_frozen(lora_params, lora_spec)
    optimizer = optax.adamw(learning_rate=1e-4, weight_decay=0.01)
    opt_state = optimizer.init(trainable_params)

    # Wrapped model
    lora_model = lorax.lora(model)

    # Training loss function (instruction following with proper labels)
    def loss_fn(trainable_params, frozen_params, input_ids_batch, attention_mask_batch, labels_batch):
        # Merge trainable and frozen params for forward pass
        merged_params = lorax.merge_trainable_frozen(trainable_params, frozen_params)

        # Forward pass with input_ids AND attention_mask (important!)
        logits = lora_model(input_ids_batch, attention_mask=attention_mask_batch, params=merged_params).logits

        # CORRECT: Shift logits and labels for proper causal prediction
        # logits[i] should predict labels[i+1] (next token prediction)
        shift_logits = logits[:, :-1, :]  # Remove last prediction (B, T-1, V)
        shift_labels = labels_batch[:, 1:]  # Remove first token (B, T-1)
        
        # Cross-entropy loss with proper label masking  
        logprobs = jax.nn.log_softmax(shift_logits, axis=-1)
        
        # Create one-hot encoding for shifted labels
        vocab_size = logprobs.shape[-1]
        
        # Mask out ignored labels (-100) - these are prompt tokens
        valid_mask = (shift_labels != -100)
        masked_labels = jnp.where(valid_mask, shift_labels, 0)  # Replace -100 with 0 for one-hot
        
        # Create one-hot encoding  
        one_hot = jax.nn.one_hot(masked_labels, num_classes=vocab_size, dtype=logprobs.dtype)
        target_logprobs = jnp.sum(logprobs * one_hot, axis=-1)  # (B, T-1)
        
        # Apply mask and compute loss (only on response tokens)
        masked_loss = -(target_logprobs * valid_mask.astype(jnp.float32))
        loss = jnp.sum(masked_loss) / jnp.sum(valid_mask.astype(jnp.float32))

        return loss

    # JIT compiled training step
    @jax.jit
    def train_step(trainable_params, frozen_params, opt_state, input_ids_batch, attention_mask_batch, labels_batch):
        loss, grads = jax.value_and_grad(loss_fn, argnums=0)(trainable_params, frozen_params, input_ids_batch, attention_mask_batch, labels_batch)
        updates, new_opt_state = optimizer.update(grads, opt_state, trainable_params)
        new_params = optax.apply_updates(trainable_params, updates)
        return new_params, new_opt_state, loss

    # JIT compiled validation step (no gradient computation)
    @jax.jit
    def validation_step(trainable_params, frozen_params, input_ids_batch, attention_mask_batch, labels_batch):
        loss = loss_fn(trainable_params, frozen_params, input_ids_batch, attention_mask_batch, labels_batch)
        return loss

    def run_validation(trainable_params, frozen_params, validation_dataset):
        """Run validation on the entire validation dataset."""
        print("🔍 Running validation...")
        
        # Convert validation dataset to JAX arrays
        val_input_ids = []
        val_attention_mask = []
        val_labels = []
        
        for item in validation_dataset:
            val_input_ids.append(np.array(item['input_ids']))
            val_attention_mask.append(np.array(item['attention_mask']))
            val_labels.append(np.array(item['labels']))
        
        # Convert to JAX arrays
        val_input_ids = jnp.array(val_input_ids)
        val_attention_mask = jnp.array(val_attention_mask)
        val_labels = jnp.array(val_labels)
        
        # Create validation batches
        val_input_id_batches = create_batches(val_input_ids, batch_size=training_config.batch_size)
        val_attention_mask_batches = create_batches(val_attention_mask, batch_size=training_config.batch_size)
        val_label_batches = create_batches(val_labels, batch_size=training_config.batch_size)
        
        # Run validation on all batches
        val_losses = []
        num_val_batches = len(val_input_id_batches)
        
        for batch_idx in range(num_val_batches):
            val_input_ids_batch = val_input_id_batches[batch_idx]
            val_attention_mask_batch = val_attention_mask_batches[batch_idx]
            val_labels_batch = val_label_batches[batch_idx]
            
            val_loss = validation_step(
                trainable_params, frozen_params, 
                val_input_ids_batch, val_attention_mask_batch, val_labels_batch
            )
            val_losses.append(float(val_loss))
        
        avg_val_loss = np.mean(val_losses)
        print(f"✅ Validation completed: {num_val_batches} batches, Avg Loss = {avg_val_loss:.4f}")
        return avg_val_loss



    # Training loop with wandb logging
    print("🎯 Starting training on SST dataset...")
    global_step = 0
    last_10_losses = []
    
    try:
        for epoch in range(training_config.num_epochs):
            epoch_losses = []
            
            num_batches = len(input_id_batches)
            
            for batch_idx in range(num_batches):
                input_ids = input_id_batches[batch_idx]
                attention_mask = attention_mask_batches[batch_idx]
                labels = label_batches[batch_idx]

                # Use the JIT-compiled train_step function (cleaner and faster!)
                trainable_params, opt_state, loss = train_step(
                    trainable_params, frozen_params, opt_state, input_ids, attention_mask, labels
                )

                current_loss = float(loss)
                epoch_losses.append(current_loss)
                last_10_losses.append(current_loss)
                global_step += 1

                # 📊 Graph 1: Log EVERY single loss step
                log_to_wandb({
                    "step_loss": current_loss,  # Individual loss at each step
                    "epoch": epoch + 1,
                    "batch": batch_idx + 1,
                }, step=global_step)

                # 📈 Graph 2: Log average of 10 losses every 10 steps  
                if len(last_10_losses) == 10:
                    avg_10_loss = np.mean(last_10_losses)
                    log_to_wandb({
                        "avg_10_loss": avg_10_loss,  # Average of last 10 losses (separate graph)
                    }, step=global_step)
                    print(f"📈 Epoch {epoch+1}, Batch {batch_idx+1:2d}: Loss = {current_loss:.4f} | Avg 10 = {avg_10_loss:.4f} [logged to wandb]")
                    last_10_losses = []  # Reset for next 10
                else:
                    print(f"Epoch {epoch+1}, Batch {batch_idx+1:2d}: Loss = {current_loss:.4f} ({len(last_10_losses)}/10)")

            # Log epoch summary
            avg_epoch_loss = np.mean(epoch_losses)
            
            # 🔍 Run validation after each epoch
            avg_val_loss = run_validation(trainable_params, frozen_params, validation_dataset)
            
            # Log both training and validation metrics
            log_to_wandb({
                "epoch_avg_loss": avg_epoch_loss,  # Training loss (third graph)
                "val_loss": avg_val_loss,          # Validation loss (fourth graph)
                "train_val_diff": avg_val_loss - avg_epoch_loss,  # Overfitting indicator
            }, step=global_step)
            
            print(f"📊 Epoch {epoch+1} Results:")
            print(f"   Training Avg Loss: {avg_epoch_loss:.4f}")
            print(f"   Validation Loss:   {avg_val_loss:.4f}")
            print(f"   Difference:        {avg_val_loss - avg_epoch_loss:+.4f}")
            if avg_val_loss > avg_epoch_loss:
                print("   📈 Model may be overfitting")
            else:
                print("   📉 Model generalizing well")
            print()

        # Log training completion
        log_to_wandb({
            "training_completed": True,
            "total_steps": global_step,
        }, step=global_step)
        
        print("🎉 TRAINING COMPLETED - All metrics logged to wandb!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        log_to_wandb({"error": str(e), "training_failed": True})
        raise
    
    finally:
        # Finish wandb run
        wandb.finish()
        print("🔗 Finished wandb run")


if __name__ == "__main__":
    main()