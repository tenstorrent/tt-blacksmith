#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Real LoRA training on Llama 3.2-1B with SST dataset.
Sentiment classification training with proper attention masking!
Focuses on MLP layers only as originally requested.
NOW RUNNING ON TT DEVICE! 🚀
"""
import warnings

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
jax.config.update("jax_platforms", "tt,cpu")

# Wandb configuration
DEFAULT_EXPERIMENT_NAME = "Llama-TT-LoRA-Training"
DEFAULT_RUN_NAME = "llama-3.2-1b-sst2-tt-lorax"

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
            "device": "tt",
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
    print("🎯 TT Device Status:")
    print(f"   Available devices: {jax.devices()}")
    print(f"   Default backend: {jax.default_backend()}")
    cpu_device = jax.devices("cpu")[0]
    tt_device = jax.devices("tt")[0]

    # Quick device test
    test_array = jnp.array([1.0, 2.0, 3.0])
    print(f"   Test computation device: {test_array.device}")
    print()

    # Load model and tokenizer
    print("🤖 Loading Llama 3.2-1B model...")
    # Force model init and PRNG ops to CPU to avoid unsupported TT PRNG ops
    with jax.default_device(cpu_device):
        config = AutoConfig.from_pretrained(MODEL_NAME)
        # Training doesn't need KV cache; turn it off to save memory
        config.use_cache = False
        config.num_hidden_layers = 16
        config.dtype = jnp.bfloat16
        # (Optional) if you still hit OOM, clamp the context a bit more:
        # config.max_position_embeddings = 8192  # or 16384, etc.

        model = FlaxAutoModelForCausalLM.from_pretrained(MODEL_NAME, config=config, from_pt=False,dtype=jnp.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Move model params to TT device
    model.params = jax.tree_util.tree_map(lambda x: jax.device_put(x, tt_device), model.params)

    # Create training configuration for SST dataset
    training_config = TrainingConfig(
        model_name=MODEL_NAME,
        dataset_id="stanfordnlp/sst2",
        max_length=128,
        learning_rate=1e-4,
        batch_size=4,
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
    # Initialize LoRA params on CPU to avoid TT PRNG issues
    with jax.default_device(cpu_device):
        lora_params = lorax.init_lora(model.params, lora_spec, jax.random.PRNGKey(42))
    # Move LoRA params to TT device
    lora_params = jax.tree_util.tree_map(lambda x: jax.device_put(x, tt_device), lora_params)

    # Split into trainable and frozen pytrees
    print("🔄 Splitting parameters into trainable and frozen pytrees...")
    trainable_params, frozen_params = lorax.split_trainable_frozen(lora_params, lora_spec)
    with jax.default_device(cpu_device):
        optimizer = optax.adamw(learning_rate=1e-4, weight_decay=0.01)
        # Move trainable_params to CPU for optimizer init
        trainable_params_cpu = jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu_device), trainable_params)
        opt_state = optimizer.init(trainable_params_cpu)
        # Keep opt_state on CPU for fully CPU optimizer

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

    @jax.jit
    def compute_grads_tt(trainable_params_tt, frozen_params_tt, input_ids_batch, attention_mask_batch, labels_batch):
        loss, grads = jax.value_and_grad(loss_fn, argnums=0)(trainable_params_tt, frozen_params_tt, input_ids_batch, attention_mask_batch, labels_batch)
        return loss, grads

    # JIT compiled training step
    @jax.jit
    def train_step(trainable_params, frozen_params, opt_state, input_ids_batch, attention_mask_batch, labels_batch):
        loss, grads = jax.value_and_grad(loss_fn, argnums=0)(trainable_params, frozen_params, input_ids_batch, attention_mask_batch, labels_batch)
        updates, new_opt_state = optimizer.update(grads, opt_state, trainable_params)
        new_params = optax.apply_updates(trainable_params, updates)
        return new_params, new_opt_state, loss



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

                # Save tensors for debugging (first epoch only)

                # Compute gradients on TT device
                loss, grads = compute_grads_tt(trainable_params, frozen_params, input_ids, attention_mask, labels)
                
                # Move grads to CPU for fully CPU optimizer
                with jax.default_device(cpu_device):
                    grads_cpu = jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu_device), grads)
                    trainable_params_cpu = jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu_device), trainable_params)
                    updates, new_opt_state = optimizer.update(grads_cpu, opt_state, trainable_params_cpu)
                    # Apply updates on CPU then move back to TT device
                    new_params_cpu = optax.apply_updates(trainable_params_cpu, updates)

                # Move updated params back to TT device
                trainable_params = jax.tree_util.tree_map(lambda x: jax.device_put(x, tt_device), new_params_cpu)
                opt_state = new_opt_state  # Keep opt_state on CPU

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

    # Test generation
    print("\n🔮 Testing sentiment classification generation...")
 

if __name__ == "__main__":
    main()
