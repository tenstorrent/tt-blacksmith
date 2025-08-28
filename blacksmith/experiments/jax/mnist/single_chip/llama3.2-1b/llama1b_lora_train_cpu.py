#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Real LoRA training on Llama with WikiText-2 dataset.
Using standard Wikipedia articles for more stable training.
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
from split_and_merge import split_trainable_frozen, merge_trainable_frozen
import lorax
from lorax import LORA_FULL, LORA_FREEZE


MODEL_NAME = "Erland/Llama-3.2-1B-JAX"


def load_wikitext_data(tokenizer, max_length=128, num_samples=1000):
    """Load and tokenize WikiText-2 data - more stable and standard dataset."""
    import random

    print("📚 Loading WikiText-2 dataset...")

    # Load WikiText-2 dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    # Get article texts and filter out short ones
    texts = []
    for item in dataset:
        text = item["text"].strip()
        # Skip empty lines, headers, and very short texts
        if len(text) > 64 and not text.startswith("=") and not text.startswith("@"):
            texts.append(text)
        if len(texts) >= num_samples * 3:  # Get more samples for better shuffling
            break

    # Shuffle the texts for better training diversity
    print(f"🔀 Shuffling {len(texts)} texts...")
    random.shuffle(texts)

    # Take the requested number of samples after shuffling
    texts = texts[:num_samples]

    print(f"📊 Loaded {len(texts)} shuffled Wikipedia article samples")
    print(f"📝 Sample text: {texts[0][:2560]}...")

    # Tokenize with better settings
    print("🔤 Tokenizing Wikipedia articles...")
    tokenized = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="np",
    )

    input_ids = tokenized["input_ids"]
    print(f"✅ Tokenized data shape: {input_ids.shape}")

    return input_ids


def create_batches(data, batch_size=8):
    """Create training batches."""
    num_batches = len(data) // batch_size
    batched_data = data[: num_batches * batch_size].reshape(num_batches, batch_size, -1)
    return batched_data


def jax_to_json_serializable(pytree, cpu_device):
    """Convert JAX arrays in a pytree to JSON-serializable format with only first/last 10 elements."""

    def convert_array(x):
        if hasattr(x, "tolist"):  # JAX/numpy array
            # Move to CPU first to avoid TT device OOM when flattening
            x_cpu = jax.device_put(x, cpu_device)
            flat_data = x_cpu.flatten()
            if len(flat_data) <= 20:
                # If tensor has 20 or fewer elements, take all
                return {"data": flat_data.tolist(), "shape": list(x.shape), "dtype": str(x.dtype)}
            else:
                # Take first 10 and last 10
                first_10 = flat_data[:10].tolist()
                last_10 = flat_data[-10:].tolist()
                return {"first_10": first_10, "last_10": last_10, "shape": list(x.shape), "dtype": str(x.dtype)}
        return x

    return jax.tree_util.tree_map(convert_array, pytree)


def main():
    print("🚀 Starting GPT-2 LoRA Training on TinyStories")
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
    print("🤖 Loading GPT-2 model...")
    # Force model init and PRNG ops to CPU to avoid unsupported TT PRNG ops
    with jax.default_device(cpu_device):
        config = AutoConfig.from_pretrained(MODEL_NAME)

        model = FlaxAutoModelForCausalLM.from_pretrained(MODEL_NAME, config=config, from_pt=False)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Move model params to TT device
    model.params = jax.tree_util.tree_map(lambda x: jax.device_put(x, tt_device), model.params)

    # Load real Wikipedia data
    text_data = load_wikitext_data(tokenizer, max_length=256, num_samples=640)
    train_batches = create_batches(text_data, batch_size=64)
    print(f"📦 Created {len(train_batches)} training batches")

    # LoRA spec: ONLY MLP layers (as originally requested!)
    def decision_fn(path, param):
        path_str = ".".join(str(k.key) if hasattr(k, "key") else str(k) for k in path)
        # Llama MLP params use gate/up/down proj, no bias
        if ".mlp." in path_str and (
            ".gate_proj.kernel" in path_str or ".up_proj.kernel" in path_str or ".down_proj.kernel" in path_str
        ):
            rank = 16
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
    trainable_params, frozen_params = split_trainable_frozen(lora_params, lora_spec)
    with jax.default_device(cpu_device):
        # Lower learning rate for more stable training with larger batches
        optimizer = optax.adamw(learning_rate=2e-4, weight_decay=0.01)
        # Move trainable_params to CPU for optimizer init
        trainable_params_cpu = jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu_device), trainable_params)
        opt_state = optimizer.init(trainable_params_cpu)
        # Keep opt_state on CPU for fully CPU optimizer

    # Wrapped model
    lora_model = lorax.lora(model)

    # Training loss function (next-token prediction)
    def loss_fn(trainable_params, frozen_params, batch):
        # Merge trainable and frozen params for forward pass
        merged_params = merge_trainable_frozen(trainable_params, frozen_params)

        input_ids = batch[:, :-1]  # Input: all tokens except last
        targets = batch[:, 1:]  # Target: all tokens except first

        # Forward pass with merged parameters
        logits = lora_model(input_ids, params=merged_params).logits

        # Cross-entropy loss
        logprobs = jax.nn.log_softmax(logits, axis=-1)
        # === replace take_along_axis with one-hot dot === ISSUE ISSUE ISSUE ISSUE ISSUE ISSUE ISSUE ISSUE
        one_hot = jax.nn.one_hot(targets, num_classes=logprobs.shape[-1], dtype=logprobs.dtype)
        target_logprobs = jnp.sum(logprobs * one_hot, axis=-1)  # (B, T)
        # ===============================================

        # Mask padding tokens (assuming tokenizer.pad_token_id is the pad token)
        pad_mask = (targets != tokenizer.pad_token_id).astype(jnp.float32)
        loss = -(target_logprobs * pad_mask).sum() / pad_mask.sum()

        return loss

    # Non-JIT function to handle device placement then call JIT gradient computation
    def compute_grads_tt(trainable_params_tt, frozen_params_tt, batch_tt):
        # Move inputs to CPU for computation (outside JIT)
        trainable_params_cpu = jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu_device), trainable_params_tt)
        frozen_params_cpu = jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu_device), frozen_params_tt)
        batch_cpu = jax.device_put(batch_tt, cpu_device)

        # Call JIT function with CPU inputs
        return compute_grads_cpu_jit(trainable_params_cpu, frozen_params_cpu, batch_cpu)

    # JIT compiled gradient computation on CPU
    @jax.jit
    def compute_grads_cpu_jit(trainable_params_cpu, frozen_params_cpu, batch_cpu):
        loss, grads = jax.value_and_grad(loss_fn, argnums=0)(trainable_params_cpu, frozen_params_cpu, batch_cpu)
        return loss, grads

    # JIT compiled training step
    @jax.jit
    def train_step(trainable_params, frozen_params, opt_state, batch):
        # razdovjim na freeze i ne freeze ( 2 pytree)

        loss, grads = jax.value_and_grad(loss_fn, argnums=0)(trainable_params, frozen_params, batch)
        # TODO: on cpu
        updates, new_opt_state = optimizer.update(grads, opt_state, trainable_params)
        new_params = optax.apply_updates(trainable_params, updates)
        return new_params, new_opt_state, loss

    def save_tensors(combined_data):
        import torch
        import os

        os.makedirs("lora_tensors", exist_ok=True)

        # Extract the three components in order
        trainable_params = combined_data["trainable_params"]
        frozen_params = combined_data["frozen_params"]
        batch = combined_data["batch"]

        # Flatten and save each component separately with clear naming
        trainable_flat, _ = jax.tree.flatten(trainable_params)
        frozen_flat, _ = jax.tree.flatten(frozen_params)
        batch_flat, _ = jax.tree.flatten(batch)

        trainable_flat_cpu = jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu_device), trainable_flat)
        frozen_flat_cpu = jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu_device), frozen_flat)
        batch_flat_cpu = jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu_device), batch_flat)

        # Save trainable params first
        for i, tensor_data in enumerate(trainable_flat_cpu):
            tensor = torch.tensor(tensor_data)
            torch.save(tensor, f"lora_tensors/{i}.pt")
            print(tensor.shape)

        # Save frozen params next
        offset = len(trainable_flat)
        for i, tensor_data in enumerate(frozen_flat_cpu):
            tensor = torch.tensor(tensor_data)
            torch.save(tensor, f"lora_tensors/{offset+i}.pt")
            print(tensor.shape)

        # Save batch last
        offset += len(frozen_flat)
        for i, tensor_data in enumerate(batch_flat_cpu):
            tensor = torch.tensor(tensor_data)
            torch.save(tensor, f"lora_tensors/{offset+i}.pt")
            print(tensor.shape)

        print(
            f"Saved {len(trainable_flat_cpu)} trainable + {len(frozen_flat_cpu)} frozen + {len(batch_flat_cpu)} batch tensors to lora_tensors/"
        )

    # Training loop
    print("🎯 Starting training on real text data...")
    loss_history = []
    for epoch in range(50):  # Train for 10 epochs
        epoch_losses = []

        for batch_idx, batch in enumerate(train_batches[:20]):  # Train on first 20 batches
            # trainable_params, opt_state, loss = train_step(trainable_params, frozen_params, opt_state, batch)

            # Merge trainable, frozen, and batch into one structure in specified order
            """combined_data = {
                "trainable_params": trainable_params,
                "frozen_params": frozen_params,
                "batch": batch
            }"""
            # save_tensors(combined_data)

            loss, grads = compute_grads_tt(trainable_params, frozen_params, batch)
            loss_history.append(loss)
            # print(f"grads: {grads}")
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

            epoch_losses.append(float(loss))
            print(f"train step {batch_idx} done")
            print(f"Epoch {epoch+1}, Batch {batch_idx:2d}: Loss = {loss:.4f}")
            """if epoch == 0:
                import os
                os.makedirs("trainable_params", exist_ok=True)
                import json
                #save trainable params to file in folder trainable_params in json format
                with open(f"trainable_params/trainable_params_{epoch+1}_{batch_idx}.json", "w") as f:
                    json.dump(jax_to_json_serializable(trainable_params, cpu_device), f)
                #save frozen params to file in folder frozen_params in json format
                os.makedirs("frozen_params", exist_ok=True)
                with open(f"frozen_params/frozen_params_{epoch+1}_{batch_idx}.json", "w") as f:
                    json.dump(jax_to_json_serializable(frozen_params, cpu_device), f)
                print("trainable_params saved")"""

        print(f"loss history: {loss_history}")
        # save losses to file
        with open("loss_history.txt", "w") as f:
            for loss in loss_history:
                f.write(f"{loss}\n")
        avg_loss = np.mean(epoch_losses)
        print(f"📊 Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    print("TRAINING COMPLETED")
    print("TRAINING COMPLETED")
    print("TRAINING COMPLETED")
    print("TRAINING COMPLETED")
    print("TRAINING COMPLETED")
    print("TRAINING COMPLETED")
    print("TRAINING COMPLETED")
    print("TRAINING COMPLETED")
    print("TRAINING COMPLETED")
    print("TRAINING COMPLETED")
    # Test generation
    print("\n🔮 Testing text generation...")
    test_prompt = "The history of artificial intelligence began in"
    test_tokens = tokenizer.encode(test_prompt, return_tensors="np")

    # Merge parameters for generation
    final_lora_params = merge_trainable_frozen(trainable_params, frozen_params)

    # Generate a few tokens
    generated_logits = lora_model(test_tokens, params=final_lora_params).logits
    next_token_id = jnp.argmax(generated_logits[0, -1])
    next_token = tokenizer.decode([next_token_id])

    print(f"📝 Input: '{test_prompt}'")
    print(f"🎯 Next predicted token: '{next_token}'")

    # Merge LoRA back into regular weights
    print("\n🔄 Merging LoRA parameters...")
    merged_params = lorax.merge_params(final_lora_params)

    # Verify merge works
    orig_logits = model(test_tokens, params=merged_params).logits
    lora_logits = lora_model(test_tokens, params=final_lora_params).logits
    merge_error = jnp.max(jnp.abs(orig_logits - lora_logits))

    print(f"✅ LoRA merge verification - Max error: {merge_error:.2e}")
    print("🎉 LoRA training completed successfully!")


if __name__ == "__main__":
    main()
