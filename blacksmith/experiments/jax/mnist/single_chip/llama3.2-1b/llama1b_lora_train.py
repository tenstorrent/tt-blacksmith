#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Real LoRA training on GPT-2 with TinyStories dataset.
Fun, engaging stories instead of boring Wikipedia!
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


def load_tinystories_data(tokenizer, max_length=128, num_samples=1000):
    """Load and tokenize TinyStories data - much more engaging than Wikipedia!"""
    print("📚 Loading TinyStories dataset...")

    # Load dataset - these are simple, fun stories!
    dataset = load_dataset("roneneldan/TinyStories", split="train")

    # Get story texts
    texts = [item["text"] for item in dataset if len(item["text"].strip()) > 100]
    texts = texts[:num_samples]  # Limit for demo

    print(f"📊 Loaded {len(texts)} story samples")
    print(f"📝 Sample story: {texts[0][:300]}...")

    # Tokenize
    print("🔤 Tokenizing stories...")
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
        # Training doesn't need KV cache; turn it off to save memory
        config.use_cache = False
        # (Optional) if you still hit OOM, clamp the context a bit more:
        # config.max_position_embeddings = 8192  # or 16384, etc.

        model = FlaxAutoModelForCausalLM.from_pretrained(MODEL_NAME, config=config, from_pt=False)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Move model params to TT device
    model.params = jax.tree_util.tree_map(lambda x: jax.device_put(x, tt_device), model.params)

    # Load real story data
    text_data = load_tinystories_data(tokenizer, max_length=64, num_samples=100)
    train_batches = create_batches(text_data, batch_size=4)
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

    # Setup optimizer ONLY for trainable parameters (much more efficient!)
    optimizer = optax.adamw(learning_rate=1e-4, weight_decay=0.01)
    opt_state = optimizer.init(trainable_params)
    # Ensure optimizer state is on TT device
    opt_state = jax.tree_util.tree_map(lambda x: jax.device_put(x, tt_device), opt_state)

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

    # JIT compiled training step
    @jax.jit
    def train_step(trainable_params, frozen_params, opt_state, batch):
        # razdovjim na freeze i ne freeze ( 2 pytree)
        loss, grads = jax.value_and_grad(loss_fn, argnums=0)(trainable_params, frozen_params, batch)
        # TODO: on cpu
        updates, new_opt_state = optimizer.update(grads, opt_state, trainable_params)
        new_params = optax.apply_updates(trainable_params, updates)
        return new_params, new_opt_state, loss

    # Training loop
    print("🎯 Starting training on real text data...")
    for epoch in range(5):  # Train for 10 epochs
        epoch_losses = []

        for batch_idx, batch in enumerate(train_batches[:20]):  # Train on first 20 batches
            trainable_params, opt_state, loss = train_step(trainable_params, frozen_params, opt_state, batch)

            epoch_losses.append(float(loss))

            print(f"Epoch {epoch+1}, Batch {batch_idx:2d}: Loss = {loss:.4f}")
            if epoch == 0:
                import os

                os.makedirs("trainable_params", exist_ok=True)
                import json

                # save trainable params to file in folder trainable_params in json format
                with open(f"trainable_params/trainable_params_{epoch+1}_{batch_idx}.json", "w") as f:
                    json.dump(trainable_params, f)
                # save frozen params to file in folder frozen_params in json format
                os.makedirs("frozen_params", exist_ok=True)
                with open(f"frozen_params/frozen_params_{epoch+1}_{batch_idx}.json", "w") as f:
                    json.dump(frozen_params, f)

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
    print("\n🔮 Testing story generation...")
    test_prompt = "Once upon a time, there was a little"
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
