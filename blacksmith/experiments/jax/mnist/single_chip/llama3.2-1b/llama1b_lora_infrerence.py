#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LoRA Inference on Llama 3.2-1B with TT Device 🚀
Text generation and interactive inference with trained LoRA model.
"""
import warnings
import jax
import jax.numpy as jnp
import numpy as np
from transformers import FlaxAutoModelForCausalLM, AutoTokenizer, AutoConfig

import lorax
from lorax import LORA_FULL, LORA_FREEZE


MODEL_NAME = "Erland/Llama-3.2-1B-JAX"


def setup_model_and_lora():
    """Setup model, tokenizer, and LoRA configuration - same as training."""
    print("🤖 Loading Llama 3.2-1B model...")

    # Get devices
    cpu_device = jax.devices("cpu")[0]
    tt_device = jax.devices("tt")[0]

    # Force model init and PRNG ops to CPU to avoid unsupported TT PRNG ops
    with jax.default_device(cpu_device):
        config = AutoConfig.from_pretrained(MODEL_NAME)
        config.use_cache = False  # Not needed for inference, saves memory

        model = FlaxAutoModelForCausalLM.from_pretrained(MODEL_NAME, config=config, from_pt=False)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Move model params to TT device
    model.params = jax.tree_util.tree_map(lambda x: jax.device_put(x, tt_device), model.params)

    # LoRA spec: ONLY MLP layers (same as training)
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

    # Wrapped model for LoRA inference
    lora_model = lorax.lora(model)

    return model, tokenizer, lora_model, lora_params, cpu_device, tt_device


def generate_text(lora_model, lora_params, tokenizer, prompt, max_length=5):
    """Simple greedy text generation using the LoRA model."""
    print(f"🔮 Generating text for prompt: '{prompt}'")

    # Tokenize input
    input_ids = tokenizer.encode(prompt, return_tensors="np")
    input_ids = jnp.array(input_ids)

    # Generate tokens one by one with greedy sampling
    generated_tokens = input_ids[0].tolist()

    for step in range(max_length):
        # Get current sequence
        current_seq = jnp.array([generated_tokens])

        # Forward pass - simple, no gradients needed
        logits = lora_model(current_seq, params=lora_params).logits
        next_token_logits = logits[0, -1]  # Last token logits

        # Greedy sampling - just pick the highest probability token
        next_token_id = int(jnp.argmax(next_token_logits))

        # Add to sequence
        generated_tokens.append(next_token_id)

        # Stop if we hit EOS
        if next_token_id == tokenizer.eos_token_id:
            break

    # Decode full sequence
    generated_text = tokenizer.decode(generated_tokens)
    return generated_text


def main():
    print("🚀 Llama 3.2-1B LoRA Inference on TT Device")
    print("🎯 Device Status:")
    print(f"   Available devices: {jax.devices()}")
    print(f"   Default backend: {jax.default_backend()}")

    # Quick device test
    test_array = jnp.array([1.0, 2.0, 3.0])
    print(f"   Test computation device: {test_array.device}")
    print()

    # Setup everything
    model, tokenizer, lora_model, lora_params, cpu_device, tt_device = setup_model_and_lora()

    print(f"✅ Model loaded and ready on {tt_device}")
    print(f"📊 Model vocab size: {tokenizer.vocab_size}")

    # Simple text generation tests
    print("\n🎯 Running text generation...")

    test_prompts = ["Once upon a time, there was a brave", "The little girl walked through the forest and"]

    for i, prompt in enumerate(test_prompts):
        print(f"\n📖 Test {i+1}: '{prompt}'")
        generated = generate_text(lora_model, lora_params, tokenizer, prompt)
        print(f"🎯 Generated: '{generated}'")

    print("\n🎉 Text generation complete!")


if __name__ == "__main__":
    main()
