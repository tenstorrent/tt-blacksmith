# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
from model import FlaxLLaMAForCausalLM
from transformers import AutoTokenizer
from transformers.generation import GenerationConfig
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jaxtyping import PyTree
from flax import struct
from typing import List, Optional


class LLaMA(struct.PyTreeNode):
    params: PyTree
    model: FlaxLLaMAForCausalLM = struct.field(pytree_node=False)
    tokenizer: AutoTokenizer = struct.field(pytree_node=False)
    mesh: Optional[Mesh] = struct.field(pytree_node=False, default=None)

    def generate_logits_only(
        self,
        tokens: jnp.ndarray,
        max_gen_len: int,
        temperature: float = 0.0,
        top_p: float = 1,
    ) -> jnp.ndarray:
        """
        Generate tokens using logits-only approach, bypassing Transformers generation pipeline
        """
        print(f"🎯 Starting logits-only generation...")
        print(f"   Input tokens shape: {tokens.shape}")
        print(f"   Max gen length: {max_gen_len}")

        batch_size, seq_len = tokens.shape
        generated_tokens = tokens.copy()  # Start with input tokens

        # Generate tokens one by one
        for step in range(max_gen_len):
            print(f"   Step {step + 1}/{max_gen_len}: Getting logits...")

            # Get logits for current sequence within mesh context
            with self.mesh:
                # Forward pass through model - just get logits
                model_outputs = self.model(
                    input_ids=generated_tokens,
                    params=self.params,
                )

                # Get logits for next token (last position)
                logits = model_outputs.logits[:, -1, :]  # Shape: [batch, vocab_size]
                print(f"   Got logits shape: {logits.shape}")

                # Simple greedy sampling (take argmax)
                if temperature == 0.0:
                    next_token = jnp.argmax(logits, axis=-1, keepdims=True)
                else:
                    # TODO: Implement temperature sampling if needed
                    next_token = jnp.argmax(logits, axis=-1, keepdims=True)

                print(f"   Next token: {next_token}")

                # Append next token to sequence
                generated_tokens = jnp.concatenate([generated_tokens, next_token], axis=1)

                # Stop if we hit EOS token
                if next_token[0, 0] == self.tokenizer.eos_token_id:
                    print(f"   Hit EOS token, stopping generation")
                    break

        print(f"✅ Logits-only generation complete! Final shape: {generated_tokens.shape}")
        return generated_tokens

    def generate(
        self,
        tokens: jnp.ndarray,
        attention_mask: jnp.ndarray,
        max_gen_len: int,
        temperature: float = 0.0,
        top_p: float = 1,
        do_sample: bool = False,
    ) -> jnp.ndarray:
        # Use logits-only generation instead of Transformers pipeline
        return self.generate_logits_only(tokens, max_gen_len, temperature, top_p)

    def generate_from_str(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.0,
        top_p: float = 1,
        do_sample: bool = False,
    ) -> jnp.ndarray:
        prompt_tokens = [
            [self.tokenizer.bos_token_id] + self.tokenizer.encode(x, add_special_tokens=False) for x in prompts
        ]

        max_prompt_size = max([len(t) for t in prompt_tokens])

        # Create tokens on CPU to avoid TT backend scatter issues
        # TODO: This is a hack to avoid TT backend scatter issues. We should find a better way to do this.
        # TODO: We probably dont nee to put back on tt because later during computation we will be on tt anyway.
        with jax.default_device(jax.devices("cpu")[0]):
            tokens = jnp.full((len(prompts), max_prompt_size), self.tokenizer.pad_token_id).astype(jnp.int32)

            for i, t in enumerate(prompt_tokens):
                tokens = tokens.at[i, -len(t) :].set(t)
            attention_mask = (tokens != self.tokenizer.eos_token_id).astype(jnp.int32)

        out_tokens = self.generate(tokens, attention_mask, max_gen_len, temperature, top_p, do_sample=False)

        return out_tokens
