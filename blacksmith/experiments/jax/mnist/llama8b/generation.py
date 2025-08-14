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

    def generate(
        self,
        tokens: jnp.ndarray,
        attention_mask: jnp.ndarray,
        max_gen_len: int,
        temperature: float = 0.0,
        top_p: float = 1,
        do_sample: bool = False,
    ) -> jnp.ndarray:

        with jax.default_device(jax.devices("cpu")[0]):
            prng_key = jax.random.PRNGKey(0)

        # Ensure generation runs within mesh context for tensor parallel ops
        with self.mesh:
            generations = self.model.generate(
                input_ids=tokens,
                attention_mask=attention_mask,
                params=self.params,
                prng_key=prng_key,
                generation_config=GenerationConfig(
                    num_beams=1,
                    do_sample=do_sample,
                    max_length=max_gen_len + tokens.shape[1],
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    temperature=temperature,
                    top_p=top_p,
                ),
            )
        return generations.sequences

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
