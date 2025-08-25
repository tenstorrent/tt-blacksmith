# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import gc
from blacksmith.tools.jax_utils import init_device
import jax
import jax.numpy as jnp

init_device()

import numpy as np
from flax.core.frozen_dict import freeze
import psutil
import time
import threading
from collections import defaultdict
from model import FlaxLLaMAForCausalLM
from convert_weights import convert_llama_weights
from transformers import AutoTokenizer
from generation import LLaMA
from jax.sharding import Mesh, PartitionSpec as P
import os
from pathlib import Path

ROOT = Path(__file__).parent


def jax_load(
    model_id: str,
    ckpt_dir: str,
    tokenizer_path: str,
    mesh,
    cpu_devices: list[jax.Device] = None,
    tt_devices: list[jax.Device] = None,
    max_seq_length: int = 16,
    n_layers: int = 1,
) -> LLaMA:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    full_weights, jax_config = convert_llama_weights(
        ckpt_dir=ckpt_dir,
        tokenizer=tokenizer,
        max_seq_len=max_seq_length,
        n_layers=n_layers,
        verbose=True,
        mesh=mesh,
        cpu_devices=cpu_devices,
        tt_devices=tt_devices,
    )

    full_weights = freeze(full_weights)

    # Create model on CPU to avoid TT backend PRNG issues (shift_right_logical not supported)
    with jax.default_device(cpu_devices[0]):
        model = FlaxLLaMAForCausalLM(config=jax_config, _do_init=False)

    llama = LLaMA(params=full_weights, model=model, tokenizer=tokenizer, mesh=mesh)

    del full_weights
    gc.collect()
    return llama


def main(
    model_id="meta-llama/Meta-Llama-3.1-8B",
    ckpt_dir=str(ROOT / "llama3.1-8B/8B/original"),
    tokenizer_path=str(ROOT / "llama3.1-8B/original/original/tokenizer.model"),
    prompt=("How much is 10 squared?"),
    max_gen_len: int = 5,
    temperature: float = 0.0,
    top_p: float = 1.0,
    n_layers: int = 1,
    max_seq_length: int = 16,
    print_hlo: bool = False,
    monitor_memory: bool = True,
):

    # Get TT devices for parallel execution
    all_devices = jax.devices()

    tt_devices = jax.devices("tt")
    cpu_devices = jax.devices("cpu")
    print(f"All devices: {all_devices}")
    print(f"TT devices: {tt_devices}")
    print(f"Number of TT devices: {len(tt_devices)}")

    if len(tt_devices) == 0:
        raise RuntimeError("No TT devices found! Make sure TT devices are properly initialized.")

    mesh_devices = tt_devices[:2]
    mesh = Mesh(mesh_devices, axis_names=("mp",))
    print(f"✅ Created mesh with 2 TT devices: {mesh_devices}")

    print("🚀 Loading LLaMA...")
    llama = jax_load(
        model_id,
        ckpt_dir,
        tokenizer_path,
        mesh,
        cpu_devices,
        tt_devices,
        max_seq_length=max_seq_length,
        n_layers=n_layers,
    )

    print("✍️ Generating...")

    with mesh:
        results = llama.generate_from_str(
            [prompt],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            do_sample=False,
        )

    print("✅ Generation complete.")
    print("🧠 Output:", llama.tokenizer.decode(results[0]))

    if not os.path.isdir("results"):
        os.mkdir("results")

    np.savetxt("results/multi_chip.txt", results, fmt="%d")


if __name__ == "__main__":

    main()
