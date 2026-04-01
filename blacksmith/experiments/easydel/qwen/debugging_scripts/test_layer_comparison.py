# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Compare per-layer hidden states: CPU vs TT, with and without GQA patch.

Usage:
    python test_layer_comparison.py              # with GQA attention patch
    python test_layer_comparison.py --no-patch   # without patch (original 5D reshapes)
"""

import argparse
import os

os.environ.setdefault("PJRT_DEVICE", "TT")
os.environ.setdefault("XLA_STABLEHLO_COMPILE", "1")

import logging

import jax
import jax.numpy as jnp
import numpy as np
from easydel import AutoEasyDeLModelForCausalLM
from flax import nnx
from transformers import AutoTokenizer

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3-0.6B"
PROMPT = "The capital of France is"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-patch", action="store_true", help="Skip GQA attention patch")
    args = parser.parse_args()

    if not args.no_patch:
        from blacksmith.experiments.easydel.qwen.attention_patch import apply_gqa_workaround
        apply_gqa_workaround()
        logger.info("GQA 4D patch: ENABLED")
    else:
        logger.info("GQA 4D patch: DISABLED (using original 5D attention)")

    platform = jax.devices()[0].platform
    logger.info(f"Platform: {platform}")

    model = AutoEasyDeLModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=jnp.bfloat16,
        config_kwargs={"mask_max_position_embeddings": 128},
    )

    tt_devices = tuple(jax.devices(platform)[:1])
    mesh = jax.make_mesh((1,), ("X",), devices=tt_devices)
    model.config.set_model_mesh(mesh)

    graphdef, state = nnx.split(model)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ids = tokenizer.encode(PROMPT, add_special_tokens=False)
    input_ids = jnp.array([ids], dtype=jnp.uint32)

    cpu = jax.devices("cpu")[0]

    def to_np(x):
        return np.array(jax.device_put(x, cpu), dtype=np.float32)

    # --- CPU reference ---
    logger.info("Computing CPU reference (full forward)...")
    cpu_mesh = jax.make_mesh((1,), ("X",), devices=(cpu,))
    cpu_model = nnx.merge(graphdef, jax.device_put(state, cpu))
    cpu_model.config.set_model_mesh(cpu_mesh)
    with cpu_mesh:
        cpu_out = cpu_model(input_ids=jax.device_put(input_ids, cpu), output_hidden_states=True)
    ref_hidden = [to_np(h) for h in cpu_out.hidden_states]
    ref_logits = to_np(cpu_out.logits)
    del cpu_model, cpu_out

    # --- TT forward ---
    logger.info("Computing TT forward (full forward via JIT)...")

    @jax.jit
    def forward(state, ids):
        m = nnx.merge(graphdef, state)
        out = m(input_ids=ids, output_hidden_states=True)
        return out.logits, out.hidden_states

    with mesh:
        tt_logits, tt_hidden = forward(state, input_ids)

    # --- Compare ---
    print(f"\n{'Layer':<10} {'max_diff':>10} {'mean_diff':>12} {'relative%':>10}")
    print("-" * 46)

    for i, tt_h in enumerate(tt_hidden):
        tt_np = to_np(tt_h)
        ref = ref_hidden[i]
        max_diff = np.max(np.abs(tt_np - ref))
        mean_diff = np.mean(np.abs(tt_np - ref))
        ref_scale = np.max(np.abs(ref)) + 1e-8
        rel_pct = (max_diff / ref_scale) * 100
        label = "embed" if i == 0 else f"layer {i}"
        print(f"{label:<10} {max_diff:>10.4f} {mean_diff:>12.6f} {rel_pct:>9.2f}%")

    logits_diff = np.max(np.abs(to_np(tt_logits) - ref_logits))
    logits_mean = np.mean(np.abs(to_np(tt_logits) - ref_logits))
    logits_rel = (logits_diff / (np.max(np.abs(ref_logits)) + 1e-8)) * 100
    print(f"{'logits':<10} {logits_diff:>10.4f} {logits_mean:>12.6f} {logits_rel:>9.2f}%")
    print()


if __name__ == "__main__":
    main()
