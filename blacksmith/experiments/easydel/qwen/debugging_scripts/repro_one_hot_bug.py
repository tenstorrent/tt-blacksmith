# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Reproducer for ttnn.eq one-hot value-doubling bug.

Bug: inside the Qwen3 model JIT graph, ``jax.nn.one_hot`` produces value 2.0
(instead of 1.0) at the correct label position for ALL even-valued labels.
Odd labels are unaffected.

Root cause (TTNN lowering):
  jax.nn.one_hot(labels, V)
    -> stablehlo.iota + stablehlo.compare EQ (uint32) -> convert (i1 -> f32)
    -> ttnn.eq(labels_u32, iota_u32) -> bf16   [fused by TT-MLIR]

The fused ``ttnn.eq`` compares uint32 values and writes a bf16 result.
For EVEN equal uint32 values it writes 2.0_bf16 (0x4000) instead of
1.0_bf16 (0x3F80).  Odd values are correct.

The bug only manifests inside sufficiently large JIT graphs (e.g. the
Qwen3-0.6B forward pass).  Standalone one-hot JITs do not trigger it.

Workaround: compute one-hot labels on the CPU host and pass them into
the device-side JIT as a pre-computed input.

Usage:
    python repro_one_hot_bug.py              # Qwen3-0.6B pretrained on TT
    python repro_one_hot_bug.py --random     # random weights (does NOT repro)
    python repro_one_hot_bug.py --cpu        # CPU reference  (should PASS)
"""

import inspect
import os
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

if "--cpu" in sys.argv:
    os.environ["JAX_PLATFORMS"] = "cpu"
else:
    os.environ.setdefault("PJRT_DEVICE", "TT")
    os.environ.setdefault("XLA_STABLEHLO_COMPILE", "1")

from easydel.modules.qwen3.modeling_qwen3 import Qwen3ForCausalLM  # noqa: E402, F401

from blacksmith.experiments.easydel.qwen.attention_patch import (  # noqa: E402
    apply_gqa_workaround,
)

apply_gqa_workaround()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from flax import nnx  # noqa: E402

BATCH = 4
SEQ = 128
VOCAB = 151936


def load_pretrained(tt_device):
    """Load Qwen3-0.6B with pretrained weights + LoRA."""
    from easydel import AutoEasyDeLModelForCausalLM

    cpu_device = jax.devices("cpu")[0]
    mesh = jax.make_mesh((1,), ("X",), devices=(tt_device,))

    jax.config.update("jax_default_device", tt_device)
    model = AutoEasyDeLModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        dtype=jnp.bfloat16,
        config_kwargs={"mask_max_position_embeddings": SEQ},
    )
    model.config.set_model_mesh(mesh)

    with jax.default_device(cpu_device):
        model = model.apply_lora_to_layers(lora_rank=32, lora_pattern=".*(q_proj|v_proj).*")

    return model, mesh


def load_random(tt_device):
    """Create Qwen3-0.6B-architecture model with random weights + LoRA."""
    from easydel.modules.qwen3.qwen3_configuration import Qwen3Config

    config = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=40960,
        mask_max_position_embeddings=SEQ,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        tie_word_embeddings=True,
    )

    mesh = jax.make_mesh((1,), ("X",), devices=(tt_device,))
    config.set_model_mesh(mesh)

    cpu_device = jax.devices("cpu")[0]
    with jax.default_device(cpu_device):
        model = Qwen3ForCausalLM(
            config=config,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            rngs=nnx.Rngs(0),
        )
        model = model.apply_lora_to_layers(lora_rank=32, lora_pattern=".*(q_proj|v_proj).*")

    return model, mesh


def run_test(model, mesh, input_ids_jnp):
    """Run forward + one_hot and check for the 2.0 bug."""
    graphdef, lora_params, frozen_state = nnx.split(model, nnx.LoRAParam, ...)
    call_sig = inspect.signature(model.__call__)

    @jax.jit
    def forward_one_hot(lora_params, frozen_state, input_ids):
        m = nnx.merge(graphdef, lora_params, frozen_state)
        kwargs = {"input_ids": input_ids}
        if "train" in call_sig.parameters:
            kwargs["train"] = False
        if "deterministic" in call_sig.parameters:
            kwargs["deterministic"] = True
        out = m(**kwargs)
        V = out.logits.shape[-1]
        shift_labels = input_ids[:, 1:]
        oh = jax.nn.one_hot(shift_labels, V)
        oh_f32 = oh.astype(jnp.float32)
        oh_sums = jnp.sum(oh_f32, axis=-1)
        oh_row0 = oh_f32[0, 0, :]
        return oh_sums, shift_labels, oh_row0

    with mesh:
        oh_sums, labels, oh_row0 = forward_one_hot(lora_params, frozen_state, input_ids_jnp)

    sums_np = np.array(oh_sums)
    labels_np = np.array(labels)
    oh_row0_np = np.array(oh_row0)

    bad = sums_np > 1.0 + 1e-3
    n_bad = int(np.sum(bad))
    n_total = sums_np.size

    print(f"  one_hot sums: min={sums_np.min():.3f}, max={sums_np.max():.3f}")
    print(f"  Result: {'FAIL' if n_bad > 0 else 'PASS'} ({n_bad}/{n_total} bad)")

    if n_bad > 0:
        bad_labels = labels_np[bad]
        good_labels = labels_np[~bad]
        n_bad_even = int(np.sum(bad_labels % 2 == 0))
        n_bad_odd = int(np.sum(bad_labels % 2 == 1))
        n_good_even = int(np.sum(good_labels % 2 == 0))
        n_good_odd = int(np.sum(good_labels % 2 == 1))
        print(f"  Parity: Bad={n_bad_even} even + {n_bad_odd} odd | " f"Good={n_good_even} even + {n_good_odd} odd")

        label_0 = int(labels_np[0, 0])
        nonzero = np.where(np.abs(oh_row0_np) > 1e-6)[0]
        print(f"  Raw one_hot[0,0] (label={label_0}):")
        for pos in nonzero:
            tag = " <-- LABEL" if pos == label_0 else ""
            print(f"    pos={pos}: {oh_row0_np[pos]:.1f}{tag}")

    return n_bad > 0


def main():
    device = jax.devices()[0]
    print(f"Device: {device.platform}")
    print(f"Config: BATCH={BATCH}, SEQ={SEQ}, VOCAB={VOCAB}")
    print()

    rng = np.random.RandomState(42)
    input_ids_np = rng.randint(0, VOCAB, size=(BATCH, SEQ)).astype(np.uint32)
    input_ids = jnp.array(input_ids_np, dtype=jnp.uint32)

    use_random = "--random" in sys.argv

    if use_random:
        print("=== Qwen3-0.6B architecture, RANDOM weights ===")
        model, mesh = load_random(device)
    else:
        print("=== Qwen3-0.6B, PRETRAINED weights ===")
        model, mesh = load_pretrained(device)

    print("  Compiling forward + one_hot...")
    failed = run_test(model, mesh, input_ids)

    print()
    if failed:
        print("BUG REPRODUCED: ttnn.eq writes 2.0_bf16 instead of 1.0")
        print("for even uint32 label values inside the Qwen3 forward JIT.")
    else:
        print("Bug did NOT reproduce with this configuration.")


if __name__ == "__main__":
    main()
