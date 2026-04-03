# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Per-layer divergence diagnostic: compare CPU vs TT hidden states across
all transformer layers of Qwen3-0.6B.

Mode 1 (layer-by-layer): runs each layer as an individual JIT call.
Mode 2 (full-model + LoRA): replicates the training script's exact eval path —
    loads model, applies LoRA, nnx.split, JIT-compiles eval_loss_fn as ONE
    function, and runs it on a real validation batch.

Usage:
    python test_layer_divergence.py --cpu          # layer-by-layer CPU
    python test_layer_divergence.py --tt           # layer-by-layer TT
    python test_layer_divergence.py --compare      # compare saved results
    python test_layer_divergence.py --all          # run both + compare

    python test_layer_divergence.py --lora-cpu     # full-model+LoRA CPU
    python test_layer_divergence.py --lora-tt      # full-model+LoRA TT
    python test_layer_divergence.py --lora-compare # compare LoRA results
    python test_layer_divergence.py --lora-all     # run both + compare
"""

import os
import sys
import subprocess

RESULTS_DIR = "/tmp/tt_layer_divergence"
MODEL_NAME = "Qwen/Qwen3-0.6B"
SEQ_LEN = 128  # match training config; change to test length sensitivity


def _setup_env(platform: str):
    if platform == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    else:
        os.environ.setdefault("PJRT_DEVICE", "TT")
        os.environ.setdefault("XLA_STABLEHLO_COMPILE", "1")


def save(name, arr):
    import numpy as np
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.save(os.path.join(RESULTS_DIR, name), arr)


def load(name):
    import numpy as np
    return np.load(os.path.join(RESULTS_DIR, name + ".npy"))


def to_np(x):
    import jax
    import numpy as np
    cpu = jax.devices("cpu")[0]
    return np.array(jax.device_put(x, cpu), dtype=np.float32)


def stat(label, arr_np):
    print(f"  {label:<18} shape={str(arr_np.shape):<25} "
          f"mean={arr_np.mean():>10.4f}  std={arr_np.std():>10.4f}")


def run_forward(platform: str):
    """Run model forward pass layer-by-layer and save hidden states."""
    _setup_env(platform)

    import jax
    import jax.numpy as jnp
    from transformers import AutoTokenizer
    from easydel import AutoEasyDeLModelForCausalLM
    from easydel.layers.caching.transformer.cache import TransformerCache
    from blacksmith.experiments.easydel.qwen.attention_patch import apply_gqa_workaround

    apply_gqa_workaround()

    tag = platform
    device = jax.devices()[0]
    seq_len = SEQ_LEN
    print(f"Platform: {device.platform} ({tag}), seq_len={seq_len}")

    import numpy as np_host
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Use real WikiText text to get a representative 128-token input
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    full_text = " ".join(t for t in ds["text"] if t.strip())
    all_ids = tokenizer(full_text, return_tensors="np")["input_ids"][0]
    input_ids_np = all_ids[:seq_len].reshape(1, seq_len)
    print(f"Input tokens: {input_ids_np.shape} (first {seq_len} of WikiText validation)\n")

    print("Loading model...")
    model = AutoEasyDeLModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=jnp.bfloat16,
        config_kwargs={"mask_max_position_embeddings": 128},
    )
    devices_for_mesh = tuple(jax.devices()[:1])
    mesh = jax.make_mesh((1,), ("X",), devices=devices_for_mesh)
    model.config.set_model_mesh(mesh)

    num_layers = model.config.num_hidden_layers
    print(f"Model loaded — {num_layers} layers\n")

    qwen_model = model.model
    input_ids = jnp.array(input_ids_np, dtype=jnp.int32)
    save(f"{tag}_input_ids", input_ids_np)

    # Pre-compute cached properties
    _ = qwen_model.frequencies
    _ = qwen_model.causal_mask

    with mesh:
        # Step 1: Embeddings
        @jax.jit
        def get_embeddings(input_ids):
            return qwen_model.embed_tokens(input_ids.astype("i4"))

        print("Running layer-by-layer forward pass...\n")
        hidden_states = get_embeddings(input_ids)
        arr = to_np(hidden_states)
        save(f"{tag}_00_embeddings", arr)
        stat("00_embeddings", arr)

        # Prepare attention inputs
        batch_size = 1
        attention_mask = jnp.ones((batch_size, seq_len), dtype="b1")
        position_ids = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
        causal_mask = qwen_model.causal_mask
        frequencies = qwen_model.frequencies
        past_key_values = TransformerCache.init_empty(num_layers)

        # Step 2: Each decoder layer
        for layer_idx in range(num_layers):
            layer = qwen_model.layers[layer_idx]
            cache_view = past_key_values.views[layer_idx]

            @jax.jit
            def run_layer(hs, layer_idx_static=layer_idx):
                out = qwen_model.layers[layer_idx_static](
                    hidden_states=hs,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    mode="train",
                    cache_view=cache_view,
                    cache_metadata=None,
                    causal_mask=causal_mask,
                    output_attentions=False,
                    segment_ids=None,
                    frequencies=frequencies,
                )
                return out.hidden_states

            hidden_states = run_layer(hidden_states)
            arr = to_np(hidden_states)
            label = f"{layer_idx + 1:02d}_layer_{layer_idx:02d}"
            save(f"{tag}_{label}", arr)
            stat(label, arr)

        # Step 3: Final RMSNorm
        @jax.jit
        def run_norm(hs):
            return qwen_model.norm(hs)

        hidden_states = run_norm(hidden_states)
        arr = to_np(hidden_states)
        save(f"{tag}_{num_layers + 1:02d}_final_norm", arr)
        stat(f"{num_layers + 1:02d}_final_norm", arr)

        # Step 4: LM head (logits)
        @jax.jit
        def run_lm_head(hs):
            return model.apply_lm_head(hs)

        logits = run_lm_head(hidden_states)
        arr = to_np(logits)
        save(f"{tag}_{num_layers + 2:02d}_logits", arr)
        stat(f"{num_layers + 2:02d}_logits", arr)

    print(f"\nSaved to {RESULTS_DIR}/{tag}_*.npy")


LORA_RANK = 32
LORA_PATTERN = ".*(q_proj|v_proj).*"


def run_with_lora(platform: str):
    """Replicate the training script's exact eval path: load model, apply LoRA,
    nnx.split, JIT-compile eval_loss_fn as one function, run on validation data."""
    _setup_env(platform)

    import inspect
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from flax import nnx
    from transformers import AutoTokenizer
    from datasets import load_dataset
    from easydel import AutoEasyDeLModelForCausalLM
    from blacksmith.experiments.easydel.qwen.attention_patch import apply_gqa_workaround

    apply_gqa_workaround()

    tag = f"lora_{platform}"
    cpu_device = jax.devices("cpu")[0]
    device = jax.devices()[0]
    seq_len = SEQ_LEN
    print(f"Platform: {device.platform} ({tag}), seq_len={seq_len}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    full_text = " ".join(t for t in ds["text"] if t.strip())
    all_ids = tokenizer(full_text, return_tensors="np")["input_ids"][0]
    input_ids_np = all_ids[:seq_len].reshape(1, seq_len)
    print(f"Input tokens: {input_ids_np.shape}")

    print("Loading model...")
    model = AutoEasyDeLModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=jnp.bfloat16,
        config_kwargs={"mask_max_position_embeddings": 128},
    )
    devices_for_mesh = tuple(jax.devices()[:1])
    mesh = jax.make_mesh((1,), ("X",), devices=devices_for_mesh)
    model.config.set_model_mesh(mesh)
    print(f"Model loaded — {model.config.num_hidden_layers} layers")

    print(f"Applying LoRA (rank={LORA_RANK})...")
    with jax.default_device(cpu_device):
        model = model.apply_lora_to_layers(
            lora_rank=LORA_RANK,
            lora_pattern=LORA_PATTERN,
            verbose=True,
        )

    graphdef, lora_params, frozen_state = nnx.split(model, nnx.LoRAParam, ...)
    call_signature = inspect.signature(model.__call__)

    # Save LoRA params for comparison
    lora_flat = jax.tree.leaves(lora_params)
    lora_concat = np.concatenate([np.array(jax.device_put(x, cpu_device)).ravel()
                                  for x in lora_flat])
    save(f"{tag}_lora_params", lora_concat)
    print(f"  LoRA params: {len(lora_flat)} arrays, {lora_concat.shape[0]:,} total scalars")
    print(f"  LoRA mean={lora_concat.mean():.6f}  std={lora_concat.std():.6f}")

    input_ids = jnp.array(input_ids_np, dtype=jnp.uint32)

    def eval_loss_fn(lora_params, frozen_state, input_ids):
        m = nnx.merge(graphdef, lora_params, frozen_state)
        kwargs = {"input_ids": input_ids}
        if "train" in call_signature.parameters:
            kwargs["train"] = False
        if "deterministic" in call_signature.parameters:
            kwargs["deterministic"] = True
        out = m(**kwargs)
        logits_f32 = out.logits.astype(jnp.float32)
        shift_logits = logits_f32[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        one_hot = jax.nn.one_hot(shift_labels, shift_logits.shape[-1])
        per_token_loss = optax.softmax_cross_entropy(
            shift_logits,
            one_hot.astype(jnp.float32),
        )
        return jnp.mean(per_token_loss), out.logits

    jit_eval = jax.jit(eval_loss_fn)

    # Test A: Fused (model + loss in one JIT) — replicates training script
    print("Test A: Fused JIT (model forward + loss)...")
    with mesh:
        loss, logits = jit_eval(lora_params, frozen_state, input_ids)

    loss_val = float(loss)
    logits_np = to_np(logits)
    print(f"  Fused loss:  {loss_val:.6f}")
    print(f"  Logits shape: {logits_np.shape}, mean={logits_np.mean():.4f}, std={logits_np.std():.4f}")

    # Test B: Split (model forward in one JIT, loss in a second JIT)
    def forward_only(lora_params, frozen_state, input_ids):
        m = nnx.merge(graphdef, lora_params, frozen_state)
        kwargs = {"input_ids": input_ids}
        if "train" in call_signature.parameters:
            kwargs["train"] = False
        if "deterministic" in call_signature.parameters:
            kwargs["deterministic"] = True
        return m(**kwargs).logits

    def loss_only(logits, input_ids):
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        one_hot = jax.nn.one_hot(shift_labels, shift_logits.shape[-1])
        per_token_loss = optax.softmax_cross_entropy(
            shift_logits.astype(jnp.float32),
            one_hot.astype(jnp.float32),
        )
        return jnp.mean(per_token_loss)

    jit_fwd = jax.jit(forward_only)
    jit_loss = jax.jit(loss_only)

    print("Test B: Split JIT (forward then loss separately)...")
    with mesh:
        logits_b = jit_fwd(lora_params, frozen_state, input_ids)
        loss_b = jit_loss(logits_b, input_ids)

    loss_b_val = float(loss_b)
    logits_b_np = to_np(logits_b)
    print(f"  Split loss:  {loss_b_val:.6f}")
    print(f"  Logits match fused: {np.allclose(logits_np, logits_b_np, atol=1e-5)}")

    # Test C: Compute loss on CPU from the TT logits
    cpu = jax.devices("cpu")[0]
    logits_cpu = jax.device_put(logits_b, cpu)
    with jax.default_device(cpu):
        loss_c = loss_only(logits_cpu, jax.device_put(input_ids, cpu))
    loss_c_val = float(loss_c)
    print(f"Test C: Loss computed on CPU from TT logits: {loss_c_val:.6f}")

    # Test D: Round-trip logits through CPU and back, then compute loss on device
    logits_roundtrip = jnp.array(to_np(logits_b))
    with mesh:
        loss_d = jit_loss(logits_roundtrip, input_ids)
    loss_d_val = float(loss_d)
    print(f"Test D: Loss on {platform} from round-tripped logits: {loss_d_val:.6f}")

    # Test E: Cast bf16 logits to f32 via separate JIT (crosses JIT boundary)
    @jax.jit
    def loss_with_explicit_cast(logits_bf16, input_ids):
        logits_f32 = logits_bf16.astype(jnp.float32)
        shift_logits = logits_f32[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        one_hot = jax.nn.one_hot(shift_labels, shift_logits.shape[-1])
        per_token_loss = optax.softmax_cross_entropy(shift_logits, one_hot.astype(jnp.float32))
        return jnp.mean(per_token_loss)

    with mesh:
        loss_e = loss_with_explicit_cast(logits_b, input_ids)
    loss_e_val = float(loss_e)
    print(f"Test E: Loss with cast-first on {platform}: {loss_e_val:.6f}")

    # Test F: Force f32 with multiplication instead of cast (inside fused JIT)
    def eval_loss_mul(lora_params, frozen_state, input_ids):
        m = nnx.merge(graphdef, lora_params, frozen_state)
        kwargs = {"input_ids": input_ids}
        if "train" in call_signature.parameters:
            kwargs["train"] = False
        if "deterministic" in call_signature.parameters:
            kwargs["deterministic"] = True
        out = m(**kwargs)
        logits_f32 = out.logits * jnp.float32(1.0)
        shift_logits = logits_f32[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        one_hot = jax.nn.one_hot(shift_labels, shift_logits.shape[-1])
        per_token_loss = optax.softmax_cross_entropy(
            shift_logits, one_hot.astype(jnp.float32),
        )
        return jnp.mean(per_token_loss)

    jit_eval_mul = jax.jit(eval_loss_mul)
    print("Test F: Fused JIT with multiply-by-1.0 f32 cast...")
    with mesh:
        loss_f = jit_eval_mul(lora_params, frozen_state, input_ids)
    loss_f_val = float(loss_f)
    print(f"  Multiply-cast loss: {loss_f_val:.6f}")

    # Test G: Force f32 with jnp.add (bf16 + 0.0f32 = f32)
    def eval_loss_add(lora_params, frozen_state, input_ids):
        m = nnx.merge(graphdef, lora_params, frozen_state)
        kwargs = {"input_ids": input_ids}
        if "train" in call_signature.parameters:
            kwargs["train"] = False
        if "deterministic" in call_signature.parameters:
            kwargs["deterministic"] = True
        out = m(**kwargs)
        logits_f32 = jnp.add(out.logits, jnp.float32(0.0))
        shift_logits = logits_f32[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        one_hot = jax.nn.one_hot(shift_labels, shift_logits.shape[-1])
        per_token_loss = optax.softmax_cross_entropy(
            shift_logits, one_hot.astype(jnp.float32),
        )
        return jnp.mean(per_token_loss)

    jit_eval_add = jax.jit(eval_loss_add)
    print("Test G: Fused JIT with add-zero f32 cast...")
    with mesh:
        loss_g = jit_eval_add(lora_params, frozen_state, input_ids)
    loss_g_val = float(loss_g)
    print(f"  Add-zero-cast loss: {loss_g_val:.6f}")

    print(f"\nSummary:")
    print(f"  A: Fused astype (model+loss on {platform}):   {loss_val:.6f}")
    print(f"  B: Split astype ({platform}+{platform}):         {loss_b_val:.6f}")
    print(f"  C: CPU loss (from {platform} logits):       {loss_c_val:.6f}")
    print(f"  D: Round-trip (logits→CPU→{platform}):     {loss_d_val:.6f}")
    print(f"  E: Cast-first separate JIT ({platform}):   {loss_e_val:.6f}")
    print(f"  F: Fused multiply-by-1.0 ({platform}):     {loss_f_val:.6f}")
    print(f"  G: Fused add-zero ({platform}):            {loss_g_val:.6f}")

    save(f"{tag}_loss", np.array([loss_val]))
    save(f"{tag}_loss_split", np.array([loss_b_val]))
    save(f"{tag}_loss_cpu", np.array([loss_c_val]))
    save(f"{tag}_logits", logits_np)
    save(f"{tag}_input_ids", input_ids_np)

    print(f"\nSaved to {RESULTS_DIR}/{tag}_*.npy")


def compare_lora():
    """Compare CPU vs TT results from full-model+LoRA eval."""
    import numpy as np

    print(f"\n{'=' * 91}")
    print(f"  FULL-MODEL + LORA DIVERGENCE: CPU vs TT")
    print(f"{'=' * 91}\n")

    cpu_loss = np.load(os.path.join(RESULTS_DIR, "lora_cpu_loss.npy"))[0]
    tt_loss = np.load(os.path.join(RESULTS_DIR, "lora_tt_loss.npy"))[0]
    cpu_logits = np.load(os.path.join(RESULTS_DIR, "lora_cpu_logits.npy"))
    tt_logits = np.load(os.path.join(RESULTS_DIR, "lora_tt_logits.npy"))
    cpu_lora = np.load(os.path.join(RESULTS_DIR, "lora_cpu_lora_params.npy"))
    tt_lora = np.load(os.path.join(RESULTS_DIR, "lora_tt_lora_params.npy"))

    print(f"  LoRA params match: {np.array_equal(cpu_lora, tt_lora)}")
    if not np.array_equal(cpu_lora, tt_lora):
        diff = np.abs(cpu_lora - tt_lora)
        print(f"    max_diff={diff.max():.6f}  mean_diff={diff.mean():.6f}")

    logit_diff = np.abs(cpu_logits.astype(np.float32) - tt_logits.astype(np.float32))
    cpu_flat = cpu_logits.ravel().astype(np.float32)
    tt_flat = tt_logits.ravel().astype(np.float32)
    cos_sim = np.dot(cpu_flat, tt_flat) / (
        np.linalg.norm(cpu_flat) * np.linalg.norm(tt_flat) + 1e-12
    )

    print(f"\n  Logits: max_diff={logit_diff.max():.4f}  mean_diff={logit_diff.mean():.4f}  cos_sim={cos_sim:.6f}")
    print(f"\n  Loss (on-device) — CPU: {cpu_loss:.6f},  TT: {tt_loss:.6f},  gap: {tt_loss - cpu_loss:+.6f}")

    # Recompute loss in numpy float64 from saved logits to isolate on-device issue
    from transformers import AutoTokenizer
    from datasets import load_dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    full_text = " ".join(t for t in ds["text"] if t.strip())
    all_ids = tokenizer(full_text, return_tensors="np")["input_ids"][0]
    input_ids = all_ids[:SEQ_LEN].reshape(1, SEQ_LEN)

    def xent_numpy(logits, labels):
        logits = logits[0, :-1, :].astype(np.float64)
        labels = labels[0, 1:]
        shifted = logits - logits.max(axis=-1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        return -np.mean(log_probs[np.arange(len(labels)), labels])

    cpu_np_loss = xent_numpy(cpu_logits, input_ids)
    tt_np_loss = xent_numpy(tt_logits, input_ids)
    print(f"  Loss (numpy f64)  — CPU: {cpu_np_loss:.6f},  TT: {tt_np_loss:.6f},  gap: {tt_np_loss - cpu_np_loss:+.6f}")
    print(f"\n  ** TT on-device loss error: {tt_loss - tt_np_loss:+.6f} "
          f"(on-device={tt_loss:.4f} vs correct={tt_np_loss:.4f}) **")
    print()


def test_loss_ops(platform: str):
    """Test individual loss operations on a given platform using saved TT logits.
    This isolates which specific operation in softmax_cross_entropy breaks on TT."""
    _setup_env(platform)

    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from transformers import AutoTokenizer
    from datasets import load_dataset

    device = jax.devices()[0]
    print(f"Testing loss operations on: {device.platform}")

    logits_np = np.load(os.path.join(RESULTS_DIR, "lora_cpu_logits.npy"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    full_text = " ".join(t for t in ds["text"] if t.strip())
    all_ids = tokenizer(full_text, return_tensors="np")["input_ids"][0]
    input_ids_np = all_ids[:SEQ_LEN].reshape(1, SEQ_LEN)

    # Use the same CPU logits on this platform
    logits = jnp.array(logits_np)
    input_ids = jnp.array(input_ids_np, dtype=jnp.uint32)

    # Test 1: Full optax.softmax_cross_entropy (as in training script)
    @jax.jit
    def loss_optax_f32(logits, input_ids):
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        one_hot = jax.nn.one_hot(shift_labels, shift_logits.shape[-1])
        per_token = optax.softmax_cross_entropy(
            shift_logits.astype(jnp.float32),
            one_hot.astype(jnp.float32),
        )
        return jnp.mean(per_token)

    # Test 2: Manual log_softmax + cross-entropy
    @jax.jit
    def loss_manual_f32(logits, input_ids):
        shift_logits = logits[:, :-1, :].astype(jnp.float32)
        shift_labels = input_ids[:, 1:]
        log_probs = jax.nn.log_softmax(shift_logits, axis=-1)
        one_hot = jax.nn.one_hot(shift_labels, shift_logits.shape[-1]).astype(jnp.float32)
        return -jnp.mean(jnp.sum(log_probs * one_hot, axis=-1))

    # Test 3: Sparse cross-entropy (no one_hot, uses gather instead)
    @jax.jit
    def loss_sparse_f32(logits, input_ids):
        shift_logits = logits[:, :-1, :].astype(jnp.float32)
        shift_labels = input_ids[:, 1:]
        log_probs = jax.nn.log_softmax(shift_logits, axis=-1)
        batch_idx = jnp.arange(shift_labels.shape[0])[:, None]
        seq_idx = jnp.arange(shift_labels.shape[1])[None, :]
        target_log_probs = log_probs[batch_idx, seq_idx, shift_labels]
        return -jnp.mean(target_log_probs)

    # Test 4: Same as test 1 but in bfloat16 (no upcast)
    @jax.jit
    def loss_optax_bf16(logits, input_ids):
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        one_hot = jax.nn.one_hot(shift_labels, shift_logits.shape[-1])
        per_token = optax.softmax_cross_entropy(shift_logits, one_hot)
        return jnp.mean(per_token)

    # Test 5: Use int32 labels instead of uint32
    @jax.jit
    def loss_optax_f32_int32(logits, input_ids_i32):
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids_i32[:, 1:]
        one_hot = jax.nn.one_hot(shift_labels, shift_logits.shape[-1])
        per_token = optax.softmax_cross_entropy(
            shift_logits.astype(jnp.float32),
            one_hot.astype(jnp.float32),
        )
        return jnp.mean(per_token)

    input_ids_i32 = jnp.array(input_ids_np, dtype=jnp.int32)

    # Reference: numpy float64
    logits_f64 = logits_np[0, :-1, :].astype(np.float64)
    labels = input_ids_np[0, 1:]
    shifted = logits_f64 - logits_f64.max(axis=-1, keepdims=True)
    log_probs_ref = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    ref_loss = -np.mean(log_probs_ref[np.arange(len(labels)), labels])

    results = {}
    results["numpy_f64"] = ref_loss
    results["optax_f32_uint32"] = float(loss_optax_f32(logits, input_ids))
    results["manual_f32"] = float(loss_manual_f32(logits, input_ids))
    results["sparse_f32"] = float(loss_sparse_f32(logits, input_ids))
    results["optax_bf16"] = float(loss_optax_bf16(logits, input_ids))
    results["optax_f32_int32"] = float(loss_optax_f32_int32(logits, input_ids_i32))

    print(f"\n  {'Method':<25} {'Loss':>10} {'Error vs numpy':>15}")
    print("  " + "-" * 52)
    for name, val in results.items():
        err = val - ref_loss
        print(f"  {name:<25} {val:>10.6f} {err:>+15.6f}")
    print()


def compare():
    """Compare CPU vs TT activations at every layer."""
    import numpy as np

    print(f"\n{'=' * 91}")
    print(f"  LAYER-BY-LAYER DIVERGENCE: CPU vs TT")
    print(f"{'=' * 91}\n")

    cpu_files = sorted(
        f for f in os.listdir(RESULTS_DIR)
        if f.startswith("cpu_") and f.endswith(".npy")
    )

    if not cpu_files:
        print(f"ERROR: No CPU results in {RESULTS_DIR}. Run with --cpu first.")
        sys.exit(1)

    labels = [f.replace("cpu_", "").replace(".npy", "") for f in cpu_files]

    print(f"  {'Checkpoint':<22} {'Shape':<25} {'MaxDiff':>10} {'MeanDiff':>10} "
          f"{'CosSim':>10} {'Verdict':>8}")
    print("  " + "-" * 87)

    for label in labels:
        cpu_path = os.path.join(RESULTS_DIR, f"cpu_{label}.npy")
        tt_path = os.path.join(RESULTS_DIR, f"tt_{label}.npy")

        if not os.path.exists(tt_path):
            print(f"  {label:<22} MISSING TT file — run with --tt first")
            continue

        cpu_arr = np.load(cpu_path)
        tt_arr = np.load(tt_path)

        diff = np.abs(cpu_arr - tt_arr)
        max_diff = diff.max()
        mean_diff = diff.mean()

        cpu_flat = cpu_arr.ravel()
        tt_flat = tt_arr.ravel()
        cos_sim = np.dot(cpu_flat, tt_flat) / (
            np.linalg.norm(cpu_flat) * np.linalg.norm(tt_flat) + 1e-12
        )

        if cos_sim > 0.9999:
            verdict = "OK"
        elif cos_sim > 0.999:
            verdict = "WARN"
        elif cos_sim > 0.99:
            verdict = "BAD"
        else:
            verdict = "BROKEN"

        print(f"  {label:<22} {str(cpu_arr.shape):<25} {max_diff:>10.4f} {mean_diff:>10.4f} "
              f"{cos_sim:>10.6f} {verdict:>8}")

    # Compute loss from logits
    cpu_logits_path = sorted(f for f in os.listdir(RESULTS_DIR) if f.startswith("cpu_") and "logits" in f)
    tt_logits_path = sorted(f for f in os.listdir(RESULTS_DIR) if f.startswith("tt_") and "logits" in f)

    if cpu_logits_path and tt_logits_path:
        from transformers import AutoTokenizer
        from datasets import load_dataset
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        full_text = " ".join(t for t in ds["text"] if t.strip())
        all_ids = tokenizer(full_text, return_tensors="np")["input_ids"][0]
        input_ids = all_ids[:SEQ_LEN].reshape(1, SEQ_LEN)

        cpu_logits = np.load(os.path.join(RESULTS_DIR, cpu_logits_path[0]))
        tt_logits = np.load(os.path.join(RESULTS_DIR, tt_logits_path[0]))

        def xent_loss(logits, labels):
            logits = logits[0, :-1, :].astype(np.float64)
            labels = labels[0, 1:]
            shifted = logits - logits.max(axis=-1, keepdims=True)
            log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
            return -np.mean(log_probs[np.arange(len(labels)), labels])

        cpu_loss = xent_loss(cpu_logits, input_ids)
        tt_loss = xent_loss(tt_logits, input_ids)
        print(f"\n  Cross-entropy loss — CPU: {cpu_loss:.4f},  TT: {tt_loss:.4f},  "
              f"gap: {tt_loss - cpu_loss:+.4f}")

    print()


def main():
    args = set(sys.argv[1:])

    if "--all" in args:
        script = os.path.abspath(__file__)
        print("=" * 91)
        print("  STEP 1: CPU forward pass")
        print("=" * 91)
        subprocess.run([sys.executable, script, "--cpu"], check=True)
        print("\n" + "=" * 91)
        print("  STEP 2: TT forward pass")
        print("=" * 91)
        subprocess.run([sys.executable, script, "--tt"], check=True)
        print()
        compare()
    elif "--lora-all" in args:
        script = os.path.abspath(__file__)
        print("=" * 91)
        print("  STEP 1: Full-model + LoRA on CPU")
        print("=" * 91)
        subprocess.run([sys.executable, script, "--lora-cpu"], check=True)
        print("\n" + "=" * 91)
        print("  STEP 2: Full-model + LoRA on TT")
        print("=" * 91)
        subprocess.run([sys.executable, script, "--lora-tt"], check=True)
        print()
        compare_lora()
    elif "--cpu" in args:
        run_forward("cpu")
    elif "--tt" in args:
        run_forward("tt")
    elif "--compare" in args:
        compare()
    elif "--lora-cpu" in args:
        run_with_lora("cpu")
    elif "--lora-tt" in args:
        run_with_lora("tt")
    elif "--lora-compare" in args:
        compare_lora()
    elif "--test-loss-cpu" in args:
        test_loss_ops("cpu")
    elif "--test-loss-tt" in args:
        test_loss_ops("tt")
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
