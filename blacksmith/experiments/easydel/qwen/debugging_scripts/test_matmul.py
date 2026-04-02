# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Trace activations through the first transformer layer of the real
Qwen3-0.6B model and compare TT vs CPU at every checkpoint.

This isolates whether the loss=29 bug comes from:
  (a) float precision issues, or
  (b) wrong math in a specific EasyDel operation on TT.

Usage:
    python test_matmul.py --trace-cpu   # run on CPU, save activations
    python test_matmul.py --trace-tt    # run on TT,  save activations
    python test_matmul.py --trace-all   # both as subprocesses, then compare
"""

import sys
import subprocess
import os
import numpy as np

RESULTS_DIR = "/tmp/tt_layer_trace"
MODEL_NAME = "Qwen/Qwen3-0.6B"
SAMPLE_TEXT = "The capital of France is Paris and the largest city in Germany is"


def save(name, arr):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.save(os.path.join(RESULTS_DIR, name), arr)


def load(name):
    return np.load(os.path.join(RESULTS_DIR, name + ".npy"))


def to_np(x):
    return np.array(x, dtype=np.float32)


def stat(name, arr_np):
    print(f"  {name:<35} shape={str(arr_np.shape):<25} "
          f"min={arr_np.min():>12.4f}  max={arr_np.max():>12.4f}  "
          f"mean={arr_np.mean():>10.4f}  std={arr_np.std():>10.4f}")


def run_trace(platform):
    """Load real Qwen3-0.6B and trace activations through layer 0."""
    if platform == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    else:
        os.environ.setdefault("PJRT_DEVICE", "TT")
        os.environ.setdefault("XLA_STABLEHLO_COMPILE", "1")

    import jax
    import jax.numpy as jnp
    from transformers import AutoTokenizer
    from easydel import AutoEasyDeLModelForCausalLM

    # Apply the same GQA patch we use in training
    from blacksmith.experiments.easydel.qwen.attention_patch import apply_gqa_workaround
    apply_gqa_workaround()

    tag = platform
    device = jax.devices()[0]
    print(f"Platform: {device.platform} ({tag})")
    print(f"Model: {MODEL_NAME}")
    print(f"Input: \"{SAMPLE_TEXT}\"\n")

    # ── Tokenize ──────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokens = tokenizer(SAMPLE_TEXT, return_tensors="np")
    input_ids_np = tokens["input_ids"]
    seq_len = input_ids_np.shape[1]
    print(f"Token IDs ({seq_len} tokens): {input_ids_np[0].tolist()}\n")
    save(f"input_ids_{tag}", input_ids_np)

    # ── Load model ────────────────────────────────────────────
    print("Loading model...")
    model = AutoEasyDeLModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=jnp.bfloat16,
        config_kwargs={"mask_max_position_embeddings": 128},
    )

    devices_for_mesh = tuple(jax.devices()[:1])
    mesh = jax.make_mesh((1,), ("X",), devices=devices_for_mesh)
    model.config.set_model_mesh(mesh)
    print("Model loaded.\n")

    input_ids = jnp.array(input_ids_np)

    # Pre-compute cached properties to avoid tracer leaks inside JIT
    _ = model.model.frequencies
    _ = model.model.causal_mask

    with mesh:
        # Skip full forward pass (too large for TT flatbuffer).
        # Focus on per-step layer trace which compiles each op separately.
        print(f"{'=' * 70}")
        print(f"  LAYER 0 ACTIVATION TRACE [{tag}]")
        print("=" * 70)

        qwen_model = model.model
        layer0 = qwen_model.layers[0]
        attn = layer0.self_attn

        num_heads = attn.config.num_attention_heads
        num_kv_heads = attn.config.num_key_value_heads
        head_dim = attn.head_dim
        num_reps = num_heads // num_kv_heads
        sm_scale = head_dim ** -0.5
        print(f"  Config: num_heads={num_heads}, num_kv_heads={num_kv_heads}, "
              f"head_dim={head_dim}, num_reps={num_reps}\n")

        # ── 1. Embedding ──────────────────────────────────────
        @jax.jit
        def step_embedding(ids):
            return qwen_model.embed_tokens(ids.astype("i4"))

        embeddings = step_embedding(input_ids)
        save(f"embedding_{tag}", to_np(embeddings))
        stat("1  embedding", to_np(embeddings))

        # ── 2. Input LayerNorm ────────────────────────────────
        @jax.jit
        def step_input_norm(x):
            return layer0.input_layernorm(x)

        normed = step_input_norm(embeddings)
        save(f"input_norm_{tag}", to_np(normed))
        stat("2  input_layernorm", to_np(normed))

        # ── 3. Q, K, V projections ───────────────────────────
        @jax.jit
        def step_qkv(x):
            return attn.q_proj(x), attn.k_proj(x), attn.v_proj(x)

        q_raw, k_raw, v_raw = step_qkv(normed)
        save(f"q_raw_{tag}", to_np(q_raw))
        save(f"k_raw_{tag}", to_np(k_raw))
        save(f"v_raw_{tag}", to_np(v_raw))
        stat("3  q_proj (via EasyDel)", to_np(q_raw))
        stat("3  k_proj (via EasyDel)", to_np(k_raw))
        stat("3  v_proj (via EasyDel)", to_np(v_raw))

        # ── 3b. DIAGNOSTIC: raw weight inspection + manual matmul ─
        print(f"\n  --- DIAGNOSTIC: q_proj weight & manual matmul ---")

        # Extract raw weights from the q_proj layer
        q_kernel = attn.q_proj.kernel.value  # should be (1024, 2048)
        q_kernel_np = to_np(q_kernel)
        save(f"q_kernel_{tag}", q_kernel_np)
        stat("3b q_kernel (raw weight)", q_kernel_np)
        print(f"       q_kernel dtype={q_kernel.dtype}, shape={q_kernel.shape}")

        # Manual matmul: normed @ q_kernel (bypassing EasyDel entirely)
        @jax.jit
        def manual_matmul_bf16(x, w):
            x_bf16 = x.astype(jnp.bfloat16)
            w_bf16 = w.astype(jnp.bfloat16)
            return x_bf16 @ w_bf16

        q_manual = manual_matmul_bf16(normed, q_kernel)
        q_manual_np = to_np(q_manual)
        save(f"q_manual_{tag}", q_manual_np)
        stat("3b q_manual (x @ w)", q_manual_np)

        # Also try with einsum (same as EasyDel uses)
        @jax.jit
        def manual_einsum_bf16(x, w):
            x_bf16 = x.astype(jnp.bfloat16)
            w_bf16 = w.astype(jnp.bfloat16)
            return jnp.einsum("...ik,...kj->...ij", x_bf16, w_bf16)

        q_einsum = manual_einsum_bf16(normed, q_kernel)
        save(f"q_einsum_{tag}", to_np(q_einsum))
        stat("3b q_einsum (einsum)", to_np(q_einsum))

        # Compare EasyDel vs manual on same device
        diff_easydel_vs_manual = float(jnp.max(jnp.abs(q_raw - q_manual)))
        diff_easydel_vs_einsum = float(jnp.max(jnp.abs(q_raw - q_einsum)))
        print(f"       EasyDel vs manual @:   max_diff = {diff_easydel_vs_manual:.6f}")
        print(f"       EasyDel vs einsum:     max_diff = {diff_easydel_vs_einsum:.6f}")
        print(f"       manual @ vs einsum:    max_diff = {float(jnp.max(jnp.abs(q_manual - q_einsum))):.6f}")

        # ── 3c. DIAGNOSTIC: Is it a device transfer bug or real scramble? ─
        print(f"\n  --- DIAGNOSTIC: device transfer round-trip test ---")

        # Create a known test matrix on CPU, put it on device, read back
        test_w = jnp.arange(1024 * 2048, dtype=jnp.float32).reshape(1024, 2048)
        test_w_back = to_np(test_w)
        roundtrip_diff = np.max(np.abs(test_w_back - np.arange(1024*2048, dtype=np.float32).reshape(1024, 2048)))
        print(f"       arange(1024,2048) roundtrip max_diff = {roundtrip_diff:.6f}")

        # Now test: create weight on CPU, manually do matmul on CPU, compare
        # Use the CPU's kernel values (loaded from file if available) with TT's normed
        cpu_kernel_path = os.path.join(RESULTS_DIR, "q_kernel_cpu.npy")
        if os.path.exists(cpu_kernel_path):
            cpu_q_kernel = np.load(cpu_kernel_path)
            cpu_q_kernel_jax = jnp.array(cpu_q_kernel)

            @jax.jit
            def matmul_with_cpu_kernel(x, w):
                return x.astype(jnp.bfloat16) @ w.astype(jnp.bfloat16)

            q_with_cpu_kernel = matmul_with_cpu_kernel(normed, cpu_q_kernel_jax)
            q_wck_np = to_np(q_with_cpu_kernel)
            save(f"q_with_cpu_kernel_{tag}", q_wck_np)
            stat("3c q using CPU's kernel", q_wck_np)

            # If this matches CPU's q_raw, then the matmul is correct
            # and the weight loading is the problem
            cpu_q_raw_path = os.path.join(RESULTS_DIR, "q_raw_cpu.npy")
            if os.path.exists(cpu_q_raw_path):
                cpu_q_raw = np.load(cpu_q_raw_path)
                diff = np.max(np.abs(q_wck_np - cpu_q_raw))
                cos = np.dot(q_wck_np.flatten(), cpu_q_raw.flatten()) / (
                    np.linalg.norm(q_wck_np.flatten()) * np.linalg.norm(cpu_q_raw.flatten()) + 1e-12)
                print(f"       q(TT normed @ CPU kernel) vs CPU q_raw: max_diff={diff:.4f} cos={cos:.6f}")
        else:
            print(f"       (run --trace-cpu first to get CPU kernel for cross-check)")
        print()

        # ── 4. Reshape to 4D ─────────────────────────────────
        @jax.jit
        def step_reshape(q, k, v):
            b, s, _ = q.shape
            return (q.reshape(b, s, num_heads, head_dim),
                    k.reshape(b, s, num_kv_heads, head_dim),
                    v.reshape(b, s, num_kv_heads, head_dim))

        q_4d, k_4d, v_4d = step_reshape(q_raw, k_raw, v_raw)
        save(f"q_4d_{tag}", to_np(q_4d))
        save(f"k_4d_{tag}", to_np(k_4d))
        stat("4  q_reshaped (4D)", to_np(q_4d))
        stat("4  k_reshaped (4D)", to_np(k_4d))
        stat("4  v_reshaped (4D)", to_np(v_4d))

        # ── 5. Q-norm and K-norm (Qwen3 per-head RMSNorm) ────
        @jax.jit
        def step_qk_norms(q, k):
            return attn.q_norm(q), attn.k_norm(k)

        q_normed, k_normed = step_qk_norms(q_4d, k_4d)
        save(f"q_normed_{tag}", to_np(q_normed))
        save(f"k_normed_{tag}", to_np(k_normed))
        stat("5  q_norm", to_np(q_normed))
        stat("5  k_norm", to_np(k_normed))

        # ── 6. RoPE ──────────────────────────────────────────
        freqs = qwen_model.frequencies  # already pre-computed above

        @jax.jit
        def step_rope(q, k):
            pos = jnp.arange(seq_len)[None, :]
            return attn.rotary(positions=pos, query=q, key=k,
                               frequencies=freqs)

        q_rope, k_rope = step_rope(q_normed, k_normed)
        save(f"q_rope_{tag}", to_np(q_rope))
        save(f"k_rope_{tag}", to_np(k_rope))
        stat("6  q_rope", to_np(q_rope))
        stat("6  k_rope", to_np(k_rope))

        # ── 7. GQA repeat (K,V: 8 heads -> 16 heads) ────────
        @jax.jit
        def step_gqa_repeat(k, v):
            return (jnp.repeat(k, num_reps, axis=2),
                    jnp.repeat(v, num_reps, axis=2))

        k_rep, v_rep = step_gqa_repeat(k_rope, v_4d)
        save(f"k_repeated_{tag}", to_np(k_rep))
        save(f"v_repeated_{tag}", to_np(v_rep))
        stat("7  k_repeated (GQA)", to_np(k_rep))
        stat("7  v_repeated (GQA)", to_np(v_rep))

        # ── 8. Attention weights QK^T * scale ────────────────
        @jax.jit
        def step_attn_weights(q, k):
            return jnp.einsum("bshd,bmhd->bhsm", q * sm_scale, k)

        aw = step_attn_weights(q_rope, k_rep)
        save(f"attn_weights_raw_{tag}", to_np(aw))
        stat("8  attn_weights (QK^T)", to_np(aw))

        # ── 9. Causal mask ───────────────────────────────────
        @jax.jit
        def step_causal_mask(aw):
            mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
            return jnp.where(mask[None, None], aw, jnp.finfo(aw.dtype).min)

        aw_masked = step_causal_mask(aw)
        save(f"attn_weights_masked_{tag}", to_np(aw_masked))
        stat("9  attn_weights (masked)", to_np(aw_masked))

        # ── 10. Softmax (f32 upcast, same as attention_patch) ─
        @jax.jit
        def step_softmax(aw):
            return jax.nn.softmax(aw.astype(jnp.float32), axis=-1).astype(jnp.bfloat16)

        aw_soft = step_softmax(aw_masked)
        save(f"attn_probs_{tag}", to_np(aw_soft))
        stat("10 attn_probs (softmax)", to_np(aw_soft))

        # ── 11. Weighted sum of V ────────────────────────────
        @jax.jit
        def step_attn_output(aw, v):
            return jnp.einsum("bhsm,bmhd->bshd", aw, v)

        attn_out = step_attn_output(aw_soft, v_rep)
        save(f"attn_output_4d_{tag}", to_np(attn_out))
        stat("11 attn_output (4D)", to_np(attn_out))

        # ── 12. Merge heads + O projection ───────────────────
        @jax.jit
        def step_o_proj(x):
            merged = x.reshape((*x.shape[:2], -1))
            return attn.o_proj(merged)

        o_out = step_o_proj(attn_out)
        save(f"o_proj_{tag}", to_np(o_out))
        stat("12 o_proj", to_np(o_out))

        # ── 13. Residual 1 (embedding + attn) ────────────────
        residual1 = jnp.array(to_np(embeddings) + to_np(o_out), dtype=jnp.bfloat16)
        # Use jit to be fair
        @jax.jit
        def step_residual(x, y):
            return x + y

        residual1 = step_residual(embeddings, o_out)
        save(f"residual1_{tag}", to_np(residual1))
        stat("13 residual1 (emb+attn)", to_np(residual1))

        # ── 14. Post-attention LayerNorm ──────────────────────
        @jax.jit
        def step_post_norm(x):
            return layer0.post_attention_layernorm(x)

        post_norm = step_post_norm(residual1)
        save(f"post_attn_norm_{tag}", to_np(post_norm))
        stat("14 post_attn_layernorm", to_np(post_norm))

        # ── 15. MLP ──────────────────────────────────────────
        @jax.jit
        def step_mlp(x):
            return layer0.mlp(x)

        mlp_out = step_mlp(post_norm)
        save(f"mlp_out_{tag}", to_np(mlp_out))
        stat("15 mlp_output", to_np(mlp_out))

        # ── 16. Residual 2 (layer 0 final output) ────────────
        residual2 = step_residual(residual1, mlp_out)
        save(f"residual2_{tag}", to_np(residual2))
        stat("16 residual2 (layer0 out)", to_np(residual2))

    print(f"\n{'=' * 70}")
    print(f"  DONE [{tag}] - saved to {RESULTS_DIR}")
    print("=" * 70)


def compare_traces():
    """Compare TT vs CPU activation traces."""
    print("\n" + "=" * 70)
    print("  ACTIVATION TRACE COMPARISON: TT vs CPU")
    print("=" * 70)

    checkpoints = [
        ("embedding", "1  Embedding"),
        ("input_norm", "2  Input LayerNorm"),
        ("q_kernel", "3b Q weight (raw kernel)"),
        ("q_manual", "3b Q manual (x @ w)"),
        ("q_einsum", "3b Q einsum"),
        ("q_raw", "3  Q proj (EasyDel)"),
        ("k_raw", "3  K proj (EasyDel)"),
        ("v_raw", "3  V proj (EasyDel)"),
        ("q_4d", "4  Q reshaped"),
        ("k_4d", "4  K reshaped"),
        ("q_normed", "5  Q-norm (head RMSNorm)"),
        ("k_normed", "5  K-norm (head RMSNorm)"),
        ("q_rope", "6  Q after RoPE"),
        ("k_rope", "6  K after RoPE"),
        ("k_repeated", "7  K after GQA repeat"),
        ("v_repeated", "7  V after GQA repeat"),
        ("attn_weights_raw", "8  QK^T (raw attn weights)"),
        ("attn_weights_masked", "9  QK^T (causal masked)"),
        ("attn_probs", "10 Attn softmax probs"),
        ("attn_output_4d", "11 Attn output (4d)"),
        ("o_proj", "12 O projection"),
        ("residual1", "13 Residual 1 (emb + attn)"),
        ("post_attn_norm", "14 Post-attn LayerNorm"),
        ("mlp_out", "15 MLP output"),
        ("residual2", "16 Layer 0 final output"),
    ]

    print(f"\n  {'checkpoint':<35} {'max_diff':>12} {'mean_diff':>12} {'cos_sim':>10} {'verdict':>10}")
    print(f"  {'-' * 82}")

    for key, label in checkpoints:
        try:
            tt = load(f"{key}_tt")
            cpu = load(f"{key}_cpu")
            max_d = np.max(np.abs(tt - cpu))
            mean_d = np.mean(np.abs(tt - cpu))

            tt_flat = tt.flatten()
            cpu_flat = cpu.flatten()
            cos = np.dot(tt_flat, cpu_flat) / (np.linalg.norm(tt_flat) * np.linalg.norm(cpu_flat) + 1e-12)

            if max_d < 0.01:
                verdict = "OK"
            elif max_d < 0.5:
                verdict = "WARN"
            elif max_d < 5.0:
                verdict = "BAD"
            else:
                verdict = "BROKEN"

            marker = " <<<" if verdict in ("BAD", "BROKEN") else ""
            print(f"  {label:<35} {max_d:>12.4f} {mean_d:>12.6f} {cos:>10.6f} {verdict:>10}{marker}")
        except FileNotFoundError:
            print(f"  {label:<35} {'MISSING':>12}")

    print()  # blank line at end


def run_trace_all():
    script = __file__
    for mode, label in [("--trace-cpu", "CPU"), ("--trace-tt", "TT")]:
        print(f"\n{'#' * 70}")
        print(f"#  Running trace on {label}: python test_matmul.py {mode}")
        print(f"{'#' * 70}")
        r = subprocess.run([sys.executable, script, mode], text=True)
        if r.returncode != 0:
            print(f"  *** {mode} failed with code {r.returncode} ***")
            return
    compare_traces()


def run_weight_diagnostic():
    """Determine exactly where weights get scrambled during EasyDel loading.

    Tests:
    1. Load raw safetensors weight -> jnp.array() on TT -> readback
    2. Load via EasyDel -> extract weight -> readback
    3. Compare both with ground truth from safetensors
    """
    import jax
    import jax.numpy as jnp
    from safetensors import safe_open

    print("=" * 70)
    print("  WEIGHT LOADING DIAGNOSTIC")
    print("=" * 70)

    # 1. Load ground-truth weight directly from safetensors
    from huggingface_hub import hf_hub_download
    target_key = "model.layers.0.self_attn.q_proj.weight"

    # Try single file first, then sharded index
    try:
        shard_path = hf_hub_download(MODEL_NAME, "model.safetensors")
    except Exception:
        import json
        idx_path = hf_hub_download(MODEL_NAME, "model.safetensors.index.json")
        with open(idx_path) as f:
            weight_map = json.load(f)["weight_map"]
        shard_file = weight_map[target_key]
        shard_path = hf_hub_download(MODEL_NAME, shard_file)

    with safe_open(shard_path, framework="numpy") as f:
        q_weight_np = f.get_tensor(target_key)  # PyTorch layout: (out, in)

    print(f"\n  Ground truth from safetensors:")
    print(f"    key: {target_key}")
    print(f"    shape: {q_weight_np.shape}, dtype: {q_weight_np.dtype}")
    print(f"    [0,:5]: {q_weight_np[0,:5]}")
    print(f"    [1,:5]: {q_weight_np[1,:5]}")

    # PyTorch stores weights as (out_features, in_features)
    # JAX/EasyDel stores as (in_features, out_features) => need transpose
    q_weight_T = q_weight_np.T.copy()
    print(f"\n  After transpose (JAX layout): shape={q_weight_T.shape}")
    print(f"    [0,:5]: {q_weight_T[0,:5]}")
    print(f"    [1,:5]: {q_weight_T[1,:5]}")

    # 2. Place this weight on TT via jnp.array and read it back
    q_jax = jnp.array(q_weight_T)
    q_roundtrip = np.array(q_jax, dtype=np.float32)
    diff_direct = np.max(np.abs(q_roundtrip - q_weight_T))
    print(f"\n  Direct jnp.array() roundtrip: max_diff = {diff_direct:.6f}")
    print(f"    roundtrip [0,:5]: {q_roundtrip[0,:5]}")

    # 3. Also test: does jnp.array on TT work for float32 2D?
    test_small = np.arange(12, dtype=np.float32).reshape(3, 4)
    test_jax = jnp.array(test_small)
    test_back = np.array(test_jax, dtype=np.float32)
    print(f"\n  Small (3,4) arange roundtrip: max_diff = {np.max(np.abs(test_back - test_small)):.6f}")
    print(f"    original: {test_small.flatten()[:6]}")
    print(f"    roundtrip: {test_back.flatten()[:6]}")

    # 4. Test larger matrices at various sizes
    for shape in [(32, 32), (32, 64), (64, 32), (128, 128), (1024, 1024), (1024, 2048)]:
        test = np.arange(shape[0]*shape[1], dtype=np.float32).reshape(shape)
        back = np.array(jnp.array(test), dtype=np.float32)
        d = np.max(np.abs(back - test))
        eq = "OK" if d == 0 else f"BROKEN (max_diff={d:.1f})"
        print(f"  roundtrip {str(shape):>14}: {eq}")

    # 5. Now load EasyDel model and compare its weights with ground truth
    print(f"\n  --- Loading EasyDel model on this platform ---")
    from flax import nnx as nn
    from easydel import AutoEasyDeLModelForCausalLM

    model = AutoEasyDeLModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, 1, 1, 1),
        config_kwargs={"mask_max_position_embeddings": 128},
        auto_shard_model=True,
        verbose=False,
    )

    attn = model.model.layers[0].self_attn
    q_kernel = attn.q_proj.kernel.value  # (in_features, out_features)
    q_kernel_np = np.array(q_kernel, dtype=np.float32)

    print(f"\n  EasyDel q_proj kernel: shape={q_kernel_np.shape}, dtype={q_kernel.dtype}")
    print(f"    [0,:5]: {q_kernel_np[0,:5]}")
    print(f"    [1,:5]: {q_kernel_np[1,:5]}")

    diff = np.max(np.abs(q_kernel_np - q_weight_T))
    cos = np.dot(q_kernel_np.flatten(), q_weight_T.flatten()) / (
        np.linalg.norm(q_kernel_np.flatten()) * np.linalg.norm(q_weight_T.flatten()) + 1e-12)
    print(f"\n  EasyDel kernel vs ground-truth (transposed safetensors):")
    print(f"    max_diff = {diff:.6f}")
    print(f"    cos_sim  = {cos:.6f}")

    # Check if sorted values match (same data, different order)
    sorted_diff = np.max(np.abs(np.sort(q_kernel_np.flatten()) - np.sort(q_weight_T.flatten())))
    print(f"    sorted values max_diff = {sorted_diff:.8f}")

    # 6. CRITICAL TEST: What does safe_flax actually return?
    print(f"\n  --- Testing safe_flax loading path directly ---")
    import safetensors.flax as safe_flax
    with safe_flax.safe_open(shard_path, framework="flax") as f:
        raw_flax_tensor = f.get_tensor(target_key)
    print(f"  safe_flax tensor: shape={raw_flax_tensor.shape}, dtype={raw_flax_tensor.dtype}")
    print(f"    device: {raw_flax_tensor.devices()}")
    raw_flax_np = np.array(raw_flax_tensor, dtype=np.float32)
    print(f"    [0,:5]: {raw_flax_np[0,:5]}")

    # Ground truth is (2048, 1024) in PyTorch layout, bfloat16
    # safe_flax returns the same shape as stored: (2048, 1024)
    gt_for_comparison = q_weight_np  # original (2048, 1024) from numpy loader
    diff_flax = np.max(np.abs(raw_flax_np - gt_for_comparison.astype(np.float32)))
    cos_flax = np.dot(raw_flax_np.flatten(), gt_for_comparison.flatten().astype(np.float32)) / (
        np.linalg.norm(raw_flax_np.flatten()) * np.linalg.norm(gt_for_comparison.flatten().astype(np.float32)) + 1e-12)
    print(f"  safe_flax vs numpy ground truth: max_diff={diff_flax:.6f} cos_sim={cos_flax:.6f}")

    # Now test: can we reproduce the scramble by transposing the safe_flax tensor?
    raw_flax_T = raw_flax_tensor.T
    raw_flax_T_np = np.array(raw_flax_T, dtype=np.float32)
    diff_T_flax = np.max(np.abs(raw_flax_T_np - q_weight_T.astype(np.float32)))
    print(f"  safe_flax.T vs ground_truth.T: max_diff={diff_T_flax:.6f}")
    print(f"    safe_flax.T [0,:5]: {raw_flax_T_np[0,:5]}")
    print(f"    ground_truth.T [0,:5]: {q_weight_T[0,:5].astype(np.float32)}")

    # Test bfloat16 roundtrip specifically
    print(f"\n  --- Testing bfloat16 roundtrip on TT ---")
    bf16_np = q_weight_T.astype(np.float32)  # ground truth is already bf16 values
    bf16_jax = jnp.array(bf16_np).astype(jnp.bfloat16)
    bf16_back = np.array(bf16_jax, dtype=np.float32)
    diff_bf16 = np.max(np.abs(bf16_back - bf16_np))
    print(f"  f32->bf16 roundtrip (1024,2048): max_diff={diff_bf16:.6f}")

    # Test with actual bfloat16 numpy data
    import ml_dtypes
    bf16_raw = q_weight_np.view(ml_dtypes.bfloat16)  # keep original bfloat16 bytes
    bf16_jax2 = jnp.array(bf16_raw)  # place bfloat16 on TT
    bf16_back2 = np.array(bf16_jax2, dtype=np.float32)
    diff_bf16_2 = np.max(np.abs(bf16_back2 - q_weight_np.astype(np.float32)))
    print(f"  native bf16 roundtrip (2048,1024): max_diff={diff_bf16_2:.6f}")

    # The EasyDel loading transposes PyTorch weights (out, in) -> (in, out)
    # Does doing .T on a TT bfloat16 array cause problems?
    bf16_jax2_T = bf16_jax2.T
    bf16_T_back = np.array(bf16_jax2_T, dtype=np.float32)
    diff_T = np.max(np.abs(bf16_T_back - q_weight_T.astype(np.float32)))
    print(f"  bf16 transpose roundtrip (1024,2048): max_diff={diff_T:.6f}")
    print(f"    bf16.T [0,:5]: {bf16_T_back[0,:5]}")
    print(f"    expected [0,:5]: {q_weight_T[0,:5].astype(np.float32)}")

    # 7. CRITICAL: Manually replicate the exact EasyDel conversion pipeline
    print(f"\n  --- Manual replication of EasyDel conversion pipeline ---")
    import torch
    from transformers import AutoModelForCausalLM

    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    pt_weight = hf_model.state_dict()["model.layers.0.self_attn.q_proj.weight"]
    print(f"  Step 0: PyTorch weight: shape={pt_weight.shape}, dtype={pt_weight.dtype}")
    print(f"    [0,:5]: {pt_weight[0,:5].tolist()}")

    pt_permuted = pt_weight.permute(1, 0)
    print(f"  Step 1: After permute(1,0): shape={pt_permuted.shape}")
    print(f"    [0,:5]: {pt_permuted[0,:5].tolist()}")
    print(f"    is_contiguous: {pt_permuted.is_contiguous()}")

    np_array = pt_permuted.cpu().detach().numpy()
    print(f"  Step 2: numpy: shape={np_array.shape}, dtype={np_array.dtype}")
    print(f"    [0,:5]: {np_array[0,:5]}")

    # Verify numpy is correctly transposed
    np_gt = q_weight_T.astype(np.float16)  # ground truth in float16
    diff_np = np.max(np.abs(np_array.astype(np.float32) - np_gt.astype(np.float32)))
    print(f"    numpy vs ground_truth.T: max_diff={diff_np:.6f}")

    jax_array = jnp.asarray(np_array, dtype=jnp.float32)
    jax_back = np.array(jax_array, dtype=np.float32)
    print(f"  Step 3: jnp.asarray (dtype=float32): shape={jax_array.shape}")
    print(f"    [0,:5]: {jax_back[0,:5]}")
    diff_jax = np.max(np.abs(jax_back - np_array.astype(np.float32)))
    print(f"    jax roundtrip vs numpy: max_diff={diff_jax:.6f}")

    # Now apply shard function (if applicable)
    from easydel.modules.auto.auto_configuration import AutoShardAndGatherFunctions
    shard_fns, _ = AutoShardAndGatherFunctions.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME,
        sharding_axis_dims=(1, 1, 1, 1, 1),
        sharding_axis_names=("dp", "fsdp", "ep", "tp", "sp"),
    )
    shard_fns_flat = {}
    from easydel.utils.traversals import flatten_dict
    if shard_fns:
        shard_fns_flat = flatten_dict(shard_fns)

    target_key = ("model", "layers", 0, "self_attn", "q_proj", "kernel")
    shard_fn = shard_fns_flat.get(target_key, None)
    print(f"  Step 4: shard_fn for q_proj.kernel: {shard_fn}")
    if shard_fn is not None:
        sharded = shard_fn(jax_array)
        sharded_back = np.array(sharded, dtype=np.float32)
        diff_sharded = np.max(np.abs(sharded_back - np_array.astype(np.float32)))
        print(f"    after shard: shape={sharded.shape}, max_diff_vs_numpy={diff_sharded:.6f}")
        print(f"    [0,:5]: {sharded_back[0,:5]}")
    else:
        print(f"    No shard function for this key")

    del hf_model, pt_weight

    # 8. Test: Load without auto_shard, compare
    print(f"\n  --- Loading EasyDel model WITHOUT auto_shard ---")
    model2 = AutoEasyDeLModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, 1, 1, 1),
        config_kwargs={"mask_max_position_embeddings": 128},
        auto_shard_model=False,
        verbose=False,
    )
    q_kernel2 = model2.model.layers[0].self_attn.q_proj.kernel.value
    q_kernel2_np = np.array(q_kernel2, dtype=np.float32)

    diff2 = np.max(np.abs(q_kernel2_np - q_weight_T))
    cos2 = np.dot(q_kernel2_np.flatten(), q_weight_T.flatten()) / (
        np.linalg.norm(q_kernel2_np.flatten()) * np.linalg.norm(q_weight_T.flatten()) + 1e-12)
    print(f"  No-shard kernel vs ground-truth: max_diff={diff2:.6f}  cos_sim={cos2:.6f}")
    print(f"    [0,:5]: {q_kernel2_np[0,:5]}")

    # 7. Compare auto_shard vs no_shard
    diff_shard = np.max(np.abs(q_kernel_np - q_kernel2_np))
    print(f"  auto_shard vs no_shard: max_diff={diff_shard:.6f}")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    if "--trace-tt" in sys.argv:
        run_trace("tt")
    elif "--trace-cpu" in sys.argv:
        run_trace("cpu")
    elif "--trace-compare" in sys.argv:
        compare_traces()
    elif "--trace-all" in sys.argv:
        run_trace_all()
    elif "--weight-diag" in sys.argv:
        run_weight_diagnostic()
    else:
        print("Usage: python test_matmul.py [--trace-tt|--trace-cpu|--trace-all|--trace-compare|--weight-diag]")
