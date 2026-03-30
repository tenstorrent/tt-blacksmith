# Debugging Plan: `norm=inf` in GPT-OSS 20B Finetuning

## Problem Statement

During partial-freeze finetuning of `openai/gpt-oss-20b` (layers 12–18 trainable), gradient
norms of `inf` appear starting from `accumulation_step=2`. The problem is a heisenbug:
adding print statements / `.detach()` calls changes the XLA compiled graph and the bug
disappears.

---

## Root Cause (under investigation, 2026-03-29)

**The TT-XLA compiler's memory planner assigns overlapping DRAM addresses to the
embedding lookup table (`%2145`, `routing_weights.grad` reshaped as `[4096×1]`) and some
other intermediate tensor in the step 2 backward graph. When that other tensor is
overwritten with large gradient values during execution, the embedding table reads those
values instead of the correct `routing_weights.grad`, producing `min=-3.69e+19`.**

The zero-copy reshape OOB hypothesis was **disproved** by an eager TTNN test
(`test_reshape_oob.py`): `ttnn.reshape([128×32]→[4096×1])` correctly allocates and copies
all 4096 positions in eager mode.

**The bug is data-dependent** — it only triggers for certain input data (not every batch).
This confirms the root cause is about WHAT VALUES a buffer-aliased tensor contains, not
whether the alias itself exists. The alias is a static compile-time fact; the explosion
depends on whether the aliased tensor holds large or benign values for the specific input.

### Gradient explosion chain (step 2, layer 15)

```
routing_weights.grad           norm=1.54e-03  ← NORMAL: scatter backward INPUT
  │
  │  scatter backward  (TT/XLA computes INCORRECTLY)
  ▼
topk_values_post_softmax.grad  min=-3.69e+19  ← EXPLOSION: scatter backward OUTPUT
  │
  │  softmax backward  (correct, but operates on exploded input)
  ▼
topk_values_pre_softmax.grad   norm=1.04e+19  ← propagated explosion
  │
  │  topk backward
  ▼
router_logits.grad             norm=1.04e+19  ← propagated
  │
  │  linear backward
  ▼
router_input.grad              norm=2.90e+18  ← propagated
```

Layers 16, 17, 18 are clean at step 2. Layers 12–14 gradient parameters explode in step 2
because they receive the exploded hidden state gradient propagating backward through the
residual stream.

---

## Evidence Chain

### E1: Not auxiliary loss
`output_router_logits: false` in `config.json`. The `if output_router_logits:` guard in
`GptOssForCausalLM.forward` (line 693) means no aux loss is computed. The decoder layer
does `hidden_states, _ = self.mlp(hidden_states)` (line 389), discarding router scores.
There is no second gradient path into `router_scores`.

### E2: Not OOB topk indices
Experiment 1 (2026-03-28): `topk_indices_raw` logged at all 7 trainable layers (12–18)
for both steps. **`oob_count=0` everywhere.** TT topk returns valid indices (0–31).

### E3: Not a CPU/PyTorch bug
Experiment 2 (2026-03-28): `use_tt: False`. CPU run, full 20B model, 2 steps. All 7
layers both steps produce `topk_values_post_softmax.grad` ~1–3e-3. **Zero inf/nan.**

### E4: Not a scatter backward kernel bug in isolation
Experiment 2b (2026-03-28): Minimal reproducer — `[128,32]` router logits, `[128,4]` topk,
same dtypes, 2-step gradient accumulation on TT. **Both steps PASS on TT** (matches CPU).
The scatter backward kernel computes correctly when compiled as a standalone graph.

### E5: Heisenbug confirms XLA compilation dependency
Experiment 3 (2026-03-28): Adding `t.detach()` calls to print forward activation stats
changes the XLA traced autograd graph. With those calls present, the explosion disappears
entirely — layer 15 step 2 is clean on TT. The bug depends on the exact XLA graph
structure. Different debug code → different kernel fusions → bug appears or disappears.

---

## Conclusion

The bug is a **compiled-graph memory plan aliasing bug** in the TT-XLA compiler. The
scatter backward computes `routing_weights.grad` ([128×32]) and builds the embedding
lookup table ([4096×1]) for the gather backward:

```
all_gather → reshape([128×32] → [4096×1]) → to_layout(row_major) → embedding
```

The individual ops are correct in isolation (verified by eager TTNN test with real data,
Exp 6). The bug only manifests in compiled flatbuffer execution.

**Root cause is still unknown.** The "memory plan aliasing" theory is one hypothesis
but has NOT been confirmed. What IS confirmed:
- The explosion happens between `routing_weights.grad` (clean, ~1.54e-3 norm) and
  `topk_values_post_softmax.grad` (exploded, min=-3.69e+19)
- This subgraph is: `all_gather → reshape → to_layout → embedding`
- Each of these ops works correctly in eager ttnn with real data
- The bug is data-dependent and layer-specific (layer 15 only)
- The bug is a heisenbug (graph changes make it disappear)

---

## Exact TTNN op sequence (from IR analysis, 2026-03-29)

The explosion originates at the embedding lookup for `topk_values_post_softmax.grad`.
All ops are in `ttnn_1774741201517.mlir` (step 2, layer 15 router backward):

```
# Lines 8677–8692, step 2 TTNN IR
%2138 = "ttnn.all_gather"(%2133)       # tensor<128x32xbf16> — routing_weights.grad
%2139 = "ttnn.reshape"(%2138, [4096,1])# tensor<4096x1xbf16> — table for embedding
"ttnn.deallocate"(%2138)
... (index computation: typecast→matmul→reshape→typecast→to_layout for flat indices)
%2145 = "ttnn.to_layout"(%2139, row_major)  # [4096×1] bf16 row-major embedding table
"ttnn.deallocate"(%2139)
%2146 = "ttnn.embedding"(%2144, %2145)      # OUTPUT min = -3.69e+19 ← EXPLOSION ORIGIN
```

- Each individual op is correct in isolation (verified: eager TTNN test, Exp 2b)
- The static IR structure is identical between step 1 and step 2
- The `scatter.7166`/`scatter.7171` locs are **attention backward ops**, not router ops

**The bug is in the compiled graph's execution**, not in the individual ops.

---

## Next Steps (2026-03-29)

1. **Revert C++ runtime changes** and sanity-check the bug still reproduces
2. **Extract the faulty subgraph** — the ops between `routing_weights.grad` (clean)
   and `topk_values_post_softmax.grad` (exploded) from the emitpy-generated Python:
   ```
   all_gather → reshape([128,32]→[4096,1]) → to_layout(row_major) → embedding
   ```
   This is lines 6946–6961 of `step2_emitpy.py`.
3. **Get the real inputs** — capture the FRESH `routing_weights.grad` (the embedding
   table) and `topk_indices` from the training run at step 2. The topk indices are in
   the exported tensors. The fresh routing_weights.grad can be captured via the existing
   Python debug hooks (which already print its norm=1.54e-3, so we know they work).
4. **Build a standalone ttnn reproducer** that runs this exact subgraph on 8 devices
   with the real inputs. If it reproduces the explosion: we have the minimal repro.
   If it doesn't: the bug requires more graph context (buffer pressure from surrounding
   ops).
5. If (4) doesn't reproduce: incrementally add surrounding ops from the emitpy script
   until the bug appears, narrowing down which context triggers it.

---

## What to report to TT

1. **Compiled-graph execution bug** in step 2 backward
2. **Data-dependent** — only triggers for certain input data
3. **Step 2 only** — step 1's compiled graph works correctly
4. **Heisenbug** — graph changes make it disappear
5. **Layer 15 specific** — layers 16–18 same pattern but clean
6. **IR files**: `irs_debug/irs/ttnn_1774740861345.mlir` (step 1, works) and
   `irs_debug/irs/ttnn_1774741201517.mlir` (step 2, fails). Failing embedding at
   line 8692.
7. **Environment**: `torch.compile(backend="tt")` + `torch_xla.sync(wait=True)`, BF16,
   `fp32_dest_acc_en=True`, `math_fidelity=hifi4`, gradient accumulation, 8-chip mesh.
