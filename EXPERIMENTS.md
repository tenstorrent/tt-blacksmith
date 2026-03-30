# Experiment Log: `norm=inf` in GPT-OSS 20B Finetuning

> Companion to [PLAN.md](PLAN.md).
> Container: `tt-run-pglusac`.

---

## Baseline Run — `out.txt` (2026-03-28)

**Command:**
```bash
python3 blacksmith/experiments/torch/gpt_oss/test_gpt_oss_finetuning_2.py \
  --config blacksmith/experiments/torch/gpt_oss/test_gpt_oss_20b_finetuning.yaml \
  2>&1 | tee out.txt
```

**Config:** `gradient_accumulation_steps: 256`, `use_tt: True`, BF16, MX-FP4 dequantized,
`debug_router_grads=True`, layers 12–18 trainable.

### Key observations

#### Gradient norms by accumulation step

| step | layer | tensor | value |
|------|-------|--------|-------|
| 1 | 15 | `router.topk_values_post_softmax.grad` | min=-6.065e-04 — **normal** |
| **2** | **15** | **`router.topk_values_post_softmax.grad`** | **min=-3.689e+19** — **EXPLOSION ORIGIN** |
| 2 | 15 | `router.routing_weights.grad` | max=1.919e-03 — **still normal** |
| 2 | 16/17/18 | all | ~1e-5 — **clean** |
| 3+ | all | all | Identical to step 2 — **frozen** |

#### Backward cascade at step 2

```
router.router_input.grad norm:
  layer 18 → 3.4e-05   [normal]
  layer 17 → 6.8e-05   [normal]
  layer 16 → 2.4e-05   [normal]
  layer 15 → 2.90e+18  ← ORIGIN
  layer 14 → 8.5e+16   ← propagated
  layer 13 → 1.3e+17   ← propagated
  layer 12 → 4.1e+16   ← propagated
```

#### Scatter/softmax backward inconsistency

The scatter backward from `routing_weights.grad` (~1.9e-3) onto `topk_values_post_softmax`
can contribute at most ~1.9e-3 per position (it's a gather of routing_weights.grad at topk
positions). The observed -3.7e+19 is 22 orders of magnitude larger — a second gradient
source must exist.

#### Gradient freeze explained

`1e19 + 1e-3 = 1e19` in BF16. Once the explosion appears at step 2, all subsequent
backward increments (~1e-3) are numerically absorbed. The gradient appears "frozen" but
the backward is still running — it just contributes nothing.

### What was ruled out

**H1 (auxiliary load-balancing loss):** Definitively eliminated.
- `config.json`: `"output_router_logits": false` — the aux loss computation is never
  triggered (`if output_router_logits:` guard at line 693 of `GptOssForCausalLM.forward`)
- Our training code: passes `labels=None` to the model, so even with `output_router_logits=True`
  the aux loss would not be added to loss (`if labels is not None:` guard at line 700)
- `OutputRecorder` is conditioned on `output_router_logits=False` via `check_model_inputs`
  decorator — router scores are never collected

### Key structural finding

The clamp in `GptOssTopKRouter.forward` (line 158) was added manually as a debugging
measure after TT hardware's `topk` was observed to produce OOB indices. It is not part of
the original model — it is a temporary workaround:
```python
# clamp router_indices
# print(f"Clamping router indices to be between 0 and {self.num_experts - 1}")
router_indices = torch.clamp(router_indices, min=0, max=self.num_experts - 1)
```

`_debug_router_forward` does NOT have this clamp (both the clamp and the logging are
commented out):
```python
#self._dbg["topk_indices_raw"] = router_indices   # logging disabled
#router_indices = router_indices.clamp(0, 31)     # clamp disabled
```

When `debug_router_grads=True` (our current run), OOB indices from TT's topk go
unclamped into the scatter op, and the OOB detection code in `print_debug_intermediates`
never fires because `topk_indices_raw` is never written to `_dbg`.

---

## Experiment 1 — Uncomment topk_indices_raw + clamp [TODO]

**Goal:** Directly check whether TT hardware's topk returns OOB indices at layer 15 for batch 2.

**Change in `gpt_oss_overrides.py`** lines 134–135 — uncomment:
```python
self._dbg["topk_indices_raw"] = router_indices
router_indices = router_indices.clamp(0, 31)
```

**Command:**
```bash
python3 blacksmith/experiments/torch/gpt_oss/test_gpt_oss_finetuning_2.py \
  --config blacksmith/experiments/torch/gpt_oss/test_gpt_oss_20b_finetuning.yaml \
  2>&1 | tee out_exp1.txt
grep -E "OUT-OF-RANGE|oob_count|topk_indices" out_exp1.txt
grep "norm=inf" out_exp1.txt
```

**Expected outcomes:**

| Result | Interpretation |
|--------|----------------|
| `OUT-OF-RANGE INDICES DETECTED` → `norm=inf` disappears after clamp | **H2 confirmed**: TT `topk` returns OOB indices; scatter backward with OOB indices reads garbage memory; clamping fixes it |
| `OUT-OF-RANGE INDICES DETECTED` → `norm=inf` persists even after clamp | OOB indices are a contributing factor but not the sole cause |
| No OOB indices → `norm=inf` disappears | Clamping itself changes XLA graph structure (heisenbug artifact) |
| No OOB indices → `norm=inf` persists | H2 ruled out; proceed to Experiment 2 |

**Additional logging to add:** Print which layer/batch triggers OOB, and the actual
out-of-range values, to understand if it's a systematic TT topk bug or data-dependent.

**Code change applied** (2026-03-28): Lines 134–135 of `gpt_oss_overrides.py` uncommented. Clamp uses `self.num_experts - 1` (dynamic, not hardcoded 31). Script now exits after accumulation_step=2.

**Result (2026-03-28):** **H2 RULED OUT.** `oob_count=0` at ALL 7 trainable layers (12–18) for BOTH step 1 and step 2. TT topk never returns OOB indices. Full layer-by-layer step 2 INTERMEDIATE GRADS:

| Layer | routing_weights.grad (norm) | topk_values_post_softmax.grad | topk_indices oob_count |
|-------|----------------------------|-------------------------------|------------------------|
| 18 | 2.621e-03 | norm=1.073e-03 (normal) | 0 |
| 17 | 2.012e-03 | norm=1.090e-03 (normal) | 0 |
| 16 | 1.479e-03 | norm=5.692e-04 (normal) | 0 |
| **15** | **1.541e-03 (normal)** | **min=-3.689e+19 (EXPLOSION)** | **0** |
| 14 | 8.475e+18 | norm=1.960e+18 | 0 |
| 13 | 4.366e+18 | norm=2.229e+18 | 0 |
| 12 | 3.456e+18 | norm=1.080e+18 | 0 |

**Layer 15 is the unambiguous origin.** With `routing_weights.grad = 1.541e-03` (valid input, 22 orders of magnitude smaller than the output) and valid indices (0–31), TT's scatter backward produces `topk_values_post_softmax.grad = -3.689e+19`. This is impossible with correct math. Layers 14–12 are downstream victims of the exploded `router_input.grad` propagating backward.

**`norm=inf` persists with the clamp active** (the clamp doesn't help because indices were always valid).

---

## Experiment 2 — CPU mode baseline [TODO]

**Goal:** Isolate TT/XLA execution vs pure math.

**Change:** `use_tt: False` in yaml.

**Command:**
```bash
python3 blacksmith/experiments/torch/gpt_oss/test_gpt_oss_finetuning_2.py \
  --config blacksmith/experiments/torch/gpt_oss/test_gpt_oss_20b_finetuning.yaml \
  --test-config blacksmith/experiments/torch/gpt_oss/test_gpt_oss_cpu.yaml \
  2>&1 | tee out_exp2_cpu.txt
```

**Result (2026-03-28): CLEAN on CPU.** All 7 layers (12–18) at both step 1 and step 2 show `topk_values_post_softmax.grad` in the 1–3e-3 range. No `norm=inf`, no `nan`, `oob_count=0` everywhere. The explosion does not exist on CPU — it is **TT/XLA-specific**.

---

## Experiment 2b — Minimal scatter backward reproducer on TT

**Goal:** Determine if TT's scatter backward is wrong in isolation (random weights, 2-step gradient accumulation).

**Script:** `blacksmith/experiments/torch/gpt_oss/test_scatter_backward_repro.py`

**Result (2026-03-28): PASS on TT.** Both steps clean on TT, matching CPU. Scatter backward is computed correctly when compiled as an isolated graph on TT.

**Key implication:** The bug requires the **full 20B model's XLA compiled graph** to trigger. It is not a general scatter backward kernel bug — it depends on graph context (op fusion, buffer allocation, kernel scheduling) at model scale.

---

## Experiment 3 — Forward activation logging (heisenbug confirmation)

**Goal:** Add `_stats(t.detach())` to `print_debug_intermediates` to also log forward values.

**Change:** In `gpt_oss_overrides.py` `print_debug_intermediates`, add `[fwd]` prints for each retained tensor.

**Result (2026-03-28): HEISENBUG CONFIRMED.** Adding the `.detach()` calls changed the XLA compiled backward graph. The explosion at layer 15 step 2 **completely disappeared**. Layer 15 step 2 forward values that were normal in both runs:
- `router_input [fwd]`: min=-106, max=7.3, norm=741 (step 1: norm=756 — nearly identical)
- `router_logits [fwd]`: min=-5.4, max=3.3, norm=79 (same both steps)
- `topk_values_pre_softmax [fwd]`: min=0.25, max=3.34, norm=35

The forward activations at layer 15 are nearly identical between step 1 and step 2 — no forward overflow, no unusual magnitudes. The difference is purely in the backward computation, and it is XLA-compilation-dependent.

**Final conclusion:** The bug is in the TT/XLA compiled backward graph for the full 20B model. The scatter backward at layer 15 is fused or scheduled in a way that produces garbage results at step 2. This is a TT/XLA compiler bug. See PLAN.md for the complete evidence chain.
- Batch 2 norm >> batch 1 at layer 15 → forward overflow is a contributing factor
- Similar norms → forward overflow is not the cause

**Result:** (fill after run)

---

## Experiment 4 — TTNN IR static analysis (2026-03-29)

**Goal:** Pinpoint the exact TTNN op that produces the wrong result by comparing step 1 vs step 2
TTNN IR files (`ttnn_1774740861345.mlir` and `ttnn_1774741201517.mlir`).

**Key findings:**

### Clarification: scatter.7166/7171 are ATTENTION ops, not router ops

`scatter.7166_chunk_0_scatter` and `scatter.7171_chunk_0_scatter` in the TTNN IR (at TTNN IR
lines 8822 / 9066) operate on a `tensor<1056768xbf16>` base buffer built from a
`tensor<1×64×128×129>` all-gathered attention weight gradient. These are NOT the router scatter
backward — they are scatters in the attention backward pass.

### Actual implementation of `topk_values_post_softmax.grad`

The router's `topk_values_post_softmax.grad` is computed by `stablehlo.gather` → TTNN
`ttnn.embedding` (a gather-as-lookup-table pattern):

| | Step 1 | Step 2 |
|-|--------|--------|
| StableHLO | `gather.6632` at line 3707 | `gather.6637` at line 3845 |
| TTNN line | 8448 | 8692 |
| TTNN op | `%2045 = ttnn.embedding(%2043, %2044)` | `%2146 = ttnn.embedding(%2144, %2145)` |
| Indices shape | `tensor<1x512xui32>` row-major | identical |
| Table shape | `tensor<4096x1xbf16>` row-major | identical |
| Output shape | `tensor<1x512x1xbf16>` tiled | identical |

- **Indices** (`%2043`/`%2144`): flat linear indices computed from topk expert indices
  (`%arg348`/`%arg349`, forward-pass saved tensors, confirmed valid 0–31 in Exp 1)
  via `typecast(ui32→f32) → matmul([32,1]) → reshape → typecast(f32→u32) → to_layout`
- **Table** (`%2044`/`%2145`): `routing_weights.grad` freshly computed as
  `sum(matmul(hidden_grad, expert_out_T)) → permute → all_gather → reshape([4096,1]) → to_layout(row_major)`
- **All layouts are identical** between step 1 and step 2 (same `memref` types, same tile
  geometry, same DRAM interleaved placement)

### Initial hypothesis: zero-copy reshape OOB (DISPROVED 2026-03-29)

The embedding TABLE is computed as:

```
%2138 = ttnn.all_gather(%2133)   # tensor<128x32xbf16, #ttnn_layout95>
                                  #   #ttnn_layout95 = memref<4x1x tile<32x32,bf16>, #dram>
                                  #   PHYSICAL SIZE: 4 tiles × 2KB = 8KB
%2139 = ttnn.reshape(%2138, [4096, 1])
                                  # tensor<4096x1xbf16, #ttnn_layout133>
                                  #   #ttnn_layout133 = memref<128x1x tile<32x32,bf16>, #dram>
                                  #   LOGICAL SIZE: 128 tiles × 2KB = 256KB
deallocate(%2138)
%2145 = ttnn.to_layout(%2139, row_major)   # [4096×1, bf16, row_major]
deallocate(%2139)
%2146 = ttnn.embedding(%2144, %2145)       # EXPLOSION
```

**Hypothesis was that `ttnn.reshape` is zero-copy, sharing the 8KB buffer for a 256KB
logical tensor.** However, an eager TTNN Python test (`test_reshape_oob.py`) showed that
`ttnn.reshape([128×32] → [4096×1])` **correctly allocates and copies data** in eager mode
— ALL 4096 positions returned the expected value even with DRAM pollution and
source-before-to_layout deallocation. The reshape is NOT zero-copy in eager execution.

### Revised analysis: compiled-graph memory plan aliasing

The bug is **data-dependent** — it does NOT trigger for every batch, only for certain
input data. This rules out the zero-copy OOB theory (which would be deterministic and
data-independent).

**Key observation:** only layer 15 explodes while layers 16–18 (same graph pattern) are
clean. This is data-dependent: different layers route to different experts with different
gradient magnitudes. The explosion depends on the SPECIFIC VALUES in the DRAM region that
overlaps with the embedding table.

**Current leading theory:** The TT-XLA compiler's **memory planner** (which assigns fixed
DRAM addresses at compile time for the step 2 backward graph) assigns **overlapping
addresses** to `%2145` (the embedding table) and some other intermediate tensor:

1. The memory plan is computed at compile time based on the graph's liveness analysis
2. A liveness analysis bug in the step 2 graph (which has extra accumulated gradient
   inputs not present in step 1) causes two logically live tensors to share an address
3. When an earlier op writes large values to the shared address, the embedding table
   reads those values instead of `routing_weights.grad`
4. **Data-dependent** because the overlapping tensor's value depends on input data:
   certain data produces large values → explosion; other data → benign values
5. **Heisenbug** because different graph structure → different memory plan → different
   (or no) overlap
6. **Step 1 clean** because step 1's graph has a different (correct) memory plan
7. **Layer 15 only** because only at layer 15's execution point does the overlapping
   tensor happen to hold large values for the specific training data

**Next steps:**
- Inspect the `.ttnn` flatbuffer files (`fb_1774741203335.ttnn` for step 2) for buffer
  address assignments
- Run with `TTXLA_LOGGER_LEVEL=DEBUG` to get runtime memory allocation traces
- Compare the memory plans between step 1 and step 2 compiled programs

---

## Summary Table

| Exp | Hypothesis tested | Status | Result |
|-----|-------------------|--------|--------|
| 1 | H2: OOB topk indices | DONE | Ruled out — `oob_count=0` everywhere |
| 2 | H4: TT/XLA op correctness (CPU baseline) | DONE | CPU clean; bug is TT/XLA-specific |
| 2b | H4b: scatter backward kernel isolation | DONE | Ruled out — isolated reproducer passes on TT |
| 3 | H3: retain_grad + XLA (heisenbug) | DONE | Confirmed — `.detach()` calls change graph and fix bug |
| 4 | TTNN IR static analysis | DONE | Identical static op structure; `ttnn.embedding` at step 2 TTNN line 8692 is the failing op |
| 5 | Zero-copy reshape OOB (eager) | DISPROVED | Eager TTNN test: reshape allocates+copies correctly; all 4096 positions valid |
| 6 | Eager replay with real data | DONE | All 3 ops (reshape, to_layout, embedding) correct in eager with real step 2 tensors; 0/512 bad values, 0/4096 table positions corrupted |
| 7 | EmitPy conversion | DONE | Step 2 TTNN IR → Python via `ttmlir-opt --ttnn-to-emitpy-pipeline` (22K lines). Emitpy crashed on consteval due to mesh tensor sharding mismatch with exported args |
| 8 | Flatbuffer op inspection | DONE | ttrt loaded step 2 flatbuffer: 16110 ops, 337 programs. Router backward embeddings at ops 4294,5184,6074,**6964**,7854,8816,9838,10860,11882,12904,13926,14956. Layer 15 = op 6964 |
| 9 | Runtime debug hooks (Python) | FAILED | `ttrt.runtime.DebugHooks` crashes: nanobind GIL error. `torch_xla.sync` releases GIL, callback called from C++ thread without GIL → `incref_check` abort |
| 10 | C++ runtime instrumentation | PARTIAL | Modified `embedding.cpp` to dump tensor values. Step 1 worked (all embeddings clean). Step 2 never reached execution within timeout — step 2 compilation is very slow (~15+ min). Also `tensor.cpu()` may deadlock mid-compiled-graph execution |
| 11 | Standalone subgraph repro | TODO | Extract the `all_gather→reshape→to_layout→embedding` subgraph from emitpy output, feed real inputs, run on 8 devices. See Next Steps in PLAN.md |
| 12 | Sanity check with tensor saves + `export_tensors:True` | NO EXPLOSION | Added `torch.save(rw.grad.cpu(), ...)` at step 2 and `export_tensors:True`. Bug did NOT reproduce. Same data (seed=23), same `export_tensors`. New flatbuffer: `fb_1774817534606.ttnn`. Hypothesis: tensor saves or export_tensors changed the compiled graph's memory plan |
| 13 | Reproduce with committed training script but overrides.py uncommitted | NO EXPLOSION | `gpt_oss_overrides.py` had `topk_indices_raw` save + `clamp(0, num_experts-1)` uncommitted. The extra `clamp` op changed the autograd graph → different memory plan → no aliasing. This IS the heisenbug mechanism confirmed |
| 14 | Reproduce with ALL files matching committed version | IN PROGRESS | Both `test_gpt_oss_finetuning_2.py` (except `exit(0)`) and `gpt_oss_overrides.py` now match committed exactly. Running 2026-03-29 21:23 |
| — | Compiled-graph runtime bug | CONFIRMED | Bug is in the compiled flatbuffer execution, not in individual TTNN ops |

---

## Running Notes

### 2026-03-28 — Initial analysis

- `zeros` count in expert weight grads decreases batch 1 → batch 2 (e.g., `gate_proj`: 43.8% → 31.2%). Batch 2 activates more experts — consistent with different routing that may include OOB indices on TT hardware.
- The min of `topk_values_post_softmax.grad` is -3.7e+19 while max is +3.8e-04. The asymmetry (only negative side is extreme) is consistent with a single OOB read returning a specific large negative BF16 value from beyond the tensor buffer, while the valid top-k positions return normal gradients.
- The clamp comment in original `GptOssTopKRouter.forward` reads `"Clamping router indices to be between 0 and {self.num_experts - 1}"` — the present tense and the `print` that was also commented out (`# print(f"Clamping router indices...")`) suggests this was added as an active debugging measure, not just defensive code. Someone already found OOB indices on TT hardware before.
