# Running GATv2/PubMed on Tenstorrent — design notes & the workaround journey

This document explains *why* the TT path uses `SpMMGATv2Conv` instead of stock
`GATv2Conv`, every workaround that was tried and failed along the way, and how those
failures led to the final design. The summary lives in the README; this is the long form.

**Stack used for the work:** pjrt-plugin-tt `1.2.0.dev20260527003051`,
torch_xla `2.9.0`, torch `2.10.0`, torch_geometric `2.7.0`, Wormhole b0 N300
(single chip, `mesh_shape: null`).

---

## 0. The blocker, and why I didn't just wait for the compiler fix

Stock `GATv2Conv` aggregates messages with `scatter_add` / `scatter_reduce_`. These **do**
compile on TT, but the *lowering is inefficient*: a single logical `scatter` over a
length-`L` index lowers to a **serial chain of ~`L/256` `ttnn.scatter` ops** (one per
256-element slice). On full PubMed (88,648 edges; 108,365 with self-loops; F up to 64) the
aggregation expands to **~190,000 chained scatter ops**, and a single forward pass did not
finish compiling in 38+ minutes. Filed upstream as
**[tenstorrent/tt-mlir#8714](https://github.com/tenstorrent/tt-mlir/issues/8714)**
(fix PR **#8718** in flight).

I did not simply wait for the fix because the lands-in-nightly timing was uncertain
against the deadline, and a *working* parity run is a stronger deliverable than "deferred
until upstream." So I looked for an on-device workaround that doesn't depend on #8718.

---

## 1. Attempt A — targeted scatter-on-CPU fallback → **autograd crash**

The maintainers allow *targeted* op-level fallbacks (just not running the whole model on
CPU). So: run everything on TT except the scatter aggregation, round-tripping that one op
TT → CPU → TT.

**Why it failed.** Forward works; `loss.backward()` crashes the autograd engine the moment
the backward graph spans TT and CPU:
```
RuntimeError: 0 <= device.index() && device.index() < device_ready_queues_.size()
INTERNAL ASSERT FAILED at engine.cpp:1550
```
Both a naive `.cpu()` round-trip and a custom `autograd.Function` (keeping the CPU hop
internal) crash identically. Pure-XLA backward works fine; it is specifically CPU-in-the-
graph that breaks. **Takeaway:** the aggregation has to stay on-device and become
scatter-free — i.e. reformulate the scatter itself.

---

## 2. The reformulation — scatter ≡ matmul (SpMM)

GATv2's two scatter sites (the attention-softmax denominator, and the message aggregation)
are both keyed by the destination node, so both equal a matmul against a constant incidence
matrix `S` (`S[dst[e], e] = 1`):
```
scatter_add(msg, dst, dim_size=N)  ==  S @ msg          # aggregation
per-node denominator               ==  S @ exp(alpha)   # softmax denominator
```
A matmul lowers to `ttnn.matmul`, never `ttnn.scatter`. Bonus: a single **global-max**
softmax stabilization replaces the per-segment `scatter_max` with *no math change*
(segment-softmax is shift-invariant). Verified on CPU against stock `GATv2Conv`: **loss
identical, every per-parameter gradient cosine = 1.0** — exact, not an approximation. The
rest of the journey is making this matmul form actually *run* on TT.

---

## 3. Attempt B — dense incidence → **OOM**

Materialize `S` densely. On a small graph this gave **zero `ttnn.scatter`** with constant
matmul count as edges grew — the reformulation works. But:

- Replacing *only* the aggregation, leaving PyG's internal `x_i`/`x_j` node→edge gathers,
  was not enough: a gather's **backward is a scatter_add**, which still tiles
  (measured: `ttnn.scatter` 6 → 96 as E went 240 → 4096). So *every* node↔edge op had to
  become a matmul — a from-scratch forward that bypasses PyG's `propagate`.
- Dense `S` is `O(N·E)` = 1.75 billion elements/matrix. On full PubMed that is 8.5 GB (fp32)
  or 4.3 GB (bf16) **per matrix**, and two are needed — plus a materialized transpose. Against
  ~12.8 GB DRAM this OOMs. Dense incidence is intrinsically too big.

**Takeaway:** the matmul form is correct, but `O(N·E)` dense memory is fatal; I need an
`O(E)` way to express the same segment operations.

---

## 4. Attempt C — `O(E)` cumsum segment-sum → **bf16 cancellation**

A segment sum can be done without a dense incidence: sort edges by group node, take a prefix
sum (`cumsum`), and difference at the segment boundaries
(`group_sum[n] = prefix[hi_n] − prefix[lo_n]`). `O(E)` memory, lowers to cumsum + gather.
The structure ran end-to-end on TT — but only after peeling three more limitations, each
found from a concrete failure:

1. **Boolean-mask backward (reshape FATAL).** The loss used `out[train_mask]` — boolean
   advanced indexing, whose backward has a data-dependent shape that ttnn rejects
   (`TT_FATAL: Invalid arguments to reshape`). *Found by:* forward-only at full PubMed
   worked; only the masked train step failed. *Fix:* a **static float-mask reduction**
   `(nll * mask.float()).sum() / mask.float().sum()` — identical math, static shapes
   (`masked_nll_loss` / `masked_accuracy`).
2. **Sub-32 tiled reshape of `att` (reshape FATAL).** `att` is `[1, 8, 8]`; reshaping to
   `[1, 64]` failed because a tile-padded `[1,8,8]` is physically `[1,32,32]`. *Fix:* store
   `att` **flat** `[1, H*C]`, and do the per-head channel reduction as a **constant
   block-ones matmul** (`ConstMatmul`) instead of `.view(E,H,C).sum(-1)`.
3. **bf16 `cumsum` catastrophic cancellation (the killer).** With (1)/(2) fixed the full
   train step ran (exit 0, scatter-free) but `loss = inf`. The prefix sum grows to
   `O(E) ≈ 108k`, so the boundary difference is a small difference of large numbers; the
   segment sums quantized to steps of 512 (the bf16 ulp at ~2¹⁶) and rounded to 0, blowing
   up the softmax denominator. Forcing `.to(float32)` had **no effect** — TT's `cumsum`
   kernel is bf16 and XLA folds the cast. Prefix-sum segment reduction is numerically dead
   at this scale on TT.

**Takeaway — the decisive clue:** the dense **matmul** path (Attempt B) was numerically
*correct* on TT (`ttnn.matmul` accumulates in fp32 on Wormhole) and only failed on memory;
cumsum was the opposite (bounded memory, bad numerics). **The answer is to combine their
strengths.**

---

## 5. The solution — blocked one-hot matmul (`use_spmm: true`)

Keep the matmul (for fp32 accumulation), but never materialize the full `[N,E]` incidence —
process edges in blocks and build a small `[N, b]` one-hot on the fly:
```python
out = zeros(N, F)                                       # fp32 accumulator
for st in range(0, E, BLOCK):                           # BLOCK = 16384
    oh = (arange(N)[:, None] == group[st:st+BLOCK])     # [N, b] one-hot
    out += oh @ vals[st:st+BLOCK]                       # ttnn.matmul, fp32 accumulate
```
One more memory fix: with an **fp32** one-hot this still OOM'd (~10 GB already live at the
failure) because XLA unrolls the loop and keeps every block's one-hot alive simultaneously —
so blocking does **not** bound peak memory; **bytes-per-element** does. A **bf16** one-hot
(1.0/0.0 are exact in bf16; the matmul still accumulates in fp32) halves it to ~4.3 GB,
which fits.

**Result (full PubMed, Wormhole N300):** finite loss, **0 `ttnn.scatter`**, ~1.9 s/step,
**test accuracy 0.780** vs CPU stock `GATv2Conv` 0.776 (see [RESULTS.md](RESULTS.md)).

---

## 6. The techniques, in one table

| # | Technique (in `spmm_gatv2.py`) | Limitation it sidesteps |
|---|---|---|
| 1 | SpMM aggregation (matmul vs scatter) | scatter → ~L/256 chained `ttnn.scatter` (#8714) |
| 2 | `att` stored flat `[1, H*C]` | tile-padded sub-32 reshape `1x8x8→1x64` FATAL |
| 3 | per-head reduction via constant block-ones matmul (`ConstMatmul`) | the same `[E,H*C]↔[E,H,C]` reshape FATAL |
| 4 | blocked **bf16** one-hot | dense `[N,E]` OOM; XLA keeps blocks live → bytes-per-element is the lever |
| 5 | static masked loss (`masked_nll_loss`) | boolean-mask backward has a dynamic shape → reshape FATAL |

Plus the enabling insight: **`ttnn.matmul` accumulates in fp32** (segment sums stay
accurate) while **`ttnn.cumsum` is bf16** (the prefix-difference approach is not viable).

Standard `GATv2Conv` (`use_spmm: false`) is expected to run on TT once #8714 lands (fix PR
#8718 in flight); the SpMM path works today regardless.
