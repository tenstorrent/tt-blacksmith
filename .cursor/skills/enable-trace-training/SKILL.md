---
name: enable-trace-training
description: Enable runtime trace (enable_trace) on tt-blacksmith LoRA training workloads to eliminate host dispatch overhead. Use when the user wants to enable trace for a LoRA training script, mentions trace replay, runtime trace, host-bound training, or wants to reduce wall-clock time when the model is host-bound and device time is much smaller than wall-clock.
---

# Enable Runtime Trace for Training Workloads

Runtime trace records device command sequences and replays them without per-op host dispatch. For HOST-bound training workloads on Tenstorrent hardware, this typically yields a **~6x wall-clock speedup** by reducing per-step time from ~3.4s to ~0.5s (validated on Gemma 1.1 2B LoRA, batch_size 4, seq_len 32).

## When to use this skill

Trace pays off when the workload is **host-bound** (device time << wall-clock). Verify with the [perf-benchmark-single-chip](../perf-benchmark-single-chip/SKILL.md) skill first:

- If `(wall_clock - device_time) / wall_clock > 50%`, trace will help significantly
- If overhead is already <30%, trace gains will be marginal

Trace works for LoRA training workloads but requires a few specific code changes to overcome incompatibilities between dynamic graph behaviour and trace's requirement for identical replay shapes.

## Recommended workflow: trace → graph-break optimize → trace again

Trace and graph-break optimization are complementary. Trace eliminates **per-op** host dispatch overhead inside each graph; graph-break optimization reduces the **number of graphs** that have to be replayed per step. Combine them iteratively:

1. **Enable trace first** (this skill). Each trace replay carries some fixed dispatch cost (~50ms in our testing), so wall-clock per step ≈ device_time + N_graphs × per_replay_dispatch + Python overhead. Trace alone usually gives 4-8x speedup.

2. **If gains are smaller than expected, or you want to push further**, run the `graph-break-analysis` skill in tt-xla (`.claude/skills/graph-break-analysis/SKILL.md`) to identify and remove graph breaks. Each break you eliminate fuses two trace replays into one.

3. **Re-enable trace** on the consolidated graph. Wall-clock should drop further as N_graphs decreases.

Most workloads converge after one trace → graph-break → trace cycle. Start with step 1 — the cheap-and-big-win — before investing in steps 2-3.

## Scope

This skill targets **LoRA finetuning** of HuggingFace causal LMs. LoRA freezes embeddings by default, which sidesteps the sparse-gradient issue that breaks trace for full finetuning. If you need to train with unfrozen embeddings, you'll need additional work not covered here.

## Why trace breaks naive training code

The main issue: **dynamic attention masks**. HuggingFace's `_update_causal_mask()` constructs variable-shape mask tensors per forward pass. Trace replay requires identical shapes, so this crashes (`TT_FATAL: Host tensor has different shape`).

A secondary issue used to be the optimizer step counter (PyTorch's AdamW computes `bias_correction = 1 - beta**step` as a Python scalar that gets baked into the graph as a constant, causing a new compilation each step). tt-blacksmith optimizers are already constructed with `capturable=True`, which keeps this math as on-device tensor operations and resolves the issue. If you bring in a custom optimizer, make sure it follows the same pattern.

## The two required modifications

### 1. Patch the dynamic attention mask

The pattern follows the perf benchmarks in `tt-xla/tests/benchmark/benchmarks/llm_benchmark.py`, which avoids HF's dynamic mask path by passing pre-allocated cache tensors with fixed shapes (see `init_static_cache` and `cache_position` usage there).

For **inference** (decode/prefill): use `StaticCache` + `cache_position`. HF's `_update_causal_mask()` derives a static mask from those.

For **training** (no KV cache): pre-build a 4D causal mask and pass it as `attention_mask`. HF's `_update_causal_mask()` returns 4D masks unchanged, bypassing the dynamic construction path.

Add this helper (e.g. in your model loading module):

```python
def make_static_causal_mask_4d(batch_size, seq_len, dtype, device):
    """When HF receives a 4D attention_mask, it returns it directly,
    bypassing dynamic mask construction that breaks trace."""
    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)
```

Build it **once** before the training loop, then pass it on every forward call along with explicit `position_ids`:

```python
mask_4d = make_static_causal_mask_4d(batch_size, seq_len, dtype, device)
position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

# In the training loop:
outputs = model(input_ids=batch["input_ids"], attention_mask=mask_4d, position_ids=position_ids)
```

This requires **fixed batch size and sequence length** for the entire run.

### 2. Enable the trace compile option

Set the option **before** `torch.compile`:

```python
import torch_xla
torch_xla.set_custom_compile_options({"enable_trace": "true"})
model = torch.compile(model, backend="tt", options=compile_options)
```

This is a global setting that applies to all subsequent compilations.

## Recommended additional modifications

### 3. Increase the trace region size

Trace data is stored in DRAM. The default is too small for full models. Set:

```python
os.environ["TT_RUNTIME_TRACE_REGION_SIZE"] = str(200 * 1_000_000)  # 200 MB
```

Set this **before** any device initialization (typically in your DeviceManager setup). 200MB is sufficient for Gemma 2B; larger models may need more.

If you see `TT_FATAL: Creating trace buffers of size [...] but only [...] is allocated for trace region`, increase this value.

### 4. Use non-blocking sync between backward and optimizer

```python
loss.backward()
torch_xla.sync(wait=False)            # dispatch fwd+bwd, don't block host
device_manager.optimizer_step(optimizer)
running_loss += loss.item()           # only read scalar after optimizer dispatched
```

The `sync(wait=False)` lets the host continue dispatching the optimizer graph while the device executes fwd+bwd, reducing pipeline bubbles.

## Gradient accumulation

If your training loop uses gradient accumulation (multiple fwd+bwd micro-steps before each optimizer step), trace works but requires one additional change.

### Pattern

```python
optimizer.zero_grad(set_to_none=False)        # IMPORTANT: see below
for micro_step in range(accumulation_steps):
    outputs = model(input_ids=..., attention_mask=mask_4d, position_ids=position_ids)
    loss = loss_fn(outputs.logits, labels) / accumulation_steps
    loss.backward()
    torch_xla.sync(wait=False)                # dispatch fwd+bwd micro-step
device_manager.optimizer_step(optimizer)      # consumes accumulated grads
```

Each accumulation cycle produces three graphs:
- **fwd+bwd graph** — replayed N times (once per micro-step). Reads inputs and current `p.grad`, writes accumulated `p.grad`.
- **optimizer graph** — replayed once per cycle.
- **zero_grad graph** — replayed once per cycle (fills `p.grad` with zeros).

### Required: use `set_to_none=False` in zero_grad

PyTorch's `optimizer.zero_grad()` defaults to `set_to_none=True`, which sets `p.grad = None` instead of zeroing the tensor. This breaks trace because:

- **First micro-step** of each cycle sees `p.grad is None` and **allocates a fresh tensor**
- **Subsequent micro-steps** see `p.grad` as a tensor and **read-add-write**

These are two structurally different graphs. The first-micro-step graph also gets recompiled every cycle. Pass `set_to_none=False` to keep `p.grad` as a fixed-shape zero tensor across the entire run.

### Performance characteristic

Gradient accumulation is the **best case** for trace: device utilization stays high because the host pipelines N back-to-back trace replays before any blocking sync. With enough micro-steps, wall-clock per cycle approaches `N × device_time` (the device-bound floor).

## Implementation checklist

Apply changes in this order:

```
- [ ] Add `enable_trace: bool = False` and `trace_region_size_mb: int = 200` to your config schema
- [ ] In device setup: if enable_trace, set TT_RUNTIME_TRACE_REGION_SIZE env var
- [ ] In model loading: if enable_trace, call torch_xla.set_custom_compile_options(...) before torch.compile
- [ ] Add make_static_causal_mask_4d() helper
- [ ] In train(): pre-build mask_4d and position_ids if enable_trace
- [ ] In train loop: pass mask_4d, position_ids to model() if enable_trace
- [ ] Switch sync to sync(wait=False) between backward and optimizer step
- [ ] Add a YAML config override for testing (enable_trace: true, trace_region_size_mb: 200)
```

## Verification

After applying changes, verify trace is working with three checks:

### Check 1: No TT_FATAL errors

Run training for ~30 steps. Watch for `TT_FATAL` errors in the output:
- `Host tensor has different shape` → static mask not applied or wrong shape
- `Creating trace buffers of size ...` → trace region too small, increase it

### Check 2: Compilations stabilize

Add a one-time diagnostic to the training loop:

```python
import torch_xla.debug.metrics as met

if global_step <= 5:
    cached = torch_xla._XLAC._xla_get_num_cached_compilation_graph()
    uncached = met.counter_value("UncachedCompile") or 0
    print(f"[DIAG] Step {global_step}: cached={cached}, uncached={uncached}")
```

Expected output: `uncached` should stop growing after step 1-2 (it stabilizes at ~9 graphs for a typical HF training script: validation graphs + fwd, bwd+loss, optimizer). If `uncached` keeps growing, a graph is being recompiled each step — usually caused by a Python scalar leaking into the graph.

### Check 3: Step time drops to near device time

Use the [perf-benchmark-single-chip](../perf-benchmark-single-chip/SKILL.md) skill to measure wall-clock per step. Compare to baseline:

| Metric | Without trace | With trace (target) |
|---|---|---|
| Wall-clock / step | baseline | should drop 4-8x |
| Device time | constant | constant |
| Overhead % | 70-90% | 20-40% |

The theoretical floor is `device_time + ~150ms` (for 3 graph dispatches + Python overhead). If you don't see at least a 3x speedup, recheck Check 2 — graphs are likely being recompiled.

## Known limitations

| Limitation | Workaround |
|---|---|
| **Tracy device profiling crashes** with `TT_FATAL: !trace_id_.has_value()` | Profile the non-trace baseline with Tracy; measure trace runs with wall-clock timers only |
| **Variable shapes not supported** (e.g. variable seq_len, padding) | Pad to a fixed `max_length` for all batches |
| **Per-step checkpointing grows trace region** | Set `save_strategy: "epoch"` while benchmarking trace |
| **Allocator warning** `Allocating device buffers is unsafe due to the existence of an active trace` appears each step | Cosmetic warning — does not affect correctness in our testing |

## Related skills

- [perf-benchmark-single-chip](../perf-benchmark-single-chip/SKILL.md) — measure baseline before and after
- `graph-break-analysis` (in tt-xla, `.claude/skills/graph-break-analysis/SKILL.md`) — investigate excessive graph generation if trace gains are smaller than expected
