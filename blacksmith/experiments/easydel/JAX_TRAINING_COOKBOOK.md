# JAX + EasyDeL + Tenstorrent training cookbook

A practitioner's guide to writing training scripts for large language models
with [EasyDeL](https://easydel.readthedocs.io/) (Flax NNX) on Tenstorrent
hardware via the TT-XLA PJRT plugin. Focuses on **where arrays live**, **when
they move**, and **why the code is structured the way it is** — using the Qwen3
LoRA fine-tuning experiment in this repo as the running example.

If you are new to JAX + NNX + TT, read top to bottom. If you are debugging a
memory / compile / sharding issue, jump to
[§ 8 — Memory model and OOM debugging](#8--memory-model-and-oom-debugging).

---

## Table of contents

1. [Mental model — devices, JIT, and where arrays live](#1--mental-model--devices-jit-and-where-arrays-live)
2. [Tensor lifecycle — CPU → TT, step by step](#2--tensor-lifecycle--cpu--tt-step-by-step)
3. [Loading the model and applying LoRA](#3--loading-the-model-and-applying-lora)
4. [`nnx.split` / `nnx.merge` — anatomy of the pytrees](#4--nnxsplit--nnxmerge--anatomy-of-the-pytrees)
5. [`jax.value_and_grad` with LoRA — what gets differentiated](#5--jaxvalue_and_grad-with-lora--what-gets-differentiated)
6. [Data parallelism — `shard_map`, `pmean`, `ShardingConfig`](#6--data-parallelism--shard_map-pmean-shardingconfig)
7. [Structuring the training step — one jit, two jits, or none](#7--structuring-the-training-step--one-jit-two-jits-or-none)
8. [Memory model and OOM debugging](#8--memory-model-and-oom-debugging)
9. [Best-practice checklist](#9--best-practice-checklist)

---

## 1 — Mental model: devices, JIT, and where arrays live

Four facts that every JAX-on-TT training script is built around.

### 1.1 — Every `jnp.ndarray` has a device; every device has its own memory.

A `jnp.ndarray` is a handle to a buffer on one specific device (or replicated /
sharded across several). `numpy.ndarray` lives in host RAM and has no device
affinity. Transfers are **explicit** (`jax.device_put`) or **implicit** (JAX
auto-places inputs to a JIT on the JIT's default device).

### 1.2 — `jax.default_device` only affects *new* arrays.

```python
jax.config.update("jax_default_device", tt_device)
```

sets the **default** device for future `jnp.*` ops. It does **not** migrate any
existing arrays. This is the single biggest footgun — changing the default
device does not move your model; you still need `jax.device_put` to move
pre-existing weights.

### 1.3 — JIT traces first, runs second.

When you call a `@jax.jit`-decorated function the first time with a given input
signature, JAX **traces** it (no device work), lowers to XLA HLO, hands it to
the TT-XLA plugin for compilation, and only then **executes**. Subsequent calls
skip straight to execution (with caching). This means:

- Compilation is slow (tens of seconds to minutes for a 4 B-param model on TT).
- Anything you do **outside** JIT with device-resident arrays runs **eagerly**
  — one op at a time, each round-tripping to the device. Avoid this on TT: the
  per-op latency is huge.
- Arrays created inside a JIT are fused into the compiled program. Arrays
  created outside (e.g. `jnp.array(...)` on host numpy) dispatch one kernel
  each.

### 1.4 — Pure functions, pytrees, and closures are JAX's currency.

- JAX transforms (`jit`, `grad`, `shard_map`, `vmap`) want **pure functions of
  pytrees of arrays**.
- A **pytree** is any nested structure of Python containers (`dict`, `list`,
  `tuple`, dataclass, NNX `State`) whose **leaves** are arrays.
- **Static** Python data (shapes, config objects, module class trees) belong in
  closures or `static_argnums` — never in the pytree of arrays.

Corollary: you want your model parameters (dynamic, arrays) cleanly separated
from your module structure (static, Python). That's exactly what `nnx.split`
does — see [§ 4](#4--nnxsplit--nnxmerge--anatomy-of-the-pytrees).

---

## 2 — Tensor lifecycle: CPU → TT, step by step

The single hardest thing to keep straight in a TT training script is **where
each array lives at each moment**. Here is the full lifecycle for the Qwen3-4B
LoRA run in this repo.

### 2.1 — The rule of thumb

> Build things on CPU first. Move them to TT explicitly, once, as late as
> possible. Inside the training loop, cross the CPU ↔ TT boundary at most
> once per step (the batch), and never for anything else.

Why:

- CPU operations are cheap and predictable; you don't want per-tensor TT
  dispatch during initialization (each dispatch pays full PJRT overhead).
- A single, bulk move (big `jax.tree.map(jax.device_put, ...)`) is one TT
  allocation per leaf — far cheaper than N small eager dispatches.
- Debugging is easier if you know the exact line where CPU → TT happens.

### 2.2 — Where this happens in the Qwen script

Using `blacksmith/experiments/easydel/qwen/test_qwen_fine_tuning_easydel.py` as
the reference. Line numbers approximate.

**Step A — Load the full model on CPU.**

```python
with jax.default_device(cpu_device):
    model = load_model(training_config.model_name, ...)
```

`jax.default_device(cpu_device)` is a context manager that pins all `jnp`
allocations inside it to the CPU. Result: all 4.4 B bf16 weights sit in host
RAM.

**Step B — Apply LoRA (still on CPU).**

```python
with jax.default_device(cpu_device):
    model = model.apply_lora_to_layers(lora_rank=..., lora_pattern=...)
```

The newly created LoRA A/B tensors are born on CPU too. Critical for TT — if
they were created on TT, the lazy initialization would eagerly allocate ~144
small buffers.

**Step C — Split the model into static + dynamic.**

```python
graphdef, lora_params, frozen_state = nnx.split(model, nnx.LoRAParam, ...)
```

No device movement — `nnx.split` just re-groups references to arrays that
already live on CPU.

**Step D — Build the optimizer on CPU.**

```python
tx = optax.adamw(learning_rate=schedule)
opt_state = tx.init(lora_params)   # arrays born on CPU, tracking lora_params' shape
```

**Step E — Flip the default device to TT.**

```python
jax.config.update("jax_default_device", current_device)  # TT
```

Now *new* `jnp` arrays default to TT. Existing ones — `frozen_state`,
`lora_params`, `opt_state` — are still on CPU.

**Step F — Explicitly move the params / opt-state to TT (DP path).**

```python
if sharding_config is not None:
    lora_params   = jax.tree.map(lambda x: jax.device_put(x, sharding_config.param_sharding), lora_params)
    frozen_state  = jax.tree.map(lambda x: jax.device_put(x, sharding_config.param_sharding), frozen_state)
    opt_state     = jax.tree.map(lambda x: jax.device_put(x, sharding_config.param_sharding), opt_state)
```

This is the **only** place the ~8.2 GiB of frozen weights crosses the PCIe
boundary. `param_sharding = NamedSharding(mesh, PartitionSpec())` — i.e.
**replicated** across every chip. That replication is a conscious choice for
pure data parallelism; see [§ 6](#6--data-parallelism--shard_map-pmean-shardingconfig)
for how to extend it.

On single-chip (no `sharding_config`), this block is skipped; weights migrate
implicitly on the first JIT call, because after step E inputs default to TT.

**Step G — Per-step batch transfer.**

Each batch has three phases:

```python
# (i) Stays as host numpy forever — no TT allocation at dataset-prep time.
train_batches = [{"input_ids": np.asarray(...), "labels": np.asarray(...), ...}, ...]

# (ii) Per-step label prep (one_hot, label_mask) on CPU.
with jax.default_device(cpu):
    one_hot = jax.nn.one_hot(labels, vocab).astype(jnp.float32)
    label_mask = (labels != -100).astype(jnp.float32)

# (iii) The actual CPU → TT transfer for this step.
input_ids, one_hot, label_mask, attention_mask = _place_batch_on_sharding(
    sharding_config, input_ids, one_hot, label_mask, attention_mask,
)
# Internally:
#   jax.device_put(x, NamedSharding(mesh, PartitionSpec("data")))
# so chip 0 gets batch row 0, chip 1 gets row 1, etc.
```

### 2.3 — The four golden rules

1. **Dataset batches stay as `np.ndarray`** until the step that uses them.
   Never preload batches to device — you'll eagerly allocate thousands of small
   tensors and run out of DRAM before training even starts.

2. **Model weights migrate once**, in bulk, via `jax.tree.map(jax.device_put, ...)`
   after `jax.config.update("jax_default_device", tt)`.

3. **Inside `jit`, never `device_put`.** It's a no-op inside tracing and tells
   XLA nothing. Do all placement outside.

4. **Outside `jit`, never compute on device-resident arrays.** Each op becomes
   an eager dispatch. Either `jit` the computation or do it on CPU.

---

## 3 — Loading the model and applying LoRA

### 3.1 — `AutoEasyDeLModelForCausalLM.from_pretrained`

EasyDeL loads HuggingFace checkpoints, converts them to a Flax NNX module, and
initializes each parameter using `jax.numpy` (so they respect
`jax.default_device`). Two knobs that matter on TT:

- `dtype=jnp.bfloat16` — compute precision.
- `param_dtype=jnp.bfloat16` — **storage precision**. Omit this and params are
  stored in f32 and your model is twice as big in DRAM. This is the difference
  between "fits on 2 chips" and "fits on 4". Always pin `param_dtype`.

### 3.2 — `model.apply_lora_to_layers(...)`

This is EasyDeL's built-in LoRA wrapper. It walks the NNX module tree, finds
all `nnx.Linear` sub-modules whose dotted path matches `lora_pattern` (regex),
and **replaces** each one with a wrapper that adds two low-rank adapters.

For a matched `Linear(F_in, F_out)`:

```
before:
  Linear
   └── kernel: nnx.Param        shape (F_in, F_out)   bf16

after (with lora_rank=r):
  LoRALinear
   └── base.kernel: nnx.Param          shape (F_in, F_out)   bf16   ← the original weight, now frozen-by-convention
   └── base.bias:   nnx.Param          shape (F_out,)         bf16  ← frozen (if the linear had one)
   └── lora_A:      nnx.LoRAParam      shape (F_in, r)        bf16  ← trainable, Gaussian init
   └── lora_B:      nnx.LoRAParam      shape (r, F_out)       bf16  ← trainable, ZERO init
```

The forward becomes `y = x @ W + (x @ A) @ B`. Because `B` is initialized to
zero, the adapter contributes **nothing at step 0** — your starting point is
exactly the pretrained model, guaranteed.

Key points:

- "Frozen" vs "trainable" is **not a flag** on the variable. It's purely the
  **NNX type** of the variable (`nnx.Param` vs `nnx.LoRAParam`). The separation
  becomes physical only at `nnx.split`.
- `lora_pattern` is a *regex matched against the dotted path in the module
  tree*. For Qwen, `.*(q_proj|v_proj).*` targets the attention Q and V
  projections in every transformer layer. 36 layers × 2 projections × 2
  adapters (A + B) = **144 LoRA tensors**.
- The wrapper is still a plain NNX Module. After LoRA, `model` behaves
  identically as a callable; only the internals have changed.

### 3.3 — Parameter counting

After LoRA, you'll want to sanity-check:

```python
def _count_params(state):
    return sum(x.size for x in jax.tree.leaves(state) if hasattr(x, "size"))

n_lora   = _count_params(lora_params)     # ~2.9 M for Qwen3-4B r=8, q+v
n_frozen = _count_params(frozen_state)    # ~4.41 B for Qwen3-4B
```

Trainable fraction for Qwen3-4B with rank=8 on q+v: ~0.067 %. That's why LoRA
is attractive — the optimizer state, gradient memory, and grad aggregation
traffic all scale with this tiny number, not with the 4.4 B base.

---

## 4 — `nnx.split` / `nnx.merge` — anatomy of the pytrees

This is the core separation of concerns for JAX + NNX: **static graph** on one
side, **dynamic arrays** on the other.

### 4.1 — What `nnx.split` returns

```python
graphdef, lora_params, frozen_state = nnx.split(
    model,
    nnx.LoRAParam,   # filter 1 → lora_params
    ...,             # filter 2 (Ellipsis = "everything else") → frozen_state
)
```

`nnx.split(module, *filters)` returns `(graphdef, state_1, state_2, ..., state_N)`.

**`graphdef: nnx.GraphDef`**
- Pure Python object; contains **no arrays**.
- Stores: module class tree, sub-module paths, non-array attributes (config,
  activation fns, dtypes), and **back-references** from tree positions into
  state pytrees.
- Hashable → safe to close over inside a `@jax.jit`. JAX treats it as static.
- Works as "construction instructions" for `nnx.merge` later.

**`lora_params: nnx.State`**
- A pytree (nested, dict-like) containing only the `nnx.LoRAParam` variables.
- Shape:
  ```
  State {
    'layers': {
      '0': {'self_attn': {
          'q_proj': {'lora_A': Array(F_in, r), 'lora_B': Array(r, F_out)},
          'v_proj': {'lora_A': Array(F_in, r), 'lora_B': Array(r, F_out)},
      }},
      '1': { ... },
      ...
      '35': { ... },
    }
  }
  ```
- `jax.tree.leaves(lora_params)` → 144 JAX arrays for Qwen3-4B.
- This is the object you pass to `optax.*.init` and to `jax.value_and_grad`.

**`frozen_state: nnx.State`**
- Same shape as the full module tree minus the LoRA fields.
- Contains ~4.4 B parameters (embeddings, all non-LoRA Linear kernels,
  layernorm gains, rotary caches, etc.).

### 4.2 — What `nnx.split` does *not* do

- **It does not copy arrays.** The arrays in `lora_params` and `frozen_state`
  are the exact same buffers as in the original `model`. If `model` lived on
  CPU, so do the states.
- **It does not "freeze" anything.** It only groups variables by NNX type.
  Freezing happens as a consequence of how you later feed things to
  `value_and_grad` — see [§ 5](#5--jaxvalue_and_grad-with-lora--what-gets-differentiated).
- **It is fully reversible.** `nnx.merge(graphdef, lora_params, frozen_state)`
  reconstructs an identical module.

### 4.3 — Why split at all?

Three reasons, each essential:

1. **`@jax.jit` wants pytrees of arrays as input, not Python objects.** You
   can't pass `model` directly to a jitted function; you pass
   `(lora_params, frozen_state)` and close over `graphdef`.
2. **Autodiff wants to differentiate only some arrays.** If trainable and
   frozen are mixed in one state, you can't tell `value_and_grad` to
   differentiate only the LoRA leaves. Splitting makes it trivial:
   `argnums=0` is the trainable state, argnums 1..N are constants.
3. **Sharding wants a clean per-leaf mapping.** You can apply different
   `NamedSharding` to different branches by mapping over each state.

### 4.4 — The reassembly pattern inside a step

Inside any jitted step:

```python
def loss_fn(lora_params, frozen_state, inputs):
    model = nnx.merge(graphdef, lora_params, frozen_state)
    outputs = model(**inputs)
    ...
```

Under JAX tracing, `nnx.merge` does **not** rebuild arrays — it wires tracers
from `lora_params` and `frozen_state` into the NNX module structure. Forward
computations use those tracers; autodiff tracks them back to the original
pytree leaves. No runtime overhead; this is all tracing-time Python.

### 4.5 — Multiple filters

You can split into more than two buckets, e.g. for partial LoRA plus full-tune
on layernorms:

```python
graphdef, lora_state, ln_state, frozen_state = nnx.split(
    model,
    nnx.LoRAParam,
    lambda path, v: "layernorm" in path,
    ...,
)
```

Then pass both `lora_state` and `ln_state` as `argnums=(0, 1)` to
`value_and_grad`.

---

## 5 — `jax.value_and_grad` with LoRA: what gets differentiated

### 5.1 — The call

```python
loss, grads = jax.value_and_grad(local_loss, argnums=0)(
    lora_params,       # arg 0 — differentiable
    frozen_state,      # arg 1 — constant
    input_ids,         # arg 2 — constant
    one_hot_labels,    # arg 3 — constant
    label_mask,        # arg 4 — constant
    attention_mask,    # arg 5 — constant
)
```

`argnums=0` says: trace `local_loss`, then run reverse-mode autodiff with
respect to argument 0 only. Everything else is frozen from the perspective of
differentiation.

### 5.2 — Mechanism, step by step

1. **Tracing.** JAX runs `local_loss` with placeholder tracers in place of
   every array argument. Each primitive op records a node in the JAXpr. The
   model's forward pass produces a scalar `loss` tracer.

2. **Backward traversal.** JAX walks the JAXpr backward from `loss`, applying
   VJP rules. At each primitive, it pushes cotangents toward inputs. *All*
   inputs in the graph (LoRA and frozen) receive intermediate cotangents
   during this traversal — that's unavoidable, because LoRA adapters sit
   downstream of frozen projections and the gradient must chain through them.

3. **Output filtering.** At the end, JAX **returns only the cotangents rooted
   in `argnums=0`**. Cotangents for the frozen backbone, inputs, labels, etc.
   are computed but discarded.

4. **Result shape.** `grads` has **exactly the same pytree structure as
   `lora_params`** — 144 leaves, each matching shape and dtype. Because
   `lora_B` starts at zero, its grad at step 0 is trivially computed but its
   *parameter value* stays pinned at zero until `optax.apply_updates` adds the
   first non-zero update. From step 1 onward both A and B update normally.

### 5.3 — Important consequence for memory

LoRA training is "parameter efficient" — only 2.9 M params update per step —
but it is **not activation-efficient**. The backward pass still traverses the
entire 36-layer stack and must keep forward activations live to compute
gradients through the frozen weights along the way. This is why Qwen3-4B
LoRA OOMs during the backward pass despite only 2.9 M trainable params.
Mitigations: gradient checkpointing (`nnx.remat`), smaller sequence length, or
sharding the frozen weights (not just the batch).

### 5.4 — Cross-device reductions for data parallelism

Inside a `shard_map` running DP on 2 chips:

```python
loss, grads = jax.value_and_grad(local_loss, argnums=0)(lora_params, ...)
loss  = lax.pmean(loss,  "data")
grads = jax.tree.map(lambda g: lax.pmean(g, "data"), grads)
```

Each chip has computed its *local* gradient from its *local* batch half.
`pmean("data")` averages each of the 144 grad tensors (and the scalar loss)
across the `"data"` mesh axis. After the pmean, every chip has the same
averaged gradient, the same Adam state update, and the same new `lora_params`
— replication is preserved step by step.

> Intuition: `pmean` is the all-reduce that makes DP work. Without it, each
> chip would drift apart because each applied a different local gradient.

Technical footgun — keep collectives *outside* the differentiated function
on TT. If you put `lax.psum` / `lax.pmean` inside `local_loss`,
`jax.value_and_grad` differentiates them and produces an additional backward
collective; TT-MLIR's current lowering of two adjacent collectives on the same
axis emits an `AggregateTensorOp` with duplicate axis names and crashes with
`TT_FATAL: dims must be unique`. Rule:

> Keep `local_loss` purely local. Apply `pmean` **once per quantity**,
> **outside** `value_and_grad`.

---

## 6 — Data parallelism: `shard_map`, `pmean`, `ShardingConfig`

### 6.1 — The minimum viable DP setup

```python
mesh           = jax.sharding.Mesh(np.array(jax.devices("tt")).reshape(N,), axis_names=("data",))
param_sharding = jax.sharding.NamedSharding(mesh, PartitionSpec())          # replicated
data_sharding  = jax.sharding.NamedSharding(mesh, PartitionSpec("data"))    # batch split across chips

# Placement:
params = jax.tree.map(lambda x: jax.device_put(x, param_sharding), params)
batch  = jax.device_put(batch,  data_sharding)

# Step:
#   inside shard_map: local compute → pmean → replicated grads
```

Three invariants must hold at all times:

1. `params` and `opt_state` have **replicated** sharding (`P()`).
2. Batches have **data-axis-sharded** sharding (`P("data")`).
3. Everything returned from a DP step must end up with a **replicated**
   sharding so the next step starts from a consistent state.

### 6.2 — Why `shard_map`?

`shard_map` lets you write the per-device program (what one chip runs on its
local shard) and have JAX lift it to the full mesh:

```python
sm = shard_map.shard_map(
    compute_loss_and_grads,                           # per-device function
    mesh=mesh,
    in_specs=(P(), P(), P("data"), P("data"), P("data"), P("data")),
    out_specs=(P(), grad_out_specs),                  # loss + grads, both replicated
    check_rep=False,
)
```

Inside the body, every array is already the *local* shard. To aggregate
anything across devices you need an explicit collective (`pmean`, `psum`,
`all_gather`).

### 6.3 — Why `batch_size` must be divisible by `num_devices`

The `data_sharding` is `P("data")` — JAX splits the leading axis across the
mesh's `"data"` axis. If `batch_size % num_devices != 0` the shard is
uneven and JAX errors out. Pre-validate in your script rather than let it blow
up deep in the stack.

### 6.4 — Pure DP replicates the whole model

That's the trade-off. A 4.4 B-param model in bf16 is ~8.2 GiB on **each**
chip regardless of mesh size. On TT (~12 GiB per chip), this leaves only
~4 GiB per chip for activations, gradients, and per-op intermediates. Pure DP
scales **throughput** but not **model size**. For model-size scaling you need
FSDP or tensor parallelism — shard the weights, not just the batch.

---

## 7 — Structuring the training step: one jit, two jits, or none

There are three reasonable shapes for a training step. Each has failure modes.

### 7.1 — One fused `jit` (forward + backward + optimizer)

```python
@jax.jit
def train_step(params, opt_state, batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    updates, new_opt = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, new_opt
```

Pros: one XLA program, maximal fusion, smallest per-step Python overhead.

Cons on TT:
- A fused `jit` that **contains a `shard_map`** currently trips a tt-mlir
  bug: output-sharding count from the shard_map boundary (e.g. 1 + 144 grad
  leaves = 145) doesn't match the outer jit's output count (e.g. 1 + 144
  params + ~290 opt-state leaves + 2 stats = 437). Compilation fails with
  `INTERNAL: Error code: 13`.

Use this shape for **single-chip** and for GPU/CPU dev runs.

### 7.2 — Two chained jits (fwd+bwd one, optimizer the other)

```python
jit_loss_grads = jax.jit(shard_map(...fwd+bwd+pmean...))
jit_apply_opt  = jax.jit(apply_opt)

def train_step_dp(params, opt_state, batch):
    loss, grads = jit_loss_grads(params, frozen, batch)
    new_params, new_opt, stats = jit_apply_opt(params, opt_state, grads)
    return loss, new_params, new_opt, stats
```

Pros: each jit's output pytree exactly matches its recorded sharding, which
sidesteps the tt-mlir output-count bug. Optimizer state sharding is trivially
replicated.

Cons: two XLA programs → slightly more PJRT dispatch overhead per step. Tiny
in practice on TT because each program is long.

This is what we use for **multi-chip DP** in the Qwen experiment.

### 7.3 — One jit for fwd+bwd, optimizer *on CPU* (DistilBERT pattern)

```python
loss, grads = jit_loss_grads(params, batch)        # on TT
grads_cpu   = jax.device_put(grads, cpu)
params_cpu  = jax.device_put(params, cpu)
with jax.default_device(cpu):
    updates, opt_state = tx.update(grads_cpu, opt_state, params_cpu)
    params_cpu = optax.apply_updates(params_cpu, updates)
params = jax.device_put(params_cpu, param_sharding)
```

Pros:
- No optimizer-on-TT headaches. Works around
  [tt-metal issue #27072](https://github.com/tenstorrent/tt-metal/issues/27072).
- `opt_state` can live in f32 on CPU without eating device DRAM.

Cons:
- Two CPU↔TT transfers per step (grads out, new params in). For LoRA this is
  tiny (~6 MB each way), so no real downside. For full fine-tuning this would
  be expensive.

Good default when the optimizer-on-TT path is broken or when optimizer state
would dominate device memory.

### 7.4 — Which to pick

| case | recommended shape |
|---|---|
| Single chip dev run | §7.1 (one fused jit) |
| Multi-chip DP, LoRA (tiny trainable set) | §7.2 or §7.3 |
| Multi-chip DP, full fine-tune (optimizer state huge) | §7.3 |
| TT runtime bug in optimizer path | §7.3 |
| GPU/CPU benchmarking | §7.1 |

---

## 8 — Memory model and OOM debugging

### 8.1 — The budget for a Wormhole chip

- ~12 GiB DRAM per chip, split across 12 banks × ~1022 MiB.
- Every allocation comes from a specific bank, with bank-wise contiguity
  requirements. Fragmentation matters: you can have 100 MiB "free" across
  banks but no 10 MiB contiguous slot.

### 8.2 — What lives on-chip during a LoRA fine-tune of Qwen3-4B

Per chip, with pure DP and bf16:

| item | size | sharded? |
|---|---|---|
| Frozen Qwen3-4B weights | ~8.22 GiB | **replicated** (every chip full) |
| LoRA params (144 leaves) | ~6 MB | replicated |
| AdamW state for LoRA (mu, nu) | ~12–24 MB (bf16 or f32) | replicated |
| `input_ids` / `attention_mask` (per step, per-chip shard) | < 1 KB | sharded on `"data"` |
| `one_hot` labels f32 | ~19 MB per chip | sharded on `"data"` |
| `shift_logits.astype(f32)` inside loss | ~19 MB | shard-local |
| Activations saved for backward across 36 layers | 200–500 MB | shard-local |
| Per-op concat/transpose intermediates | tens of MB each | shard-local |

Headroom = 12 GiB - 8.22 GiB - ~60 MB optimizer/batch = ~3.7 GiB for
activations + per-op intermediates. This is tight.

### 8.3 — Why eval can fit but training doesn't

Forward-only: no save-for-backward. Activations are free-able as soon as the
next layer consumes them. Residency peaks at ~100 MiB on top of weights.

Forward + backward: every forward activation must live until its backward VJP
consumes it. On a 36-layer transformer, that's ~36× more activation memory at
peak. Plus the backward pass materializes new intermediate tensors. Peak
residency can be 1.5 – 2.5 GiB on top of weights.

> If your forward-only eval fits and your training step OOMs, you are almost
> always hitting the save-for-backward amplification. The fix is gradient
> checkpointing, not shrinking the model.

### 8.4 — How to read a `TT_FATAL: Out of Memory`

```
Not enough space to allocate 31539200 B DRAM buffer across 12 banks,
where each bank needs to store 2636480 B,
but bank size is 1071821792 B (allocated: 1068570496 B,
free: 3251296 B, largest free block: 2629632 B)
```

Decoding:
- Requested = 31,539,200 B ≈ **30 MiB contiguous**.
- Total bank size = 1,071,821,792 B ≈ 1.02 GiB.
- Allocated = 1,068,570,496 B ≈ 99.7 % of bank full.
- Largest free block = 2,629,632 B ≈ **2.5 MiB** per bank.

Interpretation: DRAM is full *and* fragmented. A 30 MiB op has nowhere to land.
Check the backtrace — it names the op (`permute`, `transpose`, `concat`, ...)
and the call chain through `ttnn::*`, which tells you where in the graph the
allocator is stuck.

### 8.5 — Tools for locating the offender

- `TT_RUNTIME_MEMORY_LOG_LEVEL=operation` — logs per-op allocations. Feed the
  log into the TT memory profiler.
- `TTMLIR_RUNTIME_LOGGER_LEVEL=DEBUG` + `TTXLA_LOGGER_LEVEL=DEBUG` + `XLA_HLO_DEBUG=1`
  — runtime verbose logging (op names, buffer sizes, kernel launches).
- `torch_xla.set_custom_compile_options({"export_path": "./irs", "export_tensors": True})`
  at program start — dumps MLIR IRs you can inspect offline.

### 8.6 — OOM mitigation ranked by impact

1. **Gradient checkpointing (`nnx.remat`)** — typically reclaims 500 MB – 1 GB
   per chip. Biggest single lever for LoRA.
2. **`optax.softmax_cross_entropy_with_integer_labels`** instead of
   `one_hot + softmax_cross_entropy` — avoids the `(B, T, V)` one-hot tensor.
   Saves ~20 MB per step per chip for a 64-seq × 152 K-vocab model, plus
   a lot of host-side prep.
3. **Remove the f32 cast of `shift_logits`** inside the loss — saves ~19 MB
   per chip at a small accuracy cost. Optional.
4. **Pin `param_dtype=bf16`** when loading the model. Already doubles your
   headroom; non-negotiable.
5. **FSDP / tensor parallelism** — shard frozen weights across chips. Moves
   the 8.2 GiB replicated baseline down to 4.1 GiB / chip on a 2-chip mesh.
   Biggest structural change; requires a different `ShardingConfig`.
6. **Shrink `max_length`** — if your task tolerates it, every halving of
   sequence length halves activations and logits memory.

---

## 9 — Best-practice checklist

Use as a pre-flight before every new training script.

### Device placement
- [ ] Model loaded inside `with jax.default_device(cpu_device):`.
- [ ] LoRA applied inside `with jax.default_device(cpu_device):` on TT.
- [ ] `param_dtype` passed to `from_pretrained` (e.g. `bfloat16`).
- [ ] `jax.config.update("jax_default_device", tt)` called **after** CPU
      initialization, **before** moving params.
- [ ] Params / opt-state moved to TT via a single `jax.tree.map(jax.device_put, ...)`.
- [ ] Dataset batches stay as `np.ndarray` until the step that uses them.
- [ ] Per-step batch placement uses `NamedSharding(mesh, P("data"))`.

### Module plumbing
- [ ] Model split with `nnx.split(model, nnx.LoRAParam, ...)`.
- [ ] `graphdef` closed over in the step function, not passed as an argument.
- [ ] `nnx.merge(graphdef, trainable, frozen)` is the **only** reconstruction
      inside the step.

### Autodiff
- [ ] `jax.value_and_grad(..., argnums=0)` with trainable state as arg 0.
- [ ] `local_loss` is **purely local** — no `pmean` / `psum` inside.
- [ ] Cross-device collectives applied **once per quantity, outside
      `value_and_grad`**.

### Data parallelism
- [ ] Mesh axis name is `"data"` (convention in this repo).
- [ ] `batch_size % num_devices == 0` validated up front.
- [ ] `shard_map` `in_specs` / `out_specs` match the actual leaf sharding.
- [ ] All returned quantities have `P()` (replicated) out-sharding.

### JIT structure
- [ ] On multi-chip TT: **split** the fused step into
      `jit_loss_grads` + `jit_apply_opt` (or optimizer-on-CPU).
- [ ] Every `jax.jit` on multi-chip has an explicit `out_shardings=` argument.

### Debugging
- [ ] Env: `TTXLA_LOGGER_LEVEL=DEBUG`, `TTMLIR_RUNTIME_LOGGER_LEVEL=DEBUG`,
      `XLA_HLO_DEBUG=1`, `TT_RUNTIME_MEMORY_LOG_LEVEL=operation` for deep dives.
- [ ] Output redirected to `out.txt` via `2>&1 | tee out.txt` for post-mortem.
- [ ] Validation loss computed once **before** the first training step — if
      that fits but training OOMs, the culprit is save-for-backward /
      activation memory, not the model size.

---

## Appendix — Glossary

- **Pytree**: a tree-shaped Python structure (`dict`, `list`, `tuple`,
  dataclass, NNX `State`) whose leaves JAX treats as array-typed.
- **Tracer**: an abstract stand-in for a real array that JAX uses during
  function tracing; has shape + dtype but no value.
- **HLO**: "High Level Operations", XLA's IR.
- **MLIR**: "Multi-Level Intermediate Representation"; tt-mlir is the TT
  compiler stack that lowers HLO to TTNN kernels.
- **PJRT**: "Portable JAX Runtime"; the device-plugin API JAX uses to talk to
  TT, GPU, TPU, etc.
- **Shard**: the local piece of a tensor on one device.
- **Replicated**: a "sharding" in which every device holds the full tensor.
- **`pmean` / `psum` / `all_gather`**: cross-device collectives along a named
  mesh axis.
- **`nnx.Param` / `nnx.LoRAParam`**: NNX variable subclasses; their *type* is
  what `nnx.split` filters on.
- **`nnx.State`**: the pytree that carries a subset of a module's variables.
- **`nnx.GraphDef`**: the static half of an NNX split — module class tree and
  non-array config.
