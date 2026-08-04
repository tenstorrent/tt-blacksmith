# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Performance / MFU utilities for training experiments.

tt-mlir emits a per-graph FLOP report (a ``flops`` JSON section) when the compile
option ``ttnn_perf_metrics_enabled`` is set. These helpers read that report back
and combine it with the measured step time to report Model FLOPs Utilization.

The denominator's per-chip half comes straight from the report:
``peak_flops_per_sec`` is the flops-weighted peak the graph actually ran at (right
whether the matmuls ran LoFi or HiFi4, not just at one assumed fidelity). Nothing
here re-derives it or picks a fidelity from
``peak_flops_per_sec_by_fidelity`` — that table is reference material.

Everything in the report is **per chip for one graph invocation**; the report
carries no chip count, because the only sharding information reaching TTNN is a
binary ``ttcore.shard_status`` per argument. So the caller supplies ``num_chips``
(from the mesh it built) and this module scales to mesh-wide.

For the numerator there are two choices, and they differ under tensor parallelism:

* ``analytical_step_flops`` — from the model's parameter count. Never looks at the
  graph, so replicated compute cannot inflate it. This is ``mfu_pct``, and it is
  the number to trust.
* the graph's own ``total_flops * num_chips`` — what the hardware executed.
  Correct as model work under data parallelism, where each chip's share is
  distinct. Under tensor parallelism an op left out of the sharding spec is
  recomputed identically on several chips and this counts it once per chip. tt-mlir
  cannot tell the two apart (see the ``shard_status`` note above), so this is
  reported as ``hfu_pct`` (hardware FLOPs utilization), not as MFU.

Under pure data parallelism the two agree up to the accuracy of the estimate; a
growing gap on a tensor-parallel mesh is replicated compute.
"""
import glob
import json
import os

# Base name for tt-mlir's per-graph theoretical-FLOP report. When
# ttnn_perf_metrics_enabled is set, tt-mlir writes <base>_<graphidx>.json into
# the process CWD.
MFU_PERF_METRICS_FILE = "tt_blacksmith_train_perf_metrics"


def clear_perf_metrics_files(base_name=MFU_PERF_METRICS_FILE):
    """Remove stale ``<base>*.json`` reports from the CWD before a run.

    tt-mlir auto-numbers graphs per process, so a previous run's leftover files
    could otherwise pollute the max-FLOPs graph pick in
    :func:`read_training_step_flops`.
    """
    for stale in glob.glob(f"{base_name}*.json"):
        try:
            os.remove(stale)
        except OSError:
            pass


def read_training_step_flops(base_name=MFU_PERF_METRICS_FILE):
    """Read tt-mlir's ``flops`` report for the fused training-step graph.

    A training run compiles several graphs (optimizer-state init, the eval
    forward, and the fused forward+backward+optimizer step). tt-mlir writes one
    ``<base>_<idx>.json`` per graph; the fused training step does by far the
    most work, so we pick the graph with the largest ``total_flops``.

    Returns the raw ``flops`` dict (``total_flops``, ``peak_flops_per_sec``,
    ``peak_flops_per_sec_by_fidelity``, ...) for that graph, or ``None`` when no
    report with a ``flops`` section exists (e.g. an older tt-mlir without FLOP
    support).
    """
    best = None
    for path in sorted(glob.glob(f"{base_name}*.json")):
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        flops = data.get("flops")
        if not isinstance(flops, dict) or not flops.get("total_flops"):
            continue
        if best is None or flops["total_flops"] > best["total_flops"]:
            best = flops
    return best


def _lookup_param_ids(model):
    """``id()``s of parameters a per-parameter FLOP estimate must not charge for.

    Only the input embedding, which is a row lookup rather than a matmul: no
    floating-point work in the forward pass, and no input gradient in the
    backward, its input being integer token ids. Its ``vocab_size * hidden`` rows
    are large enough to move MFU by several percent if charged as dense work
    (~6.5% on Llama-3.1-8B). Skipping it also keeps this numerator comparable to
    tt-mlir's ``total_flops``, which counts ops and so never sees
    ``ttnn.embedding``.

    Skipped only when the embedding is *untied*. Under weight tying the embedding
    and the LM head are one tensor, which ``Module.parameters()`` yields once;
    that single count is the head's genuine ``6*V*H``, so dropping it would
    understate the step by exactly as much as counting the embedding overstates
    it. Returns nothing for a model that does not expose the HF embedding
    accessors, which leaves the estimate as it was.
    """
    get_input = getattr(model, "get_input_embeddings", None)
    get_output = getattr(model, "get_output_embeddings", None)
    if not callable(get_input):
        return frozenset()
    try:
        input_weight = getattr(get_input(), "weight", None)
        # None for a model with no LM head, e.g. a sequence-classification one.
        output_weight = getattr(get_output(), "weight", None) if callable(get_output) else None
    except (AttributeError, NotImplementedError):
        return frozenset()
    if input_weight is None or input_weight is output_weight:
        return frozenset()
    return frozenset({id(input_weight)})


def flops_per_step(model, seq_len, batch_size, gradient_accumulation_steps):
    """Analytical (PaLM-style) training FLOPs per step, from model parameters.

    Per token: ``6*T_p + 4*F_p + 12*L*H*Q*T``.

    A trainable parameter costs 6 flops/token (2 forward, 2 for its input
    gradient, 2 for its weight gradient). A frozen one costs only 4: the backward
    pass still propagates a gradient through it, but computes no weight gradient.
    LoRA freezes the whole base model, so charging the usual flat ``6*N`` would
    overstate the numerator by ~50% on a small-adapter run. ``12*L*H*Q*T`` is the
    attention term (L layers, H heads, Q head dim, T seq len), which is
    weight-free and so unaffected by freezing.

    The input embedding is excluded from both parameter counts, tying permitting;
    see :func:`_lookup_param_ids`.

    A step processes ``batch_size * seq_len * grad_accum`` tokens.

    ``batch_size`` must be the GLOBAL (whole-mesh) batch — true for torch_xla
    SPMD, where the input is a global tensor sharded across the data-parallel
    axis — so the result is the mesh-wide FLOPs for one optimizer step, matching
    the mesh-wide peak in :func:`compute_mfu_metrics`.

    Hardware- and compiler-independent. Returns 0 if the model config lacks the
    needed fields.
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        return 0
    skip = _lookup_param_ids(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad and id(p) not in skip)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad and id(p) not in skip)
    layers = getattr(cfg, "num_hidden_layers", 0) or 0
    heads = getattr(cfg, "num_attention_heads", 0) or 0
    hidden = getattr(cfg, "hidden_size", 0) or 0
    head_dim = getattr(cfg, "head_dim", None) or (hidden // heads if heads else 0)
    flops_per_token = 6 * trainable + 4 * frozen + 12 * layers * heads * head_dim * seq_len
    tokens_per_step = batch_size * seq_len * gradient_accumulation_steps
    return flops_per_token * tokens_per_step


def compute_mfu_metrics(perf_report, analytical_step_flops, step_elapsed, gradient_accumulation_steps=1, num_chips=1):
    """Combine tt-mlir's ``flops`` report with measured step time into MFU.

    ``analytical_step_flops`` is already a whole mesh-wide optimizer step, so two
    scalings are applied to the report, which is per-chip and per-invocation:
    ``num_chips`` to reach the mesh-wide peak, and ``grad_accum`` on the
    graph-counted FLOPs, since a measured step runs ``grad_accum`` invocations.

    ``num_chips`` is the caller's mesh size (``DeviceManager.num_chips``) — the
    report cannot supply it, see the module docstring.

    Returns a dict; any value may be ``None`` when its inputs are missing:

    * ``mfu_pct``          analytical model FLOPs / (step * mesh peak) — the metric
    * ``hfu_pct``          FLOPs the hardware executed / (step * mesh peak). Equal
      to ``mfu_pct`` under data parallelism up to the estimate's accuracy; higher
      under tensor parallelism, where it counts replicated work once per chip.
    * ``achieved_tflops``  model FLOPs per second
    """
    result = {
        "mfu_pct": None,
        "hfu_pct": None,
        "achieved_tflops": None,
    }
    if perf_report is None or step_elapsed <= 0:
        return result

    grad_accum = max(1, gradient_accumulation_steps)
    num_chips = max(1, num_chips)
    # Per-chip, fidelity-weighted peak from the report. tt-mlir emits 0 for a graph
    # with no matrix-engine work, which `or 0` also covers.
    chip_peak = perf_report.get("peak_flops_per_sec", 0) or 0
    mesh_peak = chip_peak * num_chips

    if analytical_step_flops and analytical_step_flops > 0:
        result["achieved_tflops"] = analytical_step_flops / step_elapsed / 1e12
        if mesh_peak > 0:
            result["mfu_pct"] = analytical_step_flops / (step_elapsed * mesh_peak) * 100.0

    # Every chip runs the same program, so the mesh executes total_flops per chip.
    executed = (perf_report.get("total_flops", 0) or 0) * grad_accum * num_chips
    if executed > 0 and mesh_peak > 0:
        result["hfu_pct"] = executed / (step_elapsed * mesh_peak) * 100.0

    return result


def format_mfu_summary(mfu):
    """Render the :func:`compute_mfu_metrics` dict as a one-line human summary."""

    def pct(v):
        return f"{v:.2f}%" if v is not None else "n/a"

    achieved = f"{mfu['achieved_tflops']:.2f} TFLOP/s" if mfu.get("achieved_tflops") is not None else "n/a"
    return f"MFU {pct(mfu.get('mfu_pct'))} | " f"HFU {pct(mfu.get('hfu_pct'))} | " f"{achieved}"
