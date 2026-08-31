# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for blacksmith.tools.performance_utils (MFU / FLOP helpers).

Pure-Python, no hardware: feed synthetic tt-mlir ``flops`` reports and assert the
FLOP counting, the MFU math, and the internal invariants that guard against
regressions like the earlier peak-contamination bug.

The reports mirror tt-mlir's contract: everything is per chip for one graph
invocation — ``total_flops`` is one chip's program, ``peak_flops_per_sec`` is that
chip's fidelity-weighted peak — and the chip count is not in the report at all, so
tests pass ``num_chips`` explicitly the way the training loop does.
"""
import json

import pytest

from blacksmith.tools.performance_utils import (
    clear_perf_metrics_files,
    compute_mfu_metrics,
    flops_per_step,
    format_mfu_summary,
    read_training_step_flops,
)


def _report(total_flops, chip_peak):
    return {
        "total_flops": total_flops,
        "peak_flops_per_sec": chip_peak,
        "peak_flops_per_sec_by_fidelity": {
            "lofi": 4 * chip_peak,
            "hifi2": 2 * chip_peak,
            "hifi3": int(4 * chip_peak / 3),
            "hifi4": chip_peak,
        },
    }


# --- read_training_step_flops -------------------------------------------------


def test_reader_picks_max_flops_graph(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # 3 graphs like a real run: tiny init, eval fwd, and the big fused step.
    (tmp_path / "r_0.json").write_text(json.dumps({"flops": {"total_flops": 10}}))
    (tmp_path / "r_1.json").write_text(json.dumps({"flops": {"total_flops": 500}}))
    (tmp_path / "r_2.json").write_text(json.dumps({"flops": {"total_flops": 999}}))
    best = read_training_step_flops("r")
    assert best["total_flops"] == 999


def test_reader_returns_none_without_flops_section(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "r_0.json").write_text(json.dumps({"summary": {"total_ops": 3}}))
    assert read_training_step_flops("r") is None


def test_reader_skips_corrupt_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "r_0.json").write_text("{not valid json")
    (tmp_path / "r_1.json").write_text(json.dumps({"flops": {"total_flops": 42}}))
    assert read_training_step_flops("r")["total_flops"] == 42


def test_clear_perf_metrics_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "r_0.json").write_text("{}")
    (tmp_path / "r_1.json").write_text("{}")
    clear_perf_metrics_files("r")
    assert list(tmp_path.glob("r*.json")) == []


# --- compute_mfu_metrics: math + invariants -----------------------------------


def test_mfu_math_known_values():
    # analytical=1e12, mesh peak=1e14 -> compute floor 10ms. A measured 20ms step
    # (2x the floor) is 50% MFU, and 50 TFLOP/s of model work.
    rep = _report(1e12, chip_peak=1e14)
    m = compute_mfu_metrics(rep, analytical_step_flops=1e12, step_elapsed=0.020)
    assert m["mfu_pct"] == pytest.approx(50.0)
    assert m["achieved_tflops"] == pytest.approx(50.0)
    # Graph counted the same FLOPs on one chip, so HFU matches.
    assert m["hfu_pct"] == pytest.approx(50.0)


def test_mfu_uses_weighted_peak_not_the_fidelity_table():
    # The denominator must come from peak_flops_per_sec (the fidelity-weighted peak
    # the graph ran at), never from the by_fidelity reference table. Here the graph
    # ran between hifi2 and lofi, so any single table entry gives a wrong answer.
    rep = _report(7e11, chip_peak=1.75e14)
    m = compute_mfu_metrics(rep, analytical_step_flops=1e12, step_elapsed=0.05)
    implied_peak = m["achieved_tflops"] * 1e12 / (m["mfu_pct"] / 100.0)
    assert implied_peak == pytest.approx(1.75e14, rel=1e-9)


def test_grad_accum_scales_hfu_only():
    # The report covers one micro-batch, so grad_accum scales the graph-counted
    # FLOPs. The analytical numerator already folds grad_accum in, so MFU must NOT
    # be scaled again here.
    rep = _report(1e12, chip_peak=1e14)
    m1 = compute_mfu_metrics(rep, 3e12, 0.02, gradient_accumulation_steps=1)
    m4 = compute_mfu_metrics(rep, 3e12, 0.02, gradient_accumulation_steps=4)
    assert m4["hfu_pct"] == pytest.approx(4 * m1["hfu_pct"])
    assert m4["mfu_pct"] == pytest.approx(m1["mfu_pct"])


def test_num_chips_scales_the_mesh_peak():
    # The report is per-chip, so the caller's num_chips is what turns it mesh-wide.
    # Same graph and same step time on 4 chips must give 1/4 the MFU of 1 chip.
    rep = _report(1e12, chip_peak=1e14)
    m1 = compute_mfu_metrics(rep, 1e12, 0.02, num_chips=1)
    m4 = compute_mfu_metrics(rep, 1e12, 0.02, num_chips=4)
    assert m4["mfu_pct"] == pytest.approx(m1["mfu_pct"] / 4)


def test_data_parallel_mfu_is_chip_invariant():
    # Pure data parallelism: N chips each do a distinct 1/N of the batch, so the
    # analytical step FLOPs and the mesh peak both scale with N and MFU is
    # unchanged. HFU tracks it, since every chip's work is distinct.
    rep = _report(1e12, chip_peak=1e14)
    m1 = compute_mfu_metrics(rep, 4e12, 0.02, num_chips=1)
    m4 = compute_mfu_metrics(rep, 4 * 4e12, 0.02, num_chips=4)
    assert m4["mfu_pct"] == pytest.approx(m1["mfu_pct"])
    assert m4["hfu_pct"] == pytest.approx(m1["hfu_pct"])


def test_hfu_counts_every_chip_even_when_work_is_replicated():
    # tt-mlir cannot see mesh sharding, so total_flops * num_chips is what the
    # hardware executed, not necessarily model work. With a fully replicated graph
    # (4 chips recomputing one program) HFU stays flat while MFU drops 4x -- the
    # gap between them is the signal, which is why they are reported separately.
    rep = _report(1e12, chip_peak=1e14)
    m1 = compute_mfu_metrics(rep, 1e12, 0.02, num_chips=1)
    m4 = compute_mfu_metrics(rep, 1e12, 0.02, num_chips=4)
    assert m4["hfu_pct"] == pytest.approx(m1["hfu_pct"])
    assert m4["mfu_pct"] == pytest.approx(m1["mfu_pct"] / 4)


def test_guards_missing_inputs():
    assert compute_mfu_metrics(None, 1e12, 0.02)["mfu_pct"] is None
    rep = _report(1e12, chip_peak=1e14)
    assert compute_mfu_metrics(rep, 1e12, 0.0)["mfu_pct"] is None
    # No analytical flops -> no MFU, but HFU still computable from the graph.
    m = compute_mfu_metrics(rep, 0, 0.02)
    assert m["mfu_pct"] is None
    assert m["achieved_tflops"] is None
    assert m["hfu_pct"] is not None


def test_guards_report_without_a_peak():
    # tt-mlir emits peak_flops_per_sec: 0 for a graph with no matrix-engine work.
    # Report nothing rather than guessing a fidelity from the by_fidelity table.
    rep = _report(0, chip_peak=0)
    m = compute_mfu_metrics(rep, 3e12, 0.02, num_chips=4)
    assert m["mfu_pct"] is None
    assert m["hfu_pct"] is None
    # An older tt-mlir omitting the key entirely must behave the same way.
    m = compute_mfu_metrics({"total_flops": 1e12}, 3e12, 0.02, num_chips=4)
    assert m["mfu_pct"] is None
    assert m["hfu_pct"] is None


def test_format_summary_handles_none():
    s = format_mfu_summary({"mfu_pct": 8.17, "hfu_pct": None, "achieved_tflops": 12.42})
    assert "MFU 8.17%" in s
    assert "HFU n/a" in s
    assert "12.42 TFLOP/s" in s


# --- flops_per_step --------------------------------------


class _FakeParam:
    def __init__(self, n, requires_grad=True):
        self._n = n
        self.requires_grad = requires_grad

    def numel(self):
        return self._n


class _FakeConfig:
    num_hidden_layers = 2
    num_attention_heads = 4
    hidden_size = 128  # head_dim = 32


class _FakeModel:
    config = _FakeConfig()

    def parameters(self):
        return [_FakeParam(1000), _FakeParam(2000)]  # all trainable, N = 3000


def test_analytical_flops_formula_all_trainable():
    # 6*N + 12*L*H*Q*T per token, times batch*seq*grad_accum tokens.
    # N=3000, L=2, H=4, Q=32, T=16 -> per_token = 18000 + 12*2*4*32*16 = 67152
    # tokens = 8 * 16 * 2 = 256 -> 67152 * 256 = 17,190,912
    flops = flops_per_step(_FakeModel(), seq_len=16, batch_size=8, gradient_accumulation_steps=2)
    assert flops == 17_190_912


def test_analytical_flops_charges_frozen_params_less():
    # LoRA: a frozen parameter costs 4 flops/token (no weight gradient), not 6.
    # 2000 of the 3000 params frozen -> 6*1000 + 4*2000 = 14000, vs 18000 flat.
    class _LoRAModel:
        config = _FakeConfig()

        def parameters(self):
            return [_FakeParam(1000), _FakeParam(2000, requires_grad=False)]

    attention = 12 * 2 * 4 * 32 * 16
    tokens = 8 * 16 * 2
    assert flops_per_step(_LoRAModel(), 16, 8, 2) == (14000 + attention) * tokens
    # And it must be strictly below the all-trainable count, not equal to it.
    assert flops_per_step(_LoRAModel(), 16, 8, 2) < flops_per_step(_FakeModel(), 16, 8, 2)


class _FakeEmbedding:
    def __init__(self, weight):
        self.weight = weight


def _embedding_model(params, input_weight, output_weight):
    class _M:
        config = _FakeConfig()

        def parameters(self):
            return params

        def get_input_embeddings(self):
            return _FakeEmbedding(input_weight)

        def get_output_embeddings(self):
            return _FakeEmbedding(output_weight) if output_weight is not None else None

    return _M()


def test_analytical_flops_skips_untied_embedding():
    # An untied embedding is a row lookup, not a matmul, and must not be charged
    # 6 flops/token just for being in parameters().
    embed, body, head = _FakeParam(2000), _FakeParam(1000), _FakeParam(2000)
    model = _embedding_model([embed, body, head], embed, head)
    # Leaves body + head, i.e. N = 3000 -- the same step as _FakeModel.
    assert flops_per_step(model, 16, 8, 2) == 17_190_912


def test_analytical_flops_keeps_tied_embedding():
    # Tying makes the embedding and the LM head one tensor, which parameters()
    # yields once; that count is the head's real 6*V*H and has to stay.
    shared, body = _FakeParam(2000), _FakeParam(1000)
    model = _embedding_model([shared, body], shared, shared)
    assert flops_per_step(model, 16, 8, 2) == 17_190_912


def test_analytical_flops_skips_embedding_without_lm_head():
    # A sequence-classification model has no output embedding. The input one is
    # still a lookup.
    embed, body = _FakeParam(2000), _FakeParam(1000)
    model = _embedding_model([embed, body], embed, None)
    attention = 12 * 2 * 4 * 32 * 16
    assert flops_per_step(model, 16, 8, 2) == (6 * 1000 + attention) * (8 * 16 * 2)


def test_analytical_flops_skips_frozen_embedding():
    # LoRA freezes the base model, so the embedding lands in the 4*frozen term.
    # Still zero work.
    embed = _FakeParam(2000, requires_grad=False)
    body = _FakeParam(1000, requires_grad=False)
    head = _FakeParam(2000, requires_grad=False)
    model = _embedding_model([embed, body, head], embed, head)
    attention = 12 * 2 * 4 * 32 * 16
    assert flops_per_step(model, 16, 8, 2) == (4 * 3000 + attention) * (8 * 16 * 2)


def test_analytical_flops_without_embedding_accessors():
    # A model that does not expose the HF accessors is estimated as before.
    assert flops_per_step(_FakeModel(), 16, 8, 2) == 17_190_912


def test_analytical_flops_no_config():
    class M:
        config = None

        def parameters(self):
            return [_FakeParam(10)]

    assert flops_per_step(M(), 16, 1, 1) == 0
