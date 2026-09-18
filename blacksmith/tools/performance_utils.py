# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Throughput accounting for training scripts.

Two numerators for utilization: an analytical FLOPs-per-step estimate (MFU) that every
script can compute from the model config, and the exact matrix-engine FLOPs tt-mlir
reports for the graphs it compiled (HFU), when the compiler is asked to emit them.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch

# Blackhole P150. 110 is the 10x11 worker grid the device reports, not the 140 Tensix on
# the die: counting 140 understates MFU by 1.27x and breaks comparability with tt-mlir's
# own report.
P150_TENSIX_CORES = 110
P150_CLOCK_HZ = 1_350_000_000
FLOPS_PER_TILE_MATMUL = 2 * 32 * 32 * 32
CYCLES_PER_TILE_MATMUL_HIFI4 = 64

# Per-parameter FLOP cost per token: a trainable weight pays forward, input gradient and
# weight gradient; a frozen one has no weight gradient.
FLOPS_PER_TRAINABLE_PARAM = 6
FLOPS_PER_FROZEN_PARAM = 4
# Attention scores and their gradients, per layer, per head: weight-free, so freezing does
# not change it.
FLOPS_PER_ATTENTION_ELEMENT = 12


def peak_flops(
    tensix_cores: int = P150_TENSIX_CORES,
    clock_hz: int = P150_CLOCK_HZ,
    cycles_per_tile_matmul: int = CYCLES_PER_TILE_MATMUL_HIFI4,
) -> float:
    """Peak matmul FLOP/s the device can sustain at the given math fidelity."""
    return tensix_cores * clock_hz * FLOPS_PER_TILE_MATMUL / cycles_per_tile_matmul


@dataclass(frozen=True)
class FlopParams:
    """Parameter counts charged in the analytical FLOP estimate."""

    trainable: int
    frozen: int

    @property
    def total(self) -> int:
        return self.trainable + self.frozen


def count_flop_params(
    model: torch.nn.Module,
    embedding_weight: torch.Tensor,
    is_trainable: Callable[[str], bool],
    tied_embeddings: bool,
) -> FlopParams:
    """Split the model's parameters into the trainable and frozen counts the estimate charges.

    The input embedding is charged only when tied, and both branches reach the same total.
    Tied, the one tensor `named_parameters()` yields is the LM head's genuine matmul weight.
    Untied, the LM head is charged separately and the embedding is dropped rather than
    charged as frozen: it is a row lookup with no matmul, no input gradient (integer ids)
    and a scatter-add weight gradient over a handful of rows.

    Args:
        model: The model whose parameters are counted (deduplicated by identity).
        embedding_weight: The input embedding weight, dropped when `tied_embeddings` is False.
        is_trainable: Predicate on the parameter name.
        tied_embeddings: Whether the LM head shares the embedding weight.

    Returns:
        The trainable and frozen parameter counts.
    """
    counted: dict[int, tuple[str, torch.Tensor]] = {}
    for name, parameter in model.named_parameters():
        counted.setdefault(id(parameter), (name, parameter))
    if not tied_embeddings:
        counted.pop(id(embedding_weight), None)

    trainable = sum(p.numel() for name, p in counted.values() if is_trainable(name))
    frozen = sum(p.numel() for name, p in counted.values() if not is_trainable(name))
    return FlopParams(trainable=trainable, frozen=frozen)


def training_flops_per_token(
    flop_params: FlopParams,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    seq_len: int,
) -> int:
    """Analytical training FLOPs per token.

    flops/token = 6 * trainable + 4 * frozen + 12 * L * H * Q * T
    """
    attention = FLOPS_PER_ATTENTION_ELEMENT * num_layers * num_heads * head_dim * seq_len
    return FLOPS_PER_TRAINABLE_PARAM * flop_params.trainable + FLOPS_PER_FROZEN_PARAM * flop_params.frozen + attention


def training_flops_per_step(
    flops_per_token: int,
    batch_size: int,
    seq_len: int,
    gradient_accumulation_steps: int = 1,
    num_chips: int = 1,
) -> int:
    """FLOPs one optimizer step performs across all micro-batches and chips."""
    return flops_per_token * batch_size * seq_len * gradient_accumulation_steps * num_chips


@dataclass
class TtnnPerfMetrics:
    """Exact matrix-engine FLOPs tt-mlir reports for the graphs it compiled.

    tt-mlir writes `<output_file>.json` after each compile and rewrites the same path for
    every graph, so the file has to be read (and removed) right after each compile rather
    than globbed at the end. Nothing is written for a compile served from the on-disk
    cache, in which case HFU stays unreported. `total_flops` counts matrix-engine work
    only, so a purely elementwise graph (e.g. the optimizer) contributes nothing.
    """

    output_file: str
    total_flops: int = 0
    peak_flops_per_sec: float = 0.0

    @property
    def _path(self) -> Path:
        return Path(f"{self.output_file}.json")

    def clear_stale(self) -> None:
        """Remove reports left behind by an earlier run."""
        base = Path(self.output_file)
        for stale in base.parent.glob(f"{base.name}*.json"):
            stale.unlink()

    def discard(self) -> None:
        """Drop a report that belongs to a graph the step accounting must not count."""
        if self._path.exists():
            self._path.unlink()

    def collect(self) -> bool:
        """Accumulate the report of the graph compiled since the last call, if any."""
        if not self._path.exists():
            return False
        with self._path.open() as report:
            flops = json.load(report).get("flops") or {}
        self.total_flops += flops.get("total_flops") or 0
        self.peak_flops_per_sec = max(self.peak_flops_per_sec, flops.get("peak_flops_per_sec") or 0)
        self._path.unlink()
        return True

    def hfu(self, step_time_sec: float) -> Optional[float]:
        """Hardware FLOPs utilization in percent, or None when nothing was reported."""
        if not self.peak_flops_per_sec:
            return None
        return self.total_flops / step_time_sec / self.peak_flops_per_sec * 100.0
