# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable, Iterator, Optional

_phases: list[tuple[str, float]] = []
_sink: Optional[Callable[[str], None]] = None


def set_sink(sink: Callable[[str], None]) -> None:
    global _sink
    _sink = sink


def _emit(message: str) -> None:
    (_sink or print)(message)


def fmt(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{secs:04.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes):02d}m{secs:04.1f}s"


@contextmanager
def phase(name: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        _phases.append((name, elapsed))
        _emit(f"[time] {name}: {fmt(elapsed)}")


def record(name: str, seconds: float) -> None:
    _phases.append((name, seconds))
    _emit(f"[time] {name}: {fmt(seconds)}")


def summary(stage: str, total: float) -> None:
    _emit(f"[time] ─── {stage} ───")
    if not _phases:
        _emit(f"[time]   TOTAL  {fmt(total)}")
        return
    width = max(len(name) for name, _ in _phases)
    tracked = 0.0
    for name, elapsed in _phases:
        share = 100.0 * elapsed / total if total > 0 else 0.0
        _emit(f"[time]   {name:<{width}}  {fmt(elapsed):>10}  {share:5.1f}%")
        tracked += elapsed
    other = total - tracked
    if other > 0.05 * total and total > 0:
        _emit(f"[time]   {'(untimed)':<{width}}  {fmt(other):>10}  {100.0 * other / total:5.1f}%")
    _emit(f"[time]   {'TOTAL':<{width}}  {fmt(total):>10}")
