# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Extract a consolidated performance report from a tracy run.

Combines two sources:
  1. ``ops_perf_results_*.csv``    — per-op device & dispatch metrics produced
     by ``python3 -m tracy ... -p -r``.
  2. ``tracy_profile_log_host.tracy`` — host-side zone trace, exported to CSV
     via the ``csvexport-release`` binary that ships with tt-metal.

Outputs (under ``--out-dir``, defaults to the report dir):
  - ``perf_summary.json``      — all top-line metrics in a machine-readable form
  - ``perf_summary.md``        — same metrics, human-readable
  - ``host_zones.csv``         — flattened zone trace (one row per zone event)
  - ``host_zones_summary.csv`` — per-zone-name aggregate (count/total/avg/p50/p99)
  - ``ops_by_op_code.csv``     — per-op-code aggregate device times

Usage:
    python -m blacksmith.tools.perf_report \\
        --report-dir llama_3_1_8b_bh/tracy_profile_1_layer/reports/2026_05_18_09_29_08

or fully explicit:

    python -m blacksmith.tools.perf_report \\
        --ops-csv  .../ops_perf_results_2026_05_18_09_29_08.csv \\
        --tracy    .../tracy_profile_log_host.tracy \\
        --out-dir  .../perf_summary
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd


CSVEXPORT_CANDIDATES = (
    "csvexport-release",
    "/localdev/abogdanovic/tt-xla/third_party/tt-mlir/install/bin/csvexport-release",
    "/localdev/abogdanovic/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/build_Release/tools/profiler/bin/csvexport-release",
)

ZONE_BUCKETS: dict[str, tuple[str, ...]] = {
    "compile": (
        r"^JitBuild",
        r"::compile($|\W)",
        r"BuildKernel",
        r"BuildEnv",
        r"GenerateBinaries",
    ),
    "dispatch_host": (
        r"EnqueueProgram",
        r"EnqueueCommand",
        r"EnqueueWrite",
        r"EnqueueRead",
        r"CommandQueue::",
        r"DispatchCommand",
    ),
    "program_cache": (
        r"ProgramCache",
        r"program_cache",
    ),
    "trace": (
        r"BeginTrace",
        r"EndTrace",
        r"ReplayTrace",
        r"Trace::",
    ),
    "tt_xla": (
        r"^TT_",
        r"ttxla",
        r"pjrt",
    ),
    "mlir_lowering": (
        r"MLIR",
        r"Compile",
        r"Lower",
    ),
}


def _find_csvexport(user_path: Optional[str]) -> Optional[str]:
    if user_path:
        return user_path if os.path.exists(user_path) else None
    for cand in CSVEXPORT_CANDIDATES:
        resolved = shutil.which(cand) if "/" not in cand else (cand if os.path.exists(cand) else None)
        if resolved:
            return resolved
    return None


def _ns_to_ms(ns: float) -> float:
    return ns / 1e6


def _ns_to_s(ns: float) -> float:
    return ns / 1e9


@dataclass
class OpsReport:
    total_ops: int
    unique_op_codes: int
    op_code_counts: dict[str, int]

    device_kernel_total_ns: int
    device_fw_total_ns: int
    dispatch_total_ns: int
    dispatch_go_wait_total_ns: int
    op_to_op_latency_total_ns: int
    host_duration_total_ns: int

    wall_clock_ns: int
    device_utilization_pct: float

    programs_compiled: int
    program_cache_hit_rate: float
    unique_program_hashes: int

    top_ops_by_device_time: list[dict] = field(default_factory=list)
    per_op_code: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_dict(self) -> dict:
        return {
            "totals": {
                "ops": self.total_ops,
                "unique_op_codes": self.unique_op_codes,
                "wall_clock_ms": _ns_to_ms(self.wall_clock_ns),
                "device_kernel_ms": _ns_to_ms(self.device_kernel_total_ns),
                "device_fw_ms": _ns_to_ms(self.device_fw_total_ns),
                "dispatch_cq_cmd_ms": _ns_to_ms(self.dispatch_total_ns),
                "dispatch_go_wait_ms": _ns_to_ms(self.dispatch_go_wait_total_ns),
                "op_to_op_gap_ms": _ns_to_ms(self.op_to_op_latency_total_ns),
                "host_duration_ms": _ns_to_ms(self.host_duration_total_ns),
                "device_utilization_pct": self.device_utilization_pct,
            },
            "compilation": {
                "programs_compiled": self.programs_compiled,
                "program_cache_hit_rate_pct": self.program_cache_hit_rate,
                "unique_program_hashes": self.unique_program_hashes,
            },
            "top_ops_by_device_time": self.top_ops_by_device_time,
            "op_code_counts": self.op_code_counts,
        }


def _coerce_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype("int64")


def analyze_ops_csv(path: Path, top_n: int = 25) -> OpsReport:
    df = pd.read_csv(path, low_memory=False)

    for col in (
        "DEVICE KERNEL DURATION [ns]",
        "DEVICE FW DURATION [ns]",
        "DISPATCH TOTAL CQ CMD OP TIME [ns]",
        "DISPATCH GO SEND WAIT TIME [ns]",
        "OP TO OP LATENCY [ns]",
        "HOST DURATION [ns]",
        "HOST START TS",
        "HOST END TS",
    ):
        if col in df.columns:
            df[col] = _coerce_int(df[col])

    cache_hit_mask = df.get("PROGRAM CACHE HIT")
    if cache_hit_mask is not None:
        cache_hit_mask = cache_hit_mask.astype(str).str.strip().str.lower().eq("true")
        programs_compiled = int((~cache_hit_mask).sum())
        cache_hit_rate = (
            float(cache_hit_mask.sum()) / max(1, len(cache_hit_mask)) * 100.0
        )
    else:
        programs_compiled = 0
        cache_hit_rate = 0.0

    unique_program_hashes = int(df["PROGRAM HASH"].nunique()) if "PROGRAM HASH" in df.columns else 0

    if {"HOST START TS", "HOST END TS"}.issubset(df.columns):
        wall_clock_ns = int(df["HOST END TS"].max() - df["HOST START TS"].min())
    else:
        wall_clock_ns = 0

    device_kernel_total = int(df.get("DEVICE KERNEL DURATION [ns]", pd.Series(dtype="int64")).sum())
    util = (device_kernel_total / wall_clock_ns * 100.0) if wall_clock_ns else 0.0

    if {"OP CODE", "DEVICE KERNEL DURATION [ns]"}.issubset(df.columns):
        per_op = (
            df.groupby("OP CODE", dropna=False)
            .agg(
                count=("OP CODE", "size"),
                device_kernel_ns_sum=("DEVICE KERNEL DURATION [ns]", "sum"),
                device_kernel_ns_mean=("DEVICE KERNEL DURATION [ns]", "mean"),
                device_fw_ns_sum=("DEVICE FW DURATION [ns]", "sum"),
                dispatch_ns_sum=("DISPATCH TOTAL CQ CMD OP TIME [ns]", "sum"),
                op_to_op_ns_sum=("OP TO OP LATENCY [ns]", "sum"),
                host_ns_sum=("HOST DURATION [ns]", "sum"),
            )
            .sort_values("device_kernel_ns_sum", ascending=False)
        )
        per_op["device_kernel_pct"] = (
            per_op["device_kernel_ns_sum"] / max(1, device_kernel_total) * 100.0
        )
        top_ops = (
            per_op.head(top_n)
            .reset_index()
            .assign(
                device_kernel_ms_sum=lambda d: d["device_kernel_ns_sum"] / 1e6,
                device_kernel_us_mean=lambda d: d["device_kernel_ns_mean"] / 1e3,
            )[
                [
                    "OP CODE",
                    "count",
                    "device_kernel_ms_sum",
                    "device_kernel_us_mean",
                    "device_kernel_pct",
                ]
            ]
            .to_dict(orient="records")
        )
    else:
        per_op = pd.DataFrame()
        top_ops = []

    return OpsReport(
        total_ops=len(df),
        unique_op_codes=int(df["OP CODE"].nunique()) if "OP CODE" in df.columns else 0,
        op_code_counts=(
            df["OP CODE"].value_counts().to_dict() if "OP CODE" in df.columns else {}
        ),
        device_kernel_total_ns=device_kernel_total,
        device_fw_total_ns=int(df.get("DEVICE FW DURATION [ns]", pd.Series(dtype="int64")).sum()),
        dispatch_total_ns=int(
            df.get("DISPATCH TOTAL CQ CMD OP TIME [ns]", pd.Series(dtype="int64")).sum()
        ),
        dispatch_go_wait_total_ns=int(
            df.get("DISPATCH GO SEND WAIT TIME [ns]", pd.Series(dtype="int64")).sum()
        ),
        op_to_op_latency_total_ns=int(
            df.get("OP TO OP LATENCY [ns]", pd.Series(dtype="int64")).sum()
        ),
        host_duration_total_ns=int(df.get("HOST DURATION [ns]", pd.Series(dtype="int64")).sum()),
        wall_clock_ns=wall_clock_ns,
        device_utilization_pct=util,
        programs_compiled=programs_compiled,
        program_cache_hit_rate=cache_hit_rate,
        unique_program_hashes=unique_program_hashes,
        top_ops_by_device_time=top_ops,
        per_op_code=per_op,
    )


@dataclass
class HostReport:
    total_zone_events: int
    total_zone_exec_ns: int
    bucket_totals_ns: dict[str, int]
    bucket_counts: dict[str, int]
    top_zones: list[dict]
    per_zone: pd.DataFrame

    def to_dict(self) -> dict:
        return {
            "total_zone_events": self.total_zone_events,
            "total_zone_exec_ms": _ns_to_ms(self.total_zone_exec_ns),
            "buckets": {
                name: {
                    "total_ms": _ns_to_ms(ns),
                    "count": self.bucket_counts.get(name, 0),
                }
                for name, ns in self.bucket_totals_ns.items()
            },
            "top_zones_by_total_time": self.top_zones,
        }


def export_tracy_zones(tracy_file: Path, csvexport_bin: str) -> pd.DataFrame:
    """Run csvexport-release in unwrap mode and return a DataFrame of zone events."""
    proc = subprocess.run(
        [csvexport_bin, "-u", str(tracy_file)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    df = pd.read_csv(io.BytesIO(proc.stdout), low_memory=False)
    df["exec_time_ns"] = pd.to_numeric(df.get("exec_time_ns", 0), errors="coerce").fillna(0).astype("int64")
    df["ns_since_start"] = pd.to_numeric(df.get("ns_since_start", 0), errors="coerce").fillna(0).astype("int64")
    df["name"] = df["name"].astype(str)
    return df


def _bucket_for(name: str) -> Optional[str]:
    for bucket, patterns in ZONE_BUCKETS.items():
        for pat in patterns:
            if re.search(pat, name):
                return bucket
    return None


def analyze_host_zones(df: pd.DataFrame, top_n: int = 25) -> HostReport:
    df = df.copy()
    df["bucket"] = df["name"].map(_bucket_for)

    bucket_totals = (
        df.groupby("bucket", dropna=True)["exec_time_ns"].sum().to_dict()
    )
    bucket_counts = df.groupby("bucket", dropna=True).size().to_dict()

    per_zone = (
        df.groupby("name")["exec_time_ns"]
        .agg(["count", "sum", "mean", "median", lambda s: s.quantile(0.99), "max"])
        .rename(
            columns={
                "count": "count",
                "sum": "total_ns",
                "mean": "mean_ns",
                "median": "p50_ns",
                "<lambda_0>": "p99_ns",
                "max": "max_ns",
            }
        )
        .sort_values("total_ns", ascending=False)
    )

    top_zones = (
        per_zone.head(top_n)
        .reset_index()
        .assign(
            total_ms=lambda d: d["total_ns"] / 1e6,
            mean_us=lambda d: d["mean_ns"] / 1e3,
            p99_us=lambda d: d["p99_ns"] / 1e3,
        )[["name", "count", "total_ms", "mean_us", "p99_us"]]
        .to_dict(orient="records")
    )

    return HostReport(
        total_zone_events=len(df),
        total_zone_exec_ns=int(df["exec_time_ns"].sum()),
        bucket_totals_ns={k: int(v) for k, v in bucket_totals.items() if k},
        bucket_counts={k: int(v) for k, v in bucket_counts.items() if k},
        top_zones=top_zones,
        per_zone=per_zone,
    )


def render_markdown(ops: OpsReport, host: Optional[HostReport], inputs: dict) -> str:
    out = io.StringIO()
    out.write("# Performance report\n\n")
    out.write("## Inputs\n\n")
    for k, v in inputs.items():
        out.write(f"- **{k}**: `{v}`\n")
    out.write("\n## Top-line metrics (from ops CSV)\n\n")
    t = ops.to_dict()["totals"]
    out.write("| Metric | Value |\n|---|---|\n")
    out.write(f"| Total ops | {ops.total_ops} |\n")
    out.write(f"| Unique op codes | {ops.unique_op_codes} |\n")
    out.write(f"| Wall clock | {t['wall_clock_ms']:.3f} ms |\n")
    out.write(f"| Device kernel time | {t['device_kernel_ms']:.3f} ms |\n")
    out.write(f"| Device FW time | {t['device_fw_ms']:.3f} ms |\n")
    out.write(f"| Dispatch (CQ cmd) time | {t['dispatch_cq_cmd_ms']:.3f} ms |\n")
    out.write(f"| Dispatch go-send wait | {t['dispatch_go_wait_ms']:.3f} ms |\n")
    out.write(f"| Op-to-op gap (runtime overhead) | {t['op_to_op_gap_ms']:.3f} ms |\n")
    out.write(f"| Host duration (sum of per-op host time) | {t['host_duration_ms']:.3f} ms |\n")
    out.write(f"| Device utilization | {t['device_utilization_pct']:.2f} % |\n")
    out.write("\n## Compilation\n\n")
    c = ops.to_dict()["compilation"]
    out.write("| Metric | Value |\n|---|---|\n")
    out.write(f"| Programs compiled (cache misses) | {c['programs_compiled']} |\n")
    out.write(f"| Program cache hit rate | {c['program_cache_hit_rate_pct']:.2f} % |\n")
    out.write(f"| Unique program hashes | {c['unique_program_hashes']} |\n")

    if host is not None:
        out.write("\n## Host-side breakdown (from tracy)\n\n")
        out.write(f"- Total host zone events: {host.total_zone_events}\n")
        out.write(f"- Total host zone exec time: {_ns_to_ms(host.total_zone_exec_ns):.3f} ms\n\n")
        out.write("| Bucket | Total (ms) | Count |\n|---|---|---|\n")
        for bucket, ns in sorted(
            host.bucket_totals_ns.items(), key=lambda kv: kv[1], reverse=True
        ):
            out.write(
                f"| {bucket} | {_ns_to_ms(ns):.3f} | {host.bucket_counts.get(bucket, 0)} |\n"
            )

        out.write("\n### Top host zones\n\n")
        out.write("| Zone | Count | Total (ms) | Mean (us) | p99 (us) |\n|---|---|---|---|---|\n")
        for z in host.top_zones:
            out.write(
                f"| `{z['name']}` | {z['count']} | {z['total_ms']:.3f} | "
                f"{z['mean_us']:.2f} | {z['p99_us']:.2f} |\n"
            )

    out.write("\n## Top ops by device kernel time\n\n")
    out.write("| OP CODE | Count | Device kernel (ms) | Mean (us) | % of device |\n")
    out.write("|---|---|---|---|---|\n")
    for o in ops.top_ops_by_device_time:
        out.write(
            f"| `{o['OP CODE']}` | {o['count']} | {o['device_kernel_ms_sum']:.3f} | "
            f"{o['device_kernel_us_mean']:.2f} | {o['device_kernel_pct']:.2f} % |\n"
        )

    return out.getvalue()


def _autodiscover_in_report_dir(report_dir: Path) -> tuple[Optional[Path], Optional[Path]]:
    ops_csv = next(iter(sorted(report_dir.glob("ops_perf_results_*.csv"))), None)
    tracy = report_dir / "tracy_profile_log_host.tracy"
    return ops_csv, (tracy if tracy.exists() else None)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report-dir", type=Path, help="Directory containing ops_perf_results_*.csv and tracy_profile_log_host.tracy")
    p.add_argument("--ops-csv", type=Path, help="Path to ops_perf_results_*.csv")
    p.add_argument("--tracy", type=Path, help="Path to tracy_profile_log_host.tracy")
    p.add_argument("--out-dir", type=Path, help="Where to write outputs (default: report-dir or ops-csv parent)")
    p.add_argument("--csvexport-bin", type=str, help="Path to tracy csvexport-release binary")
    p.add_argument("--top-n", type=int, default=25, help="Top N rows in summary tables")
    p.add_argument("--no-host", action="store_true", help="Skip parsing the .tracy file")
    args = p.parse_args(argv)

    ops_csv: Optional[Path] = args.ops_csv
    tracy: Optional[Path] = args.tracy
    if args.report_dir:
        auto_ops, auto_tracy = _autodiscover_in_report_dir(args.report_dir)
        ops_csv = ops_csv or auto_ops
        tracy = tracy or auto_tracy

    if not ops_csv or not ops_csv.exists():
        print("error: could not locate ops_perf_results_*.csv (pass --ops-csv or --report-dir)", file=sys.stderr)
        return 2

    out_dir: Path = args.out_dir or (args.report_dir or ops_csv.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[perf_report] ops csv: {ops_csv}")
    ops = analyze_ops_csv(ops_csv, top_n=args.top_n)
    ops.per_op_code.to_csv(out_dir / "ops_by_op_code.csv")
    print(f"[perf_report] wrote {out_dir / 'ops_by_op_code.csv'}")

    host: Optional[HostReport] = None
    if tracy and tracy.exists() and not args.no_host:
        csvexport = _find_csvexport(args.csvexport_bin)
        if not csvexport:
            print(
                "warning: csvexport-release not found; skipping host trace. "
                "Pass --csvexport-bin or install tt-metal.",
                file=sys.stderr,
            )
        else:
            print(f"[perf_report] tracy:   {tracy}")
            print(f"[perf_report] using csvexport: {csvexport}")
            zones = export_tracy_zones(tracy, csvexport)
            zones.to_csv(out_dir / "host_zones.csv", index=False)
            print(f"[perf_report] wrote {out_dir / 'host_zones.csv'} ({len(zones)} events)")
            host = analyze_host_zones(zones, top_n=args.top_n)
            host.per_zone.to_csv(out_dir / "host_zones_summary.csv")
            print(f"[perf_report] wrote {out_dir / 'host_zones_summary.csv'}")
    elif args.no_host:
        print("[perf_report] --no-host set; skipping tracy")
    else:
        print("warning: tracy file not found; skipping host trace.", file=sys.stderr)

    inputs = {
        "ops_csv": str(ops_csv),
        "tracy": str(tracy) if tracy else "(not used)",
        "out_dir": str(out_dir),
    }
    payload = {"inputs": inputs, "ops": ops.to_dict()}
    if host is not None:
        payload["host"] = host.to_dict()

    (out_dir / "perf_summary.json").write_text(json.dumps(payload, indent=2))
    md = render_markdown(ops, host, inputs)
    (out_dir / "perf_summary.md").write_text(md)
    print(f"[perf_report] wrote {out_dir / 'perf_summary.json'}")
    print(f"[perf_report] wrote {out_dir / 'perf_summary.md'}")
    print()
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
