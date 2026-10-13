---
name: perf-benchmark-single-chip
description: Run device performance benchmarks for tt-blacksmith single-chip training workloads. Use when the user asks to benchmark, profile, or measure performance of a training workload on Tenstorrent hardware, or mentions tracy, tt-perf-report, or device time analysis.
---

# Single-chip perf benchmark

Benchmark a tt-blacksmith training workload by running Tracy device profiling on the actual training loop and analyzing with `tt-perf-report`. This gives per-op device timing, wall-clock timing, and a host-side trace in a single run.

## Eligible workloads

Any workload under `blacksmith/experiments/torch/` that targets a single chip. Check the YAML config: if `mesh_shape` is `null` or absent, the workload is eligible. Multi-chip workloads (TG, Galaxy, `mesh_shape` set) are out of scope.

## Prerequisites

### 1. Virtual environment

Activate the tt-blacksmith XLA virtual environment to isolate dependencies from other projects on the machine:

```bash
cd <tt-blacksmith-repo-root>
source env/activate --xla
```

### 2. Clean device state

Before running any workload, ensure no stale processes are holding the TT device. A previous process that was force-killed (kill -9) leaves the device in a bad state, causing "Timeout waiting for Ethernet core service" or "Waiting for lock 'CHIP_IN_USE_*_PCIe'" errors.

```bash
ps aux | grep -E "python3.*blacksmith|python3.*torch" | grep -v grep

# Kill any stale processes, then reset all devices:
tt-smi -r
```

If `tt-smi` is not installed, install it inside the activated venv: `pip install tt-smi`.

Always reset devices between runs (after a crash or kill) to avoid "Timeout waiting for Ethernet core" errors. Wait a few seconds after reset before starting a new process.

### 3. Tracy

Tracy is bundled with the `pjrt_plugin_tt` wheel in the xla venv, but the CLI wrapper depends on the `pjrt_plugin_tt` source package matching the installed version.

**Verify:** Run tracy with PYTHONPATH pointing to the tt-xla source package:

```bash
PYTHONPATH="<tt-xla-repo>/python_package:$PYTHONPATH" python3 -m tracy --help
```

This should show usage info including `-p`, `-r`, `--sync-host-device`. Always invoke tracy as `python3 -m tracy` with this PYTHONPATH -- the bare `tracy` CLI entry point may fail if the installed `pjrt_plugin_tt` wheel is out of sync with the source tree.

**Tracy tools folder:** Tracy needs `capture-release` and `csvexport-release` binaries. Find them in the workspace:

```bash
find "$(dirname "$(pwd)")" -name "capture-release" -type f 2>/dev/null
```

Pick the path under `build_Release/tools/profiler/bin` or similar.

### 4. tt-perf-report

`tt-perf-report` analyzes the raw per-op CSV from Tracy and classifies each op as compute-bound, memory-bound, or slow. It also generates aggregate stacked reports and charts. Install it inside the activated venv:

```bash
pip install tt-perf-report
```

## Benchmark steps

### Step 1: Instrument the training script

Make these temporary changes to the training script (all will be reverted in Step 5):

1. **Add imports:**

```python
import time
import tracy
```

2. **Skip initial validation** -- Comment out or skip the initial `validate()` / `model.eval()` block before the training loop.

3. **Add tracy signposts and wall-clock timers** around the fwd+bwd section:

Use `len(step_times)` as the step counter -- this is self-contained and avoids relying on the script's own `global_step` variable, which may start at a different value or increment at a different point in the loop.

```python
bench_step = len(step_times)
step_label = "compile" if bench_step == 0 else f"steady_{bench_step}"
tracy.signpost(f"step_{step_label}_start")
t_start = time.perf_counter()

# ... model forward, loss computation, loss.backward() ...
# ... torch_xla.sync(wait=True) ...

t_end = time.perf_counter()
tracy.signpost(f"step_{step_label}_end")

step_ms = (t_end - t_start) * 1000.0
step_times.append(step_ms)
tag = "COMPILE" if len(step_times) == 1 else "steady"
print(f"[TIMER] Step {len(step_times)}: fwd+bwd = {step_ms:.1f} ms  ({tag})")
```

**Adapting to different loop structures:**

- The sync must come **after backward**, not between forward and backward. Calls like `loss.item()` force an implicit sync -- move them after backward or remove them when benchmarking.
- If the script already has `torch_xla.sync(wait=True)` after backward -- place `t_start` before the forward call and `t_end` right after that sync.
- If the script has no explicit sync after backward -- add `torch_xla.sync(wait=True)` after `loss.backward()` and place `t_end` right after it.

**Gradient accumulation:** If the loop accumulates gradients over multiple micro-steps before an optimizer step, place the signpost/timer around each individual fwd+bwd+sync micro-step (not the full accumulation cycle). The early stopping check (step 4 below) should go inside the accumulation-complete branch, **immediately after `global_step` is incremented** and before any validation, logging, or checkpointing. Make sure `BENCH_MAX_STEPS` is large enough to include at least one full optimizer step (i.e., `BENCH_MAX_STEPS >= gradient_accumulation_steps + 1`) so the trace captures the optimizer update as well.

4. **Add a timer summary function and early stopping** after ~5 steps (keeps the Tracy trace small):

```python
step_times: list[float] = []
BENCH_MAX_STEPS = 5

def _print_timer_summary():
    if len(step_times) <= 1:
        return
    steady = step_times[1:]
    print(f"\n{'='*50}")
    print(f"[TIMER] Summary (steady-state, steps 2-{len(step_times)}):")
    print(f"  samples : {len(steady)}")
    print(f"  mean    : {sum(steady)/len(steady):.1f} ms")
    print(f"  median  : {sorted(steady)[len(steady)//2]:.1f} ms")
    print(f"  min     : {min(steady):.1f} ms")
    print(f"  max     : {max(steady):.1f} ms")
    print(f"{'='*50}\n")
```

Then in the training loop, place the early stopping check **immediately after `global_step` is incremented** -- before any logging, validation, metric commits, cache clearing, or checkpointing. If early stopping comes after these, a validation pass at the final step will run unnecessarily, wasting minutes and polluting the Tracy trace.

```python
if global_step >= BENCH_MAX_STEPS:
    tracy.signpost("benchmark_complete")
    _print_timer_summary()
    return
```

5. **Disable `xr.clear_computation_cache()`** -- If the training script calls `xr.clear_computation_cache()` between steps, comment it out. It forces XLA to recompile the graph every step, which invalidates steady-state measurements and can cause OOM.

**Important:**
- Step 1 includes XLA graph compilation -- exclude it from steady-state averages.

### Step 2: Run Tracy

```bash
source env/activate --xla
tt-smi -r

export PYTHONPATH="<tt-xla-repo>/python_package:<tt-blacksmith-repo-root>:$PYTHONPATH"
WANDB_MODE=disabled python3 -m tracy -p -r --sync-host-device \
    --tracy-tools-folder <path-to-tracy-tools-bin> \
    -o <model>_baseline/tracy_profile \
    <training_script>.py [args]
```

**Tracy flags:**

| Flag | Description |
|------|-------------|
| `-p` | Only profile explicitly enabled zones |
| `-r` | Generate ops report |
| `--sync-host-device` | Synchronize host and device timelines |
| `-o FOLDER` | Output folder for profiler artifacts |
| `--tracy-tools-folder PATH` | Path to dir containing `capture-release` and `csvexport-release` |

Tracy adds ~10-15% overhead to wall-clock time. The `[TIMER]` values will be slightly inflated vs running without Tracy; this is expected and acceptable for profiling purposes.

### Step 3: Analyze with tt-perf-report

Tracy outputs land in `<output_folder>/reports/<timestamp>/`:

| File | Description |
|------|-------------|
| `ops_perf_results_<timestamp>.csv` | Per-op device performance data |
| `tracy_profile_log_host.tracy` | Host-side trace (open in Tracy GUI for dispatch timeline) |
| `profile_log_device.csv` | Raw device profiler data |

Find the latest report and copy for analysis:

```bash
# Find the most recent report directory:
LATEST_REPORT=$(ls -td <model>_baseline/tracy_profile/reports/*/ | head -1)

mkdir -p <model>_baseline/tracy_perf
cp "$LATEST_REPORT"/ops_perf_results_*.csv \
    <model>_baseline/tracy_perf/ops_perf_results.csv
```

**Use signposts to analyze a single steady-state step.** Use the last steady step for the most stable data. With `BENCH_MAX_STEPS = N`, the last steady step is `step_steady_{N-1}`. A single call with `--csv` prints the full report to stdout and exports files:

```bash
# Example with BENCH_MAX_STEPS=5 (last steady step = steady_4):
tt-perf-report <model>_baseline/tracy_perf/ops_perf_results.csv \
    --start-signpost step_steady_4_start --end-signpost step_steady_4_end \
    --no-advice --csv <model>_baseline/tracy_perf/tt_perf_report.csv
```

This prints the per-op report and stacked summary to stdout, and generates:
- `tt_perf_report.csv` -- enriched per-op table with bound classification
- `tt_perf_report_stacked.csv` -- aggregate by op type
- `tt_perf_report_stacked.png` -- stacked bar chart

For metric details, see [reference.md](reference.md).

### Step 4: Compare device time vs wall-clock

| Metric | Source |
|---|---|
| Device time | Total from stacked bar chart title, or sum of "Device Time" column in `ops_perf_results.csv` |
| Wall-clock fwd+bwd | Steady-state mean from `[TIMER]` output (excluding step 1) |
| Overhead | `(wall_clock - device_time) / wall_clock * 100` |

Overhead sources: host dispatch latency, Python loop overhead, data transfer.

**Interpreting HOST-bound results:** If overhead is >70%, the workload is likely HOST-bound -- the device finishes quickly but host dispatch between ops dominates. Key indicators:
- Most ops have op-to-op gaps >> 6.5 μs (host is idle between dispatches)
- Device time is a small fraction of wall-clock
- Per-op host dispatch cost is roughly constant (~0.5 ms/op); compute `(wall_clock - device_time) / op_count` to verify

HOST-bound overhead scales with op count and is inversely proportional to tensor size (smaller batch/seq = less device work per op = higher overhead %). For HOST-bound workloads, the main optimization targets are reducing op count (graph-level fusion) and increasing batch/sequence length.

### Step 5: Revert the training script

After Tracy and tt-perf-report are done, revert all instrumentation changes from Step 1 (imports, signposts, timers, early stopping, skipped validation). The training script must be restored to its original state so it remains usable for actual training.

## Output summary

After completing steps 1-5, present a summary to the user with:
- Steady-state wall-clock time (mean, median, min, max)
- Device time from tt-perf-report (sum `Device_Time_Sum_us` from `tt_perf_report_stacked.csv`)
- Overhead percentage and per-op dispatch cost (`(wall_clock - device_time) / op_count`)
- Total op count per step (sum `Ops_Count` from stacked CSV)
- Device time breakdown by op category (Compute %, TM %, Other %)
- Matmul weighted mean FLOPs utilization
- Top SLOW ops (biggest optimization targets)
- Stacked bar chart location (`tt_perf_report_stacked.png`)
- Host trace location (`tracy_profile_log_host.tracy`)

## Note on what Tracy captures

Tracy device profiling gives per-op device timing. It does **not** break down host-side time (Python overhead, PJRT dispatch, data prep). The `.tracy` host trace file shows the tt-metal dispatch timeline but not the Python/PJRT layer above it.

## Troubleshooting

- **"Timeout waiting for Ethernet core service" / "Waiting for lock 'CHIP_IN_USE_*_PCIe'"**: Kill all stale Python processes, then reset all devices: `tt-smi -r`. Wait a few seconds before starting a new process.
- **Tracy `ImportError: cannot import name 'setup_tt_metal_home' from 'pjrt_plugin_tt'`**: The installed `pjrt_plugin_tt` wheel is out of sync with the source tree. Fix by adding the tt-xla source package to PYTHONPATH and using `python3 -m tracy` instead of bare `tracy`. See [Tracy setup](#3-tracy).
- **Tracy `No device operations found`**: Default analysis uses ops after the last signpost. Use `--start-signpost` and `--end-signpost` flags to select a specific step range.
- **`env/activate --xla` triggers a package upgrade**: The activate script auto-detects pjrt-plugin-tt version mismatches and runs `pip install` to upgrade. This is expected and ensures profiling uses the current wheel. It can take 1-2 minutes -- let it complete. To avoid repeating it, don't re-source the activate script between tracy and tt-perf-report steps if the venv is already active.
- **Tracy crashes with `TypeError: Invalid value ... for dtype 'str'` during report generation**: Pandas version incompatibility in `process_device_log.py`. Fix by replacing `df.iloc[:, 8] = ...` with column-name-based assignment: `col8 = df.columns[8]; df[col8] = pd.to_numeric(df[col8], errors="coerce").fillna(-1).astype(int)` (same for column 9).
- **`tt-perf-report` missing**: `pip install tt-perf-report`.
- **`tt-smi` not installed**: `pip install tt-smi`.
