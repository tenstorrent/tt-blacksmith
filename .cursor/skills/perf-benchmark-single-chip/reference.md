# tt-perf-report & Tracy reference

## tt-perf-report metrics

Each row in the enriched CSV corresponds to one device op.

| Metric | Unit | Description |
|---|---|---|
| Device Time | us | Wall-clock time on device for this op |
| Op-to-Op Gap | us | Host-side gap between consecutive ops (dispatch overhead). **Red if >6.5 us** |
| Cores | count | Number of Tensix cores used. **Red if <10** (max 64 on Wormhole) |
| DRAM | GB/s | Measured DRAM bandwidth |
| DRAM % | % | % of peak DRAM BW (288 GB/s on Wormhole) |
| FLOPs | TFLOPs | Measured compute throughput |
| FLOPs % | % | % of peak compute for the math fidelity used |
| Bound | category | Classification (see below) |
| Math Fidelity | enum | HiFi4, HiFi2, or LoFi |

## Bound classification

| Bound | Meaning | Threshold |
|---|---|---|
| **DRAM** | Memory-bandwidth bound (>65% peak BW) | DRAM % > 65 |
| **FLOP** | Compute bound (>65% peak FLOPs) | FLOPs % > 65 |
| **BOTH** | Saturating both DRAM and compute | Both > 65% |
| **SLOW** | Neither DRAM nor compute saturated | Both < 65% -- **biggest optimization target** |
| **HOST** | Bottlenecked by host dispatch | Op-to-op gap dominates |

## Wormhole peak numbers (N150, single chip)

| Math Fidelity | Peak TFLOPs | Description |
|---|---|---|
| HiFi4 | 74 | Full precision |
| HiFi2 | 148 | Half precision mantissa |
| LoFi | 262 | Lowest precision |

- **Peak DRAM BW**: 288 GB/s
- **Max cores**: 64 Tensix cores

## Blackhole peak numbers (P150, single chip)

| Math Fidelity | Peak TFLOPs | Description |
|---|---|---|
| HiFi4 | 180 | Full precision |
| HiFi2 | 359 | Half precision mantissa |
| LoFi | 719 | Lowest precision |

- **Peak DRAM BW**: 512 GB/s (GDDR6)
- **Max cores**: 130 Tensix cores

## tt-perf-report CLI flags

Run `tt-perf-report --help` for the full list.

## Tracy CLI flags

Run `PYTHONPATH="<tt-xla-repo>/python_package:$PYTHONPATH" python3 -m tracy --help` for the full list.

## Tracy output files

All output lands in `<output_folder>/reports/<timestamp>/`:

| File | Description |
|------|-------------|
| `ops_perf_results_<timestamp>.csv` | **Primary** -- per-op device metrics. Feed to `tt-perf-report`. |
| `tracy_profile_log_host.tracy` | Host-side trace. Open in Tracy GUI for dispatch timeline. |
| `profile_log_device.csv` | Raw device profiler data |

Additional files in `<output_folder>/.logs/`:

| File | Description |
|------|-------------|
| `tracy_ops_data.csv` | Tracy op metadata |
| `tracy_ops_times.csv` | Tracy op timing data |
| `cpp_device_perf_report.csv` | C++ runtime device perf analysis |
| `sync_device_info.csv` | Device sync information |

## Tracy signpost workflow

Signposts mark named boundaries in the trace. Use them to isolate specific steps for analysis.

**In training script** (signpost names are derived from `BENCH_MAX_STEPS`; with N=5, the last steady step is `steady_4`):
```python
import tracy
bench_step = len(step_times)
step_label = "compile" if bench_step == 0 else f"steady_{bench_step}"
tracy.signpost(f"step_{step_label}_start")
# ... training step ...
tracy.signpost(f"step_{step_label}_end")
```

**In tt-perf-report** (use the last steady step for most stable data):
```bash
# With BENCH_MAX_STEPS=5, last steady step = steady_4:
tt-perf-report ops_perf_results.csv \
    --start-signpost step_steady_4_start \
    --end-signpost step_steady_4_end
```

Without signpost flags, tt-perf-report analyzes ops after the last signpost. If no signpost has ops after it, you get "No device operations found" -- always use explicit signpost ranges.

## Host-side PJRT profiling with Tracy zones

tt-xla has optional Tracy zones across PJRT API entry points (compile, execute, buffer transfer). Gated behind a build flag:

```bash
# Rebuild tt-xla with PJRT zones enabled:
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DTTXLA_TRACY_ZONES=ON
cmake --build build
```

Run host-only:
```bash
tracy -p --no-device -m <script>
```

**Known issue:** `--no-device` does not save a `.tracy` file. Workaround: run `capture-release -o output.tracy` in a separate terminal before starting the profiling run.

## Alternative analysis tools

- **[TT-NN Visualizer](https://github.com/tenstorrent/ttnn-visualizer)**: `pip install ttnn-visualizer`, web UI at `localhost:8000`. Upload the perf output folder for interactive exploration.
- **Tracy GUI**: Open `tracy_profile_log_host.tracy` for host-side timeline visualization. Useful for identifying dispatch patterns and host bottlenecks.

## What to look for in results

1. **SLOW ops** -- not saturating DRAM or compute. Largest optimization targets.
2. **High op-to-op gaps** (>6.5 us) -- host dispatch overhead.
3. **Low core counts** (<10 cores) -- underutilizing available parallelism.
4. **TM (tensor manipulation) ops** -- reshape, permute, slice. These are often SLOW and add overhead without doing useful compute. Reducing TM op count/time is a common optimization.
5. **Matmul FLOPs %** -- weighted mean FLOPs utilization across matmul ops. Higher is better; indicates how efficiently the hardware is being used for the compute-heavy operations.
6. **Stacked report** (`tt_perf_report_stacked.csv`) -- aggregate time per op type. Shows where total device time is spent across op categories.
