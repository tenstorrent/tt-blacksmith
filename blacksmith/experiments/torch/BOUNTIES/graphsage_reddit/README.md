# GraphSAGE Node Classification on Reddit

This experiment trains a two-layer [GraphSAGE](https://arxiv.org/abs/1706.02216)
model for node classification on the
[Reddit dataset](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Reddit.html).
It is the workload proposed in [bounty #529](https://github.com/tenstorrent/tt-blacksmith/issues/529).

## Dataset

| Property | Value |
|---|---:|
| Nodes | 232,965 |
| Edges | 114,615,892 |
| Node features | 602 |
| Classes | 41 |
| Train / validation / test nodes | 153,431 / 23,831 / 55,703 |

The graph is too large for full-graph training on the target device. The
experiment therefore uses `NeighborLoader` mini-batches with configurable
per-hop fanouts.

This workload uses a transductive setup: neighbor sampling can access the full
graph's features and edges, including validation and test nodes. The training
mask selects the seed nodes, and only their training labels contribute to the
training loss. It does not restrict sampling to a train-only subgraph.

## Model and TT path

The model uses two mean-aggregation GraphSAGE layers:

```text
602 features -> SAGEConv -> ReLU -> Dropout -> SAGEConv -> 41 logits
```

The stock CPU baseline uses PyTorch Geometric `SAGEConv`. The matched CPU
parity and TT runs select the scatter-free `SpMMGraphSAGEConv`, which preserves
the same weights and mean aggregation but expresses node-to-edge and
edge-to-node operations with matmuls. Both backends use the same `GraphSAGE`
class and `train.py` path. The XLA matmuls use bfloat16 inputs with float32
accumulation, while the CPU path keeps float32 inputs.

NeighborLoader normally produces different graph shapes for each step. The TT
configuration pads sampled graphs, seed labels, and masks to fixed capacities
so XLA can reuse one compiled graph. Padded edges are isolated on a reserved
sentinel node and cannot affect real-node outputs.

## Setup

```bash
source env/activate --xla
pip install -r blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/requirements.txt
```

## Run

### Full runs

```bash
# Stock PyG CPU accuracy baseline
CUDA_VISIBLE_DEVICES="" python3 blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/train.py

# CPU side of the matched SpMM/static-shape workload
CUDA_VISIBLE_DEVICES="" python3 blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/train.py \
  --config blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/single_chip/graphsage_reddit_spmm_cpu.yaml

# TT side of the same workload: single-chip Wormhole
python3 blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/train.py \
  --config blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/single_chip/graphsage_reddit_tt.yaml
```

The CPU parity and TT configurations intentionally differ only in device
selection and run metadata. Their model, sampling, optimizer hyperparameters,
static-shape, epoch, seed, and determinism settings are identical. Adam's
`capturable` mode is enabled automatically for TT execution and disabled on
CPU.

`CUDA_VISIBLE_DEVICES=""` makes the two CPU commands unambiguous on hosts that
also have a CUDA device. It is not needed on CPU-only hosts.

The TT configuration selects the backend, not a specific board model. The full
results in [RESULTS.md](RESULTS.md) were recorded on one Wormhole N150. The
maintainer CI smoke run used an N300 allocated by CI.

### Bounded smoke checks

Use the same CI smoke overlay on both sides of the parity pair. The CPU command
can be run without Tenstorrent hardware.

```bash
# CPU-only smoke check
CUDA_VISIBLE_DEVICES="" python3 blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/train.py \
  --config blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/single_chip/graphsage_reddit_spmm_cpu.yaml \
  --test-config tests/configs/BOUNTIES/tt-graphsage_reddit-reddit-n300.yaml

# TT/Wormhole smoke check (hardware is selected by the runner)
python3 blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/train.py \
  --config blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/single_chip/graphsage_reddit_tt.yaml \
  --test-config tests/configs/BOUNTIES/tt-graphsage_reddit-reddit-n300.yaml
```

The overlay runs one epoch and caps every loader iteration at two batches. A
successful run therefore performs at most two initial-validation batches, two
training batches, two post-epoch validation batches, and two test batches. It
also disables W&B and checkpoint writes. The cap does not shorten the initial
Reddit download or preprocessing.

These commands are wiring and execution smoke checks. Their losses, accuracy,
and timings are not convergence, correctness-parity, or performance evidence;
the first compiled step and tiny sample dominate the measurements.

### Bounded CPU/TT timing run

After both smoke checks pass, use the shared 100-step overlay for an
intermediate CPU/TT execution and model-step timing check:

```bash
# CPU bounded timing run
CUDA_VISIBLE_DEVICES="" python3 blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/train.py \
  --config blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/single_chip/graphsage_reddit_spmm_cpu.yaml \
  --test-config tests/configs/BOUNTIES/tt-graphsage_reddit-reddit-n300-benchmark.yaml

# TT/Wormhole bounded timing run (hardware is selected by the runner)
python3 blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/train.py \
  --config blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/single_chip/graphsage_reddit_tt.yaml \
  --test-config tests/configs/BOUNTIES/tt-graphsage_reddit-reddit-n300-benchmark.yaml
```

The overlay runs one epoch and caps every loader iteration at 100 batches. It
therefore permits at most 100 initial-validation, 100 training, 100 post-epoch
validation, and 100 test batches. Initial validation may compile the forward
path before training begins. The separate `compile_and_first_step_time_s`
metric covers the first training forward/backward/optimizer step, not all
compilation. The first reported timing window averages steps 2--10 (nine
steps); later windows contain ten steps. These model-step metrics exclude
neighbor sampling and batch preparation. Keep the raw metrics, but summarize
step-20 through step-100 windows for the warm CPU/TT comparison. This bounded
run is useful for catching execution failures and comparing model-step timing,
but it is not a correctness-parity or convergence result.

### Reproduce the full-run summary and curves

Keep each completed run's raw log, `*_train.csv`, and `*_val.csv` in a separate
artifact directory. The analysis requires `git-head.txt` and verifies that both
runs used the same commit. If `checkpoints/checkpoint_history.json` is present,
it uses the checkpoint history for full-precision final/best validation
accuracy. If `start.txt` and `end.txt` are present, it also reports the run wall
time. Without checkpoint history, validation accuracy is taken from the
four-decimal raw-log summaries.

The following Bash commands capture the required raw log and CSV files while
running from the repository root. Use clean checkpoint output directories for
a new result pair so an older checkpoint history cannot affect the reported
best validation accuracy.

```bash
set -euo pipefail
artifact_root=results/graphsage-reddit
cpu_checkpoints=blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/runs/matched_cpu/checkpoints
tt_checkpoints=blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/runs/tt/checkpoints

require_clean_tree() {
  if [ -n "$(git status --porcelain --untracked-files=normal)" ]; then
    echo "Commit or remove local source changes before collecting evidence." >&2
    git status --short >&2
    exit 1
  fi
}

require_clean_tree
if [ -e "$artifact_root" ] || [ -e "$cpu_checkpoints" ] || [ -e "$tt_checkpoints" ]; then
  echo "Remove or archive existing GraphSAGE artifacts and checkpoints first." >&2
  exit 1
fi
mkdir -p "$artifact_root"/{cpu,n150}/checkpoints "$artifact_root/meta"

# Finish the shared download and preprocessing before either timed run.
python3 - <<'PY'
from blacksmith.datasets.torch.BOUNTIES.reddit.reddit_dataset import RedditDataset
from blacksmith.experiments.torch.BOUNTIES.graphsage_reddit.configs import GraphSAGEConfig

RedditDataset(GraphSAGEConfig())
PY

cp blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/requirements.txt "$artifact_root/meta/"
cp blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/single_chip/graphsage_reddit_spmm_cpu.yaml \
  "$artifact_root/meta/"
cp blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/single_chip/graphsage_reddit_tt.yaml \
  "$artifact_root/meta/"
python3 --version > "$artifact_root/meta/python-version.txt" 2>&1
python3 -m pip freeze > "$artifact_root/meta/pip-freeze.txt"
git rev-parse HEAD > "$artifact_root/meta/git-head.txt"

date --iso-8601=seconds > "$artifact_root/cpu/start.txt"
set +e
CUDA_VISIBLE_DEVICES="" python3 blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/train.py \
  --config blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/single_chip/graphsage_reddit_spmm_cpu.yaml \
  --test-log-filename-prefix graphsage_cpu_full \
  2>&1 | tee "$artifact_root/cpu/graphsage-cpu-full.log"
cpu_pipeline_status=("${PIPESTATUS[@]}")
set -e
date --iso-8601=seconds > "$artifact_root/cpu/end.txt"
printf 'python_exit_code=%s\ntee_exit_code=%s\n' \
  "${cpu_pipeline_status[0]}" "${cpu_pipeline_status[1]}" \
  >> "$artifact_root/cpu/graphsage-cpu-full.log"
test "${cpu_pipeline_status[0]}" -eq 0 || exit "${cpu_pipeline_status[0]}"
test "${cpu_pipeline_status[1]}" -eq 0 || exit "${cpu_pipeline_status[1]}"
cp tests/test_logs/graphsage_cpu_full_{train,val}.csv "$artifact_root/cpu/"
cp -R "$cpu_checkpoints"/. "$artifact_root/cpu/checkpoints/"
git rev-parse HEAD > "$artifact_root/cpu/git-head.txt"

require_clean_tree
date --iso-8601=seconds > "$artifact_root/n150/start.txt"
set +e
python3 blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/train.py \
  --config blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/single_chip/graphsage_reddit_tt.yaml \
  --test-log-filename-prefix graphsage_n150_full \
  2>&1 | tee "$artifact_root/n150/graphsage-n150-full.log"
tt_pipeline_status=("${PIPESTATUS[@]}")
set -e
date --iso-8601=seconds > "$artifact_root/n150/end.txt"
printf 'python_exit_code=%s\ntee_exit_code=%s\n' \
  "${tt_pipeline_status[0]}" "${tt_pipeline_status[1]}" \
  >> "$artifact_root/n150/graphsage-n150-full.log"
test "${tt_pipeline_status[0]}" -eq 0 || exit "${tt_pipeline_status[0]}"
test "${tt_pipeline_status[1]}" -eq 0 || exit "${tt_pipeline_status[1]}"
cp tests/test_logs/graphsage_n150_full_{train,val}.csv "$artifact_root/n150/"
cp -R "$tt_checkpoints"/. "$artifact_root/n150/checkpoints/"
git rev-parse HEAD > "$artifact_root/n150/git-head.txt"
```

```bash
python3 -m pip install -r blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/requirements-analysis.txt
python3 blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/analyze_results.py \
  --cpu-dir results/graphsage-reddit/cpu \
  --tt-dir results/graphsage-reddit/n150 \
  --tt-label "Wormhole N150" \
  --output-dir results/graphsage-reddit/analysis
```

The command writes `summary.json` and `graphsage-cpu-tt-curves.png`. The summary
includes the logged train/evaluation interval, first-step timing, steady timing,
accuracy, and parity differences. It checks that CSV steps and values align
with the raw log before calculating parity. The steady timing values use the
same method as [RESULTS.md](RESULTS.md): timing windows at step 100 or later,
excluding epoch-end partial windows, followed by a 5% trim from each tail.

## Configuration

| Setting | CPU baseline | CPU parity | TT |
|---|---:|---:|---:|
| Hidden channels | 256 | 256 | 256 |
| Dropout | 0.5 | 0.0 | 0.0 |
| Learning rate | 0.001 | 0.001 | 0.001 |
| Weight decay | 0.0005 | 0.0005 | 0.0005 |
| Batch size | 512 | 32 | 32 |
| Validation batch size | 4096 | 32 | 32 |
| Neighbor fanouts | [25, 10] | [5, 3] | [5, 3] |
| Epochs | 30 | 5 | 5 |
| Convolution | stock `SAGEConv` | scatter-free SpMM | scatter-free SpMM |
| Static sampled-graph shapes | no | yes | yes |
| Seed / deterministic | 42 / yes | 42 / yes | 42 / yes |

Training logs include first-step compile time, steady model-step time, and seed
node throughput. These metrics use the synchronized optimizer step on TT. Use
the matched CPU parity configuration for CPU-versus-TT timing comparisons; the
larger stock CPU configuration remains the accuracy baseline.

The matched CPU/TT runs disable dropout so model execution does not advance a
different CPU-versus-XLA random stream between stochastic neighbor-sampling
steps. With the same software and dataset, the shared seed makes each run
repeatable and is intended to keep their sampled workloads aligned. Download
and process Reddit before collecting timings; dataset setup time is not part of
the benchmark.

### Checkpoints and resume

`save_strategy: epoch` saves every `epoch_freq` epochs; `step` saves every
`steps_freq` synchronized optimizer steps, outside the model-step timer. Both
also save a final snapshot; `none` disables checkpoint writes. Set
`save_optim: true` to include Adam's state in every snapshot, including the
final one; the default is `false`.

Resume with `--test-checkpoint-path PATH`, or set `resume_from_checkpoint: true`
and the YAML `resume_option` (`last`, `best`, or `path`, with `checkpoint_path`).
The saved epoch counts completed epochs. A mid-epoch checkpoint restores the
model, saved optimizer state, and global step, but restarts the unfinished
epoch from its beginning. Sampler/dropout RNG state and the loader cursor are
not saved, so resume is not exact continuation and may add training steps.

## Validation status

The matched CPU/Wormhole N150 full-run results from commit `f353f93`, including
accuracy parity, timing, and limitations, are recorded in [RESULTS.md](RESULTS.md).
Final test accuracy was 0.9247 on CPU and 0.9237 on N150, a gap of about 0.10
percentage points. These are historical measurements, not a rerun of later
commits.

The maintainer N300/PyG CI job at commit `6a045cb` passed its GraphSAGE hardware
smoke case and 23 focused tests. A full N300 cloud allocation was not available,
so the N150 full-run measurements are kept clearly separate from the N300 smoke
result.
