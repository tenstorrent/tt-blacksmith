# GraphSAGE Node Classification on Reddit

This experiment trains a two-layer [GraphSAGE](https://arxiv.org/abs/1706.02216)
model for inductive node classification on the
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

## Model and TT path

The model uses two mean-aggregation GraphSAGE layers:

```text
602 features -> SAGEConv -> ReLU -> Dropout -> SAGEConv -> 41 logits
```

CPU runs use the stock PyTorch Geometric `SAGEConv`. TT runs select the
scatter-free `SpMMGraphSAGEConv`, which preserves the same weights and mean
aggregation but expresses node-to-edge and edge-to-node operations with
matmuls. Both backends use the same `GraphSAGE` class and `train.py` path.

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

```bash
# CPU baseline
python3 blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/train.py

# CPU SpMM run matched to the TT workload
python3 blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/train.py \
  --config blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/single_chip/graphsage_reddit_spmm_cpu.yaml

# Wormhole N300, single chip
python3 blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/train.py \
  --config blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/single_chip/graphsage_reddit_tt.yaml

# Two-batches-per-split N300 smoke run
python3 blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/train.py \
  --config blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/single_chip/graphsage_reddit_tt.yaml \
  --test-config tests/configs/BOUNTIES/tt-graphsage_reddit-reddit-n300.yaml
```

## Configuration

| Setting | CPU baseline | CPU parity | TT |
|---|---:|---:|---:|
| Hidden channels | 256 | 256 | 256 |
| Dropout | 0.5 | 0.0 | 0.0 |
| Learning rate | 0.001 | 0.001 | 0.001 |
| Batch size | 512 | 32 | 32 |
| Neighbor fanouts | [25, 10] | [5, 3] | [5, 3] |
| Convolution | stock `SAGEConv` | scatter-free SpMM | scatter-free SpMM |
| Static sampled-graph shapes | no | yes | yes |

Training logs include first-step compile time, steady model-step time, and seed
node throughput. These metrics use the synchronized optimizer step on TT. Use
the matched CPU parity configuration for CPU-versus-TT timing comparisons; the
larger stock CPU configuration remains the accuracy baseline.

The matched CPU/TT runs disable dropout so model execution does not advance a
different CPU-versus-XLA random stream between stochastic neighbor-sampling
steps. The shared seed then produces the same sampled workload in both runs.
Download and process Reddit before collecting timings; dataset setup time is not
part of the benchmark.

## Validation status

The CPU figures from the closed draft [PR #570](https://github.com/tenstorrent/tt-blacksmith/pull/570)
are intentionally not copied as current results because that branch used an
older stack and reported inconsistent final metrics. Fresh CPU and N300 runs,
including correctness parity and step-time measurements, must be recorded from
this branch before the workload is submitted.
