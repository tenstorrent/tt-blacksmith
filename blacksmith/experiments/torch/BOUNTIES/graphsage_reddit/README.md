# GraphSAGE on Reddit — CPU Baseline (Bounty #529, PR-1)

GraphSAGE node classification on the Reddit dataset.

## Dataset

| Property | Value |
|---|---|
| Nodes | 232,965 |
| Edges | 114,615,892 |
| Node features | 602 |
| Classes | 41 |
| Train / Val / Test | 153,431 / 23,831 / 55,703 |

Too large for full-graph training — mini-batch via `NeighborLoader` with `[25, 10]` neighbours per hop.

## Model

2-layer GraphSAGE with mean aggregation:

```
Input (602) → SAGEConv → ReLU → Dropout(0.5) → SAGEConv → 41 classes (raw logits, F.cross_entropy loss)
```

**Parameters:** ~330K
**Optimizer:** Adam, lr=0.001, weight_decay=5e-4

## Setup

```bash
source env/activate --xla
pip install -r blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/requirements.txt
```

## Run

```bash
# CPU baseline
python blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/train.py

# TT single chip
python blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/train.py \
    --config blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/tt_single_chip/graphsage_reddit_tt.yaml
```

## Results (CPU Baseline)

| Metric | Value |
|---|---|
| Best val accuracy | 96.05% (epoch 25) |
| Final test accuracy | 95.79% |
| Epochs trained | 30 |
| Training time | ~2 h (CPU) |
| Per-epoch time | ~4 min (300 batches @ batch_size=512) |
| Throughput | ~3.5K seed-nodes/s (batch_size=1024, inference) |

The model converges by epoch 4–5 and plateaus at 95.7–96.1% val accuracy through epoch 30 with no signs of overfitting (val loss tracks train loss closely, test accuracy matches val accuracy).

## Results (TT)

<!-- TODO(sarthakmangla1): Add TT results after PR-2. See https://github.com/tenstorrent/tt-blacksmith/issues/529 -->
