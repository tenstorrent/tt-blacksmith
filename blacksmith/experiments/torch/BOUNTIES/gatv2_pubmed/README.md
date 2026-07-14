# GATv2 Node Classification on PubMed

This directory contains the code for GATv2 (Graph Attention Network v2) node classification training on the PubMed citation network dataset.
GATv2 paper can be found [here](https://arxiv.org/abs/2105.14491).

## Overview

GATv2 improves upon the original GAT by using a modified attention mechanism that is strictly more expressive — it can compute dynamic attention over any pair of nodes, whereas GAT's attention is effectively static. This experiment trains a 2-layer GATv2 model for semi-supervised node classification on the PubMed citation network.

### Model Architecture

| Layer | Input | Output | Heads | Concat |
|-------|-------|--------|-------|--------|
| GATv2Conv 1 | 500 (features) | 8 per head | 8 | Yes → 64 |
| GATv2Conv 2 | 64 | 3 (classes) | 1 | No |

- Dropout (p=0.6) applied before each convolution
- ELU activation between layers
- Log-softmax output for NLLLoss

### Dataset

PubMed is a citation network dataset with:
- **19,717** nodes (scientific publications)
- **88,648** edges (citations)
- **500** features per node (TF-IDF weighted word vectors)
- **3** classes (Diabetes Mellitus Experimental, Diabetes Mellitus Type 1, Diabetes Mellitus Type 2)
- Train/Val/Test split: 60/500/1000 nodes

Source: [Planetoid (PyTorch Geometric)](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html)

## Setup

```bash
# Activate environment
source env/activate --xla

# Install the experiment-specific dependency (PyTorch Geometric)
pip install -r blacksmith/experiments/torch/BOUNTIES/gatv2_pubmed/requirements.txt
```

## Running

### CPU Baseline

```bash
python3 blacksmith/experiments/torch/BOUNTIES/gatv2_pubmed/train.py
```

### TT (scatter-free SpMM)

```bash
python3 blacksmith/experiments/torch/BOUNTIES/gatv2_pubmed/train.py \
  --config blacksmith/experiments/torch/BOUNTIES/gatv2_pubmed/single_chip/gatv2_pubmed_tt.yaml
```

A golden-loss regression case (`tt-gatv2_pubmed-pubmed-n150`, config under
`tests/configs/BOUNTIES/`) is defined in `tests/training_test_cases.py` but kept
commented out until the experiment dependency and a TT runner are wired into CI.

## Configuration

| Architecture       | mesh_shape                   | mesh_axis_names      | input_sharding_dim | dataset      | Method                      |
| ------------------ | ---------------------------- | -------------------- | ------------------ | ------------ | --------------------------- |
| [Single-Chip](single_chip/gatv2_pubmed_tt.yaml) | None        | None        | None         | PubMed      | Full Model |

## Configuration Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `model_name` | Model identifier. | "GATv2" |
| `dataset_id` | Planetoid dataset identifier. | "pubmed" |
| `in_channels` | Number of input features per node. | 500 |
| `hidden_channels` | Hidden dimension per attention head. | 8 |
| `out_channels` | Number of output classes. | 3 |
| `heads` | Number of attention heads in first layer. | 8 |
| `dropout` | Dropout probability. | 0.6 |
| `learning_rate` | Learning rate for Adam optimizer. | 0.005 |
| `weight_decay` | L2 regularization weight. | 5e-4 |
| `num_epochs` | Maximum number of training epochs. | 300 |
| `patience` | Early stopping patience (epochs without improvement). | 50 |
| `val_freq` | Validate every N epochs. | 1 |
| `seed` | Random seed for reproducibility. | 42 |
| `deterministic` | Enforce deterministic operations. | True |
| `use_tt` | Whether to run on TT device. | False |
| `use_spmm` | Use the SpMM (matmul) GATv2 conv so the model trains natively on TT (bypasses the scatter tile-padding OOM, tt-mlir#8887). | False |
| `use_wandb` | Enable Weights & Biases logging. | False |
| `checkpoint_metric` | Metric for best checkpoint selection. | "val/accuracy" |
| `checkpoint_metric_mode` | Mode for checkpoint metric. | "max" |
| `epoch_freq` | Frequency for periodic checkpointing (in epochs). | 50 |
| `save_strategy` | Checkpoint save strategy. | "epoch" |

## Expected Results

### CPU Baseline

| Metric | Expected Range |
|--------|---------------|
| Best Val Accuracy | ~79-81% |
| Test Accuracy | ~77-79% |
| Convergence | ~100-200 epochs |

The same seed (42) is used for reproducibility.

A CPU↔TT parity comparison — CPU stock `GATv2Conv` vs TT `SpMMGATv2Conv` (both
seed 42, same stack) — is summarized in **[RESULTS.md](RESULTS.md)**: the CPU
baseline reaches 0.780 test accuracy and TT lands at 0.784, within noise (the
rewrite is bit-equivalent per step). The comparison plots are attached to the
pull request description.

## TT Execution Status

### The scatter blocker (and why a workaround is needed)

Standard `GATv2Conv` aggregates messages with `scatter_add` / `scatter_reduce_`.
The original blocker — a single logical `scatter` over a length-`L` index lowering
to a **serial chain of ~`L/256` `ttnn.scatter` ops** — was filed as
[tt-mlir#8714](https://github.com/tenstorrent/tt-mlir/issues/8714) and **fixed**
by [tt-mlir#8718](https://github.com/tenstorrent/tt-mlir/pull/8718) (merged
2026-06-15). However, after that fix the `scatter` operand's reshape to 1-D
emits a TILE-layout tensor whose degenerate height (1 padded to 32) causes a
**32× DRAM blow-up that still OOMs full-PubMed GNN training**. That residual
issue is open as
[tt-mlir#8887](https://github.com/tenstorrent/tt-mlir/issues/8887), so the
scatter path (`use_spmm: false`) still cannot train on TT today.

### The SpMM solution (`spmm_gatv2.py`, enabled by `use_spmm: true`)

`SpMMGATv2Conv` keeps the **exact** GATv2 math but rewrites every node↔edge
operation (the `x_i`/`x_j` feature lookups, the attention-softmax denominator,
and the message aggregation) as a **matmul against a one-hot incidence**. This
lowers to `ttnn.matmul` and emits **zero `ttnn.scatter`**, fully bypassing the
tiling/padding issues. It is bit-equivalent to `GATv2Conv` on CPU (loss
identical, per-parameter gradient cosine 1.0) and trains full PubMed on a
Wormhole N300 to ~78% test accuracy, matching the CPU baseline. The mandatory
TTIR proof graph (`module @SyncTensorsGraph`, zero scatter) is emitted under
`TTXLA_LOGGER_LEVEL=DEBUG`.

Making it run on the current stack required five small, documented techniques:
SpMM aggregation; storing `att` flat as `[1, H*C]`; a constant block-ones matmul
for the per-head reduction; a blocked bf16 one-hot (memory); and a static masked
loss (`masked_nll_loss`) in place of boolean-mask indexing. Each sidesteps a
distinct tt-mlir / ttnn limitation; see the header docstring in
[`spmm_gatv2.py`](../../../../models/torch/gatv2_pubmed/spmm_gatv2.py) for the
per-technique rationale.

Once tt-mlir#8887 is resolved, standard `GATv2Conv` (`use_spmm: false`) is
expected to run natively on TT as well; the SpMM path works today and does not
depend on it.

## Output Artifacts

After training, the following artifacts are generated in `results/`:

- `loss_curves.png` — Training and validation loss over epochs
- `accuracy_curve.png` — Validation accuracy over epochs
- `results_summary.json` — Final metrics and hyperparameters

Checkpoints are saved in `checkpoints/`:
- `best_model.pt` — Best model by validation accuracy
