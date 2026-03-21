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

# Install PyTorch Geometric
pip install torch_geometric
```

## Running

```bash
# CPU baseline (default config)
PYTHONPATH=. python3 blacksmith/experiments/torch/BOUNTIES/gatv2_pubmed/test_gatv2_pubmed_training.py

# With custom config
PYTHONPATH=. python3 blacksmith/experiments/torch/BOUNTIES/gatv2_pubmed/test_gatv2_pubmed_training.py --config path/to/config.yaml
```

## Configuration Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `model_name` | Model identifier. | "GATv2" |
| `dataset_name` | Planetoid dataset name. | "PubMed" |
| `dataset_root` | Directory for dataset download. | "./data" |
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
| `use_wandb` | Enable Weights & Biases logging. | False |
| `checkpoint_metric` | Metric for best checkpoint selection. | "val/accuracy" |
| `checkpoint_metric_mode` | Mode for checkpoint metric. | "max" |
| `epoch_freq` | Frequency for periodic checkpointing (in epochs). | 50 |
| `save_strategy` | Checkpoint save strategy. | "epoch" |

## Expected Results (CPU Baseline)

| Metric | Expected Range |
|--------|---------------|
| Best Val Accuracy | ~79-81% |
| Test Accuracy | ~77-79% |
| Convergence | ~100-200 epochs |

## Output Artifacts

After training, the following artifacts are generated in `results/`:

- `loss_curves.png` — Training and validation loss over epochs
- `accuracy_curve.png` — Validation accuracy over epochs
- `results_summary.json` — Final metrics and hyperparameters

Checkpoints are saved in `checkpoints/`:
- `best_model.pt` — Best model by validation accuracy
