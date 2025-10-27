# JAX DoRA Training for Llama 3.2-1B

JAX-based DoRA (Low-Rank Adaptation) fine-tuning for the Llama 3.2-1B model on TT (Tenstorrent) devices.

## Overview

This directory includes:
- TT device training (`test_llama_fine_tuning_jax.py`)
- Custom LoRAx implementation in `lorax/`
- SST-2 example task and basic wandb integration

### Prerequisites
- Follow the environment setup in the top-level TT-Blacksmith documentation.
- Install Lorax dependencies (pinned versions):

```bash
pip install git+https://github.com/patrick-kidger/quax.git@8c50184a7e60835799cc5f79c9de9315ca77c875 --no-deps
pip install git+https://github.com/patrick-kidger/equinox.git@367124071570194b5d90692b2e09caa834b89ab9 --no-deps
pip install plum-dispatch==2.5.7 beartype==0.21.0 rich==14.1.0
```

### TT Device Training

Run DoRA training on Tenstorrent device:

```bash
python3 blacksmith/experiments/jax/llama_dora/test_llama_fine_tuning_jax.py
```

## Configuration Options

This script supports the following configurable parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `"Erland/Llama-3.2-1B-JAX"` | HuggingFace model identifier |
| `dataset_id` | `"stanfordnlp/sst2"` | Dataset for fine-tuning |
| `max_length` | `128` | Maximum sequence length |
| `learning_rate` | `1e-4` | Learning rate for optimizer |
| `batch_size` | `4` | Training batch size |
| `num_epochs` | `1` | Number of training epochs |
| `dora_rank` | `4` | DoRA adaptation rank |

### DoRA Target Modules

The implementation applies DoRA adaptation to MLP layers only:
- `mlp.up_proj.kernel`
- `mlp.down_proj.kernel`
