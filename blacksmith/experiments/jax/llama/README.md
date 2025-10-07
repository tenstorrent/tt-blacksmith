# JAX LoRA Training for Llama 3.2-1B

JAX-based LoRA (Low-Rank Adaptation) fine-tuning for the Llama 3.2-1B model on TT (Tenstorrent) devices.

## Overview

This directory includes:
- TT device training (`test_llama_fine_tuning_jax.py`)
- Custom LoRAx implementation in `lorax/`
- SST-2 example task and basic wandb integration

### Prerequisites
- Follow the environment setup in the top-level tt-blacksmith documentation.
- Install Lorax dependencies (pinned versions):

```bash
pip install git+https://github.com/patrick-kidger/quax.git@8c50184a7e60835799cc5f79c9de9315ca77c875 --no-deps
pip install equinox==0.13.1 --no-deps
pip install plum-dispatch==2.5.7 beartype==0.21.0 rich==14.1.0
export PYTHONPATH=/localdev/upantelic/tt-blacksmith:$PYTHONPATH
```

## Usage

### TT Device Training

Run LoRA training on Tenstorrent device:

```bash
python3 blacksmith/experiments/jax/llama/test_llama_fine_tuning_jax.py
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
| `num_epochs` | `5` | Number of training epochs |
| `lora_rank` | `4` | LoRA adaptation rank |
| `num_hidden_layers` | `16` | Number of transformer layers |

### LoRA Target Modules

The implementation applies LoRA adaptation to MLP layers only:
- `mlp.gate_proj.kernel`
- `mlp.up_proj.kernel`
- `mlp.down_proj.kernel`
