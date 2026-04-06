# EasyDel Qwen LoRA Fine-Tuning

This directory contains [LoRA](https://arxiv.org/abs/2106.09685) fine-tuning experiments for Qwen models on Tenstorrent hardware using JAX and [EasyDel](https://github.com/erfanzar/EasyDeL).

## Overview

The shared training script (`test_qwen_fine_tuning_easydel.py`) implements LoRA fine-tuning with EasyDel's native NNX LoRA support. It performs causal language modelling with gradient accumulation and optional periodic validation. Per-topology YAML configs live in subdirectories:

- **`single_chip/`** — Configs for single-device TT runs (e.g. Qwen3-0.6B on N150)
- **`gpu/`** — Configs for GPU baseline runs

The `use_tt` flag in each YAML config controls whether the experiment targets Tenstorrent hardware (`true`) or GPU (`false`).

## Prerequisites

Follow the environment setup in the top-level TT-Blacksmith documentation:

```bash
cd /path/to/tt-blacksmith

# For Tenstorrent hardware:
source env/activate --xla

# For GPU baseline:
source env/activate --gpu
```

Then install the additional EasyDel-specific dependencies:

```bash
pip install -r blacksmith/experiments/easydel/requirements.txt
```

## Training

On Tenstorrent hardware:

```bash
python3 blacksmith/experiments/easydel/qwen/test_qwen_fine_tuning_easydel.py \
  --config blacksmith/experiments/easydel/qwen/single_chip/test_qwen3_0.6b_lora.yaml
```

GPU baseline:

```bash
python3 blacksmith/experiments/easydel/qwen/test_qwen_fine_tuning_easydel.py \
  --config blacksmith/experiments/easydel/qwen/gpu/test_qwen3_0.6b_lora.yaml
```

## Data

The [WikiText-2](https://huggingface.co/datasets/wikitext) dataset (`wikitext-2-raw-v1`) is used for training and validation. The raw text is concatenated, tokenized, and chunked into fixed-length sequences of `max_length` tokens.

## Configuration

Each YAML config in the subdirectories specifies all training parameters. Alternatively, override individual fields via the CLI.

### Dataset

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `dataset_id` | The training dataset id. | `"wikitext"` |
| `dataset_configuration` | Dataset configuration/subset name. | `"wikitext-2-raw-v1"` |

### Model

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `model_name` | HuggingFace model identifier. | `"Qwen/Qwen3-0.6B"` |
| `max_length` | Maximum sequence length for tokenization. | 128 |
| `dtype` | Data type used for model parameters. | `"jnp.bfloat16"` |
| `max_position_embeddings` | Max position embeddings (None = use model default). | None |

### Training

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `learning_rate` | Learning rate for the AdamW optimizer. | 2e-4 |
| `batch_size` | Number of samples per training batch. | 4 |
| `gradient_accumulation_steps` | Number of mini-batches to accumulate before an optimizer step. | 1 |
| `num_epochs` | Total number of training epochs. | 1 |
| `val_steps_freq` | Run validation every N steps (null = disabled). | null |
| `max_val_batches` | Limit number of validation batches per eval pass (null = use all). | null |

### LoRA

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `lora_rank` | Rank of the LoRA adaptation matrices. | 16 |
| `lora_pattern` | Regex pattern matching layers to apply LoRA to. | `".*(q_proj\|v_proj).*"` |

### Other

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `steps_freq` | Log average loss every N steps. | 10 |
| `log_level` | Logging verbosity level. | `"INFO"` |
| `use_wandb` | Whether to log metrics to Weights & Biases. | True |
| `wandb_project` | Weights & Biases project name. | `"Qwen-TT-EasyDel-LoRA-Training"` |
| `wandb_run_name` | Weights & Biases run name. | `"qwen3-0.6b-wikitext-tt-easydel"` |
| `seed` | Random seed for reproducibility. | 42 |
| `use_tt` | Whether to run on Tenstorrent device. | True |
