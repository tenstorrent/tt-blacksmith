# EasyDel Qwen3-0.6B LoRA Fine-Tuning

This directory contains the code for the Qwen3-0.6B LoRA fine-tuning experiment using JAX and EasyDel.

## Overview

The experiment implements LoRA (Low-Rank Adaptation) fine-tuning for the Qwen3-0.6B model on Tenstorrent hardware using JAX and EasyDel's native NNX LoRA support. It performs causal language modelling on the WikiText-2 dataset with gradient accumulation and optional periodic validation.

## Training

```bash
python3 blacksmith/experiments/easydel/qwen/test_qwen_fine_tuning_jax.py
```

Override configuration via CLI:

```bash
python3 blacksmith/experiments/easydel/qwen/test_qwen_fine_tuning_jax.py --config path/to/custom.yaml
```

## Data

The [WikiText-2](https://huggingface.co/datasets/wikitext) dataset (`wikitext-2-raw-v1`) is used for training and validation. The raw text is concatenated, tokenized, and chunked into fixed-length sequences of `max_length` tokens.

## Configuration

In `blacksmith/experiments/easydel/qwen/test_qwen_fine_tuning_jax.yaml` you can configure all training parameters. Alternatively, override individual fields via the CLI.

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
| `num_hidden_layers` | Number of transformer layers (None = use model default). | None |
| `max_position_embeddings` | Max position embeddings (None = use model default). | None |

### Training

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `learning_rate` | Learning rate for the AdamW optimizer. | 2e-4 |
| `batch_size` | Number of samples per training batch. | 4 |
| `gradient_accumulation_steps` | Number of mini-batches to accumulate before an optimizer step. | 1 |
| `num_epochs` | Total number of training epochs. | 1 |
| `val_steps_freq` | Run validation every N steps (0 = disabled). | 0 |

### LoRA

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `lora_rank` | Rank of the LoRA adaptation matrices. | 16 |
| `lora_pattern` | Regex pattern matching layers to apply LoRA to. | `".*(q_proj\|v_proj).*"` |

### Other

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `log_level` | Logging verbosity level. | `"INFO"` |
| `model_to_wandb` | Whether to log metrics to Weights & Biases. | False |
| `seed` | Random seed for reproducibility. | 42 |
| `use_tt` | Whether to run on Tenstorrent device. | True |
