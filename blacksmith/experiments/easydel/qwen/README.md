# EasyDel Qwen LoRA Fine-Tuning

This directory contains [LoRA](https://arxiv.org/abs/2106.09685) fine-tuning experiments for Qwen models on Tenstorrent hardware using JAX and [EasyDel](https://github.com/erfanzar/EasyDeL).

## Overview

The shared training script (`test_qwen_fine_tuning_easydel.py`) implements LoRA fine-tuning with EasyDel's native NNX LoRA support on the SST-2 sentiment classification dataset, formatted as instruction-style causal language modelling.

Prompt tokens are masked (`-100`) so the loss is computed only on the response tokens (JSON label).

YAML configs live under **`single_chip/`** (and **`multi_chip/`** when present) for different device counts. Use **`use_tt`** in the config to select Tenstorrent (`true`) or GPU/CPU (`false`). The default config is TT-oriented with **`use_tt: true`**.

## Module layout

| File | Responsibility |
|------|---------------|
| `configs.py` | Pydantic `TrainingConfig` with all hyperparameters. |
| `data_loading.py` | SST-2 data loading, tokenization, batching. |
| `train_steps.py` | JIT-compiled train/eval steps, CPU f32 loss helpers, evaluation loop, prediction display. |
| `test_qwen_fine_tuning_easydel.py` | Thin orchestrator: CLI, model load, LoRA, optimizer, training loop. Uses `TrainingLogger` for stdout + W&B. |
| `blacksmith/tools/workaround_utils_jax.py` | GQA workaround for TT devices (shared). |

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

That file pins **Triton 3.2.x** to match the pinned EasyDeL revision (`triton~=3.2.0` in EasyDeL’s metadata). Using Triton **3.3+** with the current pin will make `pip` report a dependency conflict.

For **GPU baseline** runs, also install the JAX CUDA plugin (`--no-deps` avoids a cuDNN version conflict with torch):

```bash
pip install --no-deps jax-cuda12-plugin==0.7.1 jax-cuda12-pjrt==0.7.1
```

## Training

Default (Tenstorrent, `use_tt: true` in the YAML):

```bash
python3 blacksmith/experiments/easydel/qwen/test_qwen_fine_tuning_easydel.py \
  --config blacksmith/experiments/easydel/qwen/single_chip/test_qwen3_0.6b_lora.yaml
```

GPU baseline (override `use_tt`; requires GPU JAX and the CUDA plugin above):

```bash
python3 blacksmith/experiments/easydel/qwen/test_qwen_fine_tuning_easydel.py \
  --config blacksmith/experiments/easydel/qwen/single_chip/test_qwen3_0.6b_lora.yaml \
  --test_config '{"use_tt": false}'
```

When **`multi_chip/`** configs exist, pass the appropriate YAML path (they set `num_devices` > 1).

The SST-2 pipeline uses the Torch `SSTDataset` loader from `blacksmith/datasets/torch/sst2/` and formats each example as `Review: <sentence>\nOutput: {"label": "positive|negative"}`.  Prompt tokens are masked with `-100` so only the response tokens contribute to the loss.

## Data

**SST-2** (GLUE): instruction-style prompt/response pairs padded to `max_length`, with masked labels. The Hugging Face load uses `glue` / `sst2`; `dataset_id` in the config is the logical dataset tag (e.g. `sst2`), aligned with other LoRA experiment YAMLs.

## Configuration

Each YAML specifies training parameters. Override fields via `--test_config` JSON as needed.

### Dataset

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `dataset_id` | Dataset identifier (SST-2 tag; matches other LoRA configs). | `"sst2"` |

### Model

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `model_name` | HuggingFace model identifier. | `"Qwen/Qwen3-0.6B"` |
| `max_length` | Maximum sequence length for tokenization. | 128 |
| `dtype` | Data type used for model parameters. | `"bfloat16"` |
| `mask_max_position_embeddings` | Cap for pre-allocated causal mask size (None = model default). | None |

### Training

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `learning_rate` | Peak learning rate for the AdamW optimizer. | 2e-4 |
| `warmup_steps` | Linear warm-up steps before the cosine decay begins. | 0 |
| `end_learning_rate` | Final learning rate after the cosine decay. | 0.0 |
| `batch_size` | Number of samples per training batch. | 4 |
| `gradient_accumulation_steps` | Number of mini-batches to accumulate before an optimizer step. | 1 |
| `num_epochs` | Total number of training epochs. | 1 |
| `val_steps_freq` | Run validation every N steps (null = disabled). | null |
| `max_val_batches` | Limit number of validation batches per eval pass (null = use all). | null |
| `ignored_label_index` | Sentinel value for masked label positions. | `-100` |

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
| `use_wandb` | Whether to log metrics to Weights & Biases. Can also be globally toggled with `wandb disabled` / `wandb enabled` (or `WANDB_MODE=disabled`). | True |
| `wandb_project` | Weights & Biases project name. | `"Qwen-TT-EasyDel-LoRA-Training"` |
| `wandb_run_name` | Weights & Biases run name. | `"qwen3-0.6b-sst2-tt-easydel"` |
| `print_examples` | Print a few decoded training examples at the start of a run. | False |
| `seed` | Random seed for reproducibility. | 42 |
| `use_tt` | Whether to run on Tenstorrent device. | True |
| `num_devices` | Number of TT (or GPU) devices in the JAX mesh. | 1 |
