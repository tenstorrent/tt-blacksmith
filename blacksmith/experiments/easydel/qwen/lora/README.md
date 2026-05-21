# EasyDel Qwen LoRA Fine-Tuning

This directory contains [LoRA](https://arxiv.org/abs/2106.09685) fine-tuning experiments for Qwen models on Tenstorrent hardware using JAX and [EasyDel](https://github.com/erfanzar/EasyDeL).

- Qwen 3 0.6B model specification can be found [here](https://huggingface.co/Qwen/Qwen3-0.6B).
- Qwen 3 4B model specification can be found [here](https://huggingface.co/Qwen/Qwen3-4B).

## Overview

The training script (`train.py`) implements LoRA fine-tuning with EasyDel's native NNX LoRA support, formatted as instruction-style causal language modelling. Two datasets are wired up via `dataset_id`: **SST-2** (single-label sentiment, JSON response) and **Alpaca** (general instruction following). Prompt tokens are masked (`-100`) so the loss is computed only on the response tokens.

To keep the program signature constant on TT (necessary to avoid `MeshDevice` re-open crashes — see `tt-xla#1993` / `tt-xla#4809`), the train step fuses label shift / masking / one-hot / clamped cross-entropy into a single `@jax.jit`, and the eval step emits `(loss, predictions)` from a single fixed-signature JIT (no per-step micro-ops escape to TT).

YAML configs live under per-device directories:

- **`single_chip/`** — single-device runs (Qwen3-0.6B).
- **`quietbox/`** — 8-chip multi-device runs (Qwen3-4B), tensor-parallel `[8]/model` or hybrid data+tensor-parallel `[2, 4]/data,model`.
- **`loudbox/`**, **`galaxy/`** — reserved for future larger meshes.

Use **`use_tt`** in the config to select Tenstorrent (`true`) or GPU/CPU (`false`). All shipped configs are TT-oriented with **`use_tt: true`**.

## Prerequisites

Follow the environment setup in the top-level TT-Blacksmith documentation:

```bash
cd /path/to/tt-blacksmith

# For Tenstorrent hardware:
source env/activate --xla

# For GPU baseline:
source env/activate --gpu
```

EasyDel is pulled in automatically by `env/xla_requirements.txt`, pinned to the validated commit.

For **GPU baseline** runs, also install EasyDel manually (the GPU env does not pin it) plus the JAX CUDA plugin (`--no-deps` avoids a cuDNN version conflict with torch):

```bash
pip install git+https://github.com/erfanzar/EasyDeL.git@77ced9d2f2ab6a3d705936d26112eb97d9f9e64a
pip install --no-deps jax-cuda12-plugin==0.7.1 jax-cuda12-pjrt==0.7.1
```

## Training

### Qwen 3 0.6B (Single Chip)

**SST-2:**

```bash
python3 blacksmith/experiments/easydel/qwen/lora/train.py \
  --config blacksmith/experiments/easydel/qwen/lora/single_chip/qwen3_0_6b_sst2.yaml
```

GPU baseline (override `use_tt`; requires GPU JAX and the CUDA plugin above):

```bash
python3 blacksmith/experiments/easydel/qwen/lora/train.py \
  --config blacksmith/experiments/easydel/qwen/lora/single_chip/qwen3_0_6b_sst2.yaml \
  --test_config '{"use_tt": false}'
```

### Qwen 3 4B (Quietbox, 8 chips)

**Alpaca** (hybrid DP+TP, `mesh [2, 4]`, `seq_len 128`):

```bash
python3 blacksmith/experiments/easydel/qwen/lora/train.py \
  --config blacksmith/experiments/easydel/qwen/lora/quietbox/qwen3_4b_alpaca.yaml
```

**SST-2** (pure TP, `mesh [8]`, `seq_len 64`):

```bash
python3 blacksmith/experiments/easydel/qwen/lora/train.py \
  --config blacksmith/experiments/easydel/qwen/lora/quietbox/qwen3_4b_sst2.yaml
```

### Training Configurations

| Architecture | mesh_shape | mesh_axis_names | input_sharding_dim | dataset | Method |
| ------------ | ---------- | --------------- | ------------------ | ------- | ------ |
| [Single-Chip](single_chip/qwen3_0_6b_sst2.yaml) | None | None | None | SST-2 | LoRA |
| [Quietbox 8-chip](quietbox/qwen3_4b_sst2.yaml) | `[8]` | `["model"]` | None | SST-2 | LoRA (TP) |
| [Quietbox 8-chip](quietbox/qwen3_4b_alpaca.yaml) | `[2, 4]` | `["data", "model"]` | `"data"` | Alpaca | LoRA (DP+TP) |

### Mesh and Sharding Configuration

Multi-chip configs declare the parallelism strategy via `mesh_shape`, `mesh_axis_names`, and `input_sharding_dim` fields in the YAML. Parameter shardings come from two sources stitched together:

- `easydel_partition_specs_for_lora` in `blacksmith/tools/jax/easydel/partitioning.py` derives the default shardings from EasyDel's built-in `partition_rules()`.
- `model_sharding_patterns` (a list of `[regex, [axis_or_null, ...]]` entries) lets the YAML override specific parameters; the shipped Qwen3-4B configs use Megatron-style column-/row-parallel patterns on `q/k/v/o_proj`, `gate/up/down_proj`, and `lm_head` along the `model` axis.

Multi-device runs use the Shardy partitioner by default (`use_shardy_partitioner: true`) and enable the GQA workaround (`apply_gqa_workaround: true`).

## Data

**SST-2** (GLUE): instruction-style prompt/response pairs padded to `max_length`, with masked labels. The Hugging Face load uses `glue` / `sst2`; `dataset_id` in the config is the logical dataset tag. The SST-2 pipeline uses the Torch `SSTDataset` loader from `blacksmith/datasets/torch/sst2/` and formats each example as `Review: <sentence>\nOutput: {"label": "positive|negative"}`. Prompt tokens are masked with `-100` so only the response tokens contribute to the loss.

**Alpaca** (`tatsu-lab/alpaca`): general instruction-following pairs (`instruction`, optional `input`, `output`) padded to `max_length`, with masked labels. The JAX loader (`blacksmith/datasets/jax/alpaca/alpaca_dataset.py`) wraps the Torch `AlpacaDataset` from `blacksmith/datasets/torch/alpaca/` and produces `(input_ids, labels, attention_masks)` shaped `(num_batches, batch_size, seq_len)`. Prompt tokens are masked with `-100` so only the response tokens contribute to the loss.

`dataset_id` in the YAML selects between them via the `AVAILABLE_DATASETS` registry in `blacksmith/datasets/jax/dataset_utils.py` (currently `"sst2"` and `"alpaca"`).

## Configuration

Each YAML specifies training parameters. Override fields via `--test_config` JSON as needed.

### Dataset

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `dataset_id` | Dataset identifier; one of `"sst2"`, `"alpaca"`. | `"sst2"` |

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

### Checkpoint

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `resume_from_checkpoint` | Whether to resume training from a previous checkpoint. | False |
| `resume_option` | Resume method (`last`, `best`, or `path`). | `"last"` |
| `checkpoint_path` | Path to a checkpoint if `resume_option="path"`. | `""` |
| `checkpoint_metric` | Metric used to determine best checkpoint. | `"val/loss"` |
| `checkpoint_metric_mode` | Whether to minimize or maximize checkpoint metric. | `"min"` |
| `keep_last_n` | Number of most recent checkpoints to keep. | 3 |
| `keep_best_n` | Number of best checkpoints to keep. | 3 |
| `save_strategy` | Strategy for saving checkpoints (`none`, `epoch`, or `step`). | `"none"` |
| `project_dir` | Directory for experiment outputs. | `"blacksmith/experiments/easydel/qwen/lora"` |
| `save_optim` | Whether to save optimizer state. | False |

### Device

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `use_tt` | Whether to run on Tenstorrent device. | True |
| `num_devices` | Number of TT (or GPU) devices in the JAX mesh. | 1 |
| `mesh_shape` | Mesh shape for distributed training (None = single device). | None |
| `mesh_axis_names` | Axis names for the mesh (None = single device). | None |
| `input_sharding_dim` | Mesh axis for data-parallel sharding (None = no DP). | None |
| `apply_gqa_workaround` | Apply EasyDel GQA workaround required on TT multi-chip. | True |
| `use_shardy_partitioner` | Use the Shardy partitioner (False falls back to GSPMD). | True |
| `model_sharding_patterns` | List of `[regex, [axis_or_null, ...]]` overrides on top of EasyDel's `partition_rules()`. None = replicated (pure DP). | None |
| `extra_config_kwargs` | Extra kwargs forwarded to the EasyDel model config. | None |

### Logging

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `steps_freq` | Log average loss every N steps. | 10 |
| `log_level` | Logging verbosity level. | `"INFO"` |
| `use_wandb` | Whether to log metrics to Weights & Biases. | True |
| `wandb_project` | Weights & Biases project name. | `"Qwen-TT-EasyDel-LoRA-Training"` |
| `wandb_run_name` | Weights & Biases run name. | `"qwen3-0.6b-sst2-tt-easydel"` |
| `print_examples` | Print decoded prediction examples during evaluation. | False |

### Reproducibility

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `seed` | Random seed for reproducibility. | 42 |
| `deterministic` | Whether to enforce deterministic behavior. | False |
| `framework` | Training framework. | `"easydel"` |
