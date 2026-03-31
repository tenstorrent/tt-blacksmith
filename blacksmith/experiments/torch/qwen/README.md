# Qwen with LoRA Experiment

This directory contains the code for the Qwen model with LoRA fine-tuning experiment in TT-XLA.

- Qwen 2.5 0.5B model specification can be found [here](https://huggingface.co/Qwen/Qwen2.5-0.5B).
- Qwen 2.5 1.5B model specification can be found [here](https://huggingface.co/Qwen/Qwen2.5-1.5B).
- Qwen 3 4B Instruct 2507 model specification can be found [here](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).
- Qwen 3 8B Base model specification can be found [here](https://huggingface.co/Qwen/Qwen3-8B-Base).

Original LoRA paper can be found [here](https://arxiv.org/pdf/2106.09685).

## Overview

The Qwen fine-tuning experiment applies the LoRA technique to adapt a pre-trained Qwen model on the SST sentiment analysis or Text-to-SQL datasets.
The experiment is designed to run on the Huggingface framework.

## Training

The experiment supports different hardware configurations with per-model training setups.

### Mesh and Sharding Configuration

Mesh configurations define the parallelism strategy. `input_sharding_dim` defines which mesh dimension to use to shard inputs, while `model_sharding_patterns` define how we shard model weights.

Example mesh configuration in YAML:
```yaml
mesh_shape: [2, 4]  # 2 data parallel, 4 model parallel
mesh_axis_names: ["batch", "model"]
input_sharding_dim: "batch"
model_sharding_patterns:
  - ['\.self_attn\.q_proj\.base_layer$',      ["model", null]]
  - ['\.self_attn\.v_proj\.base_layer$',      ["model", null]]
  - ['\.self_attn\.o_proj$',                  [null, "model"]]
  - ['\.mlp\.gate_proj$',                     ["model", null]]
  - ['\.mlp\.up_proj$',                       ["model", null]]
  - ['\.mlp\.down_proj$',                     [null, "model"]]
```

### Qwen 2.5 0.5B Training

Qwen 2.5 0.5B is the default single chip example.

**Single Chip Training:**
```bash
python3 blacksmith/experiments/torch/qwen/test_qwen_finetuning.py --config blacksmith/experiments/torch/qwen/single_chip/test_qwen_finetuning.yaml
```

#### Qwen 2.5 0.5B Training Configuration

| Architecture       | mesh_shape | mesh_axis_names | dataset | Method |
| ------------------ | ---------- | --------------- | ------- | ------ |
| [Single-Chip](single_chip/test_qwen_finetuning.yaml)        | None       | None            | SST2    | LoRA   |

### Qwen 2.5 1.5B Training

Qwen 2.5 1.5B supports training on single chip configuration.

**Single Chip Training:**
```bash
python3 blacksmith/experiments/torch/qwen/test_qwen_finetuning.py --config blacksmith/experiments/torch/qwen/single_chip/test_qwen_1-5b_finetuning.yaml
```

#### Qwen 2.5 1.5B Training Configuration

| Architecture       | mesh_shape | mesh_axis_names | dataset | Method |
| ------------------ | ---------- | --------------- | ------- | ------ |
| [Single-Chip](single_chip/test_qwen_1-5b_finetuning.yaml)        | None       | None            | SST2    | LoRA   |

### Qwen 3 4B Instruct 2507 Training

Qwen 3 4B Instruct 2507 supports training on different configurations.

**Single Chip Training:**
```bash
python3 blacksmith/experiments/torch/qwen/test_qwen_finetuning.py --config blacksmith/experiments/torch/qwen/single_chip/test_qwen_3_4b_instruct_2507_finetuning.yaml
```

**Blackhole QuietBox Training:**
```bash
python3 blacksmith/experiments/torch/qwen/test_qwen_finetuning.py --config blacksmith/experiments/torch/qwen/quietbox/test_qwen_3_4b_instruct_2507_finetuning.yaml
```
Working mesh shapes for Blackhole QuietBox: `[1, 4]` (data, model)

#### Qwen 3 4B Instruct 2507 Training Configurations

| Architecture       | mesh_shape         | mesh_axis_names                           | dataset | Method |
| ------------------ | -----------------  | ----------------------------------------- | ------- | ------ |
| [Single-Chip](single_chip/test_qwen_3_4b_instruct_2507_finetuning.yaml)        | None               | None                                      | SST2    | LoRA   |
| [N300](quietbox/test_qwen_3_4b_instruct_2507_finetuning.yaml)             | `[1, 2]`, `[2, 1]` |  `["data", "model"]`, `["model", "data"]` | SST2    | LoRA   |
| [Blackhole QuietBox](quietbox/test_qwen_3_4b_instruct_2507_finetuning.yaml) | `[1, 4]`           | `["data", "model"]`                       | SST2    | LoRA   |

### Qwen 3 8B Base Training

Qwen 3 8B Base requires multi-chip configurations (not supported on single chip).

**Galaxy Training:**
```bash
python3 blacksmith/experiments/torch/qwen/test_qwen_finetuning.py --config blacksmith/experiments/torch/qwen/galaxy/test_qwen_3_8b_base_finetuning.yaml
```

#### Qwen 3 8B Base Training Configurations

| Architecture       | mesh_shape         | mesh_axis_names                           | dataset | Method |
| ------------------ | -----------------  | ----------------------------------------- | ------- | ------ |
| [Wormhole Galaxy](galaxy/test_qwen_3_8b_base_finetuning.yaml)        | `[8, 4]`   | `["data", "model"]`, `["model", "data"]` | SST2    | LoRA   |


## Data


GLUE, the General Language Understanding Evaluation benchmark (https://gluebenchmark.com/) is a collection of resources for training, evaluating, and analyzing natural language understanding systems.
The Stanford Sentiment Treebank consists of sentences from movie reviews and human annotations of their sentiment. The task is to predict the sentiment of a given sentence. It uses the two-way (positive/negative) class split, with only sentence-level labels.
Each example consists of a sentence from movie reviews labeled as either positive or negative sentiment.
This dataset is commonly used to evaluate the performance of natural language understanding models on sentiment analysis tasks.

Source: [Hugging Face Dataset Hub](https://huggingface.co/datasets/nyu-mll/glue)

Example
```
{
  "sentence": "A touching and insightful film.",
  "label": 1
}
```
- sentence: A short movie review or phrase.
- label: Sentiment label (1 for positive, 0 for negative).

### Configuration Paramaters

| Parameter                     | Description                                            | Default Value                       |
| ----------------------------- | ------------------------------------------------------ | ----------------------------------- |
| `dataset_id`                  | The dataset used for fine-tuning.                      | "text2sql" or "sst2"                |
| `model_name`                  | Name or path of the pre-trained model.                 | "Qwen/Qwen2.5-0.5B"                 |
| `max_length`                  | Maximum token length for inputs.                       | 128                                 |
| `dtype`                       | Data type used during training.                        | "torch.bfloat16"                    |
| `training_type`               | Which type of finetuning to do.                        | "lora"                              |
| `learning_rate`               | Learning rate for the optimizer.                       | 2e-5                                |
| `batch_size`                  | Number of samples per training batch.                  | 8                                   |
| `gradient_accumulation_steps` | Steps to accumulate gradients before updating.         | 1                                   |
| `gradient_checkpointing`      | Whether to use gradient checkpointing to save memory.  | False                               |
| `num_epochs`                  | Total number of training epochs.                       | 1                                   |
| `optim`                       | Optimizer to use for training.                         | "adamw_torch"                       |
| `log_level`                   | Logging verbosity level.                               | "INFO"                              |
| `use_wandb`                   | Whether to enable Weights & Biases logging.            | True                                |
| `wandb_project`               | Project name for Weights & Biases logging.             | "qwen-finetuning"                   |
| `wandb_run_name`              | Run name for Weights & Biases tracking.                | "tt-qwen"           |
| `wandb_tags`                  | List of tags assigned to the W&B run.                  | ["test"]                            |
| `wandb_watch_mode`            | Watch mode for model parameter logging.                | "all"                               |
| `wandb_log_freq`              | Frequency of logging to Weights & Biases (in steps).   | 1000                                |
| `model_to_wandb`              | Whether to store model checkpoint in Weights & Biases. | False                               |
| `steps_freq`                  | Frequency (in steps) for performing periodic actions.  | 25                                  |
| `val_steps_freq`              | Frequency (in steps) for performing validation actions.| 25                                  |
| `epoch_freq`                  | Frequency (in epochs) for performing periodic actions. | 1                                   |
| `print_examples`              | Whether to print example predictions during training.  | True                                |
| `ignored_index`               | Index to ignore in loss computation.                   | -100                                |
| `resume_from_checkpoint`      | Whether to resume training from a previous checkpoint. | False                               |
| `resume_option`               | Resume method (`last`, `best`, or `path`).             | "last"                              |
| `checkpoint_path`             | Path to a checkpoint if `resume_option="path"`.        | ""                                  |
| `checkpoint_metric`           | Metric used for best checkpoint.                       | "eval/loss"                         |
| `checkpoint_metric_mode`      | Mode for checkpoint metric (`min` or `max`).           | "min"                               |
| `keep_last_n`                 | Number of latest checkpoints to keep.                  | 3                                   |
| `keep_best_n`                 | Number of best checkpoints to keep.                    | 1                                   |
| `save_strategy`               | Strategy for saving checkpoints (`epoch` or `step`).   | "step"                              |
| `project_dir`                 | Directory for experiment outputs.                      | "blacksmith/experiments/torch/qwen" |
| `save_optim`                  | Whether to save optimizer state.                       | False                               |
| `storage_backend`             | Storage backend for saving checkpoints.                | "local"                             |
| `sync_to_storage`             | Whether to sync checkpoints to remote storage.         | False                               |
| `load_from_storage`           | Whether to load checkpoints from remote storage.       | False                               |
| `remote_path`                 | Remote storage path (if applicable).                   | ""                                  |
| `seed`                        | Random seed for reproducibility.                       | 23                                  |
| `deterministic`               | Whether to enforce deterministic behavior.             | False                               |
| `lora_r`                      | Rank of LoRA adaptation matrices.                      | 4                                   |
| `lora_alpha`                  | Scaling factor for LoRA updates.                       | 8                                   |
| `lora_target_modules`         | Target modules for LoRA adaptation.                    | ["all-linear"]                      |
| `lora_task_type`              | Training task type for LoRA.                           | "CAUSAL_LM"                         |
| `framework`                   | Training framework.                                    | "pytorch"                           |
| `use_tt`                      | Whether to run on TT device (or GPU otherwise).        | True                                |
| `mesh_shape`                  | Mesh shape for distributed training.                   | None                                |
| `mesh_axis_names`             | Axis names for the mesh.                               | None
