# Llama with LoRA Experiment in TT-XLA

This directory contains the code for the Llama with LoRA fine-tuning experiment in TT-XLA.
Llama model specification can be found [here](https://huggingface.co/meta-llama/Llama-3.2-1B).
Original LoRA paper can be found [here](https://arxiv.org/pdf/2106.09685).

## Overview

The LLaMA fine-tuning experiment applies the LoRA technique to adapt a pre-trained LLaMA model on the SST sentiment analysis dataset.
The experiment is designed to run on the Huggingface framework.

## Training

The experiment supports different hardware configurations with per-model training setups.

### Mesh and Sharding Configuration

Mesh configurations define the parallelism strategy. The `mesh_axis_names` can be either `["data", "model"]` or `["model", "data"]` depending on which dimension corresponds to which type of parallelism.

Example mesh configuration in YAML:
```yaml
mesh_shape: [2, 4]  # 2 data parallel, 4 model parallel
mesh_axis_names: ["data", "model"]

model_sharding_patterns:
  - ['\.self_attn\.q_proj\.base_layer$',      ["model", null]]
  - ['\.self_attn\.v_proj\.base_layer$',      ["model", null]]
  - ['\.self_attn\.o_proj$',                  [null, "model"]]
  - ['\.mlp\.gate_proj$',                     ["model", null]]
  - ['\.mlp\.up_proj$',                       ["model", null]]
  - ['\.mlp\.down_proj$',                     [null, "model"]]
```

### Llama 1B Training

Llama 1B supports training on all hardware configurations:

**Single Chip Training:**
```bash
python3 blacksmith/experiments/torch/llama/xla/test_llama_fine_tuning_pure_torch.py --config blacksmith/experiments/torch/llama/xla/lora/single_chip/test_llama_1b.yaml
```

**QuietBox Training:**
```bash
python3 blacksmith/experiments/torch/llama/xla/test_llama_fine_tuning_pure_torch.py --config blacksmith/experiments/torch/llama/xla/lora/quietbox/test_llama_1b.yaml
```
Working mesh shapes: `[1, 8]`, `[8, 1]`, `[2, 4]` (both `mesh_axis_names` orderings supported)

**Galaxy Training:**
```bash
python3 blacksmith/experiments/torch/llama/xla/test_llama_fine_tuning_pure_torch.py --config blacksmith/experiments/torch/llama/xla/lora/galaxy/test_llama_1b.yaml
```
Working mesh shape: `[8, 4]` (both `mesh_axis_names` orderings supported)

**N300 Training:**
Working mesh shapes: `[1, 2]`, `[2, 1]` (both `mesh_axis_names` orderings supported)

### Llama 8B Training

**Llama 8B requires multi-chip configurations (not supported on single chip) and must be model sharded (model dimension > 1).**

**QuietBox Training:**
```bash
python3 blacksmith/experiments/torch/llama/xla/test_llama_fine_tuning_pure_torch.py --config blacksmith/experiments/torch/llama/xla/lora/quietbox/test_llama_8b.yaml
```
Working mesh shapes: `[1, 8]` (data, model), `[8, 1]` (model, data), `[2, 4]` (data, model)

*Note: For meshes with 1 dimension, the 1 must be the data dimension (model dimension must be > 1)*

**Galaxy Training:**
```bash
python3 blacksmith/experiments/torch/llama/xla/test_llama_fine_tuning_pure_torch.py --config blacksmith/experiments/torch/llama/xla/lora/galaxy/test_llama_8b.yaml
```
Working mesh shape: `[8, 4]` (both `mesh_axis_names` orderings supported)

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

## Configuration

The experiment is configured using the configuration file `test_llama_fine_tuning_pure_torch.yaml`. The configuration file specifies the hyperparameters for the experiment, such as the number of epochs, the batch size, and the lora configuration.

Current `test_llama_fine_tuning_pure_torch.yaml` has the recommended and tested hyperparameters for the experiment.

### Configuration Paramaters

| Parameter                     | Description                                            | Default Value                       |
| ----------------------------- | ------------------------------------------------------ | ----------------------------------- |
| `dataset_id`                  | The dataset used for fine-tuning.                      | "sst2"                              |
| `model_name`                  | Name or path of the pre-trained model.                 | "meta-llama/Llama-3.2-1B".          |
| `max_length`                  | Maximum token length for inputs.                       | 128                                 |
| `dtype`                       | Data type used during training.                        | "torch.bfloat16"                    |
| `training_type`               | Which type of finetuning to do.                        | "lora"                              |
| `learning_rate`               | Learning rate for the optimizer.                       | 2e-5                                |
| `batch_size`                  | Number of samples per training batch.                  | 32                                  |
| `gradient_accumulation_steps` | Steps to accumulate gradients before updating.         | 1                                   |
| `gradient_checkpointing`      | Whether to use gradient checkpointing to save memory.  | False                               |
| `num_epochs`                  | Total number of training epochs.                       | 1                                   |
| `optim`                       | Optimizer to use for training.                         | "adamw_torch"                       |
| `log_level`                   | Logging verbosity level.                               | "INFO"                              |
| `use_wandb`                   | Whether to enable Weights & Biases logging.            | True                                |
| `wandb_project`               | Project name for Weights & Biases logging.             | "llama-finetuning"                  |
| `wandb_run_name`              | Run name for Weights & Biases tracking.                | "tt-llama"                          |
| `wandb_tags`                  | List of tags assigned to the W&B run.                  | ["test"]                            |
| `wandb_watch_mode`            | Watch mode for model parameter logging.                | "all"                               |
| `wandb_log_freq`              | Frequency of logging to Weights & Biases (in steps).   | 1000                                |
| `model_to_wandb`              | Whether to store model checkpoint in Weights & Biases. | False                               |
| `steps_freq`                  | Frequency (in steps) for performing periodic actions.  | 25                                  |
| `epoch_freq`                  | Frequency (in epochs) for performing periodic actions. | 1                                   |
| `resume_from_checkpoint`      | Whether to resume training from a previous checkpoint. | False                               |
| `resume_option`               | Resume method (`last`, `best`, or `path`).             | "last"                              |
| `checkpoint_path`             | Path to a checkpoint if `resume_option="path"`.        | ""                                  |
| `save_strategy`               | Strategy for saving checkpoints (`epoch` or `step`).   | "epoch"                             |
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
| `parallelism`                 | Parallelism strategy (`single`, `data`, `tensor`).     | "single"                            |
| `mesh_shape`                  | Mesh shape for distributed training.                   | "8,1"                               |
