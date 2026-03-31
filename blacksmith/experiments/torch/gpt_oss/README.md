# GPT-OSS with LoRA Experiment in TT-XLA

This directory contains the code for the GPT-OSS 20B LoRA fine-tuning experiment in TT-XLA.

- GPT-OSS 20B model specification can be found [here](https://huggingface.co/openai/gpt-oss-20b).

Original LoRA paper can be found [here](https://arxiv.org/pdf/2106.09685).

## Overview

The GPT-OSS fine-tuning experiment applies the LoRA technique to adapt a pre-trained GPT-OSS 20B
model on the SST-2 sentiment analysis dataset.
The experiment is designed to run on the Huggingface framework.

The model uses a Mixture-of-Experts (MoE) architecture. Expert weights are de-interleaved from
the original `gate_up_proj` layout at load time to enable BMM-based forward passes compatible with
the TT backend. LoRA is applied to the top half of transformer layers (`q_proj`, `k_proj`,
`v_proj`, `o_proj`).

## Training

The experiment supports multi-chip configurations with 1D tensor parallelism.

### Mesh and Sharding Configuration

Mesh configurations define the parallelism strategy. `input_sharding_dim` defines which mesh
dimension to use to shard inputs, while `model_sharding_patterns` and `param_sharding_patterns`
define how model weights and parameters are sharded.

The default YAML uses Megatron-style 1D tensor parallelism: QKV projections are column-parallel
(`["model", null]`), the output projection is row-parallel (`[null, "model"]`), LoRA adapters are
replicated (`[null, null]`), and expert weights are sharded across the expert dimension
(`["model", null, null]`).

Example mesh configuration in YAML:
```yaml
mesh_shape: [1, 8]
mesh_axis_names: ["batch", "model"]
input_sharding_dim: null
model_sharding_patterns:
  - ['\.self_attn\.q_proj\.base_layer$', ["model", null]]
  - ['\.self_attn\.k_proj\.base_layer$', ["model", null]]
  - ['\.self_attn\.v_proj\.base_layer$', ["model", null]]
  - ['\.self_attn\.o_proj\.base_layer$', [null, "model"]]
param_sharding_patterns:
  - ['\.self_attn\.q_proj\.(base_layer\.)?bias$', ["model"]]
  - ['\.self_attn\.k_proj\.(base_layer\.)?bias$', ["model"]]
  - ['\.self_attn\.v_proj\.(base_layer\.)?bias$', ["model"]]
  - ['\.self_attn\.o_proj\.(base_layer\.)?bias$', [null]]
```

### GPT-OSS 20B Training

**BH LoudBox Training:**
```bash
python3 blacksmith/experiments/torch/gpt_oss/test_gpt_oss_finetuning.py --config blacksmith/experiments/torch/gpt_oss/lora/loudbox/test_gpt_oss_20b_finetuning.yaml
```

#### GPT-OSS 20B Training Configurations

| Architecture | mesh_shape | mesh_axis_names | Dataset | Method |
| --- | --- | --- | --- | --- |
| [BH LoudBox](lora/loudbox/test_gpt_oss_20b_finetuning.yaml) | `[1, 8]` | `["batch", "model"]` | SST-2 | LoRA |

## Data

GLUE, the General Language Understanding Evaluation benchmark (https://gluebenchmark.com/) is a
collection of resources for training, evaluating, and analyzing natural language understanding
systems.
The Stanford Sentiment Treebank consists of sentences from movie reviews and human annotations of
their sentiment. The task is to predict the sentiment of a given sentence. It uses the two-way
(positive/negative) class split, with only sentence-level labels.
Each example consists of a sentence from movie reviews labeled as either positive or negative
sentiment.
This dataset is commonly used to evaluate the performance of natural language understanding models
on sentiment analysis tasks.

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

The experiment is configured using the configuration file `lora/loudbox/test_gpt_oss_20b_finetuning.yaml`. The
configuration file specifies the hyperparameters for the experiment, such as the number of epochs,
the batch size, and the LoRA configuration.

Current `test_gpt_oss_20b_finetuning.yaml` has the recommended and tested hyperparameters for the
experiment.

### Configuration Parameters

| Parameter | Description | Default Value |
| --- | --- | --- |
| `dataset_id` | The dataset used for fine-tuning. | `"sst2"` |
| `model_name` | Name or path of the pre-trained model. | `"openai/gpt-oss-20b"` |
| `max_length` | Maximum token length for inputs. | `128` |
| `dtype` | Data type used during training. | `"torch.bfloat16"` |
| `training_type` | Which type of fine-tuning to do. | `"lora"` |
| `learning_rate` | Learning rate for the optimizer. | `1e-4` |
| `batch_size` | Number of samples per training batch. | `1` |
| `gradient_accumulation_steps` | Steps to accumulate gradients before updating. | `4` |
| `num_epochs` | Total number of training epochs. | `1` |
| `log_level` | Logging verbosity level. | `"INFO"` |
| `use_wandb` | Whether to enable Weights & Biases logging. | `True` |
| `wandb_project` | Project name for Weights & Biases logging. | `"gpt-oss-20b-lora"` |
| `wandb_run_name` | Run name for Weights & Biases tracking. | `"tt-gpt-oss-20b-test"` |
| `wandb_tags` | List of tags assigned to the W&B run. | `["test"]` |
| `wandb_watch_mode` | Watch mode for model parameter logging. | `"all"` |
| `wandb_log_freq` | Frequency of logging to Weights & Biases (in steps). | `1000` |
| `model_to_wandb` | Whether to store model checkpoint in Weights & Biases. | `False` |
| `steps_freq` | Frequency (in steps) for performing periodic actions. | `1` |
| `val_steps_freq` | Frequency (in steps) for performing validation actions. | `10` |
| `epoch_freq` | Frequency (in epochs) for performing periodic actions. | `1` |
| `resume_from_checkpoint` | Whether to resume training from a previous checkpoint. | `False` |
| `resume_option` | Resume method (`last`, `best`, or `path`). | `"last"` |
| `checkpoint_path` | Path to a checkpoint if `resume_option="path"`. | `""` |
| `save_strategy` | Strategy for saving checkpoints (`epoch` or `step`). | `"step"` |
| `project_dir` | Directory for experiment outputs. | `"blacksmith/experiments/torch/gpt_oss"` |
| `save_optim` | Whether to save optimizer state. | `True` |
| `storage_backend` | Storage backend for saving checkpoints. | `"local"` |
| `sync_to_storage` | Whether to sync checkpoints to remote storage. | `False` |
| `load_from_storage` | Whether to load checkpoints from remote storage. | `False` |
| `remote_path` | Remote storage path (if applicable). | `""` |
| `seed` | Random seed for reproducibility. | `23` |
| `deterministic` | Whether to enforce deterministic behavior. | `False` |
| `lora_r` | Rank of LoRA adaptation matrices. | `32` |
| `lora_alpha` | Scaling factor for LoRA updates. | `64` |
| `lora_target_modules` | Target modules for LoRA adaptation. | `["q_proj", "k_proj", "v_proj", "o_proj"]` |
| `lora_task_type` | Training task type for LoRA. | `"CAUSAL_LM"` |
| `framework` | Training framework. | `"pytorch"` |
| `print_examples` | Whether to print validation examples. | `True` |
| `use_tt` | Whether to run on TT device (or GPU otherwise). | `True` |
| `mesh_shape` | Mesh shape for distributed training. | `[1, 8]` |
| `mesh_axis_names` | Axis names for the mesh. | `["batch", "model"]` |
| `input_sharding_dim` | Mesh dimension for input sharding. | `null` |
| `model_sharding_patterns` | Regex-based module sharding specs (shards `.weight`). | see YAML |
| `param_sharding_patterns` | Regex-based parameter sharding specs. | see YAML |
