# Gemma 1.1 2B with LoRA Experiment

This directory contains the code for the Gemma 1.1 2B fine-tuning experiment using LoRA.

- Gemma 1.1 2B model specification can be found [here](https://huggingface.co/google/gemma-1.1-2b-it).

Original LoRA paper can be found [here](https://arxiv.org/pdf/2106.09685).

## Overview

The Gemma 1.1 2B fine-tuning experiment applies the LoRA technique to adapt a pre-trained Gemma 1.1 2B model on a configurable dataset (SST2, SQuAD V2, or the math preference dataset).
The experiment is designed to run on the Huggingface framework and targets single-chip Tenstorrent hardware.

## Training

The dataset and hyperparameters are selected through the configuration file passed via `--config`.
If no config is provided, `single_chip/gemma11_sst2.yaml` is used by default.

### Single Chip Training

SST2 dataset:
```bash
python3 blacksmith/experiments/torch/gemma11/lora/train.py --config blacksmith/experiments/torch/gemma11/lora/single_chip/gemma11_sst2.yaml
```

SQuAD V2 dataset:
```bash
python3 blacksmith/experiments/torch/gemma11/lora/train.py --config blacksmith/experiments/torch/gemma11/lora/single_chip/gemma11_squadV2.yaml
```

Math preference (SFT) dataset:
```bash
python3 blacksmith/experiments/torch/gemma11/lora/train.py --config blacksmith/experiments/torch/gemma11/lora/single_chip/gemma11_math_preferences_sft.yaml
```

### QuietBox Training

```bash
python3 blacksmith/experiments/torch/gemma11/lora/train.py --config blacksmith/experiments/torch/gemma11/lora/quietbox/gemma11_math_preferences_sft.yaml
```

#### Training Configurations

| Architecture       | mesh_shape                   | mesh_axis_names      | input_sharding_dim | dataset      | Method                      |
| ------------------ | ---------------------------- | -------------------- | ------------------ | ------------ | --------------------------- |
| [Single-Chip](single_chip/gemma11_sst2.yaml) | None | None | None | SST2 | LoRA |
| [Single-Chip](single_chip/gemma11_squadV2.yaml) | None | None | None | SQuAD V2 | LoRA |
| [P150](single_chip/gemma11_math_preferences_sft.yaml) | None | None | None | Math Preference (SFT) | LoRA |
| [QuietBox](quietbox/gemma11_math_preferences_sft.yaml) | `[2, 2]` | `["batch", "model"]` | `"batch"` | Math Preference (SFT) | LoRA (DP + TP) |

## Data

### SST2

GLUE, the General Language Understanding Evaluation benchmark (https://gluebenchmark.com/) is a collection of resources for training, evaluating, and analyzing natural language understanding systems.
The Stanford Sentiment Treebank consists of sentences from movie reviews and human annotations of their sentiment. The task is to predict the sentiment of a given sentence. It uses the two-way (positive/negative) class split, with only sentence-level labels.
Each example consists of a sentence from movie reviews labeled as either positive or negative sentiment.
This dataset is commonly used to evaluate the performance of natural language understanding models on sentiment analysis tasks.

Source: [Hugging Face Dataset Hub](https://huggingface.co/datasets/nyu-mll/glue)

Example:
```
{
  "sentence": "A touching and insightful film.",
  "label": 1
}
```
- sentence: A short movie review or phrase.
- label: Sentiment label (1 for positive, 0 for negative).

### SQuAD V2

The Stanford Question Answering Dataset V2.0 (SQuAD V2.0) is a challenging benchmark for extractive Question Answering (QA) models.
It requires a model to read a context passage and a question, then either:
1. Extract the answer as a span of text from the passage (if the question is answerable), OR
2. Determine the question is unanswerable and abstain from answering.

SQuAD V2.0 combines the 100,000+ answerable questions from the original SQuAD with over 50,000 new, unanswerable questions, pushing models to be both accurate and robust in identifying when information is missing.

Source: [Hugging Face Dataset Hub](https://huggingface.co/datasets/rajpurkar/squad_v2)

Example:
```
{
  "context": "On December 13, 2013, Beyoncé unexpectedly released her eponymous fifth studio album on the iTunes Store without any prior announcement or promotion. The album debuted atop the Billboard 200 chart, giving Beyoncé her fifth consecutive number-one album in the US....."
  "question": "Who joined Beyonce on her On The Run Tour?"
  "answers": "Jay Z"
}
```

### Math Preference (SFT)

The math preference dataset is a collection of math instructions paired with a higher-rated (chosen) and a lower-rated (rejected) response.
In SFT mode only the chosen responses are used to supervise the model, training it to produce the preferred answer for each instruction. The same dataset is reused as the preference source for the [DPO experiment](../dpo/README.md).

Source: [Hugging Face Dataset Hub](https://huggingface.co/datasets/argilla/distilabel-math-preference-dpo)

Example:
```
{
  "instruction": "What is the derivative of f(x) = 3x^2 + 2x?",
  "chosen_response": "Using the power rule, f'(x) = 6x + 2.",
  "rejected_response": "The derivative is 3x + 2.",
  "chosen_rating": 9.0,
  "rejected_rating": 4.0
}
```
- instruction: The math problem or question.
- chosen_response / rejected_response: The preferred and dispreferred answers.
- chosen_rating / rejected_rating: Quality scores for each response.

## Configuration

The experiment is configured using the configuration files in `single_chip/`. The configuration file specifies the hyperparameters for the experiment, such as the dataset, the number of epochs, the batch size, and the LoRA configuration.

`gemma11_sst2.yaml` is the default config and has the recommended and tested hyperparameters for the experiment (using SST2 as the default dataset).

### Configuration Parameters

| Parameter | Description | Default Value |
| --- | --- | --- |
| `dataset_id` | The dataset used for fine-tuning. | "sst2" |
| `model_name` | Name or path of the pre-trained Gemma 1.1 2B model. | "google/gemma-1.1-2b-it" |
| `max_length` | Maximum token length for inputs. | 32 |
| `dtype` | Data type used during training. | "torch.bfloat16" |
| `training_model_type` | Which type of fine-tuning to do (`lora`, `adapters`, or full fine-tuning). | "lora" |
| `learning_rate` | Learning rate for the optimizer. | 6e-5 |
| `batch_size` | Number of samples per training batch. | 4 |
| `gradient_checkpointing` | Whether to use gradient checkpointing to save memory. | False |
| `num_epochs` | Total number of training epochs. | 1 |
| `optim` | Optimizer to use for training. | "adamw_torch" |
| `log_level` | Logging verbosity level. | "INFO" |
| `use_wandb` | Whether to enable Weights & Biases logging. | True |
| `wandb_project` | Project name for Weights & Biases logging. | "gemma11-finetuning" |
| `wandb_run_name` | Run name for Weights & Biases tracking. | "tt-gemma11-sst2" |
| `wandb_tags` | List of tags assigned to the W&B run. | ["test"] |
| `wandb_watch_mode` | Watch mode for model parameter logging. | "all" |
| `wandb_log_freq` | Frequency of logging to Weights & Biases (in steps). | 1000 |
| `model_to_wandb` | Whether to store model checkpoint in Weights & Biases. | False |
| `steps_freq` | Frequency (in steps) for performing periodic actions. | 10 |
| `epoch_freq` | Frequency (in epochs) for performing periodic actions. | 1 |
| `val_steps_freq` | Frequency of validation (in steps). | 50 |
| `resume_from_checkpoint` | Whether to resume training from a previous checkpoint. | False |
| `resume_option` | Resume method (`last`, `best`, or `path`). | "last" |
| `checkpoint_path` | Path to a checkpoint if `resume_option="path"`. | "" |
| `checkpoint_metric` | Metric to monitor for best checkpoint. | "eval/loss" |
| `checkpoint_metric_mode` | Mode for checkpoint metric (`min` or `max`). | "min" |
| `keep_last_n` | Number of recent checkpoints to keep. | 3 |
| `keep_best_n` | Number of best checkpoints to keep. | 3 |
| `save_strategy` | Strategy for saving checkpoints (`epoch` or `step`). | "epoch" |
| `project_dir` | Directory for experiment outputs. | "blacksmith/experiments/torch/gemma11" |
| `save_optim` | Whether to save optimizer state. | False |
| `storage_backend` | Storage backend for saving checkpoints. | "local" |
| `sync_to_storage` | Whether to sync checkpoints to remote storage. | False |
| `load_from_storage` | Whether to load checkpoints from remote storage. | False |
| `remote_path` | Remote storage path (if applicable). | "" |
| `seed` | Random seed for reproducibility. | 23 |
| `deterministic` | Whether to enforce deterministic behavior. | False |
| `mesh_shape` | Mesh shape for distributed training (e.g. `[2, 2]` for 4 chips). | None |
| `mesh_axis_names` | Axis names for the mesh (e.g. `["batch", "model"]`). | None |
| `input_sharding_dim` | Mesh axis along which input batches are sharded (data parallelism). | None |
| `model_sharding_patterns` | Tensor-parallel sharding rules: list of `(regex, partition_spec)` matched against module names. | None |
| `lora_r` | Rank of LoRA adaptation matrices. | 4 |
| `lora_alpha` | Scaling factor for LoRA updates. | 8 |
| `lora_target_modules` | Target modules for LoRA adaptation. | ["all-linear"] |
| `lora_task_type` | Training task type for LoRA. | "CAUSAL_LM" |
| `framework` | Training framework. | "pytorch" |
| `use_tt` | Whether to run on TT device (or GPU otherwise). | True |
