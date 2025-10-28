# Gemma with LoRA Experiment

This directory contains the code for the Gemma model with LoRA fine-tuning experiment.
Gemma model specification can be found [here](https://huggingface.co/google/gemma-3-1b-it).
Original LoRA paper can be found [here](https://arxiv.org/pdf/2106.09685).

## Overview

The Gemma fine-tuning experiment applies the LoRA technique to adapt a pre-trained Gemma model on the SST sentiment analysis dataset.
The experiment is designed to run on the Huggingface framework.

## Training

```bash
python3 blacksmith/experiments/torch/gemma/test_gemma_finetuning.py
```

## Data

The Stanford Sentiment Treebank 2 (SST-2) dataset is a widely-used benchmark for binary sentiment classification.
Each example consists of a sentence from movie reviews labeled as either positive or negative sentiment.
This dataset is commonly used to evaluate the performance of natural language understanding models on sentiment analysis tasks.

Source: [Hugging Face Dataset Hub](https://huggingface.co/datasets/stanfordnlp/sst2)

Example:
```
{
  "sentence": "A touching and insightful film.",
  "label": 1
}
```
- sentence: A short movie review or phrase.
- label: Sentiment label (1 for positive, 0 for negative).

## Configuration

The experiment is configured using the configuration file `test_gemma_finetuning.yaml`. The configuration file specifies the hyperparameters for the experiment, such as the number of epochs, the batch size, and the lora configuration.

Current `test_gemma_finetuning.yaml` has the recommended and tested hyperparameters for the experiment.

### Configuration Parameters

| Parameter | Description | Default Value |
| --- | --- | --- |
| `dataset_id` | The dataset used for fine-tuning. | "stanfordnlp/sst2" |
| `model_name` | Name or path of the pre-trained Gemma model. | "google/gemma-3-1b-it" |
| `max_length` | Maximum token length for inputs. | 128 |
| `dtype` | Data type used during training. | "torch.bfloat16" |
| `learning_rate` | Learning rate for the optimizer. | 2e-5 |
| `batch_size` | Number of samples per training batch. | 32 |
| `gradient_checkpointing` | Whether to use gradient checkpointing to save memory. | False |
| `num_epochs` | Total number of training epochs. | 1 |
| `optim` | Optimizer to use for training. | "adamw_torch" |
| `log_level` | Logging verbosity level. | "INFO" |
| `use_wandb` | Whether to enable Weights & Biases logging. | True |
| `wandb_project` | Project name for Weights & Biases logging. | "gemma-finetuning" |
| `wandb_run_name` | Run name for Weights & Biases tracking. | "tt-gemma-test" |
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
| `project_dir` | Directory for experiment outputs. | "blacksmith/experiments/torch/gemma" |
| `save_optim` | Whether to save optimizer state. | False |
| `storage_backend` | Storage backend for saving checkpoints. | "local" |
| `sync_to_storage` | Whether to sync checkpoints to remote storage. | False |
| `load_from_storage` | Whether to load checkpoints from remote storage. | False |
| `remote_path` | Remote storage path (if applicable). | "" |
| `seed` | Random seed for reproducibility. | 23 |
| `deterministic` | Whether to enforce deterministic behavior. | False |
| `lora_r` | Rank of LoRA adaptation matrices. | 4 |
| `lora_alpha` | Scaling factor for LoRA updates. | 8 |
| `lora_target_modules` | Target modules for LoRA adaptation. | ["all-linear"] |
| `lora_task_type` | Training task type for LoRA. | "CAUSAL_LM" |
| `framework` | Training framework. | "pytorch" |
| `use_tt` | Whether to run on TT device (or GPU otherwise). | True |
