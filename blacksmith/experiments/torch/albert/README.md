# ALBERT classification head Experiment

This directory contains the code for the ALBERT model with classification head fine-tuning experiment.
ALBERT model specification can be found [here](https://huggingface.co/albert/albert-base-v2).

## Overview

This experiment fine-tunes a classification head on top of a frozen ALBERT-base model for intent classification on the Banking77 dataset.

## Training

```bash
python3 blacksmith/experiments/torch/albert/test_albert_finetuning.py
```

## Data

mteb/banking77 is a fine-grained intent classification dataset consisting of online banking customer service queries annotated with their corresponding intents. The dataset contains 13,083 queries labeled across 77 distinct intent categories (such as 'activate_my_card', 'apple_pay', 'bank_transfer', etc.), making it significantly more challenging than previous intent detection benchmarks that typically contain fewer than 10 classes.

Source: [Hugging Face Dataset Hub](https://huggingface.co/datasets/mteb/banking77)

Example
```
{
  "text": "I am still waiting on my card?",
  "label": 11,
  "label_text": "card_arrival"
}
```


## Configuration

The experiment is configured using the configuration file `test_albert_finetuning.yaml`. The configuration file specifies the hyperparameters for the experiment, such as the number of epochs, the batch size, and the lora configuration.

Current `test_albert_finetuning.yaml` has the recommended and tested hyperparameters for the experiment.

### Configuration Paramaters

| Parameter                     | Description                                            | Default Value                        |
| ----------------------------- | ------------------------------------------------------ | ------------------------------------ |
| `dataset_id`                  | The dataset used for fine-tuning.                      | "mteb/banking77"                     |
| `model_name`                  | Name or path of the pre-trained model.                 | "albert/albert-base-v2"              |
| `max_length`                  | Maximum token length for inputs.                       | 128                                  |
| `num_labels`                  | Number of classification labels.                       | 77                                   |
| `mlp_hidden_size`             | Hidden size of the MLP classification head.            | 256                                  |
| `dtype`                       | Data type used during training.                        | "torch.bfloat16"                     |
| `learning_rate`               | Learning rate for the optimizer.                       | 1e-3                                 |
| `weight_decay`                | Weight decay for regularization.                       | 0.01                                 |
| `batch_size`                  | Number of samples per training batch.                  | 8                                    |
| `gradient_accumulation_steps` | Steps to accumulate gradients before updating.         | 1                                    |
| `gradient_checkpointing`      | Whether to use gradient checkpointing to save memory.  | False                                |
| `num_epochs`                  | Total number of training epochs.                       | 5                                    |
| `optim`                       | Optimizer to use for training.                         | "adamw_torch"                        |
| `log_level`                   | Logging verbosity level.                               | "INFO"                               |
| `use_wandb`                   | Whether to enable Weights & Biases logging.            | True                                 |
| `wandb_project`               | Project name for Weights & Biases logging.             | "albert-finetuning"                  |
| `wandb_run_name`              | Run name for Weights & Biases tracking.                | "tt-albert-test"                     |
| `wandb_tags`                  | List of tags assigned to the W&B run.                  | ["test"]                             |
| `wandb_watch_mode`            | Watch mode for model parameter logging.                | "all"                                |
| `wandb_log_freq`              | Frequency of logging to Weights & Biases (in steps).   | 1000                                 |
| `model_to_wandb`              | Whether to store model checkpoint in Weights & Biases. | False                                |
| `steps_freq`                  | Frequency (in steps) for performing periodic actions.  | 10                                   |
| `epoch_freq`                  | Frequency (in epochs) for performing periodic actions. | 1                                    |
| `resume_from_checkpoint`      | Whether to resume training from a previous checkpoint. | False                                |
| `resume_option`               | Resume method (`last`, `best`, or `path`).             | "last"                               |
| `checkpoint_path`             | Path to a checkpoint if `resume_option="path"`.        | ""                                   |
| `checkpoint_metric`           | Metric used to determine best checkpoint.              | "eval/loss"                          |
| `checkpoint_metric_mode`      | Mode for checkpoint metric (`min` or `max`).           | "min"                                |
| `keep_last_n`                 | Number of most recent checkpoints to keep.             | 3                                    |
| `keep_best_n`                 | Number of best checkpoints to keep.                    | 1                                    |
| `save_strategy`               | Strategy for saving checkpoints (`epoch` or `step`).   | "epoch"                              |
| `project_dir`                 | Directory for experiment outputs.                      | "blacksmith/experiments/torch/albert"|
| `save_optim`                  | Whether to save optimizer state.                       | False                                |
| `storage_backend`             | Storage backend for saving checkpoints.                | "local"                              |
| `sync_to_storage`             | Whether to sync checkpoints to remote storage.         | False                                |
| `load_from_storage`           | Whether to load checkpoints from remote storage.       | False                                |
| `remote_path`                 | Remote storage path (if applicable).                   | ""                                   |
| `seed`                        | Random seed for reproducibility.                       | 23                                   |
| `deterministic`               | Whether to enforce deterministic behavior.             | False                                |
| `framework`                   | Training framework.                                    | "pytorch"                            |
| `use_tt`                      | Whether to run on TT device (or GPU otherwise).        | True                                 |
