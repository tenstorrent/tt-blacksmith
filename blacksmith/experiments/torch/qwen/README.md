# Qwen with LoRA Experiment

This directory contains the code for the Qwen model with LoRA fine-tuning experiment.
Qwen model specification can be found [here](https://huggingface.co/Qwen/Qwen2.5-0.5B).
Original LoRA paper can be found [here](https://arxiv.org/pdf/2106.09685).

## Overview

The Qwen fine-tuning experiment applies the LoRA technique to adapt a pre-trained Qwen model on the Text-to-SQL dataset.

## Training

```bash
python3 blacksmith/experiments/torch/qwen/test_qwen_finetuning.py
```

## Data

gretelai/synthetic_text_to_sql is a rich dataset of high quality synthetic Text-to-SQL samples, designed and generated using Gretel Navigator, and released under Apache 2.0.
While this dataset has wide range of SQL complexity levels, including subqueries, single joins, multiple joins, aggregations, window functions, set operations, for this experiment only basic SQL queries under 128 tokens are used.

Source: [Hugging Face Dataset Hub](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql)

Example
```
{
  "id": 39325,
  "domain": "public health",
  "domain_description": "Community health statistics, infectious disease tracking data, healthcare access metrics, and public health policy analysis.",
  "sql_complexity": "aggregation",
  "sql_complexity_description": "aggregation functions (COUNT, SUM, AVG, MIN, MAX, etc.), and HAVING clause",
  "sql_task_type": "analytics and reporting",
  "sql_task_type_description": "generating reports, dashboards, and analytical insights",
  "sql_prompt": "What is the total number of hospital beds in each state?",
  "sql_context": "CREATE TABLE Beds (State VARCHAR(50), Beds INT); INSERT INTO Beds (State, Beds) VALUES ('California', 100000), ('Texas', 85000), ('New York', 70000);",
  "sql": "SELECT State, SUM(Beds) FROM Beds GROUP BY State;",
  "sql_explanation": "This query calculates the total number of hospital beds in each state in the Beds table. It does this by using the SUM function on the Beds column and grouping the results by the State column."
}
```


## Configuration

The experiment is configured using the configuration file `test_qwen_finetuning.yaml`. The configuration file specifies the hyperparameters for the experiment, such as the number of epochs, the batch size, and the lora configuration.

Current `test_qwen_finetuning.yaml` has the recommended and tested hyperparameters for the experiment.

### Configuration Paramaters

| Parameter                     | Description                                            | Default Value                       |
| ----------------------------- | ------------------------------------------------------ | ----------------------------------- |
| `dataset_id`                  | The dataset used for fine-tuning.                      | "gretelai/synthetic_text_to_sql"    |
| `model_name`                  | Name or path of the pre-trained model.                 | "Qwen/Qwen2.5-0.5B".                |
| `max_length`                  | Maximum token length for inputs.                       | 128                                 |
| `dtype`                       | Data type used during training.                        | "torch.bfloat16"                    |
| `learning_rate`               | Learning rate for the optimizer.                       | 2e-5                                |
| `batch_size`                  | Number of samples per training batch.                  | 32                                  |
| `gradient_accumulation_steps` | Steps to accumulate gradients before updating.         | 1                                   |
| `gradient_checkpointing`      | Whether to use gradient checkpointing to save memory.  | False                               |
| `num_epochs`                  | Total number of training epochs.                       | 1                                   |
| `optim`                       | Optimizer to use for training.                         | "adamw_torch"                       |
| `log_level`                   | Logging verbosity level.                               | "INFO"                              |
| `use_wandb`                   | Whether to enable Weights & Biases logging.            | True                                |
| `wandb_project`               | Project name for Weights & Biases logging.             | "qwen-finetuning"                   |
| `wandb_run_name`              | Run name for Weights & Biases tracking.                | "tt-qwen-test"                      |
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
