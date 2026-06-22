# Gemma 1.1 2B DPO Experiment

This directory contains the code for the Gemma 1.1 2B alignment experiment using DPO (Direct Preference Optimization).

- Gemma 1.1 2B model specification can be found [here](https://huggingface.co/google/gemma-1.1-2b-it).

Original DPO paper ("Direct Preference Optimization: Your Language Model is Secretly a Reward Model") can be found [here](https://arxiv.org/pdf/2305.18290).

## Overview

The Gemma 1.1 2B DPO experiment aligns a pre-trained Gemma 1.1 2B model on the math preference dataset, training the policy model to prefer the chosen response over the rejected response for each instruction.
DPO is the training objective and is always applied by this experiment; the underlying fine-tuning approach for the policy model (LoRA, adapters, or full fine-tuning) is selected via `training_model_type`.

### Pipeline

Standard DPO uses a reference model (π_ref) that is an SFT model trained on the chosen responses:

1. Train an SFT model on the chosen responses with the [LoRA experiment](../lora/README.md) using the `gemma11_math_preferences_sft.yaml` config.
2. Point `sft_checkpoint_path` in the DPO config to that SFT checkpoint, so it is loaded into both the policy and the reference model.
3. Run DPO training.

If `sft_checkpoint_path` is left empty, the base pre-trained model is used as π_ref (less ideal, but works).

## Training

The experiment is configured through `test_dpo.yaml` in this directory, which is loaded automatically by the training script.

### Single Chip Training

```bash
python3 blacksmith/experiments/torch/gemma11/dpo/train.py
```

#### Training Configuration

| Architecture | mesh_shape | mesh_axis_names | dataset | Method |
| ------------ | ---------- | --------------- | ------- | ------ |
| [P150](test_dpo.yaml) | None | None | Math Preference (DPO) | DPO + LoRA |

## Data

### Math Preference (DPO)

The math preference dataset is a collection of math instructions, each paired with a higher-rated (chosen) and a lower-rated (rejected) response.
In DPO mode both responses are used: the policy model is trained to increase the relative log-probability of the chosen response over the rejected one, with respect to the reference model.
The same dataset is used in SFT mode by the [LoRA experiment](../lora/README.md) to pre-train the reference model.

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

The experiment is configured using `test_dpo.yaml`. The configuration file specifies the hyperparameters for the experiment, such as the DPO objective parameters, the reference model checkpoint, the number of epochs, the batch size, and the LoRA configuration.

### Configuration Parameters

| Parameter | Description | Default Value |
| --- | --- | --- |
| `training_model_type` | Fine-tuning approach for the policy model (`lora`, `adapters`, or full fine-tuning). | "lora" |
| `dataset_id` | The dataset used for training. | "math_preference_dpo" |
| `model_name` | Name or path of the pre-trained Gemma 1.1 2B model. | "google/gemma-1.1-2b-it" |
| `max_length` | Maximum token length for inputs. | 128 |
| `dtype` | Data type used during training. | "torch.bfloat16" |
| `ignored_index` | Label id used to mask prompt tokens in the loss. | -100 |
| `dpo_beta` | DPO temperature parameter (higher = more conservative). | 0.2 |
| `dpo_label_smoothing` | Label smoothing for the DPO loss. | 0.0 |
| `sft_checkpoint_path` | Path to SFT checkpoint for the reference model. If empty, the base model is used. | "" |
| `learning_rate` | Learning rate for the optimizer. | 1e-5 |
| `batch_size` | Number of samples per training batch. | 1 |
| `gradient_accumulation_steps` | Steps to accumulate gradients before updating. | 8 |
| `gradient_checkpointing` | Whether to use gradient checkpointing to save memory. | False |
| `weight_decay` | Weight decay for the optimizer. | 0.0 |
| `num_epochs` | Total number of training epochs. | 2 |
| `max_steps` | Maximum number of optimizer steps (-1 means use `num_epochs`). | -1 |
| `optim` | Optimizer to use for training. | "adamw_torch" |
| `warmup_steps` | Number of learning-rate warmup steps. | 100 |
| `log_level` | Logging verbosity level. | "INFO" |
| `use_wandb` | Whether to enable Weights & Biases logging. | True |
| `wandb_project` | Project name for Weights & Biases logging. | "gemma11-dpo" |
| `wandb_run_name` | Run name for Weights & Biases tracking. | "tt-gemma11-dpo-math" |
| `wandb_tags` | List of tags assigned to the W&B run. | ["gemma11", "dpo", "math"] |
| `wandb_watch_mode` | Watch mode for model parameter logging. | "all" |
| `wandb_log_freq` | Frequency of logging to Weights & Biases (in steps). | 1000 |
| `model_to_wandb` | Whether to store model checkpoint in Weights & Biases. | False |
| `steps_freq` | Frequency (in steps) for performing periodic actions. | 10 |
| `epoch_freq` | Frequency (in epochs) for performing periodic actions. | 1 |
| `val_steps_freq` | Frequency of validation (in steps). | 32 |
| `print_examples` | Whether to print generation examples during training. | False |
| `do_validation` | Whether to run validation during training. | True |
| `resume_from_checkpoint` | Whether to resume training from a previous checkpoint. | False |
| `resume_option` | Resume method (`last`, `best`, or `path`). | "last" |
| `checkpoint_path` | Path to a checkpoint if `resume_option="path"`. | "" |
| `checkpoint_metric` | Metric to monitor for best checkpoint. | "val/accuracy" |
| `checkpoint_metric_mode` | Mode for checkpoint metric (`min` or `max`). | "max" |
| `keep_last_n` | Number of recent checkpoints to keep. | 3 |
| `keep_best_n` | Number of best checkpoints to keep. | 3 |
| `save_strategy` | Strategy for saving checkpoints (`epoch` or `step`). | "step" |
| `save_steps` | Frequency (in steps) for saving checkpoints. | 20 |
| `project_dir` | Directory for experiment outputs. | "blacksmith/experiments/torch/gemma11/dpo" |
| `save_optim` | Whether to save optimizer state. | False |
| `storage_backend` | Storage backend for saving checkpoints. | "local" |
| `sync_to_storage` | Whether to sync checkpoints to remote storage. | False |
| `load_from_storage` | Whether to load checkpoints from remote storage. | False |
| `remote_path` | Remote storage path (if applicable). | "" |
| `seed` | Random seed for reproducibility. | 23 |
| `deterministic` | Whether to enforce deterministic behavior. | False |
| `mesh_shape` | Mesh shape for distributed training. | None |
| `mesh_axis_names` | Axis names for the mesh. | None |
| `lora_r` | Rank of LoRA adaptation matrices. | 16 |
| `lora_alpha` | Scaling factor for LoRA updates. | 32 |
| `lora_target_modules` | Target modules for LoRA adaptation. | ["q_proj", "k_proj", "v_proj", "o_proj"] |
| `lora_task_type` | Training task type for LoRA. | "CAUSAL_LM" |
| `lora_dropout` | Dropout probability for LoRA layers. | 0.0 |
| `adapter_bottleneck_dim` | Bottleneck dimension for adapter layers. | 32 |
| `adapter_non_linearity` | Non-linearity used in adapter layers. | "torch.nn.GELU" |
| `adapter_layers` | Indices of layers to add adapters to. | [] |
| `framework` | Training framework. | "pytorch" |
| `use_tt` | Whether to run on TT device (or GPU otherwise). | True |
