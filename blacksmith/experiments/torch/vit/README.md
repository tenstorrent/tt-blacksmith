# ViT with LoRA fine-tuning

This directory contains the code for the ViT (Vision Transformer) model with LoRA fine-tuning experiment.
ViT model specification can be found [here](https://huggingface.co/google/vit-base-patch16-224).
Original ViT paper can be found [here](https://arxiv.org/pdf/2010.11929).
Original LoRA paper can be found [here](https://arxiv.org/pdf/2106.09685).

## Overview

The ViT fine-tuning experiment applies the LoRA fine-tuning technique to adapt a pre-trained ViT model on the StanfordCars dataset.
The experiment is designed to run on the Huggingface framework.

## Training

```bash
python3 blacksmith/experiments/torch/vit/test_vit_finetuning.py --config blacksmith/experiments/torch/vit/test_vit_stanfordcars.yaml
```

## Data

### StanfordCars

Stanford Cars is a fine-grained image classification dataset focused on car recognition. The task is to predict the model of a car.

Source: [Hugging Face Dataset Hub](https://huggingface.co/datasets/tanganke/stanford_cars)

### Configuration

| Parameter | Description | Default Value |
| --- | --- | --- |
| `dataset_id` | The dataset used for fine-tuning. | "stanfordcars" |
| `image_size` | Size of the image after preprocessing. (Unconfigurable for ViT experiment) | 224 |
| `image_mean` | The mean used for RGB channels of the image. (Unconfigurable for ViT experiment) | [0.5, 0.5, 0.5] |
| `image_std` | The std used for RGB channels of the image. (Unconfigurable for ViT experiment) | [0.5, 0.5, 0.5] |
| `model_name` | Name or path of the pre-trained ViT model. | "google/vit-base-patch16-224" |
| `dtype` | Data type used during training. | "torch.bfloat16" |
| `ignored_index` | Index to ignore in loss calculation. | -100 |
| `learning_rate` | Learning rate for the optimizer. | 1e-3 |
| `batch_size` | Number of samples per training batch. | 10 |
| `num_epochs` | Total number of training epochs. | 8 |
| `loss_fn` | Loss function to use. | "torch.nn.CrossEntropyLoss" |
| `log_level` | Logging verbosity level. | "INFO" |
| `use_wandb` | Whether to enable Weights & Biases logging. | True |
| `wandb_project` | Project name for Weights & Biases logging. | "vit-finetuning" |
| `wandb_run_name` | Run name for Weights & Biases tracking. | "tt-vit-stanfordcars" |
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
| `project_dir` | Directory for experiment outputs. | "blacksmith/experiments/torch/vit" |
| `save_optim` | Whether to save optimizer state. | False |
| `storage_backend` | Storage backend for saving checkpoints. | "local" |
| `sync_to_storage` | Whether to sync checkpoints to remote storage. | False |
| `load_from_storage` | Whether to load checkpoints from remote storage. | False |
| `remote_path` | Remote storage path (if applicable). | "" |
| `seed` | Random seed for reproducibility. | 23 |
| `deterministic` | Whether to enforce deterministic behavior. | False |
| `parallelism_strategy` | Parallelism strategy (`single`, `data_parallel`, or `tensor_parallel`). | "single" |
| `mesh_shape` | Mesh shape for parallelism (used if `parallelism_strategy != single`). | "1,2" |
| `tp_sharding_specs` | Tensor parallel sharding specifications. | {} |
| `lora_r` | Rank of LoRA adaptation matrices. | 4 |
| `lora_alpha` | Scaling factor for LoRA updates. | 8 |
| `lora_target_modules` | Target modules for LoRA adaptation. | ["all-linear"] |
| `lora_dropout` | Dropout probability for LoRA layers. | 0.1 |
| `framework` | Training framework. | "pytorch" |
| `use_tt` | Whether to run on TT device (or GPU otherwise). | False |
