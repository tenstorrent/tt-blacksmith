# Llama with LoRA Experiment

This directory contains the code for the Llama with LoRA fine-tuning experiment.
Llama model specification can be found [here](https://huggingface.co/meta-llama/Llama-3.2-1B).
Original LoRA paper can be found [here](https://arxiv.org/pdf/2106.09685).

## Overview

The LLaMA fine-tuning experiment applies the LoRA technique to adapt a pre-trained LLaMA model on the SST sentiment analysis dataset.
The experiment is designed to run on the Huggingface framework.

## Training

```bash
python3 blacksmith/experiments/torch/llama/test_llama_fine_tuning_pure_torch.py
```

## Data

The Stanford Sentiment Treebank 2 (SST-2) dataset is a widely-used benchmark for binary sentiment classification.
Each example consists of a sentence from movie reviews labeled as either positive or negative sentiment.
This dataset is commonly used to evaluate the performance of natural language understanding models on sentiment analysis tasks.

Source: [Hugging Face Dataset Hub](https://huggingface.co/datasets/stanfordnlp/sst2)

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

| Parameter | Description | Default Value|
| --- | --- | --- |
| `dataset_id` | The dataset used for fine-tuning. | "stanfordnlp/sst2" |
| `model_name` | Name or path of the pre-trained LLaMA model. | "meta-llama/Llama-3.2-1B" |
| `max_length` | Maximum token length for inputs. | 128 |
| `dtype` | Data type used during training. | "torch.bfloat16" |
| `learning_rate` | Learning rate for the optimizer. | 2e-5 |
| `batch_size` | Number of samples per training batch. | 32 |
| `gradient_accumulation_steps` | Number of steps to accumulate gradients before updating. | 1 |
| `gradient_checkpointing` | Whether to use gradient checkpointing to save memory. | False |
| `num_epochs` | Total number of training epochs. | 1 |
| `optim` | Optimizer to use for training. | "adamw_torch" |
| `lora_r` | Rank of the LoRA adaptation matrices. | 4 |
| `lora_alpha` | Scaling factor for the LoRA updates. | 8 |
| `lora_target_modules` | Target modules for applying LoRA adaptation. | "all-linear" |
| `lora_task_type` | Target training task. | "CAUSAL_LM" |
| `seed` | Random seed for reproducibility. | 23 |
| `output_dir` | Directory to save model checkpoints and logs. | "experiments/results/llama32-1b" |
| `wandb_project` | Project name for Weights & Biases logging. | "llama-finetuning" |
| `wandb_run_name` | Project run name for Weights & Biases logging. | "tt-llama" |
| `wandb_watch_mode` | Watch mode for model parameters in wandb. | "all" |
| `wandb_log_freq` | Frequency of logging to wandb (in steps). | 1000 |
| `model_to_wandb` | Whether to store model to wandb. | False |
| `save_strategy` | Strategy for saving checkpoints (epoch, steps, etc.). | "epoch" |
| `logging_steps` | Frequency of logging (in steps). | 10 |
| `do_train` | Whether to run training. | True |
| `use_tt` | Whether to run on TT device (or GPU otherwise). | True |
