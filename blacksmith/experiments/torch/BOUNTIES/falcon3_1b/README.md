# Falcon3-1B with LoRA Experiment

This directory contains the code for the Falcon3-1B-Base model with LoRA fine-tuning experiment.
Falcon3 model specification can be found [here](https://huggingface.co/tiiuae/Falcon3-1B-Base).
Original LoRA paper can be found [here](https://arxiv.org/pdf/2106.09685).

## Overview

The Falcon3-1B fine-tuning experiment applies the LoRA technique to adapt a pre-trained Falcon3-1B-Base model on the Wikitext-2 dataset for causal language modeling.
The experiment is designed to run on TT-N150 hardware using the TT-XLA framework.

### Embedding Unfreezing

Falcon3-1B-Base is pre-trained on a limited set of languages (English, French, Portuguese, and Spanish).
The Wikitext-2 dataset contains tokens from other languages (e.g., Japanese text in articles like Valkyria Chronicles)
that the frozen embedding layer cannot represent well. To address this, the experiment unfreezes the embedding layer
(`embed_tokens`) alongside LoRA adapters using PEFT's `modules_to_save` mechanism. This allows the model to adapt
its token representations during fine-tuning, significantly improving loss convergence.

## Training

```bash
python3 blacksmith/experiments/torch/BOUNTIES/falcon3_1b/test_falcon3_finetuning.py --config blacksmith/experiments/torch/BOUNTIES/falcon3_1b/test_falcon3_finetuning.yaml
```

For CPU baseline testing, set `use_tt: False` in the config file.

## Data

Wikitext-2 is a collection of over 100 million tokens extracted from the set of verified Good and Featured articles on Wikipedia. The task is causal language modeling - predicting the next token given previous context.

Source: [Hugging Face Dataset Hub](https://huggingface.co/datasets/Salesforce/wikitext)

Example:
```
{
  "text": " = Valkyria Chronicles III = \n Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it is the third game in the Valkyria series . ..."
}
```
- text: Raw text from Wikipedia articles containing full paragraphs of content.

## Configuration

The experiment is configured using the configuration file `test_falcon3_finetuning.yaml`. The configuration file specifies the hyperparameters for the experiment, such as the number of epochs, the batch size, and the LoRA configuration.

Current `test_falcon3_finetuning.yaml` has the recommended and tested hyperparameters for the experiment.

### Configuration Parameters

| Parameter | Description | Default Value |
| --- | --- | --- |
| `dataset_id` | The dataset used for fine-tuning. | "wikitext" |
| `model_name` | Name or path of the pre-trained Falcon3 model. | "tiiuae/Falcon3-1B-Base" |
| `max_length` | Maximum token length for inputs. | 128 |
| `dtype` | Data type used during training. | "torch.bfloat16" |
| `learning_rate` | Learning rate for the optimizer. | 5e-5 |
| `batch_size` | Number of samples per training batch. | 4 |
| `gradient_checkpointing` | Whether to use gradient checkpointing to save memory. | False |
| `num_epochs` | Total number of training epochs. | 3 |
| `optim` | Optimizer to use for training. | "adamw_torch" |
| `log_level` | Logging verbosity level. | "INFO" |
| `use_wandb` | Whether to enable Weights & Biases logging. | True |
| `wandb_project` | Project name for Weights & Biases logging. | "falcon3-finetuning" |
| `wandb_run_name` | Run name for Weights & Biases tracking. | "tt-falcon3-wikitext" |
| `wandb_tags` | List of tags assigned to the W&B run. | ["falcon3", "lora", "wikitext"] |
| `wandb_watch_mode` | Watch mode for model parameter logging. | "all" |
| `wandb_log_freq` | Frequency of logging to Weights & Biases (in steps). | 100 |
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
| `project_dir` | Directory for experiment outputs. | "blacksmith/experiments/torch/BOUNTIES/falcon3_1b" |
| `save_optim` | Whether to save optimizer state. | False |
| `storage_backend` | Storage backend for saving checkpoints. | "local" |
| `sync_to_storage` | Whether to sync checkpoints to remote storage. | False |
| `load_from_storage` | Whether to load checkpoints from remote storage. | False |
| `remote_path` | Remote storage path (if applicable). | "" |
| `seed` | Random seed for reproducibility. | 42 |
| `deterministic` | Whether to enforce deterministic behavior. | False |
| `lora_r` | Rank of LoRA adaptation matrices. | 32 |
| `lora_alpha` | Scaling factor for LoRA updates. | 64 |
| `lora_target_modules` | Target modules for LoRA adaptation. | ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] |
| `lora_task_type` | Training task type for LoRA. | "CAUSAL_LM" |
| `unfreeze_embeddings` | Unfreeze embedding layer for better token adaptation. | True |
| `framework` | Training framework. | "pytorch" |
| `use_tt` | Whether to run on TT device (or CPU otherwise). | True |
