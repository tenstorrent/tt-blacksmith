# GPT OSS 20B Expert-Parallel LoRA Experiment

This directory contains the PyTorch GPT OSS fine-tuning experiment using LoRA
and expert parallelism on GPU.
The model specification can be found
[here](https://huggingface.co/openai/gpt-oss-20b).
The original LoRA paper can be found
[here](https://arxiv.org/pdf/2106.09685).

## Overview

This experiment fine-tunes `openai/gpt-oss-20b` for causal language modeling on
the WikiText-2 dataset. The expert bank is sharded across GPUs while the
attention, router, embeddings, norms, and LoRA adapters remain replicated.

## Training

The distributed experiment is configured via
`blacksmith/experiments/torch/gpt_oss/distributed/test_gpt_oss_ep.yaml`.

**GPU Training:**

```bash
torchrun --nproc_per_node=4 blacksmith/experiments/torch/gpt_oss/distributed/test_gpt_oss.py --config blacksmith/experiments/torch/gpt_oss/distributed/test_gpt_oss_ep.yaml
```

## Data

WikiText-2 is a language-modeling benchmark built from curated Wikipedia
articles. This experiment tokenizes the corpus into fixed-length chunks and
trains with the standard causal language-modeling objective.

Source: [Hugging Face Dataset Hub](https://huggingface.co/datasets/wikitext)

## Configuration

### Configuration Parameters

| Parameter | Description | Default Value |
| --- | --- | --- |
| `dataset_id` | Dataset used for training. | `"wikitext"` |
| `model_name` | Hugging Face model identifier. | `"openai/gpt-oss-20b"` |
| `max_length` | Token block length for training samples. | `256` |
| `dtype` | Compute dtype used during training. | `"torch.bfloat16"` |
| `training_type` | Fine-tuning strategy. | `"lora"` |
| `learning_rate` | Optimizer learning rate. | `2e-4` |
| `weight_decay` | AdamW weight decay. | `0.1` |
| `batch_size` | Per-rank batch size. | `1` |
| `gradient_accumulation_steps` | Gradient accumulation steps before optimizer update. | `8` |
| `gradient_checkpointing` | Enables activation checkpointing. | `false` |
| `num_epochs` | Number of training epochs. | `1` |
| `max_grad_norm` | Gradient clipping norm. | `1.0` |
| `lora_r` | LoRA rank. | `16` |
| `lora_alpha` | LoRA scaling factor. | `32` |
| `lora_dropout` | LoRA dropout probability. | `0.05` |
| `lora_target_modules` | Attention modules adapted with LoRA. | `["q_proj", "k_proj", "v_proj", "o_proj"]` |
| `log_level` | Python logging verbosity. | `"INFO"` |
| `use_wandb` | Enables Weights & Biases logging. | `true` |
| `wandb_project` | Weights & Biases project name. | `"gpt-oss-20b-ep"` |
| `wandb_run_name` | Weights & Biases run name. | `"gpt-oss-20b-lora-ep"` |
| `steps_freq` | Frequency of training metric logging in optimizer steps. | `50` |
| `val_steps_freq` | Frequency of validation in optimizer steps. | `10` |
| `resume_from_checkpoint` | Resume training from an existing checkpoint. | `false` |
| `save_strategy` | Checkpoint save cadence. | `"step"` |
| `project_dir` | Output directory for experiment artifacts. | `"blacksmith/experiments/torch/gpt_oss/distributed"` |
| `save_optim` | Save optimizer state in checkpoints. | `false` |
| `storage_backend` | Checkpoint storage backend. | `"local"` |
| `seed` | Reproducibility seed. | `42` |
| `framework` | Training framework identifier. | `"pytorch"` |
| `use_tt` | Whether to target TT devices. | `false` |
