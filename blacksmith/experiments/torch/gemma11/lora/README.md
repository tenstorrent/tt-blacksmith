# Gemma 1.1 2B LoRA Fine-tuning Experiment

This directory contains the code for LoRA fine-tuning of Gemma 1.1 2B.

- Gemma 1.1 2B: [https://huggingface.co/google/gemma-1.1-2b-it](https://huggingface.co/google/gemma-1.1-2b-it)
- LoRA paper: [Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685)

## Overview

The Gemma 1.1 2B fine-tuning experiment applies the LoRA technique to adapt a pre-trained Gemma 1.1 2B model on various datasets.

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that freezes the pre-trained model weights and injects trainable low-rank matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters.

## Available Configurations

- `test_lora_sst2.yaml` - SST2 sentiment analysis dataset
- `test_lora_squadV2.yaml` - SQuADv2 question answering dataset
- `test_lora_math_sft.yaml` - Math SFT (Stage 1 of DPO pipeline, trains on chosen responses)

## Training

### SST2 Dataset
```bash
python3 blacksmith/experiments/torch/gemma11/lora/test_lora.py
```

### SQuADv2 Dataset
```bash
python3 blacksmith/experiments/torch/gemma11/lora/test_lora.py --config test_lora_squadV2.yaml
```

### Math SFT (for DPO Pipeline)
This is the first stage of the DPO pipeline - trains on chosen responses to create the reference model (π_ref).
```bash
python3 blacksmith/experiments/torch/gemma11/lora/test_lora.py --config test_lora_math_sft.yaml
```
After training, use the checkpoint path as `sft_checkpoint_path` in the DPO config.

## LoRA Configuration

Key LoRA parameters in the config:
- `lora_r`: Rank of the low-rank matrices (default: 4)
- `lora_alpha`: Scaling factor (default: 8)
- `lora_target_modules`: Which modules to apply LoRA to (default: ["q_proj", "v_proj"])
- `lora_task_type`: Task type for LoRA (default: "CAUSAL_LM")

