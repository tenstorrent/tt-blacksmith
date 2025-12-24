# Gemma 1.1 2B DPO (Direct Preference Optimization) Experiment

This directory contains the code for DPO training of Gemma 1.1 2B.

- Gemma 1.1 2B: [https://huggingface.co/google/gemma-1.1-2b-it](https://huggingface.co/google/gemma-1.1-2b-it)
- DPO paper: [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290)

## Overview

DPO (Direct Preference Optimization) is a method to align language models with human preferences without requiring a separate reward model. Instead of using RLHF, DPO directly optimizes the policy using preference pairs (chosen vs rejected responses).

The DPO loss is:
```
L_DPO = -log(sigmoid(β * (log π(y_w|x) - log π(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x))))
```

Where:
- `y_w` is the chosen (winning) response
- `y_l` is the rejected (losing) response
- `π` is the policy model being trained
- `π_ref` is the reference model (frozen)
- `β` is the temperature parameter

## Standard DPO Pipeline

**Important**: DPO requires a reference model (π_ref) that is typically an SFT model trained on chosen responses.

### Step 1: Train SFT Model on Chosen Responses

First, train a model on the chosen/winning responses using standard supervised fine-tuning. You can use the LoRA experiment in the parent directory for this.

### Step 2: Run DPO with SFT Checkpoint

Set `sft_checkpoint_path` in `test_dpo.yaml` to point to your SFT checkpoint:

```yaml
sft_checkpoint_path: "path/to/sft_checkpoint.pt"
```

Then run DPO:

```bash
python3 blacksmith/experiments/torch/gemma11/dpo/test_dpo.py
```

### Alternative: DPO from Base Model

If no SFT checkpoint is provided, the base pretrained model is used as π_ref. This is simpler but may be less effective than the standard pipeline.

## Dataset

The Math DPO dataset from Argilla contains math problems with preference pairs:
- `instruction`: The math problem/question
- `chosen_response`: The preferred (better quality) response
- `rejected_response`: The less preferred response

Source: [Hugging Face Dataset Hub](https://huggingface.co/datasets/argilla/distilabel-math-preference-dpo)

Example:
```json
{
  "instruction": "How can I simplify the algebraic expression (3x^2 - 4y^3) / (2x)?",
  "chosen_response": "To simplify... [detailed step-by-step solution]",
  "rejected_response": "To simplify... [less detailed solution]",
  "chosen_rating": 9.0,
  "rejected_rating": 7.0
}
```

## Configuration

The experiment is configured using `test_dpo.yaml`.

### Key DPO Parameters

| Parameter | Description | Default |
| --- | --- | --- |
| `training_type` | Training objective | "dpo" |
| `peft_method` | PEFT method to use | "lora" |
| `dpo_beta` | DPO temperature (higher = more conservative) | 0.1 |
| `dpo_label_smoothing` | Label smoothing for DPO loss | 0.0 |
| `dpo_reference_free` | Skip reference model | False |
| `learning_rate` | Learning rate (lower for DPO) | 5e-5 |
| `warmup_steps` | LR warmup steps | 5 |

### LoRA Parameters (when peft_method="lora")

| Parameter | Description | Default |
| --- | --- | --- |
| `lora_r` | LoRA rank (higher for DPO) | 16 |
| `lora_alpha` | LoRA alpha | 32 |
| `lora_target_modules` | Extended target modules | ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] |
| `lora_dropout` | LoRA dropout | 0.05 |

## Training Metrics

During training, the following metrics are logged:
- `dpo/loss`: The DPO loss value
- `dpo/chosen_rewards`: Implicit rewards for chosen responses
- `dpo/rejected_rewards`: Implicit rewards for rejected responses
- `dpo/accuracy`: How often the model prefers chosen over rejected






