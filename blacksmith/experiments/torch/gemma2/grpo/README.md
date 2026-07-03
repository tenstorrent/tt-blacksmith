# Gemma 2 2B-IT GRPO Experiment

This directory contains the code for training Gemma 2 2B-IT to reason about math
problems using GRPO (Group Relative Policy Optimization), implemented from
scratch (no TRL).

- Gemma 2 2B-IT model specification can be found [here](https://huggingface.co/google/gemma-2-2b-it).
- GRPO is introduced in "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" ([paper](https://arxiv.org/pdf/2402.03300)).
- Setup inspired by [this blog post](https://medium.com/@lucamassaron/training-for-reasoning-with-grpo-part-ii-a-step-by-step-explanation-f80c219e2059).

## Overview

The policy model generates several candidate answers per question, each is scored
by rule-based reward functions (format + correctness), and the policy is updated
to prefer higher-reward answers within each group, regularized by a KL penalty to
a frozen reference model. The underlying fine-tuning approach for the policy model
(LoRA, adapters, or full fine-tuning) is selected via `training_model_type`.

### Three-phase step

Each training step is split into three explicit phases so the compiled model only
ever sees two graph shapes (generation vs. training), avoiding repeated recompiles:

1. Phase A - Generation: sample `num_generations` (G) completions per prompt with
   the policy model (batched autoregressive decode with a `StaticCache`, temperature
   sampling). Completions are kept as token ids.
2. Phase B - Rewards: score every completion on host (`format_reward + correctness_reward`),
   then group-normalize into advantages `A_i = (r_i - mean) / (std + eps)`.
3. Phase C - Optimization: one policy forward+backward and one frozen-reference
   forward over the `prompt + completion` sequences, compute the GRPO loss, and step.

### GRPO loss

Per completion token, the loss maximizes the clipped advantage-weighted ratio and
subtracts a KL penalty using the DeepSeekMath unbiased (k3) estimator:

```text
loss_{i,t} = -( rho_{i,t} * A_i  -  beta * KL[ pi_theta || pi_ref ]_{i,t} )
KL[ pi_theta || pi_ref ] = ( pi_ref / pi_theta ) - log( pi_ref / pi_theta ) - 1
```

With `num_grpo_iterations = 1` (default), the behavior policy equals the current
policy, so the ratio is 1 in value (still differentiable) and the clip is inactive.

## Training

The dataset and hyperparameters are selected through the configuration file passed
via `--config`. If no config is provided, `single_chip/gemma2_gsm8k_grpo.yaml` is
used by default.

### Single Chip Training

```bash
python3 blacksmith/experiments/torch/gemma2/grpo/train.py --config blacksmith/experiments/torch/gemma2/grpo/single_chip/gemma2_gsm8k_grpo.yaml
```

### Smoke test (tiny open model, no gating)

```bash
python3 blacksmith/experiments/torch/gemma2/grpo/train.py --config blacksmith/experiments/torch/gemma2/grpo/single_chip/smoke_smollm_gsm8k_grpo.yaml
```

#### Training Configuration

| Architecture | mesh_shape | mesh_axis_names | dataset | Method |
| ------------ | ---------- | --------------- | ------- | ------ |
| [P150](single_chip/gemma2_gsm8k_grpo.yaml) | None | None | GSM8K | GRPO + LoRA |

## Data

### GSM8K

GSM8K is a dataset of grade-school math word problems, each with a chain-of-thought
solution ending in `#### <integer>`. This experiment uses the questions as prompts
and the parsed integer as the gold answer for the correctness reward.

Source: [Hugging Face Dataset Hub](https://huggingface.co/datasets/openai/gsm8k)

Example:
```
{
  "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did she sell altogether in April and May?",
  "answer": "In May, Natalia sold 48/2 = 24 clips.\nNatalia sold 48+24 = 72 clips altogether.\n#### 72"
}
```

## Configuration

The experiment is configured using `single_chip/gemma2_gsm8k_grpo.yaml`.

### Configuration Parameters

| Parameter | Description | Default Value |
| --- | --- | --- |
| `training_model_type` | Fine-tuning approach for the policy model (`lora`, `adapters`, or full). | "lora" |
| `dataset_id` | The dataset used for training. | "gsm8k" |
| `model_name` | Name or path of the pre-trained Gemma 2 2B model. | "google/gemma-2-2b-it" |
| `dtype` | Data type used during training. | "torch.bfloat16" |
| `ignored_index` | Label id used to mask prompt tokens. | -100 |
| `num_generations` | G: completions sampled per prompt (>=2). | 4 |
| `max_prompt_length` | Max prompt tokens (left-padded). | 256 |
| `max_completion_length` | Max generated tokens per completion. | 200 |
| `temperature` | Sampling temperature (0 = greedy). | 0.5 |
| `top_k` | Top-k sampling cutoff (0 = disabled). | 0 |
| `grpo_beta` | KL penalty coefficient. | 0.005 |
| `grpo_epsilon` | PPO clip range (used when num_grpo_iterations > 1). | 0.2 |
| `num_grpo_iterations` | mu: gradient updates per generated batch. | 1 |
| `advantage_eps` | Stabilizer in advantage normalization. | 1e-4 |
| `format_reward_weight` | Weight of the format reward. | 1.0 |
| `correct_reward_weight` | Weight of the correctness reward. | 2.0 |
| `learning_rate` | Learning rate for the optimizer. | 1e-5 |
| `batch_size` | Prompts per step (P); gen/train batch = P * num_generations. | 1 |
| `gradient_accumulation_steps` | Steps to accumulate gradients before updating. | 4 |
| `gradient_checkpointing` | Whether to use gradient checkpointing. | False |
| `weight_decay` | Weight decay for the optimizer. | 0.1 |
| `num_epochs` | Total number of training epochs. | 2 |
| `max_steps` | Maximum optimizer steps (-1 means use `num_epochs`). | -1 |
| `optim` | Optimizer to use for training. | "adamw_torch" |
| `warmup_steps` | Number of learning-rate warmup steps. | 10 |
| `max_grad_norm` | Gradient clipping norm. | 0.1 |
| `log_level` | Logging verbosity level. | "INFO" |
| `use_wandb` | Whether to enable Weights & Biases logging. | True |
| `wandb_project` | Project name for W&B logging. | "gemma2-grpo" |
| `wandb_run_name` | Run name for W&B tracking. | "tt-gemma2-grpo-gsm8k" |
| `wandb_tags` | List of tags assigned to the W&B run. | ["gemma2", "grpo", "gsm8k", "lora"] |
| `steps_freq` | Frequency (in steps) for logging. | 1 |
| `val_steps_freq` | Frequency of validation (in steps). | 50 |
| `print_examples` | Whether to log a sample completion each logging step. | True |
| `resume_from_checkpoint` | Whether to resume from a previous checkpoint. | False |
| `resume_option` | Resume method (`last`, `best`, or `path`). | "last" |
| `checkpoint_path` | Path to a checkpoint if `resume_option="path"`. | "" |
| `checkpoint_metric` | Metric monitored for best checkpoint. | "train/reward_mean" |
| `checkpoint_metric_mode` | Mode for checkpoint metric (`min` or `max`). | "max" |
| `keep_last_n` | Number of recent checkpoints to keep. | 3 |
| `keep_best_n` | Number of best checkpoints to keep. | 1 |
| `save_strategy` | Strategy for saving checkpoints (`epoch`, `step`, or `none`). | "step" |
| `save_steps` | Frequency (in steps) for saving checkpoints. | 50 |
| `project_dir` | Directory for experiment outputs. | "blacksmith/experiments/torch/gemma2/grpo" |
| `seed` | Random seed for reproducibility. | 23 |
| `deterministic` | Whether to enforce deterministic behavior. | False |
| `mesh_shape` | Mesh shape for distributed training. | None |
| `mesh_axis_names` | Axis names for the mesh. | None |
| `lora_r` | Rank of LoRA adaptation matrices. | 64 |
| `lora_alpha` | Scaling factor for LoRA updates. | 64 |
| `lora_target_modules` | Target modules for LoRA adaptation. | ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] |
| `lora_task_type` | Training task type for LoRA. | "CAUSAL_LM" |
| `lora_dropout` | Dropout probability for LoRA layers. | 0.0 |
| `framework` | Training framework. | "pytorch" |
| `use_tt` | Whether to run on TT device (or GPU/CPU otherwise). | True |
| `do_validation` | Whether to run validation during training. | False |
