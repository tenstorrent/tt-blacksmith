# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Configuration for GRPO (Group Relative Policy Optimization) training.

Based on the GRPO method from "DeepSeekMath: Pushing the Limits of Mathematical
Reasoning in Open Language Models" (https://arxiv.org/pdf/2402.03300).

This experiment wraps TRL's ``GRPOTrainer``: the policy model samples a group of
completions per prompt, rule-based reward functions score them, and the policy is
updated towards the group-relative advantage with a KL penalty to the frozen base
model. A fresh LoRA adapter is attached to the base model (no SFT prerequisite).
"""
from typing import Optional

from pydantic import BaseModel, Field

from blacksmith.tools.test_config import TestConfig


class GRPOTrainingConfig(BaseModel):
    """Configuration for GRPO training (Gemma 1.1 2B on GSM8K)."""

    # Fine-tuning approach for the policy model. "lora" attaches a fresh adapter to
    # the base model; any other value falls back to full fine-tuning.
    training_model_type: str = Field(default="lora")

    # Dataset settings
    dataset_id: str = Field(default="gsm8k")

    # Model settings
    model_name: str = Field(default="google/gemma-1.1-2b-it")
    # float32 is the safe default for CPU bring-up; switch to bfloat16 on GPU/TT.
    dtype: str = Field(default="torch.float32")

    # GRPO / generation hyperparameters
    grpo_beta: float = Field(default=0.005, ge=0, description="KL coefficient towards the frozen base policy")
    num_generations: int = Field(default=2, gt=1, description="Completions sampled per prompt (group size)")
    temperature: float = Field(default=0.5, gt=0, description="Sampling temperature for generation")
    max_completion_length: int = Field(default=256, gt=0)
    use_vllm: bool = Field(default=False, description="Use vLLM for generation (GPU only; keep False on CPU)")

    # Training hyperparameters
    learning_rate: float = Field(default=1e-5, gt=0)
    batch_size: int = Field(default=2, gt=0, description="per_device batch size; must be a multiple of num_generations")
    gradient_accumulation_steps: int = Field(default=4, gt=0)
    gradient_checkpointing: bool = Field(default=False)
    weight_decay: float = Field(default=0.1, ge=0)
    num_epochs: int = Field(default=1, gt=0)
    max_steps: int = Field(default=-1, description="-1 means use num_epochs")
    optim: str = Field(default="adamw_torch")
    adam_beta1: float = Field(default=0.9, gt=0, lt=1)
    adam_beta2: float = Field(default=0.99, gt=0, lt=1)
    warmup_ratio: float = Field(default=0.1, ge=0, le=1)
    max_grad_norm: float = Field(default=0.1, gt=0)
    lr_scheduler_type: str = Field(default="cosine")

    # CPU guardrail: cap the intra-op thread count so a slow CPU run does not
    # saturate every core and freeze the host. None leaves PyTorch's default.
    cpu_num_threads: Optional[int] = Field(default=None, gt=0)

    # Logging settings
    log_level: str = Field(default="INFO")
    use_wandb: bool = Field(default=False)
    wandb_project: str = Field(default="gemma11-grpo")
    wandb_run_name: str = Field(default="gemma11-grpo-gsm8k")
    wandb_tags: list[str] = Field(default_factory=lambda: ["gemma11", "grpo", "gsm8k", "lora"])
    wandb_watch_mode: str = Field(default="all")
    wandb_log_freq: int = Field(default=1000)
    model_to_wandb: bool = Field(default=False)
    logging_steps: int = Field(default=1, gt=0)
    # Log a metrics line for the very first step instead of waiting for the first
    # `logging_steps` boundary (useful on slow CPU runs to confirm progress early).
    logging_first_step: bool = Field(default=True)
    # When True, TRL prints a table of sampled completions + rewards every logging
    # step, so you can see what the model is generating during training.
    log_completions: bool = Field(default=False)
    # How many completions to print when `log_completions` is enabled (None = TRL default).
    num_completions_to_print: Optional[int] = Field(default=None, gt=0)

    # Checkpoint settings (TRL manages the training loop and writes to project_dir)
    save_steps: int = Field(default=50, gt=0)
    keep_last_n: int = Field(default=3, ge=0)
    project_dir: str = Field(default="blacksmith/experiments/torch/gemma11/grpo")

    # Reproducibility settings
    seed: int = Field(default=23)
    deterministic: bool = Field(default=False)

    # LoRA setup (blog defaults: r == alpha so the effective LoRA scaling is 1.0)
    lora_r: int = Field(default=64, gt=0)
    lora_alpha: int = Field(default=64, gt=0)
    lora_dropout: float = Field(default=0.0, ge=0, le=1)
    lora_task_type: str = Field(default="CAUSAL_LM")
    lora_target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # Other settings
    framework: str = Field(default="pytorch")
    use_tt: bool = Field(default=False, description="CPU/GPU first; TT enablement is future work")

    # Testing utils (used to limit training duration during CI runs).
    test_config: Optional[TestConfig] = Field(default=None)
