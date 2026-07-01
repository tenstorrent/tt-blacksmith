# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gemma 1.1 2B GRPO (Group Relative Policy Optimization) training script.

Based on the GRPO method from "DeepSeekMath: Pushing the Limits of Mathematical
Reasoning in Open Language Models" (https://arxiv.org/pdf/2402.03300) and the toy
example by Luca Massaron:
- https://medium.com/@lucamassaron/training-for-reasoning-with-grpo-881e1819f2df
- https://medium.com/@lucamassaron/training-for-reasoning-with-grpo-part-ii-a-step-by-step-explanation-f80c219e2059

The experiment teaches Gemma 1.1 2B to reason about GSM8K math problems via
reinforcement learning. For each prompt the policy model samples a group of
completions; two rule-based reward functions (format + correctness) score them;
the policy is updated towards the group-relative advantage with a KL penalty
towards the frozen base model. A fresh LoRA adapter is trained on top of the base
instruction-tuned model (no SFT prerequisite).

This first bring-up targets CPU (use_tt: False, use_vllm: False). The heavy
generation + RL loop is delegated to TRL's GRPOTrainer, while the surrounding
config/CLI/logging scaffolding mirrors the other tt-blacksmith experiments.

NOTE: GRPO of a 2B model on CPU is extremely slow and memory hungry (every step
samples and back-propagates through many long completions). The default CPU config
is a lightweight smoke test to validate the pipeline, not a real training run; use
GPU/TT (and ideally vLLM) for actual training.

Model: https://huggingface.co/google/gemma-1.1-2b-it
Dataset: openai/gsm8k
"""
import traceback
from pathlib import Path

import torch
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from blacksmith.datasets.torch.dataset_utils import get_dataset
from blacksmith.experiments.torch.gemma11.grpo.configs import GRPOTrainingConfig
from blacksmith.experiments.torch.gemma11.grpo.grpo_rewards import (
    correctness_reward_func,
    format_reward_func,
)
from blacksmith.tools.cli import generate_config, parse_cli_options
from blacksmith.tools.device_manager import DeviceManager
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.reproducibility_manager import ReproducibilityManager


def build_lora_config(config: GRPOTrainingConfig) -> LoraConfig:
    """Create a fresh LoRA adapter config for the policy model."""
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=config.lora_task_type,
        target_modules=config.lora_target_modules,
    )


def build_grpo_config(config: GRPOTrainingConfig, device: torch.device) -> GRPOConfig:
    """Translate the experiment config into TRL's GRPOConfig."""
    use_cpu = device.type == "cpu"
    dtype = eval(config.dtype)

    # TRL forces device_map="auto" unless one is provided, which makes accelerate
    # try to disk-offload on a CPU box. Pin the model to the target device instead.
    model_init_kwargs = {"dtype": dtype, "device_map": "cpu" if use_cpu else str(device)}

    return GRPOConfig(
        output_dir=str(Path(config.project_dir) / "outputs"),
        # Optimization
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.max_grad_norm,
        optim=config.optim,
        # Batching / schedule
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        num_train_epochs=config.num_epochs,
        max_steps=config.max_steps,
        # GRPO generation
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        temperature=config.temperature,
        beta=config.grpo_beta,
        use_vllm=config.use_vllm,
        # Precision / device (CPU bring-up keeps everything in float32)
        bf16=(not use_cpu) and dtype == torch.bfloat16,
        fp16=(not use_cpu) and dtype == torch.float16,
        use_cpu=use_cpu,
        model_init_kwargs=model_init_kwargs,
        # Logging / checkpointing
        logging_steps=config.logging_steps,
        logging_first_step=config.logging_first_step,
        log_completions=config.log_completions,
        num_completions_to_print=config.num_completions_to_print,
        save_steps=config.save_steps,
        save_total_limit=config.keep_last_n,
        report_to=["wandb"] if config.use_wandb else "none",
        seed=config.seed,
    )


def train_grpo(
    config: GRPOTrainingConfig,
    device_manager: DeviceManager,
    logger: TrainingLogger,
):
    """Run GRPO training for Gemma 1.1 2B on GSM8K using TRL's GRPOTrainer."""
    logger.info("Starting Gemma 1.1 2B GRPO training...")
    logger.info(f"GRPO beta (KL): {config.grpo_beta}")
    logger.info(f"num_generations: {config.num_generations} | temperature: {config.temperature}")

    # The per-device batch must split evenly into groups of `num_generations`
    # completions, otherwise TRL cannot form complete groups for advantage estimation.
    if config.batch_size % config.num_generations != 0:
        raise ValueError(
            f"batch_size ({config.batch_size}) must be a multiple of "
            f"num_generations ({config.num_generations}) for GRPO group formation."
        )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # GRPO trains on prompts: expose the raw HF Dataset (prompt + answer columns).
    train_dataset = get_dataset(config=config, split="train").dataset
    logger.info(f"Loaded {config.dataset_id} dataset. Train samples: {len(train_dataset)}")

    peft_config = build_lora_config(config)
    training_args = build_grpo_config(config, device_manager.device)

    logger.log_model_info(
        {
            "model_name": config.model_name,
            "training_model_type": config.training_model_type,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "num_generations": config.num_generations,
            "grpo_beta": config.grpo_beta,
        }
    )

    # A fresh LoRA adapter is attached to the base model; the frozen base model acts
    # as the GRPO reference policy via the KL term (no separate SFT checkpoint).
    trainer = GRPOTrainer(
        model=config.model_name,
        processing_class=tokenizer,
        reward_funcs=[correctness_reward_func, format_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )

    try:
        trainer.train()

        logger.info("Training complete. Merging LoRA adapter and saving final model...")
        final_model_dir = Path(config.project_dir) / "final_model"
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        logger.info(f"Saved merged model to {final_model_dir}")

        logger.log_summary({"output_dir": training_args.output_dir, "final_model_dir": str(final_model_dir)})

    except Exception as e:
        traceback_str = traceback.format_exc()
        logger.error(f"Training failed with error: {str(e)}", traceback_str)
        raise
    finally:
        logger.finish()


if __name__ == "__main__":
    # Config setup
    default_config = Path(__file__).parent / "single_chip" / "gemma11_gsm8k_grpo.yaml"
    args = parse_cli_options(default_config=default_config)
    config = generate_config(GRPOTrainingConfig, args.config, args.test_config)

    # Reproducibility setup
    repro_manager = ReproducibilityManager(config)
    repro_manager.setup()

    # CPU guardrail: cap intra-op threads so a slow CPU run does not saturate every
    # core and make the host unresponsive.
    if config.cpu_num_threads is not None:
        torch.set_num_threads(config.cpu_num_threads)

    # Logger setup
    logger = TrainingLogger(config, args.test_log_filename_prefix)

    # Device setup
    device_manager = DeviceManager(config)
    logger.info(f"Using device: {device_manager.device}")
    if config.cpu_num_threads is not None:
        logger.info(f"Capped CPU intra-op threads to {config.cpu_num_threads}")

    # Start GRPO training
    train_grpo(config, device_manager, logger)
