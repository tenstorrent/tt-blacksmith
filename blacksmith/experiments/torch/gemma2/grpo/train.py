# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gemma 2 2B-IT GRPO (Group Relative Policy Optimization) training script.

Based on the paper: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning
in Open Language Models" https://arxiv.org/pdf/2402.03300

Trains google/gemma-2-2b-it to reason about GSM8K math problems using GRPO,
implemented from scratch (no TRL). Each step is split into three explicit phases
so the compiled model only ever sees two graph shapes (generation vs. training):

    Phase A (generation): sample G completions per prompt with the policy model.
    Phase B (rewards):     score each completion, then group-normalize into advantages.
    Phase C (optimization): one policy fwd/bwd + one frozen-reference fwd, GRPO loss.

Model: https://huggingface.co/google/gemma-2-2b-it
Dataset: https://huggingface.co/datasets/openai/gsm8k
"""
import traceback
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch_xla
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from blacksmith.datasets.torch.dataset_utils import get_dataset
from blacksmith.experiments.torch.gemma2.grpo.configs import GRPOTrainingConfig
from blacksmith.models.torch.huggingface.hf_models import get_model
from blacksmith.tools.checkpoints_manager import CheckpointManager
from blacksmith.tools.cli import generate_config, parse_cli_options
from blacksmith.tools.device_manager import DeviceManager
from blacksmith.tools.grpo_utils import (
    compute_group_advantages,
    compute_grpo_loss,
    compute_rewards,
    get_per_token_logps,
)
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.reproducibility_manager import ReproducibilityManager
from blacksmith.tools.torch_helpers import (
    accumulate_metric_tensors,
    average_metric_tensors,
    generate_completions,
)


def build_training_batch(
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    completion_ids: torch.Tensor,
    completion_valid: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble the fixed-shape (prompt + completion) batch for Phase C.

    Returns ``(seq_ids, seq_attention_mask, completion_token_mask)``:
      - ``seq_ids`` (B, Lp+Lc): left-padded prompt followed by the completion.
      - ``seq_attention_mask`` (B, Lp+Lc): prompt padding mask + completion validity.
      - ``completion_token_mask`` (B, Lp+Lc): True only on real completion tokens
        (prompt and trailing pad are False). Shift by one in the loss to align
        with per-token log-probs.
    """
    seq_ids = torch.cat([prompt_input_ids, completion_ids], dim=1)
    seq_attention_mask = torch.cat([prompt_attention_mask, completion_valid.long()], dim=1)
    prompt_token_mask = torch.zeros_like(prompt_attention_mask, dtype=torch.bool)
    completion_token_mask = torch.cat([prompt_token_mask, completion_valid], dim=1)
    return seq_ids, seq_attention_mask, completion_token_mask


def _forward_logps(model, seq_ids: torch.Tensor, seq_attention_mask: torch.Tensor) -> torch.Tensor:
    """Run one full-sequence forward and return per-token log-probs (B, T-1).

    The (B, T, V) logits are dropped immediately after reduction to keep only the
    small per-token tensor (avoids holding a vocab-sized activation).
    """
    logits = model(input_ids=seq_ids, attention_mask=seq_attention_mask).logits
    logps = get_per_token_logps(logits, seq_ids)
    del logits
    return logps


def train_grpo(
    config: GRPOTrainingConfig,
    device_manager: DeviceManager,
    logger: TrainingLogger,
    checkpoint_manager: CheckpointManager,
):
    """Main GRPO training loop for Gemma 2 2B-IT on GSM8K."""
    device = device_manager.device
    num_generations = config.num_generations

    logger.info("Starting Gemma 2 2B-IT GRPO training...")
    logger.info(f"GRPO beta (KL coeff): {config.grpo_beta} | num_generations: {num_generations}")
    logger.info(f"temperature: {config.temperature} | max_completion_length: {config.max_completion_length}")

    # Tokenizer + raw model config (used to size the generation StaticCache).
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side="left", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_config = AutoConfig.from_pretrained(config.model_name)

    # Policy model (trainable, with PEFT + torch.compile).
    policy_model = get_model(config, device)
    total_params = sum(p.numel() for p in policy_model.parameters())
    trainable_params = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
    logger.info(f"Loaded {config.model_name} as policy model.")
    logger.info(f"Policy parameters: {total_params} | trainable: {trainable_params}")

    if config.resume_from_checkpoint:
        logger.info("Loading policy model from resume checkpoint.")
        checkpoint_manager.load_checkpoint(policy_model)

    # Reference model (pi_ref) for the KL penalty. For LoRA we reuse the policy
    # model with its adapters temporarily disabled, which IS the frozen base model
    # and avoids holding a second full copy of the weights on device (halves the
    # model memory - important for the 2B model + 256k-vocab logits). For other
    # fine-tuning modes we fall back to a separate frozen copy.
    use_shared_reference = config.training_model_type == "lora"
    reference_model = None
    if use_shared_reference:
        logger.info("Using policy model with adapters disabled as the reference (shared weights).")
    else:
        reference_model = get_model(config, device)
        for param in reference_model.parameters():
            param.requires_grad_(False)
        reference_model.eval()
        logger.info("Reference model loaded and frozen.")

    def compute_ref_logps(seq_ids, seq_attention_mask):
        with torch.no_grad():
            if use_shared_reference:
                with policy_model.disable_adapter():
                    return _forward_logps(policy_model, seq_ids, seq_attention_mask)
            return _forward_logps(reference_model, seq_ids, seq_attention_mask)

    logger.log_model_info(
        {
            "model_name": config.model_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "grpo_beta": config.grpo_beta,
            "num_generations": num_generations,
            "temperature": config.temperature,
        }
    )

    # GSM8K prompts (prompt-only dataset).
    train_dataset = get_dataset(config=config, split="train")
    train_dataloader = train_dataset.get_dataloader()
    logger.info(f"Loaded {config.dataset_id} dataset. Train prompts: {len(train_dataset)}")

    optimizer = torch.optim.AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        capturable=config.use_tt,
    )
    optimizer.zero_grad()

    # Per-step device sync used to bound the generation graph (no host transfer).
    generation_sync = (lambda: torch_xla.sync(wait=True)) if config.use_tt else None

    global_step = 0
    accumulation_step = 0
    running_metrics: Dict[str, torch.Tensor] = {
        "loss": None,
        "kl": None,
        "reward_mean": None,
        "reward_std": None,
        "format_frac": None,
        "correct_frac": None,
    }
    last_step_metrics: Dict[str, float] = {}

    try:
        for epoch in range(config.num_epochs):
            logger.info(f"\n=== Epoch {epoch + 1}/{config.num_epochs} ===")
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

            for batch in progress_bar:
                if config.max_steps > 0 and global_step >= config.max_steps:
                    logger.info(f"Reached max_steps ({config.max_steps}). Stopping training.")
                    break

                num_prompts = batch["prompt_input_ids"].shape[0]

                # Repeat each prompt G times so consecutive rows form one group.
                prompt_ids = batch["prompt_input_ids"].repeat_interleave(num_generations, dim=0)
                prompt_mask = batch["prompt_attention_mask"].repeat_interleave(num_generations, dim=0)
                golds = [g for g in batch["gold_answers"] for _ in range(num_generations)]

                # ---- Phase A: generation (no grad) ----
                policy_model.eval()
                completion_ids, completion_valid = generate_completions(
                    model=policy_model,
                    model_config=model_config,
                    prompt_input_ids=prompt_ids,
                    prompt_attention_mask=prompt_mask,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_prompt_length=config.max_prompt_length,
                    max_completion_length=config.max_completion_length,
                    device=device,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    dtype=eval(config.dtype),
                    sync_fn=generation_sync,
                )
                policy_model.train()

                # ---- Phase B: rewards + group-relative advantages ----
                completions_text = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
                rewards, format_flags, correct_flags = compute_rewards(
                    completions_text,
                    golds,
                    format_weight=config.format_reward_weight,
                    correct_weight=config.correct_reward_weight,
                )
                advantages = compute_group_advantages(
                    rewards, num_prompts, num_generations, eps=config.advantage_eps
                ).to(device)

                # ---- Phase C: policy optimization ----
                seq_ids, seq_attention_mask, completion_token_mask = build_training_batch(
                    prompt_ids, prompt_mask, completion_ids, completion_valid
                )
                seq_ids = seq_ids.to(device)
                seq_attention_mask = seq_attention_mask.to(device)
                # Shift completion mask to align with per-token log-probs (targets are seq[:, 1:]).
                completion_mask = completion_token_mask[:, 1:].to(device)

                # Run the frozen reference forward first (no_grad): its full-vocab
                # logits are freed immediately, so they don't coexist with the
                # policy forward's activations that must stay resident for backward.
                # This lowers peak device memory (the (B, T, vocab) logits dominate).
                ref_logps = compute_ref_logps(seq_ids, seq_attention_mask)
                logps = _forward_logps(policy_model, seq_ids, seq_attention_mask)

                loss, loss_metrics = compute_grpo_loss(
                    logps=logps,
                    ref_logps=ref_logps,
                    completion_mask=completion_mask,
                    advantages=advantages,
                    beta=config.grpo_beta,
                    epsilon=config.grpo_epsilon,
                )

                (loss / config.gradient_accumulation_steps).backward()
                accumulation_step += 1
                if config.use_tt:
                    torch_xla.sync(wait=True)

                # Accumulate metrics (all reduced to scalar tensors).
                reward_std = rewards.view(num_prompts, num_generations).std(dim=1).mean()
                accumulate_metric_tensors(
                    running_metrics,
                    {
                        "loss": loss_metrics["loss"],
                        "kl": loss_metrics["kl"],
                        "reward_mean": rewards.mean(),
                        "reward_std": reward_std,
                        "format_frac": format_flags.mean(),
                        "correct_frac": correct_flags.mean(),
                    },
                )

                # Keep one decoded completion (a plain string) for logging before the
                # underlying tensors are freed below.
                sample_completion = completions_text[0] if completions_text else ""

                # Explicitly drop per-iteration tensors so they don't pin device/host
                # memory across the gradient-accumulation window. Placed before the
                # early-continue below so it runs on every iteration. The metrics were
                # already accumulated (as detached scalars) just above.
                del prompt_ids, prompt_mask, golds
                del completion_ids, completion_valid, completions_text
                del rewards, format_flags, correct_flags, reward_std, advantages
                del seq_ids, seq_attention_mask, completion_token_mask, completion_mask
                del ref_logps, logps, loss, loss_metrics

                if accumulation_step < config.gradient_accumulation_steps:
                    continue

                # Optimizer step after accumulating gradients.
                torch.nn.utils.clip_grad_norm_(
                    [p for p in policy_model.parameters() if p.requires_grad], max_norm=config.max_grad_norm
                )
                device_manager.optimizer_step(optimizer)
                optimizer.zero_grad()
                accumulation_step = 0
                global_step += 1

                step_metrics = {}
                if global_step % config.steps_freq == 0:
                    divisor = config.steps_freq * config.gradient_accumulation_steps
                    avg = {f"train/{k}": v for k, v in average_metric_tensors(running_metrics, divisor).items()}
                    avg["train/learning_rate"] = config.learning_rate
                    avg["train/epoch"] = epoch + 1
                    last_step_metrics = avg
                    step_metrics.update(avg)

                    logger.info(
                        f"[Step {global_step}] loss: {avg['train/loss']:.4f} | kl: {avg['train/kl']:.4f} | "
                        f"reward: {avg['train/reward_mean']:.3f} | correct: {avg['train/correct_frac']:.3f}"
                    )
                    progress_bar.set_postfix(
                        {
                            "loss": f"{avg['train/loss']:.3f}",
                            "reward": f"{avg['train/reward_mean']:.3f}",
                            "correct": f"{avg['train/correct_frac']:.3f}",
                        }
                    )
                    if config.print_examples and sample_completion:
                        logger.info(f"Sample completion: {sample_completion!r}")

                    for key in running_metrics:
                        running_metrics[key] = None

                if step_metrics:
                    logger.log_metrics(step_metrics, commit=True, step=global_step)

                if global_step % config.save_steps == 0 and checkpoint_manager.should_save_checkpoint(global_step):
                    checkpoint_manager.save_checkpoint(
                        policy_model, global_step, epoch, optimizer, metrics=last_step_metrics
                    )

            # Discard leftover gradients from a partial accumulation window at epoch end.
            if accumulation_step > 0:
                optimizer.zero_grad()
                accumulation_step = 0

            if config.max_steps > 0 and global_step >= config.max_steps:
                break

            if checkpoint_manager.should_save_checkpoint(global_step, epoch):
                checkpoint_manager.save_checkpoint(
                    policy_model, global_step, epoch, optimizer, metrics=last_step_metrics
                )

        logger.info("Training complete. Saving final model...")
        final_model_path = checkpoint_manager.save_checkpoint(
            policy_model,
            global_step,
            epoch,
            optimizer,
            metrics=last_step_metrics,
            checkpoint_name="final_model.pth",
        )
        logger.log_artifact(final_model_path, artifact_type="model", name="final_model.pth")
        logger.log_summary({"total_steps": global_step, "final_epoch": epoch + 1, **last_step_metrics})
        logger.info(f"GRPO training completed. Total steps: {global_step}")

    except Exception as e:
        traceback_str = traceback.format_exc()
        logger.error(f"Training failed with error: {str(e)}", traceback_str)
        raise
    finally:
        logger.finish()


if __name__ == "__main__":
    default_config = Path(__file__).parent / "single_chip" / "gemma2_gsm8k_grpo.yaml"
    args = parse_cli_options(default_config=default_config)
    config = generate_config(GRPOTrainingConfig, args.config, args.test_config, args.test_checkpoint_path)

    repro_manager = ReproducibilityManager(config)
    repro_manager.setup()

    logger = TrainingLogger(config, args.test_log_filename_prefix)

    device_manager = DeviceManager(config)
    logger.info(f"Using device: {device_manager.device}")

    checkpoint_manager = CheckpointManager(config, logger, device_manager.device)

    train_grpo(config, device_manager, logger, checkpoint_manager)
