# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import traceback
from pathlib import Path

import torch
import torch_xla
import torch_xla.runtime as xr
from tqdm import tqdm

from blacksmith.datasets.torch.dataset_utils import get_dataset
from blacksmith.experiments.torch.gpt_oss.configs import TrainingConfig
from blacksmith.models.torch.gpt_oss_overrides import get_gpt_oss_model
from blacksmith.tools.checkpoints_manager import CheckpointManager
from blacksmith.tools.cli import generate_config, parse_cli_options
from blacksmith.tools.device_manager import DeviceManager
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.reproducibility_manager import ReproducibilityManager
from blacksmith.tools.torch_helpers import (
    collate_fn_for_causal_lm,
)
from blacksmith.tools.workaround_utils import cross_entropy_loss, transform_labels


def print_all_gradients(model, global_step, accumulation_step):
    print(f"\n{'='*80}", flush=True)
    print(f"GRADIENTS at global_step={global_step}, accumulation_step={accumulation_step}", flush=True)
    print(f"{'='*80}", flush=True)
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                grad = param.grad.float()
                g_norm = grad.norm().item()
                total_norm += g_norm ** 2
                num_zeros = (grad == 0).sum().item()
                numel = grad.numel()
                print(
                    f"  {name}: shape={list(grad.shape)}, "
                    f"min={grad.min().item():.6e}, max={grad.max().item():.6e}, "
                    f"mean={grad.mean().item():.6e}, std={grad.std().item():.6e}, "
                    f"norm={g_norm:.6e}, zeros={num_zeros}/{numel}",
                    flush=True,
                )
            else:
                print(f"  {name}: grad=None", flush=True)
    total_norm = total_norm ** 0.5
    print(f"  TOTAL grad norm: {total_norm:.6e}", flush=True)
    print(f"{'='*80}\n", flush=True)


# Training step extracted into a separate function to keep large vocab-sized
# tensors (e.g. logits) scoped locally. This ensures they do not propagate beyond
# the step via the computation graph, avoiding unnecessary and expensive
# CCLs in multi-chip setups.
# Issue itself should be investigated further.
def training_step_inner(batch, model, loss_fn, gradient_accumulation_steps):
    output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = output.logits
    shift_logits = logits[:, :-1, :].contiguous()
    loss = loss_fn(shift_logits, batch["expected_output"], batch["labels_mask"])
    # Scale loss by number of accumulation steps to get correct effective batch size.
    scaled_loss = loss / gradient_accumulation_steps
    scaled_loss.backward()
    return loss.detach()


def train(
    config: TrainingConfig,
    device_manager: DeviceManager,
    logger: TrainingLogger,
    checkpoint_manager: CheckpointManager,
):
    logger.info("Starting training (no validation)...")

    # Load model.
    model = get_gpt_oss_model(config, device_manager.device)
    logger.info(f"Loaded {config.model_name} model.")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)

    # Load checkpoint if needed.
    if config.resume_from_checkpoint:
        checkpoint_manager.load_checkpoint(model, optimizer)

    # Load dataset.
    train_dataset = get_dataset(config=config, split="train", collate_fn=collate_fn_for_causal_lm)
    train_dataloader = train_dataset.get_dataloader()
    logger.info(f"Loaded {config.dataset_id} dataset. Train dataset size: {len(train_dataloader)*config.batch_size}")

    global_step = 0
    running_loss = 0.0
    backward_hooks = []

    try:
        model.train()
        device_manager.shard_model(model)

        for epoch in range(config.num_epochs):
            accumulation_step = 0

            for batch in tqdm(train_dataloader, desc="Training"):
                # Zero out gradients at the start of accumulation cycle
                if accumulation_step == 0:
                    optimizer.zero_grad()

                # TODO: Refactor when https://github.com/tenstorrent/tt-blacksmith/issues/327 is resolved.
                expected_output, labels_mask = transform_labels(
                    batch["labels"], config.ignored_index, model.model.config.vocab_size
                )
                batch = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "expected_output": expected_output,
                    "labels_mask": labels_mask,
                }
                # Shard batch if data parallelism is used.
                batch = device_manager.prepare_batch(batch)
                # Shard model if tensor parallelism is used.
                #device_manager.shard_model(model)

                # Register backward hooks on the 6th micro-step to debug gradient explosion.
                #if accumulation_step == 5 and global_step == 0:
                #    def make_bwd_hook(layer_idx):
                    #        def hook(module, grad_input, grad_output):
                    #            print(f"\n--- BWD Layer {layer_idx} ---", flush=True)
                    #            for i, g in enumerate(grad_output):
                #                if g is not None:
                #                    gf = g.float()
                #                    print(
                #                        f"  grad_output[{i}]: shape={list(g.shape)}, "
                #                        f"norm={gf.norm().item():.6e}, "
                #                        f"min={gf.min().item():.6e}, max={gf.max().item():.6e}",
                #                        flush=True,
                #                    )
                #        for i, g in enumerate(grad_input):
                #            if g is not None:
                #                gf = g.float()
                #                print(

                #    unwrapped = model._orig_mod if hasattr(model, "_orig_mod") else model
                #    for i, layer in enumerate(unwrapped.base_model.model.model.layers):
                #        backward_hooks.append(layer.register_full_backward_hook(make_bwd_hook(i)))

                # Training step.
                loss_ = training_step_inner(batch, model, cross_entropy_loss, config.gradient_accumulation_steps)

                if config.use_tt:
                    torch_xla.sync(wait=True)

                running_loss += loss_.item()
                accumulation_step += 1

                #if accumulation_step == 6 and global_step == 0 and backward_hooks:
                #    for h in backward_hooks:
                #        h.remove()
                #    backward_hooks = []

                print(f"Current loss and step: {loss_.item()} {global_step}", flush=True)

                # Print all gradients after each micro-step.
                print_all_gradients(model, global_step, accumulation_step)

                # Only step the optimizer after accumulating gradients.
                if accumulation_step == config.gradient_accumulation_steps:
                    #torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    device_manager.optimizer_step(optimizer)

                    accumulation_step = 0
                    global_step += 1

                    if global_step % config.steps_freq == 0:
                        avg_loss = running_loss / (config.steps_freq * config.gradient_accumulation_steps)
                        logger.log_metrics({"train/loss": avg_loss}, commit=False, step=global_step)
                        running_loss = 0.0
                        # Clear XLA computation cache to avoid memory issues.
                        if config.use_tt:
                            xr.clear_computation_cache()

                    # Commit metrics to W&B.
                    logger.log_metrics({}, commit=True, step=global_step)

                    # Save step checkpoint.
                    if checkpoint_manager.should_save_checkpoint(global_step):
                        checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)

            # Save epoch checkpoint.
            if checkpoint_manager.should_save_checkpoint(global_step, epoch):
                checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)

        # Save final model.
        final_model_path = checkpoint_manager.save_checkpoint(
            model, global_step, epoch, optimizer, checkpoint_name="final_model.pth"
        )
        logger.log_artifact(final_model_path, artifact_type="model", name="final_model.pth")

    except Exception as e:
        traceback_str = traceback.format_exc()
        logger.error(f"Training failed with error: {str(e)}", traceback_str)
        raise
    finally:
        logger.finish()


if __name__ == "__main__":
    # Config setup
    default_config = Path(__file__).parent / "test_gpt_oss_20b_finetuning.yaml"
    args = parse_cli_options(default_config=default_config)
    config: TrainingConfig = generate_config(TrainingConfig, args.config, args.test_config)

    # Reproducibility setup
    repro_manager = ReproducibilityManager(config)
    repro_manager.setup()

    # Logger setup.
    logger = TrainingLogger(config, args.test_log_filename_prefix)

    # Device setup
    device_manager = DeviceManager(config)
    logger.info(f"Using device: {device_manager.device}")

    # Use highest numerical precision for stable fine-tuning convergence.
    # fp32_dest_acc_en: accumulate partial results in FP32 to avoid precision loss.
    # math_fidelity hifi4: use all 4 mantissa phases for full precision multiplications.
    if config.use_tt:
        torch_xla.set_custom_compile_options({"fp32_dest_acc_en": True, "math_fidelity": "hifi4"})

    # Checkpoint manager setup
    checkpoint_manager = CheckpointManager(config, logger, device_manager.device)

    # Start training.
    train(config, device_manager, logger, checkpoint_manager)
