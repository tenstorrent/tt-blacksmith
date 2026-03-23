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
from blacksmith.models.torch.gpt_oss.gpt_oss_model import (
    VOCAB_SIZE,
    GptOss20B,
    create_mesh,
)
from blacksmith.tools.checkpoints_manager import CheckpointManager
from blacksmith.tools.cli import generate_config, parse_cli_options
from blacksmith.tools.device_manager import DeviceManager
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.reproducibility_manager import ReproducibilityManager
from blacksmith.tools.torch_helpers import (
    collate_fn_for_causal_lm,
    collect_examples,
    show_examples,
)
from blacksmith.tools.workaround_utils import cross_entropy_loss, transform_labels


def validate(model, val_data_loader, logger, device, mesh, config, tokenizer=None):
    logger.info("Starting validation...")
    total_val_loss = 0.0
    num_val_batches = 0
    collected_examples = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(val_data_loader, desc="Validation"):
            input_ids = batch["input_ids"].to(device)

            # Shard model if tensor parallelism is used.
            model.shard(mesh)

            # Forward pass.
            out, _ = model(input_ids, mesh)

            # Shift logits for causal LM: logits[:, :-1] predicts tokens at positions 1:
            logits = model.lm_head(out)
            shift_logits = logits[:, :-1, :].contiguous()

            expected_output_one_hot, labels_mask = transform_labels(batch, config.ignored_index, VOCAB_SIZE)
            loss = cross_entropy_loss(shift_logits, expected_output_one_hot, labels_mask)

            # Predictions
            predictions = shift_logits.argmax(dim=-1)
            if config.use_tt:
                torch_xla.sync(wait=True)

            total_val_loss += loss.item()
            num_val_batches += 1

            if config.print_examples:
                expected_output = batch["labels"].to(device)
                collected_examples = collect_examples(
                    batch_size=expected_output.shape[0],
                    collected_examples=collected_examples,
                    max_examples=10,
                    input_ids=input_ids,
                    expected_output=expected_output,
                    predictions=predictions,
                    num_val_batches=num_val_batches,
                )

    if config.print_examples and tokenizer is not None:
        logger.info("Printing validation examples...")
        show_examples(collected_examples, tokenizer, config, logger)

    avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
    logger.info(f"Average validation loss: {avg_val_loss}")
    return avg_val_loss


# Training step extracted into a separate function to keep large vocab-sized
# tensors (e.g. logits) scoped locally. This ensures they do not propagate beyond
# the step via the computation graph, avoiding unnecessary and expensive
# CCLs in multi-chip setups.
# Issue itself should be investigated further.
def training_step_inner(batch, model, mesh):
    input_ids = batch["input_ids"]

    # Forward pass (no grad, saves per-layer inputs for recomputation).
    out, saved = model(input_ids, mesh)

    # LM head loss. Labels are one-hot float tensors (already on device) to
    # avoid int64 ops on TT, which trigger stablehlo.bitcast_convert.
    out_leaf = out.detach().requires_grad_(True)
    with torch.enable_grad():
        logits = model.lm_head(out_leaf)
        shift_logits = logits[:, :-1, :].contiguous()

    loss = cross_entropy_loss(shift_logits, batch["expected_output"], batch["labels_mask"])
    loss.backward()
    grad_out = out_leaf.grad.detach()
    torch_xla.sync(wait=True)

    # Backward through transformer layers (LoRA grads accumulated per layer).
    model.backward(saved, grad_out, mesh)

    return loss.detach()


def train(
    config: TrainingConfig,
    device_manager: DeviceManager,
    logger: TrainingLogger,
    checkpoint_manager: CheckpointManager,
):
    logger.info("Starting training...")

    # Build model on CPU, then move to device.
    model = GptOss20B()
    model = model.to(eval(config.dtype))
    logger.info("Deinterleaving gate_up weights (CPU)...")
    model.deinterleave()
    model = model.to(device_manager.device)

    logger.info(f"Loaded {config.model_name} model.")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Shard expert weights and attention projections across the mesh.
    mesh = device_manager.mesh
    model.shard(mesh)
    if config.use_tt:
        torch_xla.sync(wait=True)

    # Init training components (optimizer, lr scheduler, etc.)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
    )

    # Load checkpoint if needed.
    if config.resume_from_checkpoint:
        checkpoint_manager.load_checkpoint(model, optimizer)

    # Load dataset.
    train_dataset = get_dataset(config=config, split="train", collate_fn=collate_fn_for_causal_lm)
    train_dataloader = train_dataset.get_dataloader()
    logger.info(f"Loaded {config.dataset_id} dataset. Train dataset size: {len(train_dataloader)*config.batch_size}")

    eval_dataset = get_dataset(config=config, split="validation", collate_fn=collate_fn_for_causal_lm)
    eval_dataloader = eval_dataset.get_dataloader()
    logger.info(f"Loaded {config.dataset_id} dataset. Eval dataset size: {len(eval_dataloader)*config.batch_size}")

    tokenizer = train_dataset.tokenizer

    global_step = 0
    running_loss = 0.0
    try:
        model.train()
        for epoch in range(config.num_epochs):

            for batch in tqdm(train_dataloader, desc="Training"):
                # Zero out gradients.
                optimizer.zero_grad()

                # TODO: Refactor when https://github.com/tenstorrent/tt-blacksmith/issues/327 is resolved.
                # Labels are converted to float one-hot on CPU to avoid int64 on TT device,
                # which triggers stablehlo.bitcast_convert (a type-reinterpretation op
                # TT-MLIR cannot lower).
                expected_output, labels_mask = transform_labels(batch, config.ignored_index, VOCAB_SIZE)
                batch = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "expected_output": expected_output,
                    "labels_mask": labels_mask,
                }
                # Shard batch if data parallelism is used.
                batch = device_manager.prepare_batch(batch)
                # Shard model if tensor parallelism is used.
                model.shard(mesh)

                # Training step.
                loss_ = training_step_inner(batch, model, mesh)

                if config.use_tt:
                    torch_xla.sync(wait=True)

                # Optimizer step.
                device_manager.optimizer_step(optimizer)
                running_loss += loss_.item()

                global_step += 1
                if global_step % config.steps_freq == 0:
                    avg_loss = running_loss / config.steps_freq
                    logger.log_metrics({"train/loss": avg_loss}, commit=False, step=global_step)
                    running_loss = 0.0

                    # Do validation.
                    valid_loss = validate(
                        model,
                        eval_dataloader,
                        logger,
                        device_manager.device,
                        mesh,
                        config,
                        tokenizer,
                    )
                    logger.log_metrics({"val/loss": valid_loss}, step=global_step)

                    # Clear XLA computation cache to avoid memory issues.
                    xr.clear_computation_cache()

                    model.train()

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
    default_config = Path(__file__).parent / "lora" / "single_chip" / "test_gpt_oss_20b.yaml"
    args = parse_cli_options(default_config=default_config)
    config: TrainingConfig = generate_config(TrainingConfig, args.config)

    # Reproducibility setup
    repro_manager = ReproducibilityManager(config)
    repro_manager.setup()

    # Logger setup
    logger = TrainingLogger(config)

    # Checkpoint manager setup
    checkpoint_manager = CheckpointManager(config, logger)

    # Device setup
    device_manager = DeviceManager(config)
    logger.info(f"Using device: {device_manager.device}")

    # Start training
    train(config, device_manager, logger, checkpoint_manager)
