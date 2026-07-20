# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import time
import traceback
from pathlib import Path

import torch
import torch_xla
from tqdm import tqdm

from blacksmith.datasets.torch.dataset_utils import get_dataset
from blacksmith.experiments.torch.llama.configs import TrainingConfig
from blacksmith.models.torch.huggingface.hf_models import get_model
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
from blacksmith.tools.workaround_utils import (
    cross_entropy_loss,
    materialize_adamw_state,
    materialize_grads,
    transform_labels,
)


def validate(model, val_data_loader, loss_fn, logger, device, config, tokenizer=None):
    logger.info("Starting validation...")
    total_val_loss = 0.0
    num_val_batches = 0
    collected_examples = []

    with torch.no_grad():
        for batch in tqdm(val_data_loader, desc="Validation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            # Expected output must be prepared on CPU first due to an OOM issue.
            # See https://github.com/tenstorrent/tt-blacksmith/issues/455.
            expected_output = batch["labels"]

            # Shard model if tensor parallelism is used.
            device_manager.shard_model(model)

            # Forward pass.
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Shift logits for causal LM: predict next token
            # logits[:, :-1] predicts tokens at positions 1:
            shift_logits = logits[:, :-1, :].contiguous()

            expected_output_one_hot, labels_mask = transform_labels(
                expected_output, config.ignored_index, model.model.config.vocab_size
            )

            if config.use_tt:
                loss = loss_fn(shift_logits, expected_output_one_hot, labels_mask)
            else:
                loss = loss_fn(shift_logits, expected_output_one_hot.to(device), labels_mask.to(device))

            # Predictions
            predictions = shift_logits.argmax(dim=-1)
            if config.use_tt:
                torch_xla.sync(wait=True)

            total_val_loss += loss.item()
            num_val_batches += 1

            if config.print_examples:
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


def compute_loss(batch, model, loss_fn, gradient_accumulation_steps):
    output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = output.logits
    shift_logits = logits[:, :-1, :].contiguous()
    loss = loss_fn(shift_logits, batch["expected_output"], batch["labels_mask"])
    return loss / gradient_accumulation_steps


def train(
    config: TrainingConfig,
    device_manager: DeviceManager,
    logger: TrainingLogger,
    checkpoint_manager: CheckpointManager,
):
    logger.info("Starting training...")

    # Load model.
    model = get_model(config, device_manager.device, compile_model=False)

    logger.info(f"Loaded {config.model_name} model.")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, capturable=config.use_tt, lr=config.learning_rate)

    # Pre-create optimizer state so the fused step graph compiles only once. Defer its sync
    # (sync=False) so it fuses with the grad pre-seed below into one materialization graph.
    if config.use_tt and not config.resume_from_checkpoint:
        materialize_adamw_state(optimizer, sync=False)

    # Pre-seed zero grads and the loss accumulator so every fwd+bwd graph is identical (accumulate,
    # not assign/init) -- the first micro-batch reads an existing tensor just like the rest, so the
    # step compiles a single fwd+bwd graph. Infer step_loss's shape/dtype by probing the loss fn on
    # tiny CPU inputs (its output is a full reduction, so the probe's input sizes don't matter)
    # instead of hardcoding them. materialize_grads' sync flushes all pending pre-seeds (adamw state
    # when not resuming, grads, step_loss) as one materialization graph.
    with torch.no_grad():
        loss_probe = cross_entropy_loss(torch.zeros(1, 1, 1), torch.zeros(1, 1, 1), torch.zeros(1, 1))
    step_loss = torch.zeros(loss_probe.shape, dtype=loss_probe.dtype, device=device_manager.device)
    if config.use_tt:
        materialize_grads(optimizer)

    # Load checkpoint if needed. The optimizer's capturable/device state is repaired inside
    # load_checkpoint (see restore_capturable_optimizer_state) so the fused step graph stays stable.
    if config.resume_from_checkpoint:
        checkpoint_manager.load_checkpoint(model, optimizer)

    # Load dataset.
    train_dataset = get_dataset(config=config, split="train", collate_fn=collate_fn_for_causal_lm)
    train_dataloader = train_dataset.get_dataloader()
    logger.info(f"Loaded {config.dataset_id} dataset. Train dataset size: {len(train_dataloader) * config.batch_size}")

    eval_dataset = get_dataset(config=config, split="validation", collate_fn=collate_fn_for_causal_lm)
    eval_dataloader = eval_dataset.get_dataloader()
    logger.info(f"Loaded {config.dataset_id} dataset. Eval dataset size: {len(eval_dataloader) * config.batch_size}")

    tokenizer = train_dataset.tokenizer

    if config.use_tt:
        compile_options = {
            "tt_enable_torch_fx_fusion_pass": False,
            "tt_legacy_compile": True,
            "tt_lazy_execution": True,
        }
        compute_loss_fn = torch.compile(compute_loss, backend="tt", options=compile_options)
        eval_model = torch.compile(model, backend="tt", options=compile_options)
    else:
        compute_loss_fn = compute_loss
        eval_model = model

    global_step = 0

    try:
        # Initial validation
        model.eval()
        val_loss = validate(
            eval_model,
            eval_dataloader,
            cross_entropy_loss,
            logger,
            device_manager.device,
            config,
            tokenizer,
        )
        logger.log_metrics({"val/loss": val_loss}, commit=True, step=global_step)
        model.train()

        # TODO: Refactor when https://github.com/tenstorrent/tt-blacksmith/issues/602#issue-4596214372 is resolved.
        train_start = None
        step_start = None
        if config.measure_e2e_time:
            train_start = time.perf_counter()

        for epoch in range(config.num_epochs):
            # NOTE: grads and step_loss persist across epochs (reset only after an optimizer
            # step). If len(train_dataloader) is not a multiple of gradient_accumulation_steps,
            # a trailing partial window's grads/loss carry into the next epoch. Fine for the
            # divisible configs used here; revisit if a non-divisible dataset is added.
            accumulation_step = 0
            running_loss = 0.0

            for batch in tqdm(train_dataloader, desc="Training"):
                # No zero_grad() here: grads are pre-seeded to zero and re-zeroed in place
                # inside the optimizer graph, so zeroing never enters the fwd+bwd graph.
                if accumulation_step == 0 and config.measure_e2e_time:
                    step_start = time.perf_counter()

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
                device_manager.shard_model(model)

                loss_ = compute_loss_fn(batch, model, cross_entropy_loss, config.gradient_accumulation_steps)
                loss_.backward()

                # Accumulate the detached loss BEFORE the sync so the add lands in this
                # micro-batch's own fwd+bwd graph (identical for every micro-batch) instead of
                # leaking into the next one. step_loss stays a device tensor throughout (pre-seeded
                # above, reset via zeros_like after each optimizer step).
                accumulation_step += 1
                step_loss = step_loss + loss_.detach()

                if accumulation_step != config.gradient_accumulation_steps:
                    # Non-final: cut here so this is the shared fwd+bwd graph. grads/step_loss
                    # stay materialized device tensors for the next accumulation.
                    if config.use_tt:
                        torch_xla.sync(wait=True)

                # Only step the optimizer after accumulating gradients.
                else:
                    # Last micro-batch: leave fwd+bwd pending so it fuses with the optimizer
                    # update and the grad/loss re-zeroing into one graph. The sync inside
                    # optimizer_step flushes it and materializes window_loss.
                    window_loss = step_loss
                    step_loss = torch.zeros_like(step_loss)  # reset; fused into the optimizer graph
                    device_manager.optimizer_step(optimizer, zero_grad=True)

                    running_loss += window_loss.item()
                    accumulation_step = 0
                    global_step += 1

                    if config.measure_e2e_time:
                        step_elapsed = time.perf_counter() - step_start
                        logger.info(f"Step {global_step} e2e time: {step_elapsed:.3f}s")

                    if global_step % config.steps_freq == 0:
                        avg_loss = running_loss / config.steps_freq
                        logger.log_metrics({"train/loss": avg_loss}, commit=False, step=global_step)
                        running_loss = 0.0

                    # Validation
                    if global_step % config.val_steps_freq == 0:
                        model.eval()
                        val_loss = validate(
                            eval_model,
                            eval_dataloader,
                            cross_entropy_loss,
                            logger,
                            device_manager.device,
                            config,
                            tokenizer,
                        )
                        logger.log_metrics({"val/loss": val_loss}, commit=False, step=global_step)
                        model.train()

                    # Commit metrics to W&B.
                    logger.log_metrics({}, commit=True, step=global_step)

                    # Save step checkpoint.
                    if checkpoint_manager.should_save_checkpoint(global_step):
                        checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)

            # Save epoch checkpoint.
            if checkpoint_manager.should_save_checkpoint(global_step, epoch):
                checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)

        if config.measure_e2e_time:
            train_elapsed = time.perf_counter() - train_start
            logger.info(f"Training e2e time: {train_elapsed:.3f}s ({global_step} steps)")

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
    default_config = Path(__file__).parent / "lora" / "single_chip" / "llama_3_2_1b_sst2.yaml"
    args = parse_cli_options(default_config=default_config)
    config: TrainingConfig = generate_config(TrainingConfig, args.config, args.test_config, args.test_checkpoint_path)

    # Resolve a relative weight_dtype_overrides JSON path against the yaml's directory
    # so configs can live next to the yaml that references them.
    if isinstance(config.weight_dtype_overrides, str) and config.weight_dtype_overrides.endswith(".json"):
        override_path = Path(config.weight_dtype_overrides)
        if not override_path.is_absolute():
            config.weight_dtype_overrides = str((args.config.parent / override_path).resolve())

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
        compile_options = {
            "fp32_dest_acc_en": True,
            "math_fidelity": "hifi4",
            "enable_trace": config.enable_trace,
            "enable_const_eval": config.enable_const_eval,
        }
        if config.experimental_weight_dtype:
            compile_options["experimental_weight_dtype"] = config.experimental_weight_dtype
        torch_xla.set_custom_compile_options(compile_options)

    # Checkpoint manager setup
    checkpoint_manager = CheckpointManager(config, logger, device_manager.device)

    # Start training.
    train(config, device_manager, logger, checkpoint_manager)
