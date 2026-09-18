# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""tt-crank port of `train.py` (Llama LoRA / adapters fine-tuning), 1:1 with the tt-xla script.

Reads the same YAMLs as `train.py` -- single chip and multichip alike, the `mesh_shape` /
`model_sharding_patterns` block included:

    source env/activate --crank
    python blacksmith/experiments/torch/llama/xla/train_crank.py \
        --config blacksmith/experiments/torch/llama/xla/lora/single_chip/llama_3_2_1b_sst2.yaml

Diff against `train.py` to see what tt-crank changes. In short: `DeviceManager` and
`CheckpointManager` come from `blacksmith.tools.crank`; compile options are passed per
`torch.compile` call instead of one global `set_custom_compile_options`; the model is
sharded once (DTensor parameters stay DTensors) instead of re-annotated every step; and
the lazy-graph machinery -- `torch_xla.sync` fences, grad / AdamW-state pre-seeding,
`capturable=True`, the device-side `step_loss` accumulator -- is gone because tt-crank
executes eagerly.
"""
import time
import traceback
from pathlib import Path

import torch
from tqdm import tqdm

from blacksmith.datasets.torch.dataset_utils import get_dataset
from blacksmith.experiments.torch.llama.configs import TrainingConfig
from blacksmith.models.torch.huggingface.hf_models import get_model
from blacksmith.tools.cli import generate_config, parse_cli_options
from blacksmith.tools.crank.checkpoints_manager import CheckpointManager
from blacksmith.tools.crank.device_manager import DeviceManager
from blacksmith.tools.crank.torch_helpers import loss_to_float, to_host
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.reproducibility_manager import ReproducibilityManager
from blacksmith.tools.torch_helpers import (
    collate_fn_for_causal_lm,
    collect_examples,
    show_examples,
)
from blacksmith.tools.workaround_utils import cross_entropy_loss, transform_labels


def validate(
    model, compute_loss_fn, eval_model, val_data_loader, loss_fn, logger, device_manager, config, tokenizer=None
):
    """Validation loss through the *compiled forward+loss* (`compute_loss_fn`), not an eager loss on
    `eval_model`'s logits as in `train.py`. Same numbers on a single chip; on a mesh the eager DTensor
    path is wrong on the current tt-crank (`aten._to_copy` and `log_softmax` on a batch-sharded DTensor
    return bad values, while the same math inside a compiled graph is fine). `eval_model` is only used
    for the predictions shown by `print_examples`."""
    logger.info("Starting validation...")
    total_val_loss = 0.0
    num_val_batches = 0
    collected_examples = []

    with torch.no_grad():
        for batch in tqdm(val_data_loader, desc="Validation"):
            # Expected output must be prepared on CPU first due to an OOM issue.
            # See https://github.com/tenstorrent/tt-blacksmith/issues/455.
            expected_output = batch["labels"]
            expected_output_one_hot, labels_mask = transform_labels(
                expected_output, config.ignored_index, model.model.config.vocab_size
            )
            # Shard batch if data parallelism is used.
            device_batch = device_manager.prepare_batch(
                {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "expected_output": expected_output_one_hot,
                    "labels_mask": labels_mask,
                }
            )

            # Forward pass + loss, one compiled graph (see docstring).
            loss = compute_loss_fn(device_batch, model, loss_fn, 1)

            total_val_loss += loss_to_float(loss)
            num_val_batches += 1

            if config.print_examples:
                # Predictions: logits[:, :-1] predicts tokens at positions 1:
                logits = eval_model(
                    input_ids=device_batch["input_ids"], attention_mask=device_batch["attention_mask"]
                ).logits
                predictions = logits[:, :-1, :].argmax(dim=-1)
                collected_examples = collect_examples(
                    batch_size=expected_output.shape[0],
                    collected_examples=collected_examples,
                    max_examples=10,
                    input_ids=batch["input_ids"],
                    expected_output=expected_output,
                    predictions=to_host(predictions),  # data parallel: gathers the batch shards
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

    # Load model. compile_model=False: forward + loss are compiled together below so the loss
    # and its backward stay in one tt graph.
    model = get_model(config, device_manager.device, compile_model=False)
    # Shard model once, here (tensor and/or data parallelism). A DTensor parameter stays a
    # DTensor, so unlike tt-xla's SPMD annotations this does not have to be repeated per step.
    model = device_manager.shard_model(model)

    logger.info(f"Loaded {config.model_name} model.")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # No capturable=True and no pre-seeded grads / AdamW state: those kept the tt-xla fused
    # step graph's signature stable; tt-crank is eager.
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)

    # Load checkpoint if needed.
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
        # Per-callable compile options (tt-xla set them once, globally, in `__main__`).
        # dynamic=False: tt-crank cannot lower symbolic (SymInt) inputs, and torch's automatic dynamic
        # shapes would otherwise turn `gradient_accumulation_steps` into one after it is seen with two
        # values (1 in `validate`, N in training). With it off the second value just recompiles.
        compile_options = device_manager.compile_options()
        compute_loss_fn = torch.compile(compute_loss, backend="tt", dynamic=False, options=compile_options)
        # Only needed for the predictions printed by `print_examples` (see `validate`).
        eval_model = (
            torch.compile(model, backend="tt", dynamic=False, options=compile_options)
            if config.print_examples
            else None
        )
    else:
        compute_loss_fn = compute_loss
        eval_model = model

    global_step = 0

    try:
        # Multichip: plain tensors HF builds internally count as replicated DTensors.
        with device_manager.replication_context():
            # Initial validation
            model.eval()
            val_loss = validate(
                model,
                compute_loss_fn,
                eval_model,
                eval_dataloader,
                cross_entropy_loss,
                logger,
                device_manager,
                config,
                tokenizer,
            )
            logger.log_metrics({"val/loss": val_loss}, commit=True, step=global_step)
            model.train()

            train_start = None
            step_start = None
            if config.measure_e2e_time:
                train_start = time.perf_counter()

            for epoch in range(config.num_epochs):
                accumulation_step = 0
                running_loss = 0.0
                window_loss = 0.0

                for batch in tqdm(train_dataloader, desc="Training"):
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

                    loss_ = compute_loss_fn(batch, model, cross_entropy_loss, config.gradient_accumulation_steps)
                    loss_.backward()

                    accumulation_step += 1
                    window_loss += loss_to_float(loss_)

                    # Only step the optimizer after accumulating gradients.
                    if accumulation_step != config.gradient_accumulation_steps:
                        continue

                    device_manager.optimizer_step(optimizer, zero_grad=True)

                    running_loss += window_loss
                    window_loss = 0.0
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
                            model,
                            compute_loss_fn,
                            eval_model,
                            eval_dataloader,
                            cross_entropy_loss,
                            logger,
                            device_manager,
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

    # Per-tensor weight dtype overrides are a tt-xla feature (tt_torch.apply_weight_dtype_overrides);
    # tt-crank only has the compiler-wide `experimental_weight_dtype`, see DeviceManager.compile_options.
    if config.use_tt and config.weight_dtype_overrides:
        raise ValueError(
            "weight_dtype_overrides is tt-xla only; use `experimental_weight_dtype` (bfp_bf8 | bfp_bf4) on tt-crank."
        )

    # Reproducibility setup
    repro_manager = ReproducibilityManager(config)
    repro_manager.setup()

    # Logger setup.
    logger = TrainingLogger(config, args.test_log_filename_prefix)

    # Device setup. Compile options are per `torch.compile` call on tt-crank (see `train`), so
    # there is no global `set_custom_compile_options` step here.
    device_manager = DeviceManager(config)
    logger.info(f"Using device: {device_manager.device}")
    if device_manager.mesh is not None:
        logger.info(f"Mesh: {tuple(device_manager.mesh.shape)} {device_manager.mesh.mesh_dim_names}")

    # Checkpoint manager setup
    checkpoint_manager = CheckpointManager(config, logger, device_manager.device)

    # Start training.
    train(config, device_manager, logger, checkpoint_manager)
