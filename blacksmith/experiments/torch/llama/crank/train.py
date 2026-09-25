# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Llama LoRA fine-tuning on Tenstorrent hardware through tt-crank."""
import contextlib
import os
import time
import traceback
from pathlib import Path
from typing import Callable, Optional

import torch
from tqdm import tqdm

from blacksmith.datasets.torch.dataset_utils import get_dataset
from blacksmith.experiments.torch.llama.crank.configs import CrankTrainerConfig
from blacksmith.experiments.torch.llama.crank.utils import (
    LOSS_SHAPE,
    MAX_PRINTED_EXAMPLES,
    TFLOP,
    TT_BACKEND,
    TT_DEVICE,
    TrainStepContext,
    build_compile_options,
    build_functional_state,
    get_model,
    log_train_metrics,
    make_eval_fn,
    make_loss_fn,
    prepare_batch,
    read_loss,
    train_step,
)
from blacksmith.tools.checkpoints_manager import CheckpointManager
from blacksmith.tools.cli import generate_config, parse_cli_options
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.performance_utils import (
    TtnnPerfMetrics,
    count_flop_params,
    peak_flops,
    training_flops_per_step,
    training_flops_per_token,
)
from blacksmith.tools.reproducibility_manager import ReproducibilityManager
from blacksmith.tools.torch_helpers import (
    collate_fn_for_causal_lm,
    collect_examples,
    show_examples,
)


def validate(
    eval_fn: Callable,
    params: dict[str, torch.Tensor],
    val_dataloader,
    logger: TrainingLogger,
    config: CrankTrainerConfig,
    vocab_size: int,
    device: torch.device,
    tokenizer=None,
) -> float:
    logger.info("Starting validation...")
    dtype = config.torch_dtype()
    total_val_loss = 0.0
    num_val_batches = 0
    collected_examples = []

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation"):
            device_batch, _ = prepare_batch(batch, vocab_size, config.ignored_index, dtype, device)
            outputs = eval_fn(
                params,
                device_batch["input_ids"],
                device_batch["attention_mask"],
                device_batch["expected_output"],
                device_batch["loss_scale"],
            )
            loss, predictions = outputs if config.print_examples else (outputs, None)

            total_val_loss += read_loss(loss)
            num_val_batches += 1

            if config.print_examples:
                collected_examples = collect_examples(
                    batch_size=batch["labels"].shape[0],
                    collected_examples=collected_examples,
                    max_examples=MAX_PRINTED_EXAMPLES,
                    input_ids=batch["input_ids"],
                    expected_output=batch["labels"],
                    predictions=predictions,
                    num_val_batches=num_val_batches,
                )

    if config.print_examples and tokenizer is not None:
        logger.info("Printing validation examples...")
        show_examples(collected_examples, tokenizer, config, logger)

    avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
    logger.info(f"Average validation loss: {avg_val_loss}")
    return avg_val_loss


def train(
    config: CrankTrainerConfig,
    device: torch.device,
    logger: TrainingLogger,
    checkpoint_manager: CheckpointManager,
    compile_options: Optional[dict],
):
    logger.info("Starting training...")
    dtype = config.torch_dtype()

    # Load model.
    model, base_model, tied = get_model(config, device, logger)
    state = build_functional_state(model)
    hf_config = base_model.config
    vocab_size = hf_config.vocab_size

    parameter_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(t.numel() for t in state.params.values())
    logger.info(f"Loaded {config.model_name} model.")
    logger.info(f"Model parameters: {parameter_count}")
    logger.info(f"Trainable parameters: {trainable_count} in {len(state.params)} tensors")
    logger.info(f"Weight tying {'on' if tied else 'off'} after .to({device})")
    if state.aliases:
        aliases = ", ".join(f"{alias} -> {name}" for alias, name in state.aliases.items())
        logger.info(f"{len(state.aliases)} tied alias(es): {aliases}")

    # Analytical MFU numerator.
    head_dim = getattr(hf_config, "head_dim", None) or hf_config.hidden_size // hf_config.num_attention_heads
    flop_params = count_flop_params(
        model,
        embedding_weight=base_model.get_input_embeddings().weight,
        is_trainable=lambda name: name in state.params,
        tied_embeddings=tied,
    )
    flops_per_token = training_flops_per_token(
        flop_params,
        num_layers=hf_config.num_hidden_layers,
        num_heads=hf_config.num_attention_heads,
        head_dim=head_dim,
        seq_len=config.max_length,
    )
    step_flops = training_flops_per_step(
        flops_per_token,
        batch_size=config.batch_size,
        seq_len=config.max_length,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    device_peak_flops = peak_flops() if config.use_tt else None
    step_tflops = step_flops / TFLOP
    peak_info = f", peak {device_peak_flops / TFLOP:.2f} TFLOP/s" if device_peak_flops else ""
    logger.info(f"{step_tflops:.2f} TFLOP per step from {flop_params.total} charged parameters{peak_info}")

    # Stock fused AdamW. `fused=True` routes to aten::_fused_adamw_, which tt-crank lowers to
    # one fused kernel per parameter. The leaves stay put across steps so the optimizer owns
    # the moment state instead of it being threaded through the graph.
    optimizer = torch.optim.AdamW(
        list(state.params.values()),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
        fused=True,
    )

    # The leaves in `state.params` share storage with the module's parameters, so weights
    # restored into the module are what the next step trains.
    if config.checkpoint.resume_from_checkpoint:
        checkpoint_manager.load_checkpoint(model, optimizer)

    # Load dataset.
    train_dataset = get_dataset(config=config, split="train", collate_fn=collate_fn_for_causal_lm)
    train_dataloader = train_dataset.get_dataloader()
    logger.info(f"Loaded {config.dataset_id} dataset. Train dataset size: {len(train_dataloader) * config.batch_size}")

    eval_dataset = get_dataset(config=config, split="validation", collate_fn=collate_fn_for_causal_lm)
    eval_dataloader = eval_dataset.get_dataloader()
    logger.info(f"Loaded {config.dataset_id} dataset. Eval dataset size: {len(eval_dataloader) * config.batch_size}")

    tokenizer = train_dataset.tokenizer

    loss_fn = make_loss_fn(model, state, dtype)
    eval_fn = make_eval_fn(model, state, dtype, with_predictions=config.print_examples)
    if config.use_tt:
        loss_fn = torch.compile(loss_fn, backend=TT_BACKEND, dynamic=False, options=compile_options)
        eval_fn = torch.compile(eval_fn, backend=TT_BACKEND, dynamic=False, options=compile_options)

    grad_seed = torch.ones(LOSS_SHAPE, dtype=dtype).to(device)

    perf_metrics = None
    if config.use_tt and config.perf_metrics_enabled:
        perf_metrics = TtnnPerfMetrics(config.perf_metrics_file)
        perf_metrics.clear_stale()

    # While this collection is open the engine registers every compiled graph's TTIR and TTNN
    # IR. Registration order, so the forward comes before the backward and the per-parameter
    # AdamW graphs last -- which is why the collection has to stay open across
    # optimizer.step(). Capturing the TTIR costs a full module print the engine only pays
    # while a collection is active, hence dumping right after the first step.
    dump_artifacts = None
    artifacts_context = contextlib.nullcontext()
    if config.use_tt and config.artifacts_name:
        from tt_crank.torch._artifacts import collect_artifacts, dump_artifacts

        artifacts_context = collect_artifacts(config.artifacts_name)

    ctx = TrainStepContext(
        loss_fn=loss_fn,
        state=state,
        optimizer=optimizer,
        grad_seed=grad_seed,
        step_tflops=step_tflops,
        perf_metrics=perf_metrics,
        dump_artifacts=dump_artifacts,
    )
    global_step = 0
    epoch = 0

    try:
        # Initial validation
        val_loss = validate(eval_fn, state.params, eval_dataloader, logger, config, vocab_size, device, tokenizer)
        logger.log_metrics({"val/loss": val_loss}, commit=True, step=global_step)

        train_start = time.perf_counter()

        with artifacts_context:
            for epoch in range(config.num_epochs):
                running_loss = 0.0
                running_time = 0.0
                reached_max_steps = False

                for batch in tqdm(train_dataloader, desc="Training"):
                    if config.max_steps is not None and global_step >= config.max_steps:
                        reached_max_steps = True
                        break
                    step_start = time.perf_counter()
                    is_first_step = global_step == 0

                    device_batch, valid_tokens = prepare_batch(batch, vocab_size, config.ignored_index, dtype, device)
                    loss_tensor = train_step(ctx, device_batch, is_first_step, logger)

                    loss = read_loss(loss_tensor)
                    elapsed = time.perf_counter() - step_start
                    global_step += 1
                    running_loss += loss
                    running_time += elapsed

                    if global_step % config.metrics.steps_freq == 0:
                        log_train_metrics(
                            logger,
                            config,
                            global_step,
                            avg_loss=running_loss / config.metrics.steps_freq,
                            avg_step_time=running_time / config.metrics.steps_freq,
                            step_flops=step_flops,
                            device_peak_flops=device_peak_flops,
                            perf_metrics=perf_metrics,
                            valid_tokens=valid_tokens,
                        )
                        running_loss = 0.0
                        running_time = 0.0

                    # Validation
                    if global_step % config.val_steps_freq == 0:
                        val_loss = validate(
                            eval_fn, state.params, eval_dataloader, logger, config, vocab_size, device, tokenizer
                        )
                        logger.log_metrics({"val/loss": val_loss}, commit=False, step=global_step)

                    # Commit metrics to W&B.
                    logger.log_metrics({}, commit=True, step=global_step)

                    # Save step checkpoint.
                    if checkpoint_manager.should_save_checkpoint(global_step):
                        checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)

                # Save epoch checkpoint.
                if checkpoint_manager.should_save_checkpoint(global_step, epoch):
                    checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)

                if reached_max_steps:
                    break

        if config.measure_e2e_time:
            train_elapsed = time.perf_counter() - train_start
            logger.info(f"Training e2e time: {train_elapsed:.3f}s ({global_step} steps)")

        # Save final model.
        final_checkpoint_name = config.checkpoint.final_checkpoint_name
        final_model_path = checkpoint_manager.save_checkpoint(
            model, global_step, epoch, optimizer, checkpoint_name=final_checkpoint_name
        )
        logger.log_artifact(final_model_path, artifact_type="model", name=final_checkpoint_name)

    except Exception as e:
        traceback_str = traceback.format_exc()
        logger.error(f"Training failed with error: {str(e)}", traceback_str)
        raise
    finally:
        logger.finish()


if __name__ == "__main__":
    # Config setup
    default_config = Path(__file__).parent / "lora" / "single_chip" / "llama_3_1_8b_sst2.yaml"
    args = parse_cli_options(default_config=default_config)
    config: CrankTrainerConfig = generate_config(
        CrankTrainerConfig, args.config, args.test_config, args.test_checkpoint_path
    )

    # Reproducibility setup
    repro_manager = ReproducibilityManager(config)
    repro_manager.setup()

    # Logger setup.
    logger = TrainingLogger(config.logging, args.test_log_filename_prefix)

    # Device setup. tt-crank reads TT_CRANK_ARTIFACTS_DIR into a static at library load, so
    # it has to be set before the extension module is imported -- setting it later is
    # silently ignored.
    compile_options = None
    if config.use_tt:
        if config.artifacts_name:
            os.environ.setdefault("TT_CRANK_ARTIFACTS_DIR", config.artifacts_dir)

        # tt-metal decides per MetalContext whether Blackhole's DRAM RISC cores count as a
        # programmable core type (opt-in, gated on a firmware capability check). The runtime's
        # device context and the mock-device context the optimizer's op model opens can
        # disagree, and GraphTracker::track_program_l1 then indexes a program's per-core-type
        # kernel table with the other context's count: every op constraint query throws
        # std::out_of_range and OperationValidationAndFallback fails the compile with
        # opt_level >= 1. Pinning the flag makes both contexts agree; the DRAM cores are not
        # used for compute either way. Set it to "1" to opt in.
        os.environ.setdefault("TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES", "0")

        import tt_crank.torch  # noqa: F401  (registers the tt device and the "tt" dynamo backend)

        device = torch.device(TT_DEVICE)
        compile_options = build_compile_options(config)
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Checkpoint manager setup
    checkpoint_manager = CheckpointManager(config.checkpoint, logger, device)

    # Start training.
    train(config, device, logger, checkpoint_manager, compile_options)
