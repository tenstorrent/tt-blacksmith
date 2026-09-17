# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""LoRA fine-tuning of Llama on Tenstorrent hardware through tt-crank.

Run:
    python blacksmith/experiments/torch/llama/train.py \
        --config blacksmith/experiments/torch/llama/single_chip/llama_3_2_1b_sst2.yaml

The same script covers single chip and multichip; the only difference is the
`mesh:` block in the YAML.

Structure mirrors `blacksmith_xla/experiments/torch/llama/xla/train.py` so the
two can be diffed; what differs is what tt-crank makes unnecessary (lazy-graph
fences, grad/AdamW pre-materialization, per-step re-sharding) and how compile
options are passed (per callable, not globally).
"""
import time
import traceback
from pathlib import Path

import torch
from torch.distributed.tensor import DTensor
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
    loss_to_float,
    show_examples,
)
from blacksmith.tools.workaround_utils import cross_entropy_loss, transform_labels


def prepare_batch(batch, device_manager, config, vocab_size):
    """Build the one-hot target on host, then move/shard the whole batch.

    One-hot on device OOMs (tt-blacksmith#455), so `transform_labels` runs on CPU.
    """
    expected_output, labels_mask = transform_labels(batch["labels"], config.ignored_index, vocab_size)
    return device_manager.prepare_batch(
        {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "expected_output": expected_output,
            "labels_mask": labels_mask,
        }
    )


def compute_loss(batch, model, loss_fn):
    output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    # Causal LM: position t predicts token t+1, so the last logit has no target.
    shift_logits = output.logits[:, :-1, :]
    return loss_fn(shift_logits, batch["expected_output"], batch["labels_mask"])


def validate(compute_loss_fn, model, val_data_loader, device_manager, config, logger, tokenizer=None):
    logger.info("Starting validation...")
    total_val_loss = 0.0
    num_val_batches = 0
    collected_examples = []
    vocab_size = model.config.vocab_size

    with torch.no_grad():
        for raw_batch in tqdm(val_data_loader, desc="Validation"):
            batch = prepare_batch(raw_batch, device_manager, config, vocab_size)
            loss = compute_loss_fn(batch, model, cross_entropy_loss)

            total_val_loss += loss_to_float(loss)
            num_val_batches += 1

            if config.print_examples:
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
                predictions = logits[:, :-1, :].argmax(dim=-1)
                if isinstance(predictions, DTensor):  # data parallel: gather the batch shards
                    predictions = predictions.full_tensor()
                collected_examples = collect_examples(
                    batch_size=raw_batch["labels"].shape[0],
                    collected_examples=collected_examples,
                    max_examples=10,
                    input_ids=raw_batch["input_ids"],
                    expected_output=raw_batch["labels"],
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
    config: TrainingConfig,
    device_manager: DeviceManager,
    logger: TrainingLogger,
    checkpoint_manager: CheckpointManager,
):
    logger.info("Starting training...")

    # Load model. compile_model=False: forward+loss are compiled together below so the loss and
    # its backward stay inside the tt graph instead of falling back to eager between the two.
    model = get_model(config, device_manager.device, compile_model=False)
    # Distribute once, here: a DTensor parameter stays a DTensor for the rest of the run
    # (tt-xla's SPMD annotations had to be re-applied every step).
    model = device_manager.shard_model(model)
    vocab_size = model.config.vocab_size

    logger.info(f"Loaded {config.model_name} model.")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    logger.watch_model(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)

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

    # NOTE(sdpa): with HF's default attention path, SDPA runs as the matmul+softmax MATH decomposition
    # in training, not tt-crank's fused sdpa_fw/sdpa_bw kernels. HF passes a per-batch bool mask
    # [B, 1, S, S]; `tt_fused_sdp_choice` (tt-crank sdpa.cpp) only accepts one [1, 1, S, S] mask and
    # falls back to MATH otherwise. The trick used in tt-kurbla to get the fused kernel at batch > 1:
    # register a custom attention implementation ("sdpa_causal") whose mask function returns None and
    # load the model with attn_implementation="sdpa_causal", so SDPA is called with is_causal=True and
    # no mask. It is exact here because the dataset right-pads and pad positions are excluded from the
    # loss by `labels_mask`, so the padding half of the mask is a no-op. Not enabled: the model is kept
    # identical to the tt-xla experiment.
    if config.use_tt:
        compute_loss_fn = torch.compile(compute_loss, backend="tt", options=device_manager.compile_options())
    else:
        compute_loss_fn = compute_loss

    global_step = 0

    try:
        # Multichip: plain tensors HF builds internally count as replicated.
        with device_manager.replication_context():
            # Initial validation
            model.eval()
            val_loss = validate(compute_loss_fn, model, eval_dataloader, device_manager, config, logger, tokenizer)
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

                for batch in tqdm(train_dataloader, desc=f"Training (epoch {epoch})"):
                    if accumulation_step == 0 and config.measure_e2e_time:
                        step_start = time.perf_counter()

                    batch = prepare_batch(batch, device_manager, config, vocab_size)

                    loss = compute_loss_fn(batch, model, cross_entropy_loss) / config.gradient_accumulation_steps
                    loss.backward()

                    accumulation_step += 1
                    window_loss += loss_to_float(loss)

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
                            compute_loss_fn, model, eval_dataloader, device_manager, config, logger, tokenizer
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
    default_config = Path(__file__).parent / "single_chip" / "llama_3_2_1b_sst2.yaml"
    args = parse_cli_options(default_config=default_config)
    config: TrainingConfig = generate_config(TrainingConfig, args.config, args.test_config, args.test_checkpoint_path)

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
