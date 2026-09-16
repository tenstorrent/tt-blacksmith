# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""LoRA fine-tuning of Llama on Tenstorrent hardware through tt-crank.

Run:
    python blacksmith/experiments/torch/llama/train.py \
        --config blacksmith/experiments/torch/llama/single_chip/llama_3_2_1b_sst2.yaml

The same script covers single chip and multichip; the only difference is the
`mesh:` block in the YAML.
"""
import time
import traceback
from pathlib import Path

import torch
from tqdm import tqdm

from blacksmith.datasets.torch.dataset_utils import get_dataset
from blacksmith.experiments.torch.llama.configs import TrainingConfig
from blacksmith.models.torch.huggingface.hf_models import get_model
from blacksmith.tools.checkpoints_manager import CheckpointManager
from blacksmith.tools.cli import generate_config, parse_cli_options
from blacksmith.tools.device_manager import DeviceManager
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.reproducibility_manager import ReproducibilityManager
from blacksmith.tools.torch_helpers import collate_fn_for_causal_lm, cross_entropy_loss, loss_to_float, transform_labels


def make_loss_fn(model, config, device_manager):
    """Build the compiled forward+loss step.

    Compiling forward *and* loss as one callable (rather than the model alone)
    keeps log_softmax and the loss reduction -- and their backward -- inside the
    tt graph, instead of falling back to eager between the two. The model is
    captured by closure rather than passed as an argument, which is the shape
    dynamo traces most cleanly.
    """

    def step(batch):
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        # Causal LM: position t predicts token t+1, so the last logit has no target.
        shift_logits = logits[:, :-1, :]
        return cross_entropy_loss(shift_logits, batch["expected_output"], batch["labels_mask"])

    if not config.use_tt:
        return step
    return torch.compile(step, backend="tt", options=device_manager.compile_options())


def prepare_batch(batch, device_manager, config, vocab_size):
    """Build the one-hot target on host, then move/shard the whole batch."""
    expected_output, labels_mask = transform_labels(batch["labels"], config.ignored_index, vocab_size)
    return device_manager.prepare_batch(
        {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "expected_output": expected_output,
            "labels_mask": labels_mask,
        }
    )


def validate(loss_fn, dataloader, device_manager, config, logger, vocab_size):
    logger.info("Starting validation...")
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            batch = prepare_batch(batch, device_manager, config, vocab_size)
            total += loss_to_float(loss_fn(batch))
            count += 1

    avg = total / count if count else 0.0
    logger.info(f"Average validation loss: {avg}")
    return avg


def train(
    config: TrainingConfig,
    device_manager: DeviceManager,
    logger: TrainingLogger,
    checkpoint_manager: CheckpointManager,
):
    logger.info("Starting training...")

    model = get_model(config, device_manager.device)
    # Distribute once, here: a DTensor parameter stays a DTensor for the rest of
    # the run. (tt-xla's SPMD annotations had to be re-applied every step.)
    model = device_manager.distribute_model(model)

    vocab_size = model.config.vocab_size
    logger.info(f"Loaded {config.model_name}.")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.learning_rate)

    train_dataloader = get_dataset(config, split="train", collate_fn=collate_fn_for_causal_lm).get_dataloader()
    eval_dataloader = get_dataset(config, split="validation", collate_fn=collate_fn_for_causal_lm).get_dataloader()
    logger.info(f"Train batches: {len(train_dataloader)} | Eval batches: {len(eval_dataloader)}")

    loss_fn = make_loss_fn(model, config, device_manager)

    global_step = 0
    try:
        # Multichip: plain tensors HF builds internally count as replicated.
        with device_manager.replication_context():
            model.eval()
            val_loss = validate(loss_fn, eval_dataloader, device_manager, config, logger, vocab_size)
            logger.log_metrics({"val/loss": val_loss}, step=global_step, commit=True)
            model.train()

            train_start = time.perf_counter() if config.measure_e2e_time else None

            for epoch in range(config.num_epochs):
                accumulation_step = 0
                running_loss = 0.0
                window_loss = 0.0
                step_start = None

                for batch in tqdm(train_dataloader, desc=f"Training (epoch {epoch})"):
                    if accumulation_step == 0 and config.measure_e2e_time:
                        step_start = time.perf_counter()

                    batch = prepare_batch(batch, device_manager, config, vocab_size)

                    loss = loss_fn(batch) / config.gradient_accumulation_steps
                    loss.backward()

                    accumulation_step += 1
                    window_loss += loss_to_float(loss)

                    if accumulation_step < config.gradient_accumulation_steps:
                        continue

                    device_manager.optimizer_step(optimizer)
                    running_loss += window_loss
                    window_loss = 0.0
                    accumulation_step = 0
                    global_step += 1

                    if config.measure_e2e_time:
                        logger.info(f"Step {global_step} e2e time: {time.perf_counter() - step_start:.3f}s")

                    if global_step % config.steps_freq == 0:
                        logger.log_metrics(
                            {"train/loss": running_loss / config.steps_freq}, step=global_step, commit=False
                        )
                        running_loss = 0.0

                    if global_step % config.val_steps_freq == 0:
                        model.eval()
                        val_loss = validate(loss_fn, eval_dataloader, device_manager, config, logger, vocab_size)
                        logger.log_metrics({"val/loss": val_loss}, step=global_step, commit=False)
                        model.train()

                    logger.log_metrics({}, step=global_step, commit=True)

                    if checkpoint_manager.should_save(global_step):
                        checkpoint_manager.save(model, global_step)

                if checkpoint_manager.should_save(global_step, epoch_end=True):
                    checkpoint_manager.save(model, global_step)

            if config.measure_e2e_time:
                logger.info(f"Training e2e time: {time.perf_counter() - train_start:.3f}s ({global_step} steps)")

            checkpoint_manager.save(model, global_step, name="final_model.pth")

    except Exception as e:
        logger.error(f"Training failed with error: {e}", traceback.format_exc())
        raise
    finally:
        logger.finish()


if __name__ == "__main__":
    default_config = Path(__file__).parent / "single_chip" / "llama_3_2_1b_sst2.yaml"
    args = parse_cli_options(default_config=default_config)
    config: TrainingConfig = generate_config(TrainingConfig, args.config, args.test_config)

    ReproducibilityManager(config).setup()

    logger = TrainingLogger(config, args.test_log_filename_prefix)
    device_manager = DeviceManager(config)
    logger.info(f"Using device: {device_manager.device}")
    if device_manager.mesh is not None:
        logger.info(f"Mesh: {tuple(device_manager.mesh.shape)} {device_manager.mesh.mesh_dim_names}")

    train(config, device_manager, logger, CheckpointManager(config, logger))
