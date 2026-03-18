# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import logging
import os
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from blacksmith.datasets.torch.BOUNTIES.wikitext.wikitext_dataset import WikitextDataset
from blacksmith.experiments.torch.gpt_oss.configs import TrainingConfig
from blacksmith.models.torch.gpt_oss.expert_parallel import (
    ExpertParallelMLP,
    build_ep_model,
    sync_replicated_gradients,
)
from blacksmith.tools.checkpoints_manager import CheckpointManager
from blacksmith.tools.cli import generate_config, parse_cli_options
from blacksmith.tools.distributed import is_main_process, setup_distributed
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.reproducibility_manager import ReproducibilityManager
from blacksmith.tools.torch_helpers import collect_examples, show_examples

# This experiment is a native PyTorch/NCCL job. In environments where
# `torch_xla` is installed, Transformers may otherwise initialize PJRT/XLA
# during import and allocate auxiliary CUDA contexts on rank 0 from all ranks.
os.environ.setdefault("USE_TORCH_XLA", "0")


def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    ep_group: dist.ProcessGroup,
    logger: TrainingLogger,
    tokenizer,
    config: TrainingConfig,
) -> float:
    logger.info("Starting validation...")

    total_loss = torch.tensor(0.0, device=device)
    n_batches = torch.tensor(0, device=device)
    collected_examples = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            total_loss += out.loss.detach()
            n_batches += 1
            if config.print_examples:
                predictions = out.logits[:, :-1, :].argmax(dim=-1)
                expected_output = labels[:, 1:]

                collected_examples = collect_examples(
                    batch_size=input_ids.shape[0],
                    collected_examples=collected_examples,
                    max_examples=1,
                    input_ids=input_ids,
                    expected_output=expected_output,
                    predictions=predictions,
                    num_val_batches=n_batches.item(),
                )

            # Clear up memory.
            del input_ids, attention_mask, labels, out
            # Free cached CUDA memory once after all validation batches.
            torch.cuda.empty_cache()

    if config.print_examples and tokenizer is not None and collected_examples:
        rank = dist.get_rank(ep_group)
        logger.info(f"[Rank {rank}] Printing validation examples...")
        show_examples(collected_examples, tokenizer, config, logger)

    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM, group=ep_group)
    dist.all_reduce(n_batches, op=dist.ReduceOp.SUM, group=ep_group)

    avg_loss = (total_loss / n_batches).item() if n_batches.item() > 0 else 0.0
    return avg_loss


def train(
    config: TrainingConfig,
    rank: int,
    device: torch.device,
    ep_group: dist.ProcessGroup,
    logger: TrainingLogger,
    checkpoint_manager: CheckpointManager,
) -> None:
    world_size = dist.get_world_size(ep_group)

    model, tokenizer = build_ep_model(config, ep_group, device)
    logger.info(f"Loaded {config.model_name} model.")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if is_main_process():
        for mod in model.modules():
            if not isinstance(mod, ExpertParallelMLP):
                continue
            logger.info(
                f"Expert parallel: {mod.num_experts_global} experts / {world_size} GPUs = {mod.n_local} per GPU"
            )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    if config.resume_from_checkpoint:
        checkpoint_manager.load_checkpoint(model, optimizer)

    train_ds = WikitextDataset(config, split="train", rank=rank, world_size=world_size)
    val_ds = WikitextDataset(config, split="validation", rank=rank, world_size=world_size)
    train_loader = train_ds.get_dataloader()
    val_loader = val_ds.get_dataloader()
    logger.info(f"Loaded {config.dataset_id} dataset. Train batches: {len(train_loader)}")
    logger.info(f"Loaded {config.dataset_id} dataset. Val batches: {len(val_loader)}")

    global_step = 0
    accumulation_step = 0

    try:
        model.eval()
        val_loss = validate(model, val_loader, device, ep_group, logger, tokenizer, config)
        logger.log_metrics({"val/loss": val_loss}, commit=True, step=global_step)
        model.train()

        for epoch in range(config.num_epochs):
            train_loader.sampler.set_epoch(epoch)

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                if accumulation_step == 0:
                    optimizer.zero_grad()
                    accumulated_loss = torch.tensor(0.0, device=device)

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = out.loss / config.gradient_accumulation_steps

                loss.backward()
                accumulated_loss += loss.detach()
                accumulation_step += 1

                if accumulation_step < config.gradient_accumulation_steps:
                    continue

                accumulation_step = 0
                global_step += 1

                # Expert params are rank-local; all others must be averaged across ranks.
                sync_replicated_gradients(model, ep_group)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

                dist.all_reduce(accumulated_loss, op=dist.ReduceOp.AVG, group=ep_group)

                logger.log_metrics({"train/loss": accumulated_loss}, commit=False, step=global_step)

                if global_step % config.val_steps_freq == 0:
                    model.eval()
                    val_loss = validate(
                        model,
                        val_loader,
                        device,
                        ep_group,
                        logger,
                        tokenizer,
                        config,
                    )
                    logger.log_metrics({"val/loss": val_loss}, commit=False, step=global_step)
                    model.train()

                if checkpoint_manager.should_save_checkpoint(global_step):
                    checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)

                dist.barrier()

            if checkpoint_manager.should_save_checkpoint(global_step, epoch):
                checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)

        final_path = checkpoint_manager.save_checkpoint(
            model,
            global_step,
            config.num_epochs - 1,
            optimizer,
            checkpoint_name="final_model.pt",
        )
        logger.log_artifact(final_path, artifact_type="model", name="final_model.pt")

    except Exception as e:
        traceback_str = traceback.format_exc()
        logger.error(f"Training failed with error: {str(e)}", traceback_str)
        raise
    finally:
        logger.finish()


if __name__ == "__main__":
    default_config = Path(__file__).parent / "test_gpt_oss_ep.yaml"
    args = parse_cli_options(default_config=default_config)
    config: TrainingConfig = generate_config(TrainingConfig, args.config, args.test_config)

    # DeviceManager is not used here because it targets XLA/TT devices with mesh
    # sharding. This experiment uses native PyTorch/NCCL for expert parallelism,
    # which requires direct control over process groups and collectives.
    rank, _local_rank, device = setup_distributed()
    ep_group = dist.group.WORLD

    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format=f"%(asctime)s [rank{rank}] %(levelname)s %(name)s: %(message)s",
    )

    repro_manager = ReproducibilityManager(config)
    repro_manager.setup()

    # Only rank-0 logs to W&B to avoid duplicate runs.
    if not is_main_process():
        config.use_wandb = False
    logger = TrainingLogger(config, args.test_log_filename_prefix)
    logger.info(f"Rank {rank}/{dist.get_world_size()} | device: {device}")

    checkpoint_manager = CheckpointManager(config, logger)

    try:
        train(config, rank, device, ep_group, logger, checkpoint_manager)
    finally:
        dist.destroy_process_group()
