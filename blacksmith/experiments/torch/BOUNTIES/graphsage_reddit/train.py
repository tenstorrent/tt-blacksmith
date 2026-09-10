# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import traceback
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import torch
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from blacksmith.datasets.torch.BOUNTIES.reddit.reddit_dataset import RedditDataset
from blacksmith.experiments.torch.BOUNTIES.graphsage_reddit.batching import (
    masked_correct,
    masked_cross_entropy,
    prepare_neighbor_batch,
    sampled_graph_capacity,
)
from blacksmith.experiments.torch.BOUNTIES.graphsage_reddit.configs import (
    GraphSAGEConfig,
)
from blacksmith.models.torch.graphsage.graphsage import GraphSAGE
from blacksmith.tools.checkpoints_manager import CheckpointManager
from blacksmith.tools.cli import generate_config, parse_cli_options
from blacksmith.tools.device_manager import DeviceManager
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.reproducibility_manager import ReproducibilityManager


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: NeighborLoader,
    device_manager: DeviceManager,
    config: GraphSAGEConfig,
    seed_capacity: int,
) -> tuple[float, float]:
    model.eval()
    total_loss = correct = total = 0
    for batch in loader:
        prepared = prepare_neighbor_batch(
            batch=batch,
            device=device_manager.device,
            seed_capacity=seed_capacity,
            num_neighbors=config.num_neighbors,
            static_shapes=config.static_shapes,
        )
        out = model(prepared.x, prepared.edge_index)[: prepared.target_capacity]
        loss = masked_cross_entropy(out, prepared.target_y, prepared.target_mask)
        total_loss += loss.item() * prepared.target_count
        correct += int(masked_correct(out, prepared.target_y, prepared.target_mask).item())
        total += prepared.target_count
    if total == 0:
        raise ValueError("evaluation loader yielded no seed nodes")
    return total_loss / total, correct / total


def train_epoch(
    model: torch.nn.Module,
    loader: NeighborLoader,
    optimizer: torch.optim.Optimizer,
    device_manager: DeviceManager,
    logger: TrainingLogger,
    config: GraphSAGEConfig,
    epoch: int,
    global_step: int,
    *,
    checkpoint_manager: CheckpointManager,
) -> tuple[int, float | None]:
    model.train()
    epoch_loss = epoch_nodes = 0
    running_loss = running_nodes = 0
    running_step_time = 0.0
    running_steps = 0

    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch:02d}/{config.num_epochs}",
        leave=False,
    )
    for batch in pbar:
        # NeighborLoader puts seed nodes first. Batch preparation preserves that
        # prefix and masks any static-shape padding out of the training loss.
        prepared = prepare_neighbor_batch(
            batch=batch,
            device=device_manager.device,
            seed_capacity=config.batch_size,
            num_neighbors=config.num_neighbors,
            static_shapes=config.static_shapes,
        )
        step_start = perf_counter()
        optimizer.zero_grad()
        out = model(prepared.x, prepared.edge_index)[: prepared.target_capacity]
        loss = masked_cross_entropy(out, prepared.target_y, prepared.target_mask)

        loss.backward()
        device_manager.optimizer_step(optimizer)
        global_step += 1
        step_time = perf_counter() - step_start

        node_count = prepared.target_count
        weighted_loss = loss.item() * node_count
        epoch_loss += weighted_loss
        epoch_nodes += node_count
        running_loss += weighted_loss
        running_nodes += node_count
        running_step_time += step_time
        running_steps += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}")

        if global_step == 1:
            logger.log_metrics(
                {"train/compile_and_first_step_time_s": step_time},
                step=global_step,
                commit=False,
            )
            running_loss = running_nodes = 0
            running_step_time = 0.0
            running_steps = 0
        elif global_step % config.steps_freq == 0:
            logger.log_metrics(
                {
                    "train/loss": running_loss / running_nodes,
                    "train/model_step_time_s": running_step_time / running_steps,
                    "train/seed_nodes_per_s": running_nodes / running_step_time,
                },
                step=global_step,
                commit=False,
            )
            running_loss = running_nodes = 0
            running_step_time = 0.0
            running_steps = 0

        if checkpoint_manager.should_save_checkpoint(global_step):
            # Only completed epochs are recorded. A mid-epoch checkpoint
            # restarts this epoch on resume; the sampler/RNG state is not saved.
            checkpoint_manager.save_checkpoint(
                model,
                step=global_step,
                epoch=epoch - 1,
                optimizer=optimizer,
            )

    if epoch_nodes == 0:
        return global_step, None

    if running_nodes > 0:
        logger.log_metrics(
            {
                "train/loss": running_loss / running_nodes,
                "train/model_step_time_s": running_step_time / running_steps,
                "train/seed_nodes_per_s": running_nodes / running_step_time,
            },
            step=global_step,
            commit=False,
        )

    return global_step, epoch_loss / epoch_nodes


def _resume_training_state(
    config: GraphSAGEConfig,
    checkpoint_manager: CheckpointManager,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    logger: TrainingLogger,
) -> tuple[int, int]:
    """Restore the training counters recorded alongside a checkpoint."""
    if not config.resume_from_checkpoint:
        return 0, 0

    checkpoint_info = checkpoint_manager.load_checkpoint(model, optimizer)
    if checkpoint_info is None:
        return 0, 0

    global_step = checkpoint_info.get("step", 0)
    completed_epoch = checkpoint_info.get("epoch", 0)
    logger.info(f"Resuming after epoch {completed_epoch} at step {global_step}")
    return global_step, completed_epoch


def train(
    config: GraphSAGEConfig,
    device_manager: DeviceManager,
    logger: TrainingLogger,
    checkpoint_manager: CheckpointManager,
) -> None:
    logger.info("Starting training...")

    dataset = RedditDataset(config)
    logger.info(f"Dataset: Reddit | Nodes: {dataset.num_nodes:,} | Edges: {dataset.num_edges:,}")
    logger.info(f"Features: {dataset.num_features} | Classes: {dataset.num_classes}")
    logger.info(f"Train: {dataset.train_nodes:,}" f" | Val: {dataset.val_nodes:,}" f" | Test: {dataset.test_nodes:,}")

    if config.static_shapes:
        train_capacity = sampled_graph_capacity(config.batch_size, config.num_neighbors)
        eval_capacity = sampled_graph_capacity(config.val_batch_size, config.num_neighbors)
        logger.info(
            "Static graph capacities (nodes, edges): " f"train={train_capacity}, validation/test={eval_capacity}"
        )

    train_loader = dataset.get_neighbour_loader("train")
    val_loader = dataset.get_neighbour_loader("val")
    test_loader = dataset.get_neighbour_loader("test")

    model = GraphSAGE(
        in_channels=dataset.num_features,
        hidden_channels=config.hidden_channels,
        out_channels=dataset.num_classes,
        dropout=config.dropout,
        use_spmm=config.use_spmm,
    ).to(device_manager.device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        capturable=config.use_tt,
    )

    global_step, completed_epoch = _resume_training_state(
        config,
        checkpoint_manager,
        model,
        optimizer,
        logger,
    )
    try:
        val_loss, val_acc = evaluate(
            model,
            val_loader,
            device_manager,
            config,
            config.val_batch_size,
        )
        logger.log_metrics({"val/loss": val_loss, "val/acc": val_acc}, step=global_step, commit=True)
        logger.info(f"Initial | val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        for epoch in range(completed_epoch + 1, config.num_epochs + 1):
            global_step, avg_epoch_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                device_manager,
                logger,
                config,
                epoch,
                global_step,
                checkpoint_manager=checkpoint_manager,
            )
            if avg_epoch_loss is None:
                break
            completed_epoch = epoch
            val_loss, val_acc = evaluate(
                model,
                val_loader,
                device_manager,
                config,
                config.val_batch_size,
            )

            logger.log_metrics(
                {
                    "train/epoch_loss": avg_epoch_loss,
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                },
                step=global_step,
                commit=True,
            )
            logger.info(
                f"Epoch {epoch:02d}/{config.num_epochs} | "
                f"train_loss={avg_epoch_loss:.4f}"
                f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            )

            if checkpoint_manager.should_save_checkpoint(epoch, epoch=epoch):
                checkpoint_manager.save_checkpoint(
                    model,
                    step=global_step,
                    epoch=epoch,
                    optimizer=optimizer,
                    metrics={"val/acc": val_acc},
                )

        test_loss, test_acc = evaluate(
            model,
            test_loader,
            device_manager,
            config,
            config.val_batch_size,
        )
        logger.log_summary({"test/loss": test_loss, "test/acc": test_acc})
        logger.info(f"Final test | loss={test_loss:.4f}  acc={test_acc:.4f}")

        if config.save_strategy != "none":
            final_path = checkpoint_manager.save_checkpoint(
                model,
                step=global_step,
                epoch=completed_epoch,
                optimizer=optimizer,
                metrics={"val/acc": val_acc},
                # Do not reuse an epoch checkpoint's second-resolution name:
                # retention could delete the newly written final checkpoint.
                checkpoint_name=f"checkpoint_step{global_step}_epoch{completed_epoch}_final_{uuid4().hex}.pt",
            )
            logger.log_artifact(final_path, artifact_type="model", name="final_model.pth")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", traceback.format_exc())
        raise
    finally:
        logger.finish()


def main() -> None:
    default_config = Path(__file__).parent / "single_chip" / "graphsage_reddit.yaml"
    args = parse_cli_options(default_config=default_config)
    config: GraphSAGEConfig = generate_config(
        GraphSAGEConfig, args.config, args.test_config, test_checkpoint_path=args.test_checkpoint_path
    )

    ReproducibilityManager(config).setup()

    logger = TrainingLogger(config, args.test_log_filename_prefix)
    device_manager = DeviceManager(config)
    logger.info(f"Using device: {device_manager.device}")

    checkpoint_manager = CheckpointManager(config, logger, device_manager.device)

    train(config, device_manager, logger, checkpoint_manager)


if __name__ == "__main__":
    main()
