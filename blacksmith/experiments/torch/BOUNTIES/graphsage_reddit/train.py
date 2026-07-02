
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from blacksmith.datasets.torch.BOUNTIES.reddit.reddit_dataset import RedditDataset
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
) -> tuple[float, float]:
    model.eval()
    total_loss = correct = total = 0
    for batch in loader:
        # NeighborLoader batches include seed nodes + sampled neighbours;
        # only the first batch_size entries are the target nodes.
        out = model(
            batch.x.to(device_manager.device),
            batch.edge_index.to(device_manager.device),
        )[: batch.batch_size]
        y = batch.y[: batch.batch_size].to(device_manager.device)
        total_loss += F.cross_entropy(out, y).item() * batch.batch_size
        correct += int((out.argmax(dim=1) == y).sum())
        total += batch.batch_size
    return total_loss / total, correct / total


def train(
    config: GraphSAGEConfig,
    device_manager: DeviceManager,
    logger: TrainingLogger,
    checkpoint_manager: CheckpointManager,
):
    logger.info("Starting training...")

    dataset = RedditDataset(config)
    logger.info(f"Dataset: Reddit | Nodes: {dataset.num_nodes:,} | Edges: {dataset.num_edges:,}")
    logger.info(f"Features: {dataset.num_features} | Classes: {dataset.num_classes}")
    logger.info(f"Train: {dataset.train_nodes:,}" f" | Val: {dataset.val_nodes:,}" f" | Test: {dataset.test_nodes:,}")

    train_loader = dataset.get_neighbour_loader("train")
    val_loader = dataset.get_neighbour_loader("val")
    test_loader = dataset.get_neighbour_loader("test")

    model = GraphSAGE(
        in_channels=dataset.num_features,
        hidden_channels=config.hidden_channels,
        out_channels=dataset.num_classes,
        dropout=config.dropout,
    ).to(device_manager.device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        capturable=config.use_tt,
    )

    if config.resume_from_checkpoint:
        checkpoint_manager.load_checkpoint(model, optimizer)

    global_step = 0
    max_steps = config.max_steps

    def _step_limit_reached() -> bool:
        return max_steps > 0 and global_step >= max_steps

    try:
        val_loss, val_acc = evaluate(model, val_loader, device_manager)
        logger.log_metrics({"val/loss": val_loss, "val/acc": val_acc}, step=global_step, commit=True)
        logger.info(f"Initial | val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        for epoch in range(1, config.num_epochs + 1):
            if _step_limit_reached():
                break

            model.train()
            epoch_loss = epoch_nodes = 0
            running_loss = running_nodes = 0

            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch:02d}/{config.num_epochs}",
                leave=False,
            )
            for batch in pbar:
                if _step_limit_reached():
                    logger.info(f"Reached max_steps={max_steps}, stopping early.")
                    break

                optimizer.zero_grad()
                out = model(
                    batch.x.to(device_manager.device),
                    batch.edge_index.to(device_manager.device),
                )[: batch.batch_size]
                y = batch.y[: batch.batch_size].to(device_manager.device)
                loss = F.cross_entropy(out, y)

                loss.backward()
                device_manager.optimizer_step(optimizer)
                global_step += 1

                n = batch.batch_size
                epoch_loss += loss.item() * n
                epoch_nodes += n
                running_loss += loss.item() * n
                running_nodes += n

                pbar.set_postfix(loss=f"{loss.item():.4f}")

                if global_step % config.steps_freq == 0:
                    step_loss = running_loss / running_nodes
                    logger.log_metrics({"train/loss": step_loss}, step=global_step, commit=False)
                    running_loss = running_nodes = 0

            if epoch_nodes == 0:
                break

            if running_nodes > 0:
                logger.log_metrics(
                    {"train/loss": running_loss / running_nodes},
                    step=global_step,
                    commit=False,
                )

            avg_epoch_loss = epoch_loss / epoch_nodes
            val_loss, val_acc = evaluate(model, val_loader, device_manager)

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

        test_loss, test_acc = evaluate(model, test_loader, device_manager)
        logger.log_summary({"test/loss": test_loss, "test/acc": test_acc})
        logger.info(f"Final test | loss={test_loss:.4f}  acc={test_acc:.4f}")

        final_path = checkpoint_manager.save_checkpoint(
            model,
            step=global_step,
            epoch=config.num_epochs,
            metrics={"val/acc": val_acc},
        )
        logger.log_artifact(final_path, artifact_type="model", name="final_model.pth")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", traceback.format_exc())
        raise
    finally:
        logger.finish()


if __name__ == "__main__":
    default_config = Path(__file__).parent / "single_chip" / "graphsage_reddit.yaml"
    args = parse_cli_options(default_config=default_config)
    config: GraphSAGEConfig = generate_config(GraphSAGEConfig, args.config, args.test_config)

    ReproducibilityManager(config).setup()

    logger = TrainingLogger(config, args.test_log_filename_prefix)
    device_manager = DeviceManager(config)
    logger.info(f"Using device: {device_manager.device}")

    checkpoint_manager = CheckpointManager(config, logger, device_manager.device)

    train(config, device_manager, logger, checkpoint_manager)
