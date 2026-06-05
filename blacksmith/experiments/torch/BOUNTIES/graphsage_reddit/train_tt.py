# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GraphSAGE Reddit — TT hardware port (feasibility run, PR-2).

Why a separate script from train.py:
  SAGEConv uses scatter_reduce_ internally (PyG message-passing framework).
  TT-XLA cannot compile that op, so we replace the two SAGEConv layers with
  DenseGraphSAGE, which implements the same mean-aggregation via torch.mm on a
  row-normalised dense adjacency matrix built from edge_index on CPU.

  Math is identical to SAGEConv(aggr='mean'):
      out_i = lin_l(mean_{j in N(i)} x_j) + lin_r(x_i)

  Memory: dense adj is O(N²) per batch. With batch_size=32 and
  num_neighbors=[5, 3] the typical batch has ~200-700 nodes, so the matrix
  is well under 2 MB and safe for TT on-chip memory.

  Dynamic shapes: each NeighborLoader batch can have a different num_nodes,
  triggering TT-XLA recompilation. For a feasibility run (max_steps=10) this
  is acceptable; future work can pad to a fixed shape to avoid it.

Usage:
    python -m blacksmith.experiments.torch.BOUNTIES.graphsage_reddit.train_tt
"""
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader

from blacksmith.datasets.torch.BOUNTIES.reddit.reddit_dataset import get_reddit_loaders
from blacksmith.experiments.torch.BOUNTIES.graphsage_reddit.configs import GraphSAGEConfig
from blacksmith.tools.checkpoints_manager import CheckpointManager
from blacksmith.tools.cli import generate_config, parse_cli_options
from blacksmith.tools.device_manager import DeviceManager
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.reproducibility_manager import ReproducibilityManager


class DenseGraphSAGE(torch.nn.Module):
    """GraphSAGE with dense mean aggregation — TT-XLA compatible.

    Matches SAGEConv(aggr='mean', root_weight=True) parameter count exactly:
        out_i = lin_l(mean_j x_j) + lin_r(x_i)
    Uses torch.mm instead of scatter ops so TT-XLA can compile the whole graph.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float,
    ):
        super().__init__()
        self.lin1_l = torch.nn.Linear(in_channels, hidden_channels, bias=True)
        self.lin1_r = torch.nn.Linear(in_channels, hidden_channels, bias=False)
        self.lin2_l = torch.nn.Linear(hidden_channels, out_channels, bias=True)
        self.lin2_r = torch.nn.Linear(hidden_channels, out_channels, bias=False)
        self.dropout = dropout

    def _sage_layer(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        lin_l: torch.nn.Linear,
        lin_r: torch.nn.Linear,
    ) -> torch.Tensor:
        agg = torch.mm(adj, x)
        return lin_l(agg) + lin_r(x)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self._sage_layer(x, adj, self.lin1_l, self.lin1_r)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self._sage_layer(x, adj, self.lin2_l, self.lin2_r)
        return x


def make_dense_adj(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Convert COO edge_index to row-normalised dense adjacency on CPU.

    NeighborLoader convention: edge_index[0]=source, edge_index[1]=target.
    SAGEConv aggregates neighbours of each target node, so adj[target, source]=1.
    """
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    adj[edge_index[1], edge_index[0]] = 1.0
    deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
    return adj / deg


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: NeighborLoader,
    device_manager: DeviceManager,
    max_batches: int = 5,
) -> tuple[float, float]:
    model.eval()
    total_loss = correct = total = 0
    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        num_nodes = batch.x.size(0)
        adj = make_dense_adj(batch.edge_index, num_nodes).to(device_manager.device)
        out = model(batch.x.to(device_manager.device), adj)[: batch.batch_size]
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
    logger.info("Starting TT feasibility training (DenseGraphSAGE)...")

    loaders = get_reddit_loaders(config)
    logger.info(
        f"Dataset: Reddit | Nodes: {loaders.num_nodes:,} | Edges: {loaders.num_edges:,}"
    )
    logger.info(f"Features: {loaders.num_features} | Classes: {loaders.num_classes}")
    logger.info(
        f"Train: {loaders.train_nodes:,}"
        f" | Val: {loaders.val_nodes:,}"
        f" | Test: {loaders.test_nodes:,}"
    )

    model = DenseGraphSAGE(
        in_channels=loaders.num_features,
        hidden_channels=config.hidden_channels,
        out_channels=loaders.num_classes,
        dropout=config.dropout,
    ).to(device_manager.device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        capturable=True,
    )

    if config.resume_from_checkpoint:
        checkpoint_manager.load_checkpoint(model, optimizer)

    global_step = 0
    max_steps = config.max_steps  # -1 = run all batches

    def _step_limit_reached() -> bool:
        return max_steps > 0 and global_step >= max_steps

    try:
        for epoch in range(1, config.num_epochs + 1):
            if _step_limit_reached():
                break

            model.train()
            epoch_loss = epoch_nodes = 0

            for batch in loaders.train_loader:
                if _step_limit_reached():
                    logger.info(f"Reached max_steps={max_steps}, stopping early.")
                    break

                num_nodes = batch.x.size(0)
                # Build dense adj on CPU; shape varies per batch (triggers recompile on TT).
                adj = make_dense_adj(batch.edge_index, num_nodes).to(device_manager.device)

                optimizer.zero_grad()
                out = model(batch.x.to(device_manager.device), adj)[: batch.batch_size]
                y = batch.y[: batch.batch_size].to(device_manager.device)
                loss = F.cross_entropy(out, y)
                loss.backward()
                device_manager.optimizer_step(optimizer)

                global_step += 1
                n = batch.batch_size
                epoch_loss += loss.item() * n
                epoch_nodes += n

                if global_step % config.steps_freq == 0:
                    logger.log_metrics(
                        {"train/loss": loss.item()}, step=global_step, commit=False
                    )
                    logger.info(
                        f"Step {global_step:4d} | nodes={num_nodes} | loss={loss.item():.4f}"
                    )

            if epoch_nodes == 0:
                break

            avg_epoch_loss = epoch_loss / epoch_nodes
            val_loss, val_acc = evaluate(model, loaders.val_loader, device_manager)

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

        logger.info("TT feasibility test complete.")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", traceback.format_exc())
        raise
    finally:
        logger.finish()


if __name__ == "__main__":
    default_config = Path(__file__).parent / "tt_single_chip" / "graphsage_reddit_tt.yaml"
    args = parse_cli_options(default_config=default_config)
    config: GraphSAGEConfig = generate_config(
        GraphSAGEConfig, args.config, args.test_config
    )

    ReproducibilityManager(config).setup()

    logger = TrainingLogger(config, args.test_log_filename_prefix)
    device_manager = DeviceManager(config)
    logger.info(f"Using device: {device_manager.device}")

    checkpoint_manager = CheckpointManager(config, logger, device_manager.device)

    train(config, device_manager, logger, checkpoint_manager)
