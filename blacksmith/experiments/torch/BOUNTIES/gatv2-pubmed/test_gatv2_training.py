# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATv2Conv

from configs import TrainingConfig
from blacksmith.tools.cli import generate_config, parse_cli_options
from blacksmith.tools.reproducibility_manager import ReproducibilityManager


LOGGER = logging.getLogger("gatv2-pubmed")


class GATv2PubMed(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: int, dropout: float):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATv2Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
        )
        self.conv2 = GATv2Conv(
            in_channels=hidden_channels * heads,
            out_channels=out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def train_step(model: GATv2PubMed, data, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index)
    loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def evaluate(model: GATv2PubMed, data) -> Dict[str, float]:
    model.eval()
    logits = model(data.x, data.edge_index)

    train_loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask]).item()
    val_loss = F.cross_entropy(logits[data.val_mask], data.y[data.val_mask]).item()
    test_loss = F.cross_entropy(logits[data.test_mask], data.y[data.test_mask]).item()

    pred = logits.argmax(dim=1)
    train_acc = float((pred[data.train_mask] == data.y[data.train_mask]).float().mean().item())
    val_acc = float((pred[data.val_mask] == data.y[data.val_mask]).float().mean().item())
    test_acc = float((pred[data.test_mask] == data.y[data.test_mask]).float().mean().item())

    return {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
    }


def save_metrics_csv(rows: List[Dict[str, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "test_loss",
        "train_acc",
        "val_acc",
        "test_acc",
        "epoch_time_sec",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_curves(rows: List[Dict[str, float]], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        LOGGER.warning("matplotlib is not available; skipping plot generation: %s", exc)
        return

    epochs = [r["epoch"] for r in rows]
    train_loss = [r["train_loss"] for r in rows]
    val_loss = [r["val_loss"] for r in rows]
    train_acc = [r["train_acc"] for r in rows]
    val_acc = [r["val_acc"] for r in rows]
    test_acc = [r["test_acc"] for r in rows]

    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("GATv2 on PubMed - CPU baseline loss curves")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curves_cpu.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_acc, label="train_acc")
    plt.plot(epochs, val_acc, label="val_acc")
    plt.plot(epochs, test_acc, label="test_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("GATv2 on PubMed - CPU baseline accuracy curves")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_curves_cpu.png", dpi=150)
    plt.close()


def run_cpu_baseline(config: TrainingConfig) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    if config.use_tt:
        raise RuntimeError(
            "This script is the PR-1 CPU baseline for issue #453. "
            "Set `use_tt: false` in config. TT-N150 execution is delivered in PR-2."
        )

    device = torch.device(config.device)
    if device.type != "cpu":
        LOGGER.warning("CPU baseline expects `device=cpu`, got `%s`.", config.device)

    dataset = Planetoid(root=config.dataset_root, name=config.dataset_name)
    data = dataset[0].to(device)

    model = GATv2PubMed(
        in_channels=dataset.num_features,
        hidden_channels=config.hidden_channels,
        out_channels=dataset.num_classes,
        heads=config.heads,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    LOGGER.info("Dataset: %s, nodes=%d, edges=%d, classes=%d", config.dataset_name, data.num_nodes, data.num_edges, dataset.num_classes)
    LOGGER.info(
        "Model: %s hidden=%d heads=%d dropout=%.2f",
        config.model_name,
        config.hidden_channels,
        config.heads,
        config.dropout,
    )
    LOGGER.info(
        "Training: epochs=%d lr=%g weight_decay=%g early_stop_patience=%d",
        config.num_epochs,
        config.learning_rate,
        config.weight_decay,
        config.early_stop_patience,
    )

    rows: List[Dict[str, float]] = []
    best_val_acc = -1.0
    best_epoch = 0
    best_metrics: Dict[str, float] = {}
    best_state = None
    no_improve = 0

    for epoch in range(1, config.num_epochs + 1):
        start = time.perf_counter()
        train_loss = train_step(model, data, optimizer)
        eval_metrics = evaluate(model, data)
        elapsed = time.perf_counter() - start

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": eval_metrics["val_loss"],
            "test_loss": eval_metrics["test_loss"],
            "train_acc": eval_metrics["train_acc"],
            "val_acc": eval_metrics["val_acc"],
            "test_acc": eval_metrics["test_acc"],
            "epoch_time_sec": elapsed,
        }
        rows.append(row)

        if epoch == 1 or epoch % config.log_interval == 0:
            LOGGER.info(
                "epoch=%d train_loss=%.6f val_loss=%.6f train_acc=%.4f val_acc=%.4f test_acc=%.4f",
                epoch,
                row["train_loss"],
                row["val_loss"],
                row["train_acc"],
                row["val_acc"],
                row["test_acc"],
            )

        if row["val_acc"] > best_val_acc:
            best_val_acc = row["val_acc"]
            best_epoch = epoch
            best_metrics = row.copy()
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= config.early_stop_patience:
            LOGGER.info("Early stopping at epoch %d (no val_acc improvement for %d epochs).", epoch, config.early_stop_patience)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    summary = {
        "best_epoch": best_epoch,
        "best_train_loss": best_metrics.get("train_loss", 0.0),
        "best_val_loss": best_metrics.get("val_loss", 0.0),
        "best_test_loss": best_metrics.get("test_loss", 0.0),
        "best_train_acc": best_metrics.get("train_acc", 0.0),
        "best_val_acc": best_metrics.get("val_acc", 0.0),
        "best_test_acc": best_metrics.get("test_acc", 0.0),
        "epochs_ran": len(rows),
    }
    return rows, summary


def main() -> None:
    default_config = Path(__file__).parent / "test_gatv2_training.yaml"
    args = parse_cli_options(default_config=default_config)
    config: TrainingConfig = generate_config(TrainingConfig, args.config, args.test_config)

    setup_logging(config.log_level)

    repro_manager = ReproducibilityManager(config)
    repro_manager.setup()

    rows, summary = run_cpu_baseline(config)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_metrics_csv(rows, output_dir / "metrics_cpu.csv")
    save_curves(rows, output_dir)

    summary_path = output_dir / "summary_cpu.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    LOGGER.info("Saved metrics CSV to %s", output_dir / "metrics_cpu.csv")
    LOGGER.info("Saved summary JSON to %s", summary_path)
    LOGGER.info(
        "Best epoch=%d val_acc=%.4f test_acc=%.4f",
        summary["best_epoch"],
        summary["best_val_acc"],
        summary["best_test_acc"],
    )


if __name__ == "__main__":
    main()
