# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import json
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

from blacksmith.experiments.torch.BOUNTIES.gatv2_pubmed.configs import TrainingConfig
from blacksmith.experiments.torch.BOUNTIES.gatv2_pubmed.model import GATv2
from blacksmith.tools.checkpoints_manager import CheckpointManager
from blacksmith.tools.cli import generate_config, parse_cli_options
from blacksmith.tools.device_manager import DeviceManager
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.reproducibility_manager import ReproducibilityManager


def load_dataset(config, logger):
    """Load PubMed dataset via Planetoid."""
    dataset = Planetoid(root=config.dataset_root, name=config.dataset_name)
    data = dataset[0]
    logger.info(f"Loaded {config.dataset_name} dataset:")
    logger.info(f"  Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    logger.info(f"  Features: {data.num_node_features}, Classes: {dataset.num_classes}")
    logger.info(f"  Train: {data.train_mask.sum()}, Val: {data.val_mask.sum()}, Test: {data.test_mask.sum()}")
    return data


def create_model(config, num_features, num_classes, device, logger):
    """Instantiate GATv2 model and move to device."""
    model = GATv2(
        in_channels=num_features,
        hidden_channels=config.hidden_channels,
        out_channels=num_classes,
        heads=config.heads,
        dropout=config.dropout,
    ).to(device)

    if config.use_tt and config.scatter_cpu_fallback:
        model.enable_cpu_fallback()
        logger.info("CPU fallback enabled for GATv2Conv (scatter ops unsupported on TT-XLA)")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params}")
    logger.info(f"Trainable parameters: {trainable_params}")
    return model


def train_epoch(model, data, optimizer, loss_fn, device_manager):
    """Single training step over the full graph."""
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    # out may be on CPU (fallback) or TT — align masks/labels
    train_mask = data.train_mask.to(out.device)
    y = data.y.to(out.device)
    loss = loss_fn(out[train_mask], y[train_mask])
    loss.backward()
    device_manager.optimizer_step(optimizer)
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask, loss_fn):
    """Evaluate model on nodes selected by mask. Returns (loss, accuracy)."""
    model.eval()
    out = model(data.x, data.edge_index)
    mask = mask.to(out.device)
    y = data.y.to(out.device)
    loss = loss_fn(out[mask], y[mask]).item()
    pred = out[mask].argmax(dim=1)
    accuracy = (pred == y[mask]).float().mean().item()
    return loss, accuracy


def generate_plots(metrics_history, output_dir):
    """Generate loss and accuracy plots."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = metrics_history["epochs"]

    # Loss curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, metrics_history["train_loss"], label="Train Loss")
    ax.plot(epochs, metrics_history["val_loss"], label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("GATv2 PubMed - Loss Curves")
    ax.legend()
    ax.grid(True)
    fig.savefig(output_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Accuracy curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, metrics_history["val_accuracy"], label="Val Accuracy", color="green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("GATv2 PubMed - Validation Accuracy")
    ax.legend()
    ax.grid(True)
    fig.savefig(output_dir / "accuracy_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def train(
    config: TrainingConfig,
    device_manager: DeviceManager,
    logger: TrainingLogger,
    checkpoint_manager: CheckpointManager,
):
    """Main training loop for GATv2 node classification on PubMed."""
    logger.info("Starting GATv2 PubMed training...")

    if config.use_tt:
        import torch_xla

        torch_xla.set_custom_compile_options({"fp32_dest_acc_en": True, "math_fidelity": "hifi4"})
        logger.info("TT device: compile options set " "(fp32 accumulation, hifi4 fidelity)")

    data = load_dataset(config, logger)
    data = data.to(device_manager.device)

    model = create_model(config, data.num_node_features, config.out_channels, device_manager.device, logger)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = F.nll_loss

    if config.resume_from_checkpoint:
        checkpoint_manager.load_checkpoint(model, optimizer)

    global_step = 0
    best_val_acc = 0.0
    patience_counter = 0
    best_model_path = None

    metrics_history = {"epochs": [], "train_loss": [], "val_loss": [], "val_accuracy": []}

    results_dir = Path(config.project_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Initial validation
        val_loss, val_acc = evaluate(model, data, data.val_mask, loss_fn)
        logger.log_metrics({"val/loss": val_loss, "val/accuracy": val_acc}, commit=True, step=global_step)

        for epoch in range(1, config.num_epochs + 1):
            global_step += 1

            # Train
            train_loss = train_epoch(model, data, optimizer, loss_fn, device_manager)

            # Log training loss
            logger.log_metrics({"train/loss": train_loss}, commit=False, step=global_step)

            # Validate
            if epoch % config.val_freq == 0:
                val_loss, val_acc = evaluate(model, data, data.val_mask, loss_fn)
                logger.log_metrics({"val/loss": val_loss, "val/accuracy": val_acc}, commit=False, step=global_step)

                metrics_history["epochs"].append(epoch)
                metrics_history["train_loss"].append(train_loss)
                metrics_history["val_loss"].append(val_loss)
                metrics_history["val_accuracy"].append(val_acc)

                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    best_model_path = checkpoint_manager.save_checkpoint(
                        model,
                        step=global_step,
                        epoch=epoch,
                        optimizer=optimizer,
                        metrics={"val/accuracy": val_acc, "val/loss": val_loss},
                        checkpoint_name="best_model.pt",
                    )
                else:
                    patience_counter += 1

                if patience_counter >= config.patience:
                    logger.info(f"Early stopping at epoch {epoch} (patience={config.patience})")
                    break

            logger.log_metrics({}, commit=True, step=global_step)

            # Periodic checkpoint
            if checkpoint_manager.should_save_checkpoint(global_step, epoch):
                checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)

        logger.info(f"Training finished. Best val accuracy: {best_val_acc:.4f}")

        # Test evaluation with best model
        if best_model_path is not None:
            checkpoint_manager.load_checkpoint_path(best_model_path, model)
            logger.info(f"Loaded best model from {best_model_path}")

        test_loss, test_acc = evaluate(model, data, data.test_mask, loss_fn)
        logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        logger.log_metrics({"test/loss": test_loss, "test/accuracy": test_acc}, commit=True, step=global_step + 1)

        # Generate artifacts
        generate_plots(metrics_history, results_dir)
        logger.info(f"Plots saved to {results_dir}")

        results_summary = {
            "model": config.model_name,
            "dataset": config.dataset_name,
            "device": str(device_manager.device),
            "use_tt": config.use_tt,
            "scatter_cpu_fallback": config.scatter_cpu_fallback,
            "best_val_accuracy": best_val_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "total_epochs": len(metrics_history["epochs"]),
            "hyperparameters": {
                "hidden_channels": config.hidden_channels,
                "heads": config.heads,
                "dropout": config.dropout,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
            },
        }
        summary_path = results_dir / "results_summary.json"
        with open(summary_path, "w") as f:
            json.dump(results_summary, f, indent=2)
        logger.info(f"Results summary saved to {summary_path}")

    except Exception as e:
        traceback_str = traceback.format_exc()
        logger.error(f"Training failed with error: {str(e)}", traceback_str)
        raise
    finally:
        logger.finish()


if __name__ == "__main__":
    # Config setup
    default_config = Path(__file__).parent / "test_gatv2_pubmed_training.yaml"
    args = parse_cli_options(default_config=default_config)
    config: TrainingConfig = generate_config(TrainingConfig, args.config)

    # Reproducibility setup
    repro_manager = ReproducibilityManager(config)
    repro_manager.setup()

    # Logger setup
    logger = TrainingLogger(config)

    # Device setup
    device_manager = DeviceManager(config)
    logger.info(f"Using device: {device_manager.device}")

    # Checkpoint manager setup
    checkpoint_manager = CheckpointManager(config, logger, device_manager.device)

    # Start training
    train(config, device_manager, logger, checkpoint_manager)
