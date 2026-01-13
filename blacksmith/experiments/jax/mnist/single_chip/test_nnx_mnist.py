# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
import wandb
from flax import nnx

from blacksmith.datasets.jax.mnist.dataloader import load_mnist_jax
from blacksmith.experiments.jax.mnist.configs import ExperimentConfig
from blacksmith.tools.cli import generate_config, parse_cli_options

NUM_CLASSES = 10
DEFAULT_MOMENTUM = 0.9
DEFAULT_GRAD_CLIP_NORM = 1.0

DEFAULT_EXPERIMENT_NAME = "NNX-MNIST"
DEFAULT_RUN_NAME = "test-run"
DEFAULT_WANDB_DIR = "./wandb_logs"


def init_configs(config: ExperimentConfig) -> ExperimentConfig:
    return config


def get_dataset() -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Returns the MNIST dataset with integer labels (not one-hot)."""
    train_images, train_labels, val_images, val_labels, test_images, test_labels = load_mnist_jax()

    train_labels = jnp.argmax(train_labels, axis=-1)
    val_labels = jnp.argmax(val_labels, axis=-1)
    test_labels = jnp.argmax(test_labels, axis=-1)

    return train_images, train_labels, val_images, val_labels, test_images, test_labels


def create_batch(
    images: jnp.ndarray, labels: jnp.ndarray, start: int, end: int, cpu_device: jax.Device, tt_device: jax.Device
) -> Dict[str, jnp.ndarray]:
    """Create a batch and move it to appropriate devices."""
    with jax.default_device(cpu_device):
        x_batch_host = images[start:end]
        y_batch_host = labels[start:end]

    x_batch = jax.device_put(x_batch_host, tt_device)
    y_batch = jax.device_put(y_batch_host, tt_device)

    return {"image": x_batch, "label": y_batch}


class MLP(nnx.Module):
    """A simple MLP model."""

    def __init__(self, *, rngs: nnx.Rngs, input_size: int = 784, hidden_size: int = 256, output_size: int = 10) -> None:
        self.linear1 = nnx.Linear(input_size, hidden_size, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.linear3 = nnx.Linear(hidden_size, output_size, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.reshape(x.shape[0], -1)
        x = nnx.relu(self.linear1(x))
        x = nnx.relu(self.linear2(x))
        x = self.linear3(x)
        return x


def loss_fn(
    model: MLP, batch: Dict[str, jnp.ndarray], cpu_device: jax.Device, tt_device: jax.Device
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the loss and logits for a batch."""
    logits = model(batch["image"])

    with jax.default_device(cpu_device):
        one_hot_labels = jax.nn.one_hot(batch["label"].astype(jnp.int32), num_classes=NUM_CLASSES)
    one_hot_labels = jax.device_put(one_hot_labels, tt_device)
    log_probs = jax.nn.log_softmax(logits)
    loss = -jnp.sum(log_probs * one_hot_labels, axis=-1).mean()

    return loss, logits


@nnx.jit(static_argnames=["cpu_device", "tt_device"])
def training_step(
    model: MLP, metrics: nnx.MultiMetric, batch: Dict[str, jnp.ndarray], cpu_device: jax.Device, tt_device: jax.Device
):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(lambda m, b: loss_fn(m, b, cpu_device, tt_device), has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    labels = batch["label"].astype(jnp.int32)
    metrics.update(loss=loss, logits=logits, labels=labels)
    return grads, loss, logits


@nnx.jit(static_argnames=["cpu_device", "tt_device"])
def evaluation_step(
    model: MLP, metrics: nnx.MultiMetric, batch: Dict[str, jnp.ndarray], cpu_device: jax.Device, tt_device: jax.Device
):
    """Evaluate the model on a batch."""
    loss, logits = loss_fn(model, batch, cpu_device, tt_device)
    labels = batch["label"].astype(jnp.int32)
    metrics.update(loss=loss, logits=logits, labels=labels)


def setup_model_and_optimizer(
    config: ExperimentConfig, cpu_device: jax.Device, tt_device: jax.Device
) -> Tuple[nnx.Module, nnx.Optimizer]:
    """Initialize model and optimizer."""
    with jax.default_device(cpu_device):
        rngs_host = nnx.Rngs(0)
        model = MLP(
            rngs=rngs_host,
            input_size=config.net_config.input_size,
            hidden_size=config.net_config.hidden_size,
            output_size=config.net_config.output_size,
        )

    learning_rate = config.training_config.lr

    # Initializing model parameters on CPU, since Jax random number generator
    # is currently not supported on device (https://github.com/tenstorrent/tt-xla/issues/420).
    graphdef, state = nnx.split(model)
    state_on_device = jax.device_put(state, tt_device)
    model = nnx.merge(graphdef, state_on_device)

    _ = model(jnp.ones((1, 28, 28, 1)))

    with jax.default_device(cpu_device):
        graphdef, state_tt = nnx.split(model)
        state_cpu = jax.device_put(state_tt, cpu_device)
        model_cpu = nnx.merge(graphdef, state_cpu)
        optimizer_fn = optax.chain(
            optax.clip_by_global_norm(DEFAULT_GRAD_CLIP_NORM), optax.adamw(learning_rate, DEFAULT_MOMENTUM)
        )
        optimizer = nnx.Optimizer(model_cpu, optimizer_fn)

    return model, optimizer


def process_metrics_to_logs(metrics: nnx.MultiMetric, prefix: str, epoch_logs: Dict[str, Any]) -> None:
    """Process metrics and add to epoch logs with given prefix."""
    for metric, value in metrics.compute().items():
        if metric == "loss":
            epoch_logs[f"{prefix}/loss"] = float(value)
        elif metric == "accuracy":
            epoch_logs[f"{prefix}/accuracy"] = float(value)
    metrics.reset()


def run_validation(
    model: MLP,
    metrics: nnx.MultiMetric,
    val_images: jnp.ndarray,
    val_labels: jnp.ndarray,
    batch_size: int,
    cpu_device: jax.Device,
    tt_device: jax.Device,
    epoch_logs: Dict[str, Any],
) -> None:
    """Run validation and add results to epoch logs."""
    val_steps = len(val_images) // batch_size
    for i in range(val_steps):
        start = i * batch_size
        end = start + batch_size
        val_batch = create_batch(val_images, val_labels, start, end, cpu_device, tt_device)
        evaluation_step(model, metrics, val_batch, cpu_device, tt_device)

    process_metrics_to_logs(metrics, "val", epoch_logs)


def log_to_wandb(data_dict: Dict[str, Any], step: Optional[int] = None) -> None:
    """Helper function to log data to wandb."""
    wandb.log(data_dict, step=step)


def setup_wandb(config: ExperimentConfig) -> Any:
    """Setup wandb with error handling."""
    wandb_run = wandb.init(
        project=DEFAULT_EXPERIMENT_NAME,
        name=DEFAULT_RUN_NAME,
        dir=DEFAULT_WANDB_DIR,
        config=config.model_dump(),
    )
    print(f"Started wandb run: {wandb_run.name}")
    return wandb_run


@nnx.jit
def update_model_with_optimizer(
    graphdef: nnx.GraphDef,
    state_cpu: nnx.State,
    optimizer: nnx.Optimizer,
    grads_cpu: Dict[str, jnp.ndarray],
) -> nnx.State:
    """Apply optimizer update and return updated model state.

    All inputs and outputs are on CPU device. Device transfers done outside.
    Pure function that can be JIT compiled.
    """
    # Optimizer step is done on CPU because TT device doesn't support optimizer state operations
    # See: https://github.com/tenstorrent/tt-xla/issues/342
    optimizer.model = nnx.merge(graphdef, state_cpu)
    optimizer.update(grads_cpu)

    _, updated_state_cpu = nnx.split(optimizer.model)
    return updated_state_cpu


def run_final_test(
    model: nnx.Module,
    test_images: jnp.ndarray,
    test_labels: jnp.ndarray,
    batch_size: int,
    cpu_device: jax.Device,
    tt_device: jax.Device,
    wandb_run: Any,
    global_step: int,
) -> None:
    """Run final test evaluation and log results."""
    test_steps = len(test_images) // batch_size
    final_test_metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average("loss"),
    )

    for i in range(test_steps):
        start = i * batch_size
        end = start + batch_size
        test_batch = create_batch(test_images, test_labels, start, end, cpu_device, tt_device)
        evaluation_step(model, final_test_metrics, test_batch, cpu_device, tt_device)

    final_test_results = final_test_metrics.compute()
    final_test_logs = {f"final_test/{metric}": float(value) for metric, value in final_test_results.items()}
    log_to_wandb(final_test_logs, global_step)


def train(config: ExperimentConfig) -> None:
    """Main training function."""

    config = init_configs(config)
    wandb_run = setup_wandb(config)

    try:
        cpu_device = jax.devices("cpu")[0]
        tt_device = jax.devices("tt")[0]

        # Load dataset on CPU because TT device has dtype restrictions (int16 not supported)
        # and one_hot encoding operations are not supported on TT device.
        with jax.default_device(cpu_device):
            train_images, train_labels, val_images, val_labels, test_images, test_labels = get_dataset()

        model, optimizer = setup_model_and_optimizer(config, cpu_device, tt_device)

        batch_size = config.training_config.batch_size
        train_steps = len(train_images) // batch_size
        epochs = config.training_config.epochs

        metrics = nnx.MultiMetric(
            accuracy=nnx.metrics.Accuracy(),
            loss=nnx.metrics.Average("loss"),
        )

        global_step = 0
        for epoch in range(epochs):
            for step in range(train_steps):
                start = step * batch_size
                end = start + batch_size

                batch = create_batch(train_images, train_labels, start, end, cpu_device, tt_device)

                grads, loss, logits = training_step(model, metrics, batch, cpu_device, tt_device)

                # Transfer everything to CPU before optimizer update
                grads_cpu = jax.device_put(grads, cpu_device)
                graphdef, state_tt = nnx.split(model)
                state_cpu = jax.device_put(state_tt, cpu_device)

                # Pure optimizer update on CPU (JIT compiled)
                updated_state_cpu = update_model_with_optimizer(graphdef, state_cpu, optimizer, grads_cpu)

                # Transfer updated state back to TT device
                updated_state_tt = jax.device_put(updated_state_cpu, tt_device)
                model = nnx.merge(graphdef, updated_state_tt)

                global_step += 1

            epoch_logs = {"epoch": epoch + 1}
            process_metrics_to_logs(metrics, "train", epoch_logs)
            run_validation(model, metrics, val_images, val_labels, batch_size, cpu_device, tt_device, epoch_logs)
            log_to_wandb(epoch_logs, global_step - 1)

        run_final_test(model, test_images, test_labels, batch_size, cpu_device, tt_device, wandb_run, global_step)

    finally:
        wandb.finish()
        print("Finished wandb run")


if __name__ == "__main__":
    default_config = Path(__file__).parent.parent / "test_mnist.yaml"
    args = parse_cli_options(default_config=default_config)
    config: ExperimentConfig = generate_config(ExperimentConfig, args.config)
    train(config)
