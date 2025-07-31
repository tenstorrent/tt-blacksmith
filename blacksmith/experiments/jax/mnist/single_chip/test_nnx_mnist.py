# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import jax
import optax
import jax.numpy as jnp

from flax import nnx
from typing import Dict

from blacksmith.tools.jax_utils import init_device
from blacksmith.tools.cli import generate_config
from blacksmith.datasets.jax.mnist.dataloader import load_mnist_jax
from blacksmith.experiments.jax.mnist.configs import ExperimentConfig


def init_configs(config_path: Optional[str] = None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "..", "test_mnist.yaml")

    config = generate_config(ExperimentConfig, config_path)
    return config


def get_dataset():
    """Returns the MNIST dataset with integer labels (not one-hot)."""
    train_images, train_labels, val_images, val_labels, test_images, test_labels = load_mnist_jax()

    train_labels = jnp.argmax(train_labels, axis=-1)
    val_labels = jnp.argmax(val_labels, axis=-1)
    test_labels = jnp.argmax(test_labels, axis=-1)

    return train_images, train_labels, val_images, val_labels, test_images, test_labels


class MLP(nnx.Module):
    """A simple MLP model."""

    def __init__(self, *, rngs: nnx.Rngs, input_size: int = 784, hidden_size: int = 256, output_size: int = 10):
        self.linear1 = nnx.Linear(input_size, hidden_size, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.linear3 = nnx.Linear(hidden_size, output_size, rngs=rngs)

    def __call__(self, x: jnp.ndarray):
        x = x.reshape(x.shape[0], -1)
        x = nnx.relu(self.linear1(x))
        x = nnx.relu(self.linear2(x))
        x = self.linear3(x)
        return x


def loss_fn(model: MLP, batch: Dict[str, jnp.ndarray], cpu_device: jax.Device, tt_device: jax.Device):
    """Compute the loss and logits for a batch."""
    logits = model(batch["image"])

    with jax.default_device(cpu_device):
        one_hot_labels = jax.nn.one_hot(batch["label"].astype(jnp.int32), num_classes=10)
    one_hot_labels = jax.device_put(one_hot_labels, tt_device)
    log_probs = jax.nn.log_softmax(logits)
    loss = -jnp.sum(log_probs * one_hot_labels, axis=-1).mean()

    return loss, logits


def create_train_step(cpu_device: jax.Device, tt_device: jax.Device):
    @nnx.jit
    def train_step(model: MLP, metrics: nnx.MultiMetric, batch: Dict[str, jnp.ndarray]):
        """Train for a single step."""
        grad_fn = nnx.value_and_grad(lambda m, b: loss_fn(m, b, cpu_device, tt_device), has_aux=True)
        (loss, logits), grads = grad_fn(model, batch)
        labels = batch["label"].astype(jnp.int32)
        metrics.update(loss=loss, logits=logits, labels=labels)
        return grads, loss, logits

    return train_step


def create_eval_step(cpu_device: jax.Device, tt_device: jax.Device):
    @nnx.jit
    def eval_step(model: MLP, metrics: nnx.MultiMetric, batch: Dict[str, jnp.ndarray]):
        """Evaluate the model on a batch."""
        loss, logits = loss_fn(model, batch, cpu_device, tt_device)
        labels = batch["label"].astype(jnp.int32)
        metrics.update(loss=loss, logits=logits, labels=labels)

    return eval_step


def train():
    """Main training function."""

    config = init_configs()

    init_device()
    cpu_device = jax.devices("cpu")[0]
    tt_device = jax.devices("tt")[0]

    train_step = create_train_step(cpu_device, tt_device)
    eval_step = create_eval_step(cpu_device, tt_device)

    with jax.default_device(cpu_device):
        train_images, train_labels, val_images, val_labels, test_images, test_labels = get_dataset()
        rngs_host = nnx.Rngs(0)
        model = MLP(
            rngs=rngs_host,
            input_size=config.net_config.input_size,
            hidden_size=config.net_config.hidden_size,
            output_size=config.net_config.output_size,
        )

    momentum = 0.9
    batch_size = config.training_config.batch_size
    learning_rate = config.training_config.lr
    train_steps = len(train_images) // batch_size
    epochs = config.training_config.epochs

    # Initializing model parameters on CPU, since Jax random number generator
    # is currently not supported on device (https://github.com/tenstorrent/tt-xla/issues/420).
    graphdef, state = nnx.split(model)
    state_on_device = jax.device_put(state, tt_device)
    model = nnx.merge(graphdef, state_on_device)

    y = model(jnp.ones((1, 28, 28, 1)))

    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average("loss"),
    )

    metrics_history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        for step in range(train_steps):
            start = step * batch_size
            end = start + batch_size

            # Batch creation is done on CPU (https://github.com/tenstorrent/tt-mlir/issues/2309)
            with jax.default_device(cpu_device):
                x_batch_host = train_images[start:end]
                y_batch_host = train_labels[start:end]

            x_batch = jax.device_put(x_batch_host, tt_device)
            y_batch = jax.device_put(y_batch_host, tt_device)

            batch = {"image": x_batch, "label": y_batch}

            grads, loss, logits = train_step(model, metrics, batch)

            # Optimizer step is done on CPU (https://github.com/tenstorrent/tt-xla/issues/342)
            with jax.default_device(cpu_device):
                grads_cpu = jax.device_put(grads, cpu_device)
                graphdef, state_tt = nnx.split(model)
                state_cpu = jax.device_put(state_tt, cpu_device)
                model_cpu = nnx.merge(graphdef, state_cpu)

                optimizer_cpu = nnx.Optimizer(model_cpu, optax.adamw(learning_rate, momentum))
                optimizer_cpu.update(grads_cpu)

                graphdef_cpu, updated_state_cpu = nnx.split(optimizer_cpu.model)
                updated_state_tt = jax.device_put(updated_state_cpu, tt_device)
                model = nnx.merge(graphdef_cpu, updated_state_tt)

        for metric, value in metrics.compute().items():
            metrics_history[f"train_{metric}"].append(value)
        metrics.reset()

        test_steps = len(test_images) // batch_size
        for i in range(test_steps):
            start = i * batch_size
            end = start + batch_size

            with jax.default_device(cpu_device):
                x_batch_host = test_images[start:end]
                y_batch_host = test_labels[start:end]

            x_batch = jax.device_put(x_batch_host, tt_device)
            y_batch = jax.device_put(y_batch_host, tt_device)

            test_batch = {"image": x_batch, "label": y_batch}
            eval_step(model, metrics, test_batch)

        for metric, value in metrics.compute().items():
            metrics_history[f"test_{metric}"].append(value)
        metrics.reset()

        print(f"Epoch {epoch + 1} Results:")
        print(
            f"  Train Loss: {metrics_history['train_loss'][-1]:.4f}, Train Accuracy: {metrics_history['train_accuracy'][-1]:.4f}"
        )
        print(
            f"  Test Loss:  {metrics_history['test_loss'][-1]:.4f}, Test Accuracy:  {metrics_history['test_accuracy'][-1]:.4f}"
        )


if __name__ == "__main__":
    train()
