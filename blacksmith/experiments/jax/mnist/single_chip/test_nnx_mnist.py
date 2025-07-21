# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import jax
import jax._src.xla_bridge as xb
import os
import sys
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

plt.ion()
from flax import nnx
from functools import partial
import jax.numpy as jnp
import optax
from blacksmith.datasets.jax.mnist.dataloader import load_mnist_jax
from jax import random


def init_device():
    priority = 500
    backend = "tt"
    path = os.path.join(os.getcwd(), "build/src/tt/pjrt_plugin_tt.so")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find tt_pjrt C API plugin at {path}")

    print("Loading tt_pjrt C API plugin", file=sys.stderr)
    xb.discover_pjrt_plugins()

    plugin = xb.register_plugin(backend, priority=priority, library_path=path, options=None)
    print("Loaded", file=sys.stderr)
    jax.config.update("jax_platforms", "tt,cpu")


def get_dataset(train_steps, batch_size):
    """Returns the MNIST dataset."""
    return load_mnist_jax()


class CNN(nnx.Module):
    """A simple CNN model."""

    def __init__(self, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(784, 256, rngs=rngs)
        self.linear2 = nnx.Linear(256, 256, rngs=rngs)
        self.linear3 = nnx.Linear(256, 10, rngs=rngs)

    def __call__(self, x):
        x = x.reshape(x.shape[0], -1)
        x = nnx.relu(self.linear1(x))
        x = nnx.relu(self.linear2(x))
        x = self.linear3(x)
        return x


def loss_fn(model: CNN, batch):
    """Compute the loss and logits for a batch."""
    logits = model(batch["image"])

    with jax.default_device(cpu_device):
        one_hot_labels = jax.nn.one_hot(batch["label"].astype(jnp.int32), num_classes=10)
    one_hot_labels = jax.device_put(one_hot_labels, tt_device)
    log_probs = jax.nn.log_softmax(logits)
    loss = -jnp.sum(log_probs * one_hot_labels, axis=-1).mean()

    return loss, logits


@nnx.jit
def train_step(model: CNN, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    labels = batch["label"].astype(jnp.int32)
    metrics.update(loss=loss, logits=logits, labels=labels)
    return grads, loss, logits


@nnx.jit
def eval_step(model: CNN, metrics: nnx.MultiMetric, batch):
    """Evaluate the model on a batch."""
    loss, logits = loss_fn(model, batch)
    labels = batch["label"].astype(jnp.int32)
    metrics.update(loss=loss, logits=logits, labels=labels)


init_device()
cpu_device = jax.devices("cpu")[0]
tt_device = jax.devices("tt")[0]


train_steps = 1500
eval_every = 200
batch_size = 32

with jax.default_device(cpu_device):
    train_images, train_labels, val_images, val_labels, test_images, test_labels = get_dataset(train_steps, batch_size)

with jax.default_device(cpu_device):
    rngs_host = nnx.Rngs(0)

with jax.default_device(cpu_device):
    model = CNN(rngs=rngs_host)

# Initializing model parameters on CPU, since Jax random number generator
# is currently not supported on device (https://github.com/tenstorrent/tt-xla/issues/420).
graphdef, state = nnx.split(model)
state_on_device = jax.device_put(state, tt_device)
model = nnx.merge(graphdef, state_on_device)

y = model(jnp.ones((1, 28, 28, 1)))

learning_rate = 0.005
momentum = 0.9

optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, momentum))
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


for step in range(train_steps):
    print(step)
    start = step * batch_size
    end = start + batch_size

    # Batch creation is done on CPU (https://github.com/tenstorrent/tt-mlir/issues/2309)
    with jax.default_device(cpu_device):
        x_batch_host = train_images[start:end]
        y_batch_host = train_labels[start:end]

    x_batch = jax.device_put(x_batch_host, tt_device)
    y_batch = jax.device_put(y_batch_host, tt_device)

    batch = {"image": x_batch, "label": y_batch}

    grads, loss, logits = train_step(model, optimizer, metrics, batch)

    loss_val = jax.device_get(loss)
    logits_val = jax.device_get(logits)

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

    if step > 0 and (step % eval_every == 0 or step == train_steps - 1):

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

        print(f"Step {step}:")
        print(
            f"  Train Loss: {metrics_history['train_loss'][-1]:.4f}, Train Accuracy: {metrics_history['train_accuracy'][-1]:.4f}"
        )
        print(
            f"  Test Loss:  {metrics_history['test_loss'][-1]:.4f}, Test Accuracy:  {metrics_history['test_accuracy'][-1]:.4f}"
        )
