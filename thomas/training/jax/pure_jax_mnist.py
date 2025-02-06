# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from keras import datasets

import jax
import jax.numpy as jnp
from jax import random
import jax._src.xla_bridge as xb

import os
import sys

import wandb

from DataLoader import load_mnist
from logg_it import init_wandb, log_metrics, save_checkpoint, load_checkpoint


def init_device():
    backend = "tt"
    path = os.path.join(os.getcwd(), "third_party/tt-xla/build/src/tt/pjrt_plugin_tt.so")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find tt_pjrt C API plugin at {path}")

    print("Loading tt_pjrt C API plugin", file=sys.stderr)
    xb.discover_pjrt_plugins()

    plugin = xb.register_plugin("tt", priority=500, library_path=path, options=None)
    print("Loaded", file=sys.stderr)
    jax.config.update("jax_platforms", "tt,cpu")


@jax.jit
def mlp_model(params, x):
    w1, b1, w2, b2 = params
    h1 = jnp.maximum(jnp.dot(x, w1) + b1, 0.0)
    logits = jnp.dot(h1, w2) + b2
    return logits


# Initialize the model parameters with specified shapes
def init_mlp_params(key, input_dim=28 * 28, hidden_dim=64, output_dim=10):
    # Define shapes for weights and biases
    w1_shape = (input_dim, hidden_dim)
    b1_shape = (1, hidden_dim)
    w2_shape = (hidden_dim, output_dim)
    b2_shape = (1, output_dim)

    w1 = random.normal(key, w1_shape)
    w1 = w1.astype(jnp.float32)
    b1 = jnp.zeros(b1_shape, dtype=jnp.float32)
    w2 = random.normal(key, w2_shape)
    w2 = w2.astype(jnp.float32)
    b2 = jnp.zeros(b2_shape, dtype=jnp.float32)

    return (w1, b1, w2, b2)


@jax.jit
def mse_loss(params, x, y):
    logits = mlp_model(params, x)
    return jnp.mean((logits - y) ** 2)


# Gradient descent update rule (simple SGD)
@jax.jit
def update(params, x_batch, y_batch, learning_rate):
    grads = jax.grad(mse_loss, argnums=0)(params, x_batch, y_batch)
    w1, b1, w2, b2 = params
    dw1, db1, dw2, db2 = grads
    updated_params = (
        w1 - learning_rate * dw1,
        b1 - learning_rate * db1,
        w2 - learning_rate * dw2,
        b2 - learning_rate * db2,
    )

    return updated_params


def argmax_on_cpu(array):
    array_cpu = jax.device_put(array, jax.devices("cpu")[0])
    with jax.default_device(jax.devices("cpu")[0]):
        argmax_result = jnp.argmax(array_cpu, axis=-1)
        argmax_result = argmax_result.astype(jnp.uint32)
    return jax.device_put(argmax_result, jax.devices()[0])


def compute_accuracy(logits, y):

    predictions = argmax_on_cpu(logits)
    true_labels = argmax_on_cpu(y)

    if predictions != true_labels:
        correct = 0
    else:
        correct = 1

    return correct


# Training loop
def train_mlp(
    x_trainh, y_trainh, x_valh, y_valh, x_testh, y_testh, key, num_epochs=10, batch_size=1, learning_rate=1e-3
):
    input_dim = x_trainh.shape[1]
    hidden_dim = 64
    output_dim = 10
    current_device = jax.devices()[0]
    # Initialize model parameters
    with jax.default_device(jax.devices("cpu")[0]):
        paramsh = init_mlp_params(key, input_dim, hidden_dim, output_dim)

    params = jax.device_put(paramsh, current_device)

    num_batches = x_trainh.shape[0] // batch_size

    config = init_wandb(
        project_name="Pure JAX MLP training", job_type="Pure JAX MLP training", dir_path="/proj_sw/user_dev/umales"
    )

    for epoch in range(num_epochs):

        batch_loss_accum = 0.0
        batch_accuracy_accum = 0.0

        for i in range(num_batches):
            with jax.default_device(jax.devices("cpu")[0]):
                x_batchh, y_batchh = (
                    x_trainh[i * batch_size : (i + 1) * batch_size],
                    y_trainh[i * batch_size : (i + 1) * batch_size],
                )

            x_batch = jax.device_put(x_batchh, current_device)
            y_batch = jax.device_put(y_batchh, current_device)

            w1, b1, w2, b2 = update(params, x_batch, y_batch, learning_rate)

            params = (w1, b1, w2, b2)
            batch_loss = mse_loss(params, x_batch, y_batch)

            logits = mlp_model(params, x_batch)
            batch_accuracy = compute_accuracy(logits, y_batch)

            batch_loss_accum += batch_loss
            batch_accuracy_accum += batch_accuracy

            if i % 100 == 0:
                avg_loss_100 = batch_loss_accum / 100.0
                avg_accuracy_100 = batch_accuracy_accum / 100.0
                wandb.log({"train loss": avg_loss_100, "train accuracy": avg_accuracy_100})
                # Reset accumulators after every 100 batches
                batch_loss_accum = 0.0
                batch_accuracy_accum = 0.0

        val_loss, val_accuracy = evaluate(params, x_valh, y_valh)
        wandb.log({"validation loss": val_loss, "validation accuracy": val_accuracy})

    test_loss, test_accuracy = evaluate(params, x_testh, y_testh)
    wandb.log({"test loss": test_loss, "test accuracy": test_accuracy})

    wandb.finish()

    return params


def evaluate(params, x_test, y_test, batch_size=1):
    total_loss = 0.0
    correct_predictions = 0.0
    num_samples = len(x_test)

    for i in range(0, num_samples, batch_size):

        with jax.default_device(jax.devices("cpu")[0]):
            x_batchh, y_batchh = x_test[i : i + batch_size], y_test[i : i + batch_size]

        x_batch = jax.device_put(x_batchh, jax.devices()[0])
        y_batch = jax.device_put(y_batchh, jax.devices()[0])

        logits = mlp_model(params, x_batch)
        batch_loss = mse_loss(params, x_batch, y_batch)
        batch_accuracy = compute_accuracy(logits, y_batch)

        total_loss += batch_loss * 1.0
        correct_predictions += batch_accuracy * 1.0

    # Calculate average loss and accuracy over the entire test set
    num_samples = num_samples * 1.0
    avg_loss = total_loss / num_samples
    avg_accuracy = correct_predictions / num_samples

    return avg_loss, avg_accuracy


if __name__ == "__main__":

    init_device()

    with jax.default_device(jax.devices("cpu")[0]):
        key = random.PRNGKey(0)
        x_trainh, y_trainh, x_valh, y_valh, x_testh, y_testh = load_mnist()

    params = train_mlp(x_trainh, y_trainh, x_valh, y_valh, x_testh, y_testh, key)
