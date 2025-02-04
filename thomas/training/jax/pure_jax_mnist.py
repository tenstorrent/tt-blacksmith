# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
import numpy as np
from keras import datasets
from jax import random

import os
import sys
import jax._src.xla_bridge as xb


def init_device():
    backend = "tt"
    print(os.getcwd())
    path = os.path.join(os.getcwd(), "third_party/tt-xla/build/src/tt/pjrt_plugin_tt.so")
    print(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find tt_pjrt C API plugin at {path}")

    print("Loading tt_pjrt C API plugin", file=sys.stderr)
    xb.discover_pjrt_plugins()

    plugin = xb.register_plugin("tt", priority=500, library_path=path, options=None)
    print("Loaded", file=sys.stderr)
    jax.config.update("jax_platforms", "tt,cpu")


# Load MNIST dataset
def load_mnist():
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

    train_images = (train_images[..., None] / 255.0).astype(jnp.float32)
    test_images = (test_images[..., None] / 255.0).astype(jnp.float32)

    train_images = train_images.reshape(-1, 28 * 28).astype(jnp.float32)
    test_images = test_images.reshape(-1, 28 * 28).astype(jnp.float32)

    train_labels = jax.nn.one_hot(train_labels, 10)
    test_labels = jax.nn.one_hot(test_labels, 10)
    train_labels = train_labels.astype(jnp.float32)
    test_labels = test_labels.astype(jnp.float32)

    return train_images, train_labels, test_images, test_labels


def mlp_model(params, x):
    w1, b1, w2, b2 = params
    h1 = jnp.maximum(jnp.dot(x, w1) + b1, 0.0)
    logits = jnp.dot(h1, w2) + b2
    return jax.nn.softmax(logits)


def init_mlp_params(key, input_dim=28 * 28, hidden_dim=64, output_dim=10):
    w1_shape = (input_dim, hidden_dim)
    b1_shape = (hidden_dim,)
    w2_shape = (hidden_dim, output_dim)
    b2_shape = (output_dim,)

    w1 = random.normal(key, w1_shape) * jnp.sqrt(2.0 / input_dim)
    w1 = w1.astype(jnp.float32)
    b1 = jnp.zeros(b1_shape, dtype=jnp.float32)
    w2 = random.normal(key, w2_shape) * jnp.sqrt(2.0 / hidden_dim)
    w2 = w2.astype(jnp.float32)
    b2 = jnp.zeros(b2_shape, dtype=jnp.float32)

    return (w1, b1, w2, b2)


# Categorical Cross-Entropy loss function
@jax.jit
def categorical_cross_entropy_loss(params, x, y):
    logits = mlp_model(params, x)
    return -jnp.mean(jnp.sum(y * jnp.log(logits + 1e-8), axis=-1))


# Gradient descent update rule (simple SGD)
@jax.jit
def update(params, x_batch, y_batch, learning_rate):
    grads = jax.grad(categorical_cross_entropy_loss, argnums=0)(params, x_batch, y_batch)
    w1, b1, w2, b2 = params
    dw1, db1, dw2, db2 = grads
    updated_params = (
        w1 - learning_rate * dw1,
        b1 - learning_rate * db1,
        w2 - learning_rate * dw2,
        b2 - learning_rate * db2,
    )
    return updated_params


def compute_accuracy(logits, y):
    # Transfer logits to CPU
    logits_cpu = jax.device_put(logits, jax.devices("cpu")[0])

    # Compute predictions on CPU
    with jax.default_device(jax.devices("cpu")[0]):
        predictionsh = jnp.argmax(logits_cpu, axis=-1)
        predictionsh = predictionsh.astype(jnp.uint32)

    predictions = jax.device_put(predictionsh, jax.devices()[0])

    yh = jax.device_put(y, jax.devices("cpu")[0])
    # Compute true labels on CPU
    with jax.default_device(jax.devices("cpu")[0]):
        true_labelsh = jnp.argmax(yh, axis=-1)
        true_labelsh = true_labelsh.astype(jnp.uint32)

    true_labels = jax.device_put(true_labelsh, jax.devices()[0])

    correct = jnp.sum(predictions == true_labels, dtype=jnp.uint32)

    num_samples = jnp.array(logits.shape[0], dtype=jnp.uint32)
    accuracy = correct / num_samples
    return accuracy


# Training loop
def train_mlp(x_trainh, y_trainh, x_testh, y_testh, key, num_epochs=20, batch_size=128, learning_rate=1e-2):
    input_dim = x_trainh.shape[1]
    hidden_dim = 256
    output_dim = 10
    current_device = jax.devices()[0]

    # Initialize model parameters
    with jax.default_device(jax.devices("cpu")[0]):
        paramsh = init_mlp_params(key, input_dim, hidden_dim, output_dim)

    params = jax.device_put(paramsh, current_device)

    num_batches = x_trainh.shape[0] // batch_size

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0

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
            batch_loss = categorical_cross_entropy_loss(params, x_batch, y_batch)
            batch_logits = mlp_model(params, x_batch)
            batch_accuracy = compute_accuracy(batch_logits, y_batch)

            epoch_loss += batch_loss
            epoch_accuracy += batch_accuracy

            # if i % 200 == 0:
            #    print(f"Epoch {epoch}, Batch {i}, Loss: {batch_loss:.4f}, Accuracy: {batch_accuracy:.4f}")

        avg_epoch_loss = epoch_loss / num_batches
        avg_epoch_accuracy = epoch_accuracy / num_batches
        print(f"Epoch {epoch}, Average Loss: {avg_epoch_loss:.4f}, Average Accuracy: {avg_epoch_accuracy:.4f}")

    return params


def evaluate(params, x_test, y_test):
    logits = mlp_model(params, x_test)
    test_loss = categorical_cross_entropy_loss(params, x_test, y_test)
    test_accuracy = compute_accuracy(logits, y_test)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    return test_loss, test_accuracy


if __name__ == "__main__":

    # init_device()
    current_device = jax.devices()[0]

    with jax.default_device(jax.devices("cpu")[0]):
        key = random.PRNGKey(0)
        x_trainh, y_trainh, x_testh, y_testh = load_mnist()

    # Train the model
    params = train_mlp(x_trainh, y_trainh, x_testh, y_testh, key)

    # Evaluate the model
    evaluate(params, x_testh, y_testh)
