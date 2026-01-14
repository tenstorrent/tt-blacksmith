# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
import optax

from blacksmith.models.jax.mnist.model import MLP


def cross_entropy(logits, labels):
    return optax.softmax_cross_entropy(logits, labels).mean()


def forward_pass(params, x):
    logits, _ = MLP().apply({"params": params}, x, mutable=["params"])
    return logits


def forward_and_compute_loss_and_logits(params, x, y):
    logits = forward_pass(params, x)
    loss = cross_entropy(logits, y)
    return loss, logits


@jax.jit
def compute_loss_grads_and_logits(params, x, y):
    (loss, logits), grads = jax.value_and_grad(forward_and_compute_loss_and_logits, has_aux=True)(params, x, y)
    return loss, grads, logits


@jax.jit
def optimizer_step(state, grads):
    return state.apply_gradients(grads=grads)


@jax.jit
def eval_step(params, x, y):
    logits = forward_pass(params, x)
    loss = cross_entropy(logits, y)
    return loss, logits


@jax.jit
def calculate_metrics_val(logits, loss, y):
    accuracy = jnp.mean(jnp.argmax(logits, 1) == jnp.argmax(y, 1))
    metrics = {
        "loss": loss,
        "accuracy": accuracy,
    }
    return metrics


def accumulate_metrics(metrics):
    keys = metrics[0].keys()

    if any(set(metric.keys()) != set(keys) for metric in metrics):
        raise ValueError("All dictionaries in 'metrics' must have the same keys.")

    count = len(metrics)
    sums = {}
    # Calculate the mean of each key (loss, accuracy)
    for d in metrics:
        for key, value in d.items():
            if key not in sums:
                sums[key] = 0
            sums[key] += value

    averages = {key: sums[key] / count for key in sums}
    return averages
