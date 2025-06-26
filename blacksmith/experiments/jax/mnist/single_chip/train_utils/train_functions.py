# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp

import optax

from blacksmith.models.jax.mnist.model import MLP


@jax.jit
def forward_pass(params, x):
    return MLP().apply({"params": params}, x, mutable=["params"])


def forward_and_compute_loss(params, x, y):
    logits, _ = forward_pass(params, x)
    loss = func_optax_loss(logits, y)
    return loss


# Currently loss is L2 and not cross entropy (https://github.com/tenstorrent/tt-xla/issues/288)
@jax.jit
def func_optax_loss(logits, labels):
    # one_hot_labels = jax.nn.one_hot(labels, num_classes=logits.shape[-1]).astype(jnp.float32)
    return optax.l2_loss(logits, labels).mean()


@jax.jit
def compute_loss_and_backward_pass(params, x, y):
    grad_fn = jax.value_and_grad(forward_and_compute_loss, has_aux=False)
    loss, grads = grad_fn(params, x, y)
    return grads, loss


@jax.jit
def update_params(state, grads):
    return state.apply_gradients(grads=grads)


@jax.jit
def train_step(state, x, y):
    grads, loss = compute_loss_and_backward_pass(state.params, x, y)
    state = update_params(state, grads)
    return state, loss, grads


@jax.jit
def eval_step(params, x):
    logits, _ = forward_pass(params, x)
    return logits


@jax.jit
def calculate_metrics_train(logits, y, loss):
    accuracy = jnp.mean(jnp.argmax(logits, 1) == jnp.argmax(y, 1))
    metrics = {
        "loss": loss,
        "accuracy": accuracy,
    }
    return metrics


@jax.jit
def calculate_metrics_val(logits, y):
    loss = func_optax_loss(logits, y)
    return calculate_metrics_train(logits, y, loss)


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
