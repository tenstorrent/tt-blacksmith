# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax import export

import optax

from flax import linen as nn
from flax.training import train_state
from flax.serialization import to_state_dict, msgpack_serialize, from_bytes

from tensorflow import keras
import wandb
import os

from model import Models, MLP
from utils import ExportSHLO
from logg_it import init_wandb, log_metrics, save_checkpoint, load_checkpoint


def load_mnist():

    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

    train_images = train_images[..., None] / 255.0
    test_images = test_images[..., None] / 255.0

    # Shuffle the training data
    perm = jax.random.permutation(jax.random.PRNGKey(0), len(train_images))
    train_images, train_labels = train_images[perm], train_labels[perm]

    # Split the training data into training and validation sets
    train_size = int(0.8 * len(train_images))
    val_size = int(0.1 * len(train_images))

    train_images, val_images = train_images[:train_size], train_images[train_size : train_size + val_size]
    train_labels, val_labels = train_labels[:train_size], train_labels[train_size : train_size + val_size]

    return train_images, train_labels, val_images, val_labels, test_images, test_labels


train_images, train_labels, eval_images, eval_labels, test_images, test_labels = load_mnist()


class EarlyStopping:
    def __init__(self, patience=1):
        self.patience = patience
        self.best_accuracy = 0
        self.counter = 0

    def __call__(self, val_accuracy):
        if val_accuracy > self.best_accuracy:
            self.best_accuracy = val_accuracy
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


early_stopping = EarlyStopping(patience=1)

config = init_wandb(
    project_name="Flax mnist mlp training", job_type="Flax mnist mlp training", dir_path="/proj_sw/user_dev/umales"
)

config.learning_rate = 1e-3
config.batch_size = 64
config.num_epochs = 15
config.seed = 0


@jax.jit
def forward_pass(params, x):
    def apply_fn(x):
        return MLP().apply({"params": params}, x, mutable=["params"])

    return apply_fn(x)


def compute_loss(params, x, y):
    logits, new_model_state = forward_pass(params, x)
    loss = func_optax_loss(logits, y)
    return loss, new_model_state


@jax.jit
def func_optax_loss(logits, labels):
    # one_hot_labels = jax.nn.one_hot(labels, num_classes=logits.shape[-1]).astype(jnp.float32)
    return optax.losses.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean()


@jax.jit
def backward_pass(params, x, y):
    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
    (loss, new_model_state), grads = grad_fn(params, x, y)
    return grads, loss, new_model_state


@jax.jit
def update_params(state, grads):
    return state.apply_gradients(grads=grads)


@jax.jit
def train_step(state, x, y):
    grads, loss, new_model_state = backward_pass(state.params, x, y)
    state = update_params(state, grads)
    return state, new_model_state, loss, grads


@jax.jit
def eval_step(params, x):
    logits, _ = forward_pass(params, x)
    return logits


@jax.jit
def calculate_metrics(logits, y):
    loss = func_optax_loss(logits, y)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == y)
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
    # Calculate the mean of each key (loss, accuracy, etc.)
    for d in metrics:
        for key, value in d.items():
            if key not in sums:
                sums[key] = 0
            sums[key] += value

    averages = {key: sums[key] / count for key in sums}
    return averages


rng = random.PRNGKey(config.seed)
input_shape = (1, 28, 28, 1)
output_shape = jnp.ones((1, 10))
pred_model = Models(model_type="MLP")
params = pred_model.model.init(rng, jnp.ones(input_shape))["params"]
tx = optax.sgd(learning_rate=config.learning_rate)
state = train_state.TrainState.create(apply_fn=pred_model.model.apply, params=params, tx=tx)

batch_num = 0
num_batches = len(train_images) // config.batch_size
num_eval_batches = len(eval_images) // config.batch_size
for epoch in range(config.num_epochs):

    best_eval_loss = 1e3

    train_batch_metrics = []
    for i in range(num_batches):
        batch_images = train_images[i * config.batch_size : (i + 1) * config.batch_size]
        batch_labels = train_labels[i * config.batch_size : (i + 1) * config.batch_size]
        state, _, loss, grads = train_step(state, batch_images, batch_labels)
        batch_num = batch_num + 1

        logits = eval_step(state.params, batch_images)
        metrics = calculate_metrics(logits, batch_labels)
        train_batch_metrics.append(metrics)
    train_batch_metrics_avg = accumulate_metrics(train_batch_metrics)

    eval_batch_metrics = []
    for i in range(num_eval_batches):
        batch_images = eval_images[i * config.batch_size : (i + 1) * config.batch_size]
        batch_labels = eval_labels[i * config.batch_size : (i + 1) * config.batch_size]
        logits = eval_step(state.params, batch_images)
        metrics = calculate_metrics(logits, batch_labels)
        eval_batch_metrics.append(metrics)
    eval_batch_metrics_avg = accumulate_metrics(eval_batch_metrics)

    log_metrics(
        grads,
        state,
        train_batch_metrics_avg["loss"],
        train_batch_metrics_avg["accuracy"],
        eval_batch_metrics_avg["loss"],
        eval_batch_metrics_avg["accuracy"],
        epoch,
    )

    base_checkpoint_dir = f"/proj_sw/user_dev/umales/checkpoints/{wandb.run.name}"
    epoch_dir = f"epoch={epoch:02d}"
    checkpoint_dir = os.path.join(base_checkpoint_dir, epoch_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file_name = "checkpoint.msgpack"
    checkpoint_file_path = os.path.join(checkpoint_dir, checkpoint_file_name)
    save_checkpoint(checkpoint_file_path, state, epoch)


epoch = 0
ckpt_file = "checkpoint.msgpack"
restored_state = load_checkpoint(ckpt_file, state, epoch)
logits = eval_step(restored_state.params, test_images)
metrics = calculate_metrics(logits, test_labels)
test_batch_metrics = []
test_batch_metrics.append(metrics)
test_batch_metrics_avg = accumulate_metrics(test_batch_metrics)
wandb.log({"Test Loss": test_batch_metrics_avg["loss"], "Test Accuracy": test_batch_metrics_avg["accuracy"]})

wandb.finish()

# from utils import ExportSHLO

export_it = ExportSHLO()
# export_it.export_fwd_train_to_StableHLO_and_get_ops(forward_pass, state, input_shape, print_stablehlo=False)
# export_it.export_fwd_tst_to_StableHLO_and_get_ops(eval_step, state, input_shape, print_stablehlo=False)
# export_it.export_bwd_to_StableHLO_and_get_ops(backward_pass, state, input_shape, print_stablehlo=False)
export_it.export_loss_to_StableHLO_and_get_ops(func_optax_loss, output_shape, print_stablehlo=False)
# export_it.export_optimizer_to_StableHLO_and_get_ops(update_params, state, grads, print_stablehlo=False)
