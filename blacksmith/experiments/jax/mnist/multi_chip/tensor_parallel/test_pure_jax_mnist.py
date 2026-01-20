# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import wandb
from jax import random
from jax.experimental import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from blacksmith.datasets.jax.mnist.dataloader import load_mnist_jax
from blacksmith.experiments.jax.mnist.configs import ExperimentConfig
from blacksmith.tools.cli import generate_config, parse_cli_options


class ShardingConfig:
    def __init__(self):
        self.mesh = Mesh(np.array(jax.devices("tt")), axis_names=("tp",))
        self.data_sharding_x = NamedSharding(self.mesh, PartitionSpec())
        self.data_sharding_y = NamedSharding(self.mesh, PartitionSpec(None, "tp"))
        self.param_sharding = (
            NamedSharding(self.mesh, PartitionSpec(None, "tp")),  # w1
            NamedSharding(
                self.mesh,
                PartitionSpec(
                    "tp",
                ),
            ),  # b1
            NamedSharding(self.mesh, PartitionSpec("tp", None)),  # w2
            NamedSharding(self.mesh, PartitionSpec()),  # b2
            NamedSharding(self.mesh, PartitionSpec(None, "tp")),  # w3
            NamedSharding(
                self.mesh,
                PartitionSpec(
                    "tp",
                ),
            ),  # b3
        )
        self.scalar_sharding = NamedSharding(self.mesh, PartitionSpec())


# We use Column-Row-Column sharding strategy, so that communication
# happens only in second layer. This is because of
# metal issue (https://github.com/tenstorrent/tt-metal/issues/21987),
# and our sharding strategy avoids all-reduce on the last layer, which has
# output dimension not divisible by 32 (10); which is prohibitive by the issue.
def mlp_model(params, x):
    w1, b1, w2, b2, w3, b3 = params

    # Layer 1: Column sharded
    h1 = jnp.maximum(jnp.dot(x, w1) + b1, 0.0)

    # Layer 2: Row sharded (partial sum)
    h2_partial = jnp.dot(h1, w2)
    h2_full = lax.psum(h2_partial, "tp")
    h2_out = jnp.maximum(h2_full + b2, 0.0)

    # Layer 3: Column sharded (output fragments)
    output_logits = jnp.dot(h2_out, w3) + b3

    return output_logits


def init_mlp_params(key, input_size, hidden_size, output_size):
    w1_shape = (input_size, hidden_size)
    b1_shape = (hidden_size,)
    w2_shape = (hidden_size, hidden_size)
    b2_shape = (hidden_size,)
    w3_shape = (hidden_size, output_size)
    b3_shape = (output_size,)

    key1, key2, key3 = random.split(key, 3)

    # Lecun normal
    w1 = random.normal(key1, w1_shape) * jnp.sqrt(1.0 / w1_shape[0])
    w1 = w1.astype(jnp.float32)
    b1 = jnp.zeros(b1_shape, dtype=jnp.float32)
    w2 = random.normal(key2, w2_shape) * jnp.sqrt(1.0 / w2_shape[0])
    w2 = w2.astype(jnp.float32)
    b2 = jnp.zeros(b2_shape, dtype=jnp.float32)
    w3 = random.normal(key3, w3_shape) * jnp.sqrt(1.0 / w3_shape[0])
    w3 = w3.astype(jnp.float32)
    b3 = jnp.zeros(b3_shape, dtype=jnp.float32)

    return (w1, b1, w2, b2, w3, b3)


def cross_entropy(logits, y):
    return -jnp.mean(jnp.sum(y * jax.nn.log_softmax(logits), axis=-1))


def cross_entropy_loss_global(logits_frag, y_true_frag):
    full_logits = lax.all_gather(logits_frag, "tp", axis=1, tiled=False)
    full_y_true = lax.all_gather(y_true_frag, "tp", axis=1, tiled=False)

    full_logits = jnp.reshape(full_logits, (full_logits.shape[0], -1))
    full_y_true = jnp.reshape(full_y_true, (full_y_true.shape[0], -1))

    return cross_entropy(full_logits, full_y_true)


def compute_loss_grads_logits(params, x_batch, y_batch_sharded):
    def loss_fn(p):
        logits_frag = mlp_model(p, x_batch)
        return cross_entropy(logits_frag, y_batch_sharded), logits_frag

    (_, logits_frag), grads_frag = jax.value_and_grad(loss_fn, has_aux=True)(params)

    loss_all = cross_entropy_loss_global(logits_frag, y_batch_sharded)

    return loss_all, grads_frag, logits_frag


def update(params, grads, learning_rate):
    w1, b1, w2, b2, w3, b3 = params
    dw1, db1, dw2, db2, dw3, db3 = grads
    updated_params = (
        w1 - learning_rate * dw1,
        b1 - learning_rate * db1,
        w2 - learning_rate * dw2,
        b2 - learning_rate * db2,
        w3 - learning_rate * dw3,
        b3 - learning_rate * db3,
    )
    return updated_params


def validation_loss(params, x_batch, y_batch_sharded):
    logits_frag = mlp_model(params, x_batch)
    return logits_frag, cross_entropy_loss_global(logits_frag, y_batch_sharded)


def compute_accuracy_step(logits_frag, y_true_frag):
    # all-gather across tensor-parallel axis
    full_logits = lax.all_gather(logits_frag, "tp", axis=1, tiled=False)
    full_y_true = lax.all_gather(y_true_frag, "tp", axis=1, tiled=False)

    return jnp.mean(jnp.argmax(full_logits, axis=-1) == jnp.argmax(full_y_true, axis=-1))


def train_mlp(
    x_train_host,
    y_train_host,
    x_val_host,
    y_val_host,
    x_test_host,
    y_test_host,
    key,
    sharding_config,
    training_config,
    net_config,
    logger_config,
    early_stopping_config,
):
    input_size = net_config.input_size
    hidden_size = net_config.hidden_size
    output_size = net_config.output_size

    batch_size = training_config.batch_size
    learning_rate = training_config.lr
    num_epochs = training_config.epochs

    # Initializing model parameters on CPU, since jax.random.normal
    # is currently not supported on device (https://github.com/tenstorrent/tt-xla/issues/1105).
    with jax.default_device(jax.devices("cpu")[0]):
        params_host = init_mlp_params(key, input_size, hidden_size, output_size)

    params = jax.tree_util.tree_map(lambda p, s: jax.device_put(p, s), params_host, sharding_config.param_sharding)

    num_batches = x_train_host.shape[0] // batch_size

    param_in_specs = (
        PartitionSpec(None, "tp"),  # w1
        PartitionSpec(
            "tp",
        ),  # b1
        PartitionSpec("tp", None),  # w2
        PartitionSpec(
            None,
        ),  # b2
        PartitionSpec(None, "tp"),  # w3
        PartitionSpec(
            "tp",
        ),  # b3
    )

    def training_step(params_sharded, x_batch_replicated, y_batch_sharded):
        return shard_map.shard_map(
            lambda p, x, y: compute_loss_grads_logits(p, x, y),
            mesh=sharding_config.mesh,
            in_specs=(
                param_in_specs,
                PartitionSpec(),
                PartitionSpec(None, "tp"),
            ),
            out_specs=(
                PartitionSpec(),
                param_in_specs,  # grads
                PartitionSpec(None, "tp"),
            ),
            check_rep=False,
        )(params_sharded, x_batch_replicated, y_batch_sharded)

    training_step_jit = jax.jit(
        training_step,
        out_shardings=(
            sharding_config.scalar_sharding,  # loss
            sharding_config.param_sharding,  # grads
            sharding_config.data_sharding_y,  # logits
        ),
    )

    def optimizer_step(params, grads, learning_rate):
        return shard_map.shard_map(
            lambda p, g, lr: update(p, g, lr),
            mesh=sharding_config.mesh,
            in_specs=(param_in_specs, param_in_specs, PartitionSpec()),
            out_specs=param_in_specs,
            check_rep=False,
        )(params, grads, learning_rate)

    optimizer_step_jit = jax.jit(
        optimizer_step,
        out_shardings=sharding_config.param_sharding,
    )

    def compute_accuracy(logits_frag, y_batch_sharded):
        return shard_map.shard_map(
            lambda l, y: compute_accuracy_step(l, y),
            mesh=sharding_config.mesh,
            in_specs=(PartitionSpec(None, "tp"), PartitionSpec(None, "tp")),
            out_specs=PartitionSpec(),  # scalar accuracy per shard
            check_rep=False,
        )(logits_frag, y_batch_sharded)

    compute_accuracy_jit = jax.jit(compute_accuracy, out_shardings=sharding_config.scalar_sharding)

    learning_rate = jax.device_put(learning_rate, sharding_config.scalar_sharding)

    best_val_loss = 1000.0
    epochs_no_improvement = 0

    wandb.init(
        project="TP - Pure JAX MLP training",
        job_type="TP - Pure JAX MLP training",
        dir=logger_config.checkpoint.checkpoint_dir,
    )
    print("\n--- Starting Training (Tensor Parallel MLP) ---")
    for epoch in range(num_epochs):
        batch_loss_sum = 0.0
        batch_accuracy_sum = 0.0

        for i in range(num_batches):
            # Batch creation is done on CPU, since we have to load dataset to CPU (https://github.com/tenstorrent/tt-mlir/issues/2768)
            # and we would have to replicate it across devices if we wanted on-device batching,
            # because single chip batching within Shardy context produces empty mesh, which is currently not supported (https://github.com/tenstorrent/tt-mlir/issues/4636).
            with jax.default_device(jax.devices("cpu")[0]):
                x_batch_host, y_batch_host = (
                    x_train_host[i * batch_size : (i + 1) * batch_size],
                    y_train_host[i * batch_size : (i + 1) * batch_size],
                )

            x_batch = jax.device_put(x_batch_host, sharding_config.data_sharding_x)
            y_batch_sharded = jax.device_put(y_batch_host, sharding_config.data_sharding_y)

            loss, grads_frag, logits_frag = training_step_jit(params, x_batch, y_batch_sharded)

            params = optimizer_step_jit(params, grads_frag, learning_rate)

            print(logits_frag.sharding)
            print(y_batch_sharded.sharding)
            accuracy = compute_accuracy_jit(logits_frag, y_batch_sharded)

            loss_host = jax.device_put(loss, jax.devices("cpu")[0])
            batch_loss_sum += loss_host

            accuracy_host = jax.device_put(accuracy, jax.devices("cpu")[0])
            batch_accuracy_sum += accuracy_host

            if (i + 1) % logger_config.log_every_n_steps == 0:
                avg_loss = batch_loss_sum / logger_config.log_every_n_steps
                avg_accuracy = batch_accuracy_sum / logger_config.log_every_n_steps

                wandb.log({"train loss": avg_loss, "train accuracy": avg_accuracy})
                print(f"Epoch {epoch}, Batch {i +1}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

                batch_loss_sum = 0.0
                batch_accuracy_sum = 0.0

        val_loss_global, val_acc = evaluate(params, x_val_host, y_val_host, sharding_config, param_in_specs)

        wandb.log({"validation loss": val_loss_global, "validation accuracy": val_acc})
        print(f"Epoch {epoch}, Validation Loss: {val_loss_global:.4f}")
        print(f"Epoch {epoch}, Validation Accuracy: {val_acc:.4f}")

        if val_loss_global < best_val_loss - early_stopping_config.min_delta:
            best_val_loss = val_loss_global
            epochs_no_improvement = 0
            best_params = params
        else:
            epochs_no_improvement += 1

        if epochs_no_improvement >= early_stopping_config.patience:
            params = best_params
            break

    test_loss_global, test_accuracy = evaluate(params, x_test_host, y_test_host, sharding_config, param_in_specs)

    wandb.log({"test loss": test_loss_global, "test accuracy": test_accuracy})
    wandb.finish()
    print(f"\n--- Final Evaluation ---")
    print(f"Test Loss: {test_loss_global:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    return params


def evaluate(params, x_test, y_test, sharding_config, param_in_specs, batch_size=256):
    total_loss = 0.0
    correct_predictions = 0.0
    num_samples = len(x_test) // batch_size

    def validation_step(params_sharded, x_batch_replicated, y_batch_sharded):
        return shard_map.shard_map(
            lambda p, x, y: validation_loss(p, x, y),
            mesh=sharding_config.mesh,
            in_specs=(
                param_in_specs,
                PartitionSpec(),
                PartitionSpec(None, "tp"),
            ),
            out_specs=(
                PartitionSpec(None, "tp"),
                PartitionSpec(),
            ),
            check_rep=False,
        )(params_sharded, x_batch_replicated, y_batch_sharded)

    validation_step_jit = jax.jit(
        validation_step,
        out_shardings=(
            sharding_config.data_sharding_y,
            sharding_config.scalar_sharding,
        ),
    )

    def compute_accuracy(logits_frag, y_batch_sharded):
        return shard_map.shard_map(
            lambda l, y: compute_accuracy_step(l, y),
            mesh=sharding_config.mesh,
            in_specs=(PartitionSpec(None, "tp"), PartitionSpec(None, "tp")),
            out_specs=PartitionSpec(),  # scalar accuracy per shard
            check_rep=False,
        )(logits_frag, y_batch_sharded)

    compute_accuracy_jit = jax.jit(compute_accuracy, out_shardings=sharding_config.scalar_sharding)

    for i in range(0, len(x_test), batch_size):

        with jax.default_device(jax.devices("cpu")[0]):
            x_batch_host = x_test[i : i + batch_size]
            y_batch_host = y_test[i : i + batch_size]

        x_batch = jax.device_put(x_batch_host, sharding_config.data_sharding_x)
        y_batch_sharded = jax.device_put(y_batch_host, sharding_config.data_sharding_y)

        logits_frag, loss = validation_step_jit(params, x_batch, y_batch_sharded)

        loss_host = jax.device_put(loss, jax.devices("cpu")[0])
        total_loss += loss_host

        accuracy = compute_accuracy_jit(logits_frag, y_batch_sharded)
        accuracy_host = jax.device_put(accuracy, jax.devices("cpu")[0])
        correct_predictions += accuracy_host

    avg_loss = total_loss / num_samples
    avg_accuracy = correct_predictions / num_samples

    return avg_loss, avg_accuracy


def train_mnist(config: ExperimentConfig):
    jax.config.update("jax_use_shardy_partitioner", True)
    os.environ["WANDB_MODE"] = "online" if config.logger_config.log_on_wandb else "disabled"

    sharding_config = ShardingConfig()

    key = random.PRNGKey(0)
    # Loading data on CPU, since data loader contains jax.random.permutation
    # which is currently not supported on device, as it fails on stablehlo.custom_call operation
    # (https://github.com/tenstorrent/tt-mlir/issues/2768).
    with jax.default_device(jax.devices("cpu")[0]):
        x_train_host, y_train_host, x_val_host, y_val_host, x_test_host, y_test_host = load_mnist_jax()

    train_mlp(
        x_train_host,
        y_train_host,
        x_val_host,
        y_val_host,
        x_test_host,
        y_test_host,
        key,
        sharding_config,
        config.training_config,
        config.net_config,
        config.logger_config,
        config.early_stopping,
    )


if __name__ == "__main__":
    default_config = Path(__file__).parent.parent.parent / "test_mnist.yaml"
    args = parse_cli_options(default_config=default_config)
    config: ExperimentConfig = generate_config(ExperimentConfig, args.config)
    train_mnist(config)
