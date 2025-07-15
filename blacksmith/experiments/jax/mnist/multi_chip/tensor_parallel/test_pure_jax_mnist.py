# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import shard_map
import jax.lax as lax
import numpy as np

import wandb
import os

from blacksmith.tools.cli import generate_config
from blacksmith.tools.jax_utils import init_device
from blacksmith.datasets.jax.mnist.dataloader import load_mnist_jax
from blacksmith.experiments.jax.mnist.configs import ExperimentConfig

from blacksmith.experiments.jax.mnist.logging.wandb_utils import init_wandb


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


def train_mnist():
    init_device()
    jax.config.update("jax_use_shardy_partitioner", True)

    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "test_mnist.yaml")
    config = generate_config(ExperimentConfig, config_path)

    training_config = config.training_config
    net_config = config.net_config
    logger_config = config.logger_config
    early_stopping_config = config.early_stopping
    sharding_config = ShardingConfig()

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

    @jax.jit
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
        def loss_fn(p):
            logits_frag = mlp_model(p, x_batch)
            return logits_frag, cross_entropy_loss_global(logits_frag, y_batch_sharded)

        return loss_fn(params)

    def compute_accuracy(logits, y_true):
        logits_host = jax.device_put(logits, jax.devices("cpu")[0])
        y_true_host = jax.device_put(y_true, jax.devices("cpu")[0])
        return jnp.mean(jnp.argmax(logits_host, axis=-1) == jnp.argmax(y_true_host, axis=-1))

    def train_mlp(
        x_train_host,
        y_train_host,
        x_val_host,
        y_val_host,
        x_test_host,
        y_test_host,
        key,
        sharding_config,
        num_epochs=training_config.epochs,
        batch_size=training_config.batch_size,
        learning_rate=training_config.lr,
        early_stopping_config=early_stopping_config,
    ):
        input_size = net_config.input_size
        hidden_size = net_config.hidden_size
        output_size = net_config.output_size

        # Initializing model parameters on CPU, since Jax random number generator
        # is currently not supported on device (https://github.com/tenstorrent/tt-xla/issues/420).
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

        if logger_config.log_on_wandb:
            config = init_wandb(
                project_name="TP - Pure JAX MLP training",
                job_type="TP - Pure JAX MLP training",
                dir_path=logger_config.checkpoint.checkpoint_dir,
            )

        best_val_loss = 1000.0
        epochs_no_improvement = 0
        early_stop_triggered = False

        print("\n--- Starting Training (Tensor Parallel MLP) ---")
        for epoch in range(num_epochs):
            batch_loss_sum = 0.0
            batch_accuracy_sum = 0.0

            for i in range(num_batches):
                # Batch creation is done on CPU (https://github.com/tenstorrent/tt-mlir/issues/2309)
                with jax.default_device(jax.devices("cpu")[0]):
                    x_batch_host, y_batch_host = (
                        x_train_host[i * batch_size : (i + 1) * batch_size],
                        y_train_host[i * batch_size : (i + 1) * batch_size],
                    )

                x_batch = jax.device_put(x_batch_host, sharding_config.data_sharding_x)
                y_batch_sharded = jax.device_put(y_batch_host, sharding_config.data_sharding_y)

                loss, grads_frag, logits_frag = training_step_jit(params, x_batch, y_batch_sharded)

                params_host = jax.device_put(params, jax.devices("cpu")[0])
                grads_host = jax.device_put(grads_frag, jax.devices("cpu")[0])
                learning_rate_host = jax.device_put(learning_rate, jax.devices("cpu")[0])

                # Optimizer step is done on CPU (https://github.com/tenstorrent/tt-xla/issues/342)
                params_host_updated = update(params_host, grads_host, learning_rate_host)
                params = jax.device_put(params_host_updated, sharding_config.param_sharding)

                batch_accuracy_sum += compute_accuracy(logits_frag, y_batch_sharded)

                # print(training_step.lower(params, x_batch, y_batch_sharded, training_config.learning_rate).as_text())

                loss_host = jax.device_put(loss, jax.devices("cpu")[0])
                batch_loss_sum += loss_host

                if (i + 1) % logger_config.log_every_n_steps == 0:
                    avg_loss = batch_loss_sum / logger_config.log_every_n_steps
                    avg_accuracy = batch_accuracy_sum / logger_config.log_every_n_steps
                    if logger_config.log_on_wandb:
                        wandb.log({"train loss": avg_loss, "train accuracy": avg_accuracy})
                    else:
                        print(f"Epoch {epoch}, Batch {i +1}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
                    batch_loss_sum = 0.0
                    batch_accuracy_sum = 0.0

            val_loss_global, val_acc = evaluate(params, x_val_host, y_val_host, sharding_config, param_in_specs)
            if logger_config.log_on_wandb:
                wandb.log({"validation loss": val_loss_global, "validation accuracy": val_acc})
            else:
                print(f"Epoch {epoch}, Validation Loss: {val_loss_global:.4f}")
                print(f"Epoch {epoch}, Validation Accuracy: {val_acc:.4f}")

            if val_loss_global < best_val_loss - early_stopping_config.min_delta:
                best_val_loss = val_loss_global
                epochs_no_improvement = 0
                best_params = params
            else:
                epochs_no_improvement += 1

            if epochs_no_improvement >= early_stopping_config.patience:
                early_stop_triggered = True
                break

        if early_stop_triggered:
            params = best_params

        test_loss_global, test_accuracy = evaluate(params, x_test_host, y_test_host, sharding_config, param_in_specs)
        if logger_config.log_on_wandb:
            wandb.log({"test loss": test_loss_global, "test accuracy": test_accuracy})
            wandb.finish()
        else:
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

        for i in range(0, len(x_test), batch_size):

            with jax.default_device(jax.devices("cpu")[0]):
                x_batch_host = x_test[i : i + batch_size]
                y_batch_host = y_test[i : i + batch_size]

            x_batch = jax.device_put(x_batch_host, sharding_config.data_sharding_x)
            y_batch_sharded = jax.device_put(y_batch_host, sharding_config.data_sharding_y)

            logits_frag, loss = validation_step_jit(params, x_batch, y_batch_sharded)

            loss_host = jax.device_put(loss, jax.devices("cpu")[0])

            total_loss += loss_host
            correct_predictions += compute_accuracy(logits_frag, y_batch_sharded)

        avg_loss = total_loss / num_samples
        avg_accuracy = correct_predictions / num_samples

        return avg_loss, avg_accuracy

    with jax.default_device(jax.devices("cpu")[0]):
        key = random.PRNGKey(0)
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
    )


if __name__ == "__main__":
    train_mnist()
