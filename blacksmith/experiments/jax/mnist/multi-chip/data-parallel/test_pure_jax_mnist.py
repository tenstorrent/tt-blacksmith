# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import shard_map
import jax.lax as lax
import torchvision
import numpy as np

import wandb
import os
import sys

import jax._src.xla_bridge as xb
import jax.tree_util as tree_util

from blacksmith.tools.cli import generate_config
from blacksmith.tools.jax_utils import init_device
from blacksmith.datasets.jax.mnist.dataloader import load_mnist_jax
from blacksmith.experiments.jax.mnist.configs import ExperimentConfig

from blacksmith.experiments.jax.mnist.logging.wandb_utils import init_wandb

class ShardingConfig:
    def __init__(self):
        self.mesh = Mesh(np.array(jax.devices("tt")), axis_names=("dp",))
        self.data_sharding = NamedSharding(self.mesh, PartitionSpec("dp"))
        self.param_sharding = NamedSharding(self.mesh, PartitionSpec(None))
        self.scalar_sharding = NamedSharding(self.mesh, PartitionSpec())

def train_mnist():
    init_device()
    jax.config.update("jax_use_shardy_partitioner", True)

    config_path = os.path.join(os.path.dirname(__file__), "../..", "test_mnist.yaml")
    config = generate_config(ExperimentConfig, config_path)

    training_config = config.training_config
    net_config = config.net_config
    logger_config = config.logger_config
    sharding_config = ShardingConfig()

    def mlp_model(params, x):
        w1, b1, w2, b2, w3, b3 = params
        h1 = jnp.maximum(jnp.dot(x, w1) + b1, 0.0)
        h2 = jnp.maximum(jnp.dot(h1, w2) + b2, 0.0)
        logits = jnp.dot(h2, w3) + b3
        return logits


    def init_mlp_params(key, input_size=net_config.input_size, hidden_size=net_config.hidden_size, output_size=net_config.output_size):
        w1_shape = (input_size, hidden_size)
        b1_shape = (hidden_size,)
        w2_shape = (hidden_size, hidden_size)
        b2_shape = (hidden_size,)
        w3_shape = (hidden_size, output_size)
        b3_shape = (output_size,)

        w1 = random.normal(key, w1_shape) * jnp.sqrt(2.0 / w1_shape[0])
        w1 = w1.astype(jnp.float32)
        b1 = jnp.zeros(b1_shape, dtype=jnp.float32)
        w2 = random.normal(key, w2_shape) * jnp.sqrt(2.0 / w2_shape[0])
        w2 = w2.astype(jnp.float32)
        b2 = jnp.zeros(b2_shape, dtype=jnp.float32)
        w3 = random.normal(key, w3_shape) * jnp.sqrt(2.0 / w3_shape[0])
        w3 = w3.astype(jnp.float32)
        b3 = jnp.zeros(b3_shape, dtype=jnp.float32)

        return (w1, b1, w2, b2, w3, b3)


    def mse_loss(logits, y):
        return -jnp.mean(jnp.sum(y * jax.nn.log_softmax(logits), axis=-1))

    def compute_loss_grads_logits(params, x_batch, y_batch, lr):
        def loss_fn(p):
            logits = mlp_model(p, x_batch)
            return mse_loss(logits, y_batch), logits
        
        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        def gather_mean_grad(g):
            gathered = lax.all_gather(g, axis_name="dp")  
            return jnp.mean(gathered, axis=0)
        
        grads = tree_util.tree_map(gather_mean_grad, grads)
        
        gathered_loss = lax.all_gather(loss, axis_name="dp")  
        loss = jnp.mean(gathered_loss)  

        logits = lax.all_gather(logits, axis_name="dp")

        return loss, grads, logits

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

    def validation_loss(params, x_batch, y_batch):

        def loss_fn(p):
            logits = mlp_model(p, x_batch)
            return logits, mse_loss(logits, y_batch)

        logits, loss = loss_fn(params)

        gathered_loss = lax.all_gather(loss, axis_name="dp")  # Shape: (num_devices,)
        loss = jnp.mean(gathered_loss)  # Scalar, replicated

        logits = lax.all_gather(logits, axis_name="dp") 

        return logits, loss

    def argmax_on_cpu(array):
        array_cpu = jax.device_put(array, jax.devices("cpu")[0])
        with jax.default_device(jax.devices("cpu")[0]):
            argmax_result = jnp.argmax(array_cpu, axis=-1)
            argmax_result = argmax_result.astype(jnp.uint32)
        return argmax_result

    def compute_accuracy(logits, y):
        predictions = argmax_on_cpu(logits)
        true_labels = argmax_on_cpu(y)
        correct = jnp.mean(predictions == true_labels)
        return correct


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
    ):

        input_size = net_config.input_size
        hidden_size = net_config.hidden_size
        output_size = net_config.output_size

        # Initializing model parameters on CPU, since Jax random number generator
        # is currently not supported on device (https://github.com/tenstorrent/tt-xla/issues/420).
        with jax.default_device(jax.devices("cpu")[0]):
            params_init_host = init_mlp_params(key, input_size, hidden_size, output_size)

        params = jax.device_put(params_init_host, sharding_config.param_sharding)

        num_batches = x_train_host.shape[0] // batch_size

        def training_step(params, x_batch, y_batch, lr):
            return shard_map.shard_map(
                lambda p, x, y, lr: compute_loss_grads_logits(p, x, y, lr),
                mesh=sharding_config.mesh,
                in_specs=(
                    PartitionSpec(None),
                    PartitionSpec("dp"),
                    PartitionSpec("dp"),
                    PartitionSpec(),
                ),
                out_specs=(PartitionSpec(), PartitionSpec(None), PartitionSpec(None)),
                check_rep=False,
            )(params, x_batch, y_batch, lr)

        learning_rate = jax.device_put(
            training_config.lr, sharding_config.scalar_sharding
        )

        training_step_jit = jax.jit(training_step, out_shardings=(sharding_config.scalar_sharding, sharding_config.param_sharding, sharding_config.param_sharding))

        config = init_wandb(
            project_name="DP - Pure JAX MLP training",
            job_type="DP - Pure JAX MLP training",
            dir_path=logger_config.checkpoint.checkpoint_dir,
        )

        for epoch in range(num_epochs):
            batch_loss_accum = 0.0
            batch_accuracy_accum = 0.0

            for i in range(num_batches):

                # Batch creation is done on CPU (https://github.com/tenstorrent/tt-mlir/issues/2309)
                with jax.default_device(jax.devices("cpu")[0]):
                    x_batch_host, y_batch_host = (
                        x_train_host[i * batch_size : (i + 1) * batch_size],
                        y_train_host[i * batch_size : (i + 1) * batch_size],
                    )

                x_batch = jax.device_put(x_batch_host, sharding_config.data_sharding)
                y_batch = jax.device_put(y_batch_host, sharding_config.data_sharding)

                loss, grads, logits = training_step_jit(
                    params, x_batch, y_batch, learning_rate
                )

                params_host = jax.device_put(params, jax.devices("cpu")[0])
                grads_host = jax.device_put(grads, jax.devices("cpu")[0])
                learning_rate_host = jax.device_put(learning_rate, jax.devices("cpu")[0])

                # Optimizer step is done on CPU (https://github.com/tenstorrent/tt-xla/issues/342)
                params_host_updated = update(
                    params_host, grads_host, learning_rate_host
                )

                params = jax.device_put(params_host_updated, sharding_config.param_sharding)

                loss_host = jax.device_put(loss, jax.devices("cpu")[0])
                batch_loss_accum += loss_host

                logits_host = jax.device_put(logits, jax.devices("cpu")[0])
                # reshape logits_host from (2, 64, 10) to (128, 10)
                logits_host = logits_host.reshape(-1, y_batch_host.shape[1])
                # Accuracy calculation is done on CPU, as argmax is not supported on device (https://github.com/tenstorrent/tt-metal/issues/20570)
                accuracy = compute_accuracy(logits_host, y_batch_host)

                batch_accuracy_accum += accuracy
                if (i + 1) % logger_config.log_every_n_steps == 0:
                    avg_loss = batch_loss_accum / logger_config.log_every_n_steps
                    avg_accuracy = batch_accuracy_accum / logger_config.log_every_n_steps
                    wandb.log({"train loss": avg_loss, "train accuracy": avg_accuracy})
                    batch_loss_accum = 0.0
                    batch_accuracy_accum = 0.0

            val_loss, val_accuracy = evaluate(params, x_val_host, y_val_host, sharding_config)
            wandb.log({"validation loss": val_loss, "validation accuracy": val_accuracy})

        test_loss, test_accuracy = evaluate(
            params, x_test_host, y_test_host, sharding_config
        )
        wandb.log({"test loss": test_loss, "test accuracy": test_accuracy})

        wandb.finish()

        return params

    def evaluate(params, x_test, y_test, sharding_config, batch_size=training_config.batch_size):
        total_loss = 0.0
        correct_predictions = 0.0
        num_samples = len(x_test) // batch_size

        def validation_step(params, x_batch, y_batch):
            return shard_map.shard_map(
                lambda params, local_x, local_y: validation_loss(params, local_x, local_y),
                mesh=sharding_config.mesh,
                in_specs=(PartitionSpec(None), PartitionSpec("dp"), PartitionSpec("dp")),
                out_specs=(PartitionSpec(None), PartitionSpec()),
                check_rep=False,
            )(params, x_batch, y_batch)
            
        validation_step_jit = jax.jit(validation_step, out_shardings=(sharding_config.param_sharding, sharding_config.scalar_sharding)) 

        for i in range(0, len(x_test), batch_size):

            with jax.default_device(jax.devices("cpu")[0]):
                x_batch_host, y_batch_host = x_test[i : i + batch_size], y_test[i : i + batch_size]

            x_batch = jax.device_put(x_batch_host, sharding_config.data_sharding)
            y_batch = jax.device_put(y_batch_host, sharding_config.data_sharding)

            logits, loss = validation_step_jit(params, x_batch, y_batch)

            logits_host = jax.device_put(logits, jax.devices("cpu")[0])
            batch_loss = jax.device_put(loss, jax.devices("cpu")[0])

            logits_host = logits_host.reshape(-1, y_test.shape[1])
            batch_accuracy = compute_accuracy(logits_host, y_batch_host)

            total_loss += batch_loss * 1.0
            correct_predictions += batch_accuracy * 1.0

        num_samples = num_samples * 1.0
        avg_loss = total_loss / num_samples
        avg_accuracy = correct_predictions / num_samples

        return avg_loss, avg_accuracy
    
    with jax.default_device(jax.devices("cpu")[0]):
        key = random.PRNGKey(0)
        x_train_host, y_train_host, x_val_host, y_val_host, x_test_host, y_test_host = load_mnist_jax()

    train_mlp(
        x_train_host, y_train_host, x_val_host, y_val_host, x_test_host, y_test_host, key, sharding_config
    )

if __name__ == "__main__":
    train_mnist()