# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
from jax import random

import wandb

from thomas.tools.cli import generate_config
from thomas.tools.jax_utils import init_device

from thomas.models.jax.mnist.mnist_linear import MNISTLinearConfig

from thomas.datasets.mnist.dataloader import load_mnist

from thomas.experiments.jax.mnist.logging.wandb_utils import init_wandb
from thomas.experiments.jax.mnist.logging.logger_config import LoggerConfig, get_default_logger_config
from thomas.experiments.jax.mnist.configs.experiment_config import TrainingConfig, ExperimentConfig

from pydantic import BaseModel, Field


def train_mnist():

    config = generate_config(ExperimentConfig, "thomas/experiments/jax/mnist/configs/test_jax_mnist.yaml")

    training_config = config.training_config
    net_config = config.net_config
    logger_config = config.logger_config

    init_device()

    @jax.jit
    def mlp_model(params, x):
        w1, b1, w2, b2, w3, b3 = params
        h1 = jnp.maximum(jnp.dot(x, w1) + b1, 0.0)
        h2 = jnp.maximum(jnp.dot(h1, w2) + b2, 0.0)
        logits = jnp.dot(h2, w3) + b3
        return logits

    # Initialize the model parameters with specified shapes
    def init_mlp_params(
        key, input_size=net_config.input_size, hidden_size=net_config.hidden_size, output_size=net_config.output_size
    ):
        # Define shapes for weights and biases
        w1_shape = (input_size, hidden_size)
        b1_shape = (hidden_size,)
        w2_shape = (hidden_size, hidden_size)
        b2_shape = (hidden_size,)
        w3_shape = (hidden_size, output_size)
        b3_shape = (output_size,)

        # He initialization for weights
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

    @jax.jit
    def mse_loss(params, x, y):
        logits = mlp_model(params, x)
        loss = jnp.mean((logits - y) ** 2)
        return loss, logits

    @jax.jit
    def compute_loss_grads_logits(params, x, y):
        (loss, logits), grads = jax.value_and_grad(mse_loss, argnums=0, has_aux=True)(params, x, y)
        return loss, grads, logits

    # Gradient descent update rule (simple SGD)
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

    # Done on CPU.
    def argmax_on_cpu(array):
        array_cpu = jax.device_put(array, jax.devices("cpu")[0])

        with jax.default_device(jax.devices("cpu")[0]):
            argmax_result = jnp.argmax(array_cpu, axis=-1)
            argmax_result = argmax_result.astype(jnp.uint32)

        return argmax_result

    # Done on CPU.
    def compute_accuracy(logits, y):
        predictions = argmax_on_cpu(logits)
        true_labels = argmax_on_cpu(y)

        correct = jnp.mean(predictions == true_labels)

        return correct

    # Training loop
    def train_mlp(
        x_train_host,
        y_train_host,
        x_val_host,
        y_val_host,
        x_test_host,
        y_test_host,
        key,
        num_epochs=training_config.epochs,
        batch_size=training_config.batch_size,
        learning_rate=training_config.lr,
    ):
        input_size = net_config.input_size
        hidden_size = net_config.hidden_size
        output_size = net_config.output_size
        current_device = jax.devices()[0]
        # Initialize model parameters
        with jax.default_device(jax.devices("cpu")[0]):
            params_host = init_mlp_params(key, input_size, hidden_size, output_size)

        params = jax.device_put(params_host, current_device)

        num_batches = x_train_host.shape[0] // batch_size

        config = init_wandb(
            project_name="Pure JAX MLP training",
            job_type="Pure JAX MLP training",
            dir_path=logger_config.checkpoint.checkpoint_dir,
        )

        for epoch in range(num_epochs):

            batch_loss_accum = 0.0
            batch_accuracy_accum = 0.0

            for i in range(num_batches):
                with jax.default_device(jax.devices("cpu")[0]):
                    x_batch_host, y_batch_host = (
                        x_train_host[i * batch_size : (i + 1) * batch_size],
                        y_train_host[i * batch_size : (i + 1) * batch_size],
                    )

                x_batch = jax.device_put(x_batch_host, current_device)
                y_batch = jax.device_put(y_batch_host, current_device)

                batch_loss, grads, logits = compute_loss_grads_logits(params, x_batch, y_batch)

                w1, b1, w2, b2, w3, b3 = update(params, grads, learning_rate)
                params = (w1, b1, w2, b2, w3, b3)

                batch_accuracy = compute_accuracy(logits, y_batch)

                batch_loss_accum += batch_loss
                batch_accuracy_accum += batch_accuracy

                if (i + 1) % logger_config.log_every_n_steps == 0:
                    avg_loss = batch_loss_accum / logger_config.log_every_n_steps
                    avg_accuracy = batch_accuracy_accum / logger_config.log_every_n_steps
                    wandb.log({"train loss": avg_loss, "train accuracy": avg_accuracy})
                    batch_loss_accum = 0.0
                    batch_accuracy_accum = 0.0

            val_loss, val_accuracy = evaluate(params, x_val_host, y_val_host)
            wandb.log({"validation loss": val_loss, "validation accuracy": val_accuracy})

        test_loss, test_accuracy = evaluate(params, x_test_host, y_test_host)
        wandb.log({"test loss": test_loss, "test accuracy": test_accuracy})

        wandb.finish()

        return params

    def evaluate(params, x_test, y_test, batch_size=training_config.batch_size):
        total_loss = 0.0
        correct_predictions = 0.0
        num_samples = len(x_test) // batch_size

        for i in range(0, len(x_test), batch_size):

            with jax.default_device(jax.devices("cpu")[0]):
                x_batch_host, y_batch_host = x_test[i : i + batch_size], y_test[i : i + batch_size]

            x_batch = jax.device_put(x_batch_host, jax.devices()[0])
            y_batch = jax.device_put(y_batch_host, jax.devices()[0])

            batch_loss, logits = mse_loss(params, x_batch, y_batch)
            batch_accuracy = compute_accuracy(logits, y_batch)

            total_loss += batch_loss * 1.0
            correct_predictions += batch_accuracy * 1.0

        # Calculate average loss and accuracy over the entire test set
        num_samples = num_samples * 1.0
        avg_loss = total_loss / num_samples
        avg_accuracy = correct_predictions / num_samples

        return avg_loss, avg_accuracy

    with jax.default_device(jax.devices("cpu")[0]):
        key = random.PRNGKey(0)
        x_train_host, y_train_host, x_val_host, y_val_host, x_test_host, y_test_host = load_mnist()

    params = train_mlp(x_train_host, y_train_host, x_val_host, y_val_host, x_test_host, y_test_host, key)


if __name__ == "__main__":
    train_mnist()
