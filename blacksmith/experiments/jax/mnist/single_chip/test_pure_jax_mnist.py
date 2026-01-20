# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import wandb
from jax import random

from blacksmith.datasets.jax.mnist.dataloader import load_mnist_jax
from blacksmith.experiments.jax.mnist.configs import ExperimentConfig
from blacksmith.experiments.jax.mnist.logging.wandb_utils import init_wandb
from blacksmith.tools.cli import generate_config, parse_cli_options


def train_mnist(config: ExperimentConfig):

    training_config = config.training_config
    net_config = config.net_config
    logger_config = config.logger_config
    early_stopping_config = config.early_stopping

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

    @jax.jit
    def compute_loss_grads_logits(params, x_batch, y_batch):
        def loss_fn(p):
            logits = mlp_model(p, x_batch)
            return cross_entropy(logits, y_batch), logits

        (loss, logits), grads = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(params)

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

    @jax.jit
    def validation_loss(params, x_val, y_val):
        def loss_fn(p):
            logits = mlp_model(p, x_val)
            return cross_entropy(logits, y_val), logits

        loss, logits = loss_fn(params)
        return loss, logits

    @jax.jit
    def compute_accuracy(logits, y):
        correct = jnp.mean(jnp.argmax(logits, 1) == jnp.argmax(y, 1))
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
        early_stopping_config=early_stopping_config,
        logger_config=logger_config,
    ):
        input_size = net_config.input_size
        hidden_size = net_config.hidden_size
        output_size = net_config.output_size
        current_device = jax.devices()[0]

        # Initializing model parameters on CPU, since Jax random number generator
        # is currently not supported on device (https://github.com/tenstorrent/tt-xla/issues/420).
        with jax.default_device(jax.devices("cpu")[0]):
            params_host = init_mlp_params(key, input_size, hidden_size, output_size)

        params = jax.device_put(params_host, current_device)

        num_batches = x_train_host.shape[0] // batch_size

        if logger_config.log_on_wandb:
            config = init_wandb(
                project_name="Pure JAX MLP training",
                job_type="Pure JAX MLP training",
                dir_path=logger_config.checkpoint.checkpoint_dir,
            )

        best_val_loss = 1000.0
        epochs_no_improvement = 0
        early_stop_triggered = False

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

                x_batch = jax.device_put(x_batch_host, current_device)
                y_batch = jax.device_put(y_batch_host, current_device)

                batch_loss, grads, logits = compute_loss_grads_logits(params, x_batch, y_batch)

                params = update(params, grads, learning_rate)

                batch_accuracy = compute_accuracy(logits, y_batch)

                batch_loss_accum += batch_loss
                batch_accuracy_accum += batch_accuracy

                if (i + 1) % logger_config.log_every_n_steps == 0:
                    avg_loss = batch_loss_accum / logger_config.log_every_n_steps
                    avg_accuracy = batch_accuracy_accum / logger_config.log_every_n_steps
                    if logger_config.log_on_wandb:
                        wandb.log({"train loss": avg_loss, "train accuracy": avg_accuracy})
                    else:
                        print(f"Epoch {epoch + 1}, Step {i + 1}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
                    batch_loss_accum = 0.0
                    batch_accuracy_accum = 0.0

            val_loss, val_accuracy = evaluate(params, x_val_host, y_val_host)
            if logger_config.log_on_wandb:
                wandb.log({"validation loss": val_loss, "validation accuracy": val_accuracy})
            else:
                print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

            if val_loss < best_val_loss - early_stopping_config.min_delta:
                best_val_loss = val_loss
                epochs_no_improvement = 0
                best_params = params
            else:
                epochs_no_improvement += 1

            if epochs_no_improvement >= early_stopping_config.patience:
                early_stop_triggered = True
                break

        if early_stop_triggered:
            params = best_params

        test_loss, test_accuracy = evaluate(params, x_test_host, y_test_host)

        if logger_config.log_on_wandb:
            wandb.log({"test loss": test_loss, "test accuracy": test_accuracy})
            wandb.finish()

        else:
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

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

            batch_loss, logits = validation_loss(params, x_batch, y_batch)

            batch_accuracy = compute_accuracy(logits, y_batch)

            total_loss += batch_loss
            correct_predictions += batch_accuracy

        avg_loss = total_loss / num_samples
        avg_accuracy = correct_predictions / num_samples

        return avg_loss, avg_accuracy

    with jax.default_device(jax.devices("cpu")[0]):
        key = random.PRNGKey(0)
        x_train_host, y_train_host, x_val_host, y_val_host, x_test_host, y_test_host = load_mnist_jax()

    params = train_mlp(x_train_host, y_train_host, x_val_host, y_val_host, x_test_host, y_test_host, key)


if __name__ == "__main__":
    default_config = Path(__file__).parent.parent / "test_mnist.yaml"
    args = parse_cli_options(default_config=default_config)
    config: ExperimentConfig = generate_config(ExperimentConfig, args.config)
    train_mnist(config)
