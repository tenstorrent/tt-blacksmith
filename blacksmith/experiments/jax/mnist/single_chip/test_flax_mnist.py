# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import wandb
from flax.training import train_state
from jax import random

from blacksmith.datasets.jax.mnist.dataloader import load_mnist_jax
from blacksmith.experiments.jax.mnist.configs import ExperimentConfig
from blacksmith.experiments.jax.mnist.logging.shlo_ops_logging import ExportSHLO
from blacksmith.experiments.jax.mnist.logging.wandb_utils import (
    init_wandb,
    load_checkpoint,
    log_metrics,
    save_checkpoint,
)
from blacksmith.experiments.jax.mnist.single_chip.train_utils.train_functions import (
    accumulate_metrics,
    calculate_metrics_val,
    compute_loss_grads_and_logits,
    cross_entropy,
    eval_step,
    forward_pass,
    optimizer_step,
)
from blacksmith.models.jax.mnist.model import Models
from blacksmith.tools.cli import generate_config, parse_cli_options


def init_configs(config: ExperimentConfig):

    if config.logger_config.log_on_wandb:
        config_wandb = init_wandb(
            project_name=config.logger_config.experiment_name,
            job_type=config.logger_config.experiment_name,
            dir_path=config.logger_config.wandb_dir,
        )
        run_name = wandb.run.name
    else:
        run_name = config.logger_config.run_name

    base_checkpoint_dir = f"{config.logger_config.checkpoint.checkpoint_dir}{run_name}"
    os.makedirs(base_checkpoint_dir, exist_ok=True)

    return config, base_checkpoint_dir


def init_training(config):
    current_device = jax.devices()[0]
    cpu_device = jax.devices("cpu")[0]

    input_shape = (1, 28, 28, 1)
    output_shape = jnp.ones((1, 10))
    pred_model = Models(model_type="MLP", hidden_size=config.net_config.hidden_size)

    # Initializing model parameters on CPU, since Jax random number generator
    # is currently not supported on device (https://github.com/tenstorrent/tt-xla/issues/420).
    with jax.default_device(cpu_device):
        (
            train_images_host,
            train_labels_host,
            eval_images_host,
            eval_labels_host,
            test_images_host,
            test_labels_host,
        ) = load_mnist_jax()

        rng = random.PRNGKey(0)
        params_host = pred_model.model.init(rng, jnp.ones(input_shape))["params"]

    params = jax.device_put(params_host, current_device)

    tx = optax.sgd(learning_rate=config.training_config.lr)
    state = train_state.TrainState.create(apply_fn=pred_model.model.apply, params=params, tx=tx)

    batch_size = config.training_config.batch_size
    num_batches = len(train_images_host) // batch_size
    num_eval_batches = len(eval_images_host) // batch_size

    training_data = {
        "train_images": train_images_host,
        "train_labels": train_labels_host,
        "eval_images": eval_images_host,
        "eval_labels": eval_labels_host,
        "test_images": test_images_host,
        "test_labels": test_labels_host,
        "num_batches": num_batches,
        "num_eval_batches": num_eval_batches,
    }

    return {
        "state": state,
        "pred_model": pred_model,
        "training_data": training_data,
        "devices": {"current": current_device, "cpu": cpu_device},
        "shapes": {"input": input_shape, "output": output_shape},
    }


def train(config: ExperimentConfig):
    config, base_checkpoint_dir = init_configs(config)

    training_components = init_training(config)

    state = training_components["state"]
    training_data = training_components["training_data"]
    current_device = training_components["devices"]["current"]
    cpu_device = training_components["devices"]["cpu"]

    training_config = config.training_config
    batch_size = training_config.batch_size
    epochs = training_config.epochs

    early_stopping_config = config.early_stopping

    train_images_host = training_data["train_images"]
    train_labels_host = training_data["train_labels"]
    eval_images_host = training_data["eval_images"]
    eval_labels_host = training_data["eval_labels"]
    test_images_host = training_data["test_images"]
    test_labels_host = training_data["test_labels"]
    num_batches = training_data["num_batches"]
    num_eval_batches = training_data["num_eval_batches"]

    best_epoch = 0
    best_val_loss = 1e7
    grads = None

    for epoch in range(epochs):
        train_batch_metrics = []
        for i in range(num_batches):
            # Batch creation is done on CPU (https://github.com/tenstorrent/tt-mlir/issues/2309)
            with jax.default_device(cpu_device):
                batch_images_host = train_images_host[i * batch_size : (i + 1) * batch_size]
                batch_labels_host = train_labels_host[i * batch_size : (i + 1) * batch_size]

            batch_images = jax.device_put(batch_images_host, current_device)
            batch_labels = jax.device_put(batch_labels_host, current_device)

            loss, grads, logits = compute_loss_grads_and_logits(state.params, batch_images, batch_labels)

            state = optimizer_step(state, grads)

            accuracy = jnp.mean(jnp.argmax(logits, 1) == jnp.argmax(batch_labels, 1))
            metrics = {"loss": loss, "accuracy": accuracy}
            train_batch_metrics.append(metrics)

        train_batch_metrics_avg = accumulate_metrics(train_batch_metrics)

        eval_batch_metrics = []
        for i in range(num_eval_batches):
            with jax.default_device(cpu_device):
                batch_images_host = eval_images_host[i * batch_size : (i + 1) * batch_size]
                batch_labels_host = eval_labels_host[i * batch_size : (i + 1) * batch_size]

            batch_images = jax.device_put(batch_images_host, current_device)
            batch_labels = jax.device_put(batch_labels_host, current_device)

            loss, logits = eval_step(state.params, batch_images, batch_labels)

            accuracy = jnp.mean(jnp.argmax(logits, 1) == jnp.argmax(batch_labels, 1))

            metrics = {
                "loss": cross_entropy(logits, batch_labels),
                "accuracy": accuracy,
            }
            eval_batch_metrics.append(metrics)

        eval_batch_metrics_avg = accumulate_metrics(eval_batch_metrics)

        if config.logger_config.log_on_wandb:
            log_metrics(
                grads,
                state,
                train_batch_metrics_avg["loss"],
                train_batch_metrics_avg["accuracy"],
                eval_batch_metrics_avg["loss"],
                eval_batch_metrics_avg["accuracy"],
                epoch,
            )
        else:
            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_batch_metrics_avg['loss']:.4f}, "
                f"Train Accuracy: {train_batch_metrics_avg['accuracy']:.4f}, "
                f"Eval Loss: {eval_batch_metrics_avg['loss']:.4f}, "
                f"Eval Accuracy: {eval_batch_metrics_avg['accuracy']:.4f}"
            )

        epoch_dir = f"epoch={epoch:02d}"
        checkpoint_dir = os.path.join(base_checkpoint_dir, epoch_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file_name = "checkpoint.msgpack"
        checkpoint_file_path = os.path.join(checkpoint_dir, checkpoint_file_name)
        save_checkpoint(checkpoint_file_path, state, epoch, config.logger_config.log_on_wandb)

        if eval_batch_metrics_avg["loss"] < best_val_loss - early_stopping_config.min_delta:
            best_val_loss = eval_batch_metrics_avg["loss"]
            best_epoch = epoch
            epochs_no_improvement = 0

        else:
            epochs_no_improvement += 1

        if epochs_no_improvement >= early_stopping_config.patience:
            break

    if training_config.run_test:
        ckpt_file = os.path.join(base_checkpoint_dir, f"epoch={best_epoch:02d}", checkpoint_file_name)
        restored_state = load_checkpoint(ckpt_file, state, best_epoch, config.logger_config.log_on_wandb)
        test_labels = jax.device_put(test_labels_host, current_device)
        test_images = jax.device_put(test_images_host, current_device)
        loss, logits = eval_step(restored_state.params, test_images, test_labels)
        metrics = calculate_metrics_val(logits, loss, test_labels)
        if config.logger_config.log_on_wandb:
            wandb.log({"Test Loss": metrics["loss"], "Test Accuracy": metrics["accuracy"]})
        else:
            print(
                f"Test Loss: {metrics['loss']:.4f}, "
                f"Test Accuracy: {metrics['accuracy']:.4f} (best epoch: {best_epoch + 1})"
            )

    if config.logger_config.log_on_wandb:
        wandb.finish()

    if training_config.export_shlo:
        export_it = ExportSHLO()
        export_it.export_fwd_train_to_StableHLO_and_get_ops(
            forward_pass, state, training_components["shapes"]["input"], print_stablehlo=False
        )
        export_it.export_fwd_tst_to_StableHLO_and_get_ops(
            eval_step, state, training_components["shapes"]["input"], print_stablehlo=False
        )
        export_it.export_loss_to_StableHLO_and_get_ops(
            cross_entropy, training_components["shapes"]["output"], print_stablehlo=False
        )
        export_it.export_optimizer_to_StableHLO_and_get_ops(update_params, state, grads, print_stablehlo=False)

    return state, best_epoch, best_val_loss


if __name__ == "__main__":
    default_config = Path(__file__).parent.parent / "test_mnist.yaml"
    args = parse_cli_options(default_config=default_config)
    config: ExperimentConfig = generate_config(ExperimentConfig, args.config)
    train(config)
