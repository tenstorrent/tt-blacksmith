# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
from jax import random

from flax.training import train_state
from flax.serialization import to_state_dict, msgpack_serialize, from_bytes

import optax

import wandb
import os
import time

from blacksmith.tools.cli import generate_config
from blacksmith.tools.jax_utils import init_device

from blacksmith.models.jax.mnist.model import Models

from blacksmith.datasets.jax.mnist.dataloader import load_mnist_jax

from blacksmith.experiments.jax.mnist.logging.shlo_ops_logging import ExportSHLO
from blacksmith.experiments.jax.mnist.logging.wandb_utils import (
    init_wandb,
    log_metrics,
    save_checkpoint,
    load_checkpoint,
)
from blacksmith.experiments.jax.mnist.single_chip.train_utils.train_functions import (
    forward_pass,
    forward_and_compute_loss,
    func_optax_loss,
    compute_loss_and_backward_pass,
    update_params,
    train_step,
    eval_step,
    calculate_metrics_train,
    calculate_metrics_val,
    accumulate_metrics,
)

from blacksmith.experiments.jax.mnist.configs import ExperimentConfig


def init_configs(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "..", "test_mnist.yaml")

    config = generate_config(ExperimentConfig, config_path)

    config_wandb = init_wandb(
        project_name=config.logger_config.experiment_name,
        job_type=config.logger_config.experiment_name,
        dir_path=config.logger_config.wandb_dir,
    )

    base_checkpoint_dir = f"{config.logger_config.checkpoint.checkpoint_dir}{wandb.run.name}"
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
        params = pred_model.model.init(rng, jnp.ones(input_shape))["params"]

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


def train(config_path=None):
    config, base_checkpoint_dir = init_configs(config_path)

    training_components = init_training(config)

    state = training_components["state"]
    training_data = training_components["training_data"]
    current_device = training_components["devices"]["current"]
    cpu_device = training_components["devices"]["cpu"]

    training_config = config.training_config
    batch_size = training_config.batch_size
    epochs = training_config.epochs

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

            state, loss, grads = train_step(state, batch_images, batch_labels)
            logits = eval_step(state.params, batch_images)

            logits_host = jax.device_put(logits, cpu_device)
            batch_labels_host = jax.device_put(batch_labels, cpu_device)

            # Accuracy calculation is done on CPU, as argmax is not supported on device (https://github.com/tenstorrent/tt-metal/issues/20570)
            with jax.default_device(cpu_device):
                accuracy_host = jnp.mean(jnp.argmax(logits_host, 1) == jnp.argmax(batch_labels_host, 1))

            accuracy = jax.device_put(accuracy_host, current_device)
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

            logits = eval_step(state.params, batch_images)

            logits_host = jax.device_put(logits, cpu_device)
            batch_labels_host = jax.device_put(batch_labels, cpu_device)

            with jax.default_device(cpu_device):
                accuracy_host = jnp.mean(jnp.argmax(logits_host, 1) == jnp.argmax(batch_labels_host, 1))

            accuracy = jax.device_put(accuracy_host, current_device)
            metrics = {
                "loss": func_optax_loss(logits, batch_labels),
                "accuracy": accuracy,
            }
            eval_batch_metrics.append(metrics)

        eval_batch_metrics_avg = accumulate_metrics(eval_batch_metrics)

        if eval_batch_metrics_avg["loss"] < best_val_loss:
            best_val_loss = eval_batch_metrics_avg["loss"]
            best_epoch = epoch

        log_metrics(
            grads,
            state,
            train_batch_metrics_avg["loss"],
            train_batch_metrics_avg["accuracy"],
            eval_batch_metrics_avg["loss"],
            eval_batch_metrics_avg["accuracy"],
            epoch,
        )

        epoch_dir = f"epoch={epoch:02d}"
        checkpoint_dir = os.path.join(base_checkpoint_dir, epoch_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file_name = "checkpoint.msgpack"
        checkpoint_file_path = os.path.join(checkpoint_dir, checkpoint_file_name)
        save_checkpoint(checkpoint_file_path, state, epoch)

    if training_config.run_test:
        ckpt_file = os.path.join(base_checkpoint_dir, f"epoch={best_epoch:02d}", checkpoint_file_name)
        restored_state = load_checkpoint(ckpt_file, state, best_epoch)
        logits = eval_step(restored_state.params, test_images_host)
        metrics = calculate_metrics_val(logits, test_labels_host)
        wandb.log({"Test Loss": metrics["loss"], "Test Accuracy": metrics["accuracy"]})

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
            func_optax_loss, training_components["shapes"]["output"], print_stablehlo=False
        )
        export_it.export_optimizer_to_StableHLO_and_get_ops(update_params, state, grads, print_stablehlo=False)

    return state, best_epoch, best_val_loss


if __name__ == "__main__":
    init_device()
    train()
