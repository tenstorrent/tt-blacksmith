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

from thomas.tools.cli import generate_config
from thomas.tools.jax_utils import init_device

from thomas.models.jax.mnist.model import Models

from thomas.datasets.mnist.dataloader import load_mnist

from thomas.experiments.jax.mnist.logging.shlo_ops_logging import ExportSHLO
from thomas.experiments.jax.mnist.logging.wandb_utils import init_wandb, log_metrics, save_checkpoint, load_checkpoint
from thomas.experiments.jax.mnist.train_utils.train_functions import (
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

from thomas.experiments.jax.mnist.configs.experiment_config import ExperimentConfig


def train(run_test=False, use_export_shlo=False):

    init_device()
    current_device = jax.devices()[0]

    config_wandb = init_wandb(
        project_name="Flax mnist mlp training", job_type="Flax mnist mlp training", dir_path="/proj_sw/user_dev/umales"
    )

    config = generate_config(ExperimentConfig, "thomas/experiments/jax/mnist/configs/test_jax_mnist.yaml")
    training_config = config.training_config
    net_config = config.net_config
    logger_config = config.logger_config

    input_shape = (1, 28, 28, 1)
    output_shape = jnp.ones((1, 10))
    pred_model = Models(model_type="MLP", hidden_size=net_config.hidden_size)

    with jax.default_device(jax.devices("cpu")[0]):
        (
            train_images_host,
            train_labels_host,
            eval_images_host,
            eval_labels_host,
            test_images_host,
            test_labels_host,
        ) = load_mnist()

        rng = random.PRNGKey(0)
        params = pred_model.model.init(rng, jnp.ones(input_shape))["params"]

    tx = optax.sgd(learning_rate=training_config.lr)
    state = train_state.TrainState.create(apply_fn=pred_model.model.apply, params=params, tx=tx)

    num_batches = len(train_images_host) // training_config.batch_size
    num_eval_batches = len(eval_images_host) // training_config.batch_size

    best_epoch = 0
    best_val_loss = 1e7
    for epoch in range(training_config.epochs):

        train_batch_metrics = []
        for i in range(num_batches):

            with jax.default_device(jax.devices("cpu")[0]):
                batch_images_host = train_images_host[
                    i * training_config.batch_size : (i + 1) * training_config.batch_size
                ]
                batch_labels_host = train_labels_host[
                    i * training_config.batch_size : (i + 1) * training_config.batch_size
                ]

            batch_images = jax.device_put(batch_images_host, current_device)
            batch_labels = jax.device_put(batch_labels_host, current_device)

            state, loss, grads = train_step(state, batch_images, batch_labels)

            logits = eval_step(state.params, batch_images)

            logits_host = jax.device_put(logits, jax.devices("cpu")[0])
            batch_labels_host = jax.device_put(batch_labels, jax.devices("cpu")[0])

            with jax.default_device(jax.devices("cpu")[0]):
                accuracy_host = jnp.mean(jnp.argmax(logits_host, 1) == jnp.argmax(batch_labels_host, 1))

            accuracy = jax.device_put(accuracy_host, current_device)

            metrics = {
                "loss": loss,
                "accuracy": accuracy,
            }

            train_batch_metrics.append(metrics)
        train_batch_metrics_avg = accumulate_metrics(train_batch_metrics)

        eval_batch_metrics = []
        for i in range(num_eval_batches):

            with jax.default_device(jax.devices("cpu")[0]):
                batch_images_host = eval_images_host[
                    i * training_config.batch_size : (i + 1) * training_config.batch_size
                ]
                batch_labels_host = eval_labels_host[
                    i * training_config.batch_size : (i + 1) * training_config.batch_size
                ]

            batch_images = jax.device_put(batch_images_host, current_device)
            batch_labels = jax.device_put(batch_labels_host, current_device)

            logits = eval_step(state.params, batch_images)

            logits_host = jax.device_put(logits, jax.devices("cpu")[0])
            batch_labels_host = jax.device_put(batch_labels, jax.devices("cpu")[0])

            with jax.default_device(jax.devices("cpu")[0]):
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

        base_checkpoint_dir = f"/proj_sw/user_dev/umales/checkpoints/{wandb.run.name}"
        epoch_dir = f"epoch={epoch:02d}"
        checkpoint_dir = os.path.join(base_checkpoint_dir, epoch_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file_name = "checkpoint.msgpack"
        checkpoint_file_path = os.path.join(checkpoint_dir, checkpoint_file_name)
        save_checkpoint(checkpoint_file_path, state, epoch)

    time.sleep(2)

    if run_test:
        epoch = best_epoch
        ckpt_file = "checkpoint.msgpack"
        restored_state = load_checkpoint(ckpt_file, state, epoch)
        logits = eval_step(restored_state.params, test_images_host)
        metrics = calculate_metrics_val(logits, test_labels_host)
        wandb.log({"Test Loss": metrics["loss"], "Test Accuracy": metrics["accuracy"]})

    wandb.finish()

    if use_export_shlo:
        export_it = ExportSHLO()
        export_it.export_fwd_train_to_StableHLO_and_get_ops(forward_pass, state, input_shape, print_stablehlo=False)
        export_it.export_fwd_tst_to_StableHLO_and_get_ops(eval_step, state, input_shape, print_stablehlo=False)
        export_it.export_loss_to_StableHLO_and_get_ops(func_optax_loss, output_shape, print_stablehlo=False)
        export_it.export_optimizer_to_StableHLO_and_get_ops(update_params, state, grads, print_stablehlo=False)


def main():
    train(run_test=True, use_export_shlo=False)


if __name__ == "__main__":
    main()
