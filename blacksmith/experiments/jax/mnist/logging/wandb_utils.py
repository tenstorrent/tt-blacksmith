# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import jax
import jax.numpy as jnp

import wandb
import os

from flax.serialization import msgpack_serialize, from_bytes, to_state_dict


def init_wandb(project_name, job_type, dir_path):
    wandb.init(dir=dir_path, project=project_name, job_type=job_type)
    config = wandb.config
    return config


def log_metrics(grads, state, train_loss, train_accuracy, val_loss, val_accuracy, epoch, show_optimizer=False):

    for k, v in grads.items():
        wandb.log({f"Gradients/{k}_bias": wandb.Histogram(v["bias"].flatten(), num_bins=100)}, step=epoch)
        wandb.log({f"Gradients/{k}_kernel": wandb.Histogram(v["kernel"].flatten(), num_bins=100)}, step=epoch)

    for layer_name, layer_params in state.params.items():
        for param_name, param_values in layer_params.items():
            wandb.log(
                {f"Weights/{layer_name}_{param_name}": wandb.Histogram(param_values.flatten(), num_bins=100)},
                step=epoch,
            )

    if show_optimizer:
        log_optimizer_state(state.opt_state, state.params, epoch)

    wandb.log(
        {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
    )


def log_optimizer_state(opt_state, params, epoch, prefix=""):
    index_to_name = {1: "first_momentum", 2: "second_momentum"}

    if isinstance(opt_state, dict):
        for k, v in opt_state.items():
            new_prefix = f"{prefix}/{k}" if prefix else k
            log_optimizer_state(v, params.get(k, {}), new_prefix)
    elif isinstance(opt_state, (list, tuple)):
        for idx, v in enumerate(opt_state):
            if idx in index_to_name:
                name = index_to_name[idx]
                new_prefix = f"{prefix}/{name}" if prefix else name
                log_optimizer_state(v, params, new_prefix)
            else:
                log_optimizer_state(v, params, prefix)
    elif isinstance(opt_state, (jnp.ndarray, np.ndarray)):
        wandb.log({f"Optimizer State/{prefix}": wandb.Histogram(opt_state.flatten(), num_bins=100)}, step=epoch)


def save_checkpoint(ckpt_path, state, epoch, log_on_wandb=True):

    with open(ckpt_path, "wb") as outfile:
        outfile.write(msgpack_serialize(to_state_dict(state)))

    if log_on_wandb and not (wandb.run is None or wandb.run.disabled):
        artifact = wandb.Artifact(f"{wandb.run.name}-checkpoint-epoch-{epoch}", type="dataset")
        print(f"Uploading checkpoint to {ckpt_path}")
        artifact.add_reference(f"file://{ckpt_path}")

        wandb.log_artifact(artifact, aliases=[f"epoch_{epoch}", "latest"])
        artifact.wait()


def load_checkpoint(ckpt_file, state, epoch, log_on_wandb=True):

    if log_on_wandb and not (wandb.run is None or wandb.run.disabled):
        artifact = wandb.use_artifact(
            f"{wandb.run.name}-checkpoint-epoch-{epoch}:latest"
        )  # Reference to the specific epoch
        artifact_dir = artifact.download()

        ckpt_path = os.path.join(artifact_dir, ckpt_file)

    else:
        ckpt_path = os.path.join(os.getcwd(), ckpt_file)

    with open(ckpt_path, "rb") as data_file:
        byte_data = data_file.read()

    return from_bytes(state, byte_data)
