# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from flax.training import train_state
import optax
import numpy as np
import random as py_random
from PIL import Image

from blender import BlenderDataset, create_dataloader, create_dataloader_val
from nerf_rendering import render_rays
from nerf import Embedding, NeRF
from nerftree import NerfTree
from configs import NerfConfig, load_config
from optimizers import get_optimizer

import wandb

import time


def seed_everything(seed=0):
    py_random.seed(seed)
    np.random.seed(seed)
    return random.PRNGKey(seed)


key = seed_everything(0)


class EfficientNeRFSystem:
    def __init__(self, config: NerfConfig):
        self.config = config
        self.experiment_log_dir = os.path.join(config.training.log_dir, config.experiment_name)
        self.in_channels_xyz = 3 + config.model.num_freqs * 2 * 3
        self.in_channels_dir = config.model.in_channels_dir
        self.deg = config.model.deg
        self.dim_sh = 3 * (self.deg + 1) ** 2
        self.sigma_init = config.model.sigma_init
        self.sigma_default = config.model.sigma_default

        self.embedding_xyz = Embedding(in_channels=3, num_freqs=config.model.num_freqs)
        self.nerf_coarse = NeRF(
            depth=config.model.coarse.depth,
            width=config.model.coarse.width,
            in_channels_xyz=self.in_channels_xyz,
            in_channels_dir=self.in_channels_dir,
            deg=self.deg,
        )
        self.nerf_fine = NeRF(
            depth=config.model.fine.depth,
            width=config.model.fine.width,
            in_channels_xyz=self.in_channels_xyz,
            in_channels_dir=self.in_channels_dir,
            deg=self.deg,
        )

        coord_scope = config.model.coord_scope or 1.0
        self.nerf_tree_base = NerfTree(
            xyz_min=[[-coord_scope, -coord_scope, -coord_scope]],
            xyz_max=[[coord_scope, coord_scope, coord_scope]],
            grid_coarse=384,
            grid_fine=3,
            deg=self.deg,
            sigma_init=self.sigma_init,
            sigma_default=self.sigma_default,
        )

        # self.optimizer = optax.adam(learning_rate=config.training.lr_init if hasattr(config.training, 'lr_init') else 5e-4)
        self.optimizer = get_optimizer(config, [self.nerf_coarse, self.nerf_fine])
        self._init_state()
        self.global_step = 0
        self.current_epoch = 0

    def _init_state(self):
        global key
        key, subkey = random.split(key)
        dummy_input = jnp.ones((1, self.in_channels_xyz))
        params_coarse = self.nerf_coarse.init(subkey, dummy_input)
        params_fine = self.nerf_fine.init(subkey, dummy_input)
        self.params = {
            "nerf_coarse": params_coarse["params"],
            "nerf_fine": params_fine["params"],
        }
        # print(params_coarse['params'])
        self.state_coarse = train_state.TrainState.create(
            apply_fn=self.nerf_coarse.apply, params=self.params["nerf_coarse"], tx=self.optimizer
        )
        self.state_fine = train_state.TrainState.create(
            apply_fn=self.nerf_fine.apply, params=self.params["nerf_fine"], tx=self.optimizer
        )

    def prepare_data(self):
        dataset_kwargs = {
            "root_dir": self.config.data_loading.input_dir,
            "img_wh": tuple(self.config.data_loading.img_wh),
        }
        self.train_dataset = BlenderDataset(split="train", **dataset_kwargs)
        self.val_dataset = BlenderDataset(split="test", **dataset_kwargs)
        self.near = self.train_dataset.near
        self.far = self.train_dataset.far

        # print(len(self.train_dataset))
        # print(len(self.val_dataset))

        self.train_dataloader, self.train_steps_per_epoch = create_dataloader(
            self.train_dataset, self.config.data_loading.batch_size
        )
        self.val_dataloader, self.val_steps_per_epoch = create_dataloader_val(
            self.val_dataset, self.config.data_loading.batch_size
        )

        # print(f"Train dataset size: {len(self.train_dataset)}, steps_per_epoch: {self.train_steps_per_epoch}")
        # print(f"Val dataset size: {len(self.val_dataset)}, steps_per_epoch: {self.val_steps_per_epoch}")

    # @jax.jit
    def forward(self, rays, params, tree_data, global_step):
        if rays.ndim != 2 or rays.shape[1] != 6:
            raise ValueError(f"Expected rays shape (batch_size, 6), got {rays.shape}")

        results = {}
        batch_size = self.config.data_loading.batch_size
        num_rays = rays.shape[0]

        for ray_idx in range(0, num_rays, batch_size):
            rays_chunk = rays[ray_idx : ray_idx + batch_size]
            num_rays_in_chunk = rays_chunk.shape[0]
            num_padding_needed = batch_size - num_rays_in_chunk

            if num_padding_needed > 0:
                global key
                key, subkey = random.split(key)
                random_indices = random.randint(subkey, (num_padding_needed,), 0, num_rays)
                padding_rays = rays[random_indices]
                rays_chunk = jnp.concatenate([rays_chunk, padding_rays], axis=0)

            chunk_results = render_rays(
                config=self.config,
                rays=rays_chunk,
                embedding_xyz=self.embedding_xyz,
                tree_data=tree_data,  # Changed from nerf_tree to tree_data
                near=self.near,
                far=self.far,
                global_step=global_step,
                model_coarse=self.nerf_coarse,
                params_coarse=params["nerf_coarse"],
                model_fine=self.nerf_fine,
                params_fine=params["nerf_fine"],
            )

            for key, value in chunk_results.items():
                if num_padding_needed > 0:
                    trimmed_value = value[:-num_padding_needed]
                else:
                    trimmed_value = value
                results[key] = results.get(key, []) + [trimmed_value]

        for key, value_list in results.items():
            results[key] = jnp.concatenate(value_list, axis=0)

        return results

    def loss_fn(self, results, rgbs):
        rgbs_valid_idx = results.get("rgb_valid", None)
        coarse_loss = jnp.mean((results["rgb_coarse"][rgbs_valid_idx] - rgbs[rgbs_valid_idx]) ** 2)
        total_loss = coarse_loss
        if "rgb_fine" in results:
            fine_loss = jnp.mean((results["rgb_fine"] - rgbs) ** 2)
            total_loss = total_loss + fine_loss
        return total_loss

    def training_step(self, state, batch, global_step):
        rays, rgbs = batch["rays"], batch["rgbs"]
        nerf_tree = NerfTree(
            xyz_min=self.nerf_tree_base.xyz_min,
            xyz_max=self.nerf_tree_base.xyz_max,
            grid_coarse=self.nerf_tree_base.grid_coarse,
            grid_fine=self.nerf_tree_base.grid_fine,
            deg=self.deg,
            sigma_init=self.sigma_init,
            sigma_default=self.sigma_default,
        )
        nerf_tree.sigma_voxels_coarse = self.nerf_tree_base.sigma_voxels_coarse
        nerf_tree.index_voxels_coarse = self.nerf_tree_base.index_voxels_coarse
        nerf_tree.voxels_fine = self.nerf_tree_base.voxels_fine

        extract_time = global_step >= (self.config.training.epochs * self.train_steps_per_epoch - 1)
        if extract_time and nerf_tree.voxels_fine is None:
            index_voxels_coarse, voxels_fine = nerf_tree.create_voxels_fine(
                nerf_tree.sigma_voxels_coarse, nerf_tree.index_voxels_coarse
            )
            nerf_tree.index_voxels_coarse = index_voxels_coarse
            nerf_tree.voxels_fine = voxels_fine
            self.nerf_tree_base.index_voxels_coarse = index_voxels_coarse
            self.nerf_tree_base.voxels_fine = voxels_fine

        tree_data = {
            "sigma_voxels_coarse": nerf_tree.sigma_voxels_coarse,
            "index_voxels_coarse": nerf_tree.index_voxels_coarse,
            "voxels_fine": nerf_tree.voxels_fine,
            "xyz_min": nerf_tree.xyz_min,
            "xyz_max": nerf_tree.xyz_max,
            "grid_coarse": nerf_tree.grid_coarse,
            "grid_fine": nerf_tree.grid_fine,
            "xyz_scope": nerf_tree.xyz_max - nerf_tree.xyz_min,
        }

        @jax.jit
        def compute_grads(params, rays, rgbs, tree_data, global_step):
            def loss_fn(params):
                results = self.forward(rays, params, tree_data, global_step)
                return self.loss_fn(results, rgbs), results

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, results), grads = grad_fn(params)
            return (loss, results), grads

        time1 = time.time()

        params = {"nerf_coarse": state.state_coarse.params, "nerf_fine": state.state_fine.params}
        (loss, results), grads = compute_grads(params, rays, rgbs, tree_data, global_step)

        time2 = time.time()
        # print(time2 - time1)

        # grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        # (loss, results), grads = grad_fn({'nerf_coarse': state.state_coarse.params, 'nerf_fine': state.state_fine.params})

        if "sigma_voxels_coarse" in results:
            self.nerf_tree_base.sigma_voxels_coarse = results["sigma_voxels_coarse"]  # Update outside tracing

        # print(grads['nerf_coarse'])
        state_coarse = state.state_coarse.apply_gradients(grads=grads["nerf_coarse"])
        state_fine = state.state_fine.apply_gradients(grads=grads["nerf_fine"])

        new_state = SystemState(state_coarse, state_fine)
        return new_state, loss, results, grads

    def validation_step(self, batch, batch_idx, global_step):
        rays, rgbs = batch["rays"], batch["rgbs"]
        rays = rays.squeeze()  # (4096, 6)
        rgbs = rgbs.squeeze()  # (4096, 3)

        params = {"nerf_coarse": self.state_coarse.params, "nerf_fine": self.state_fine.params}
        tree_data = {
            "sigma_voxels_coarse": self.nerf_tree_base.sigma_voxels_coarse,
            "index_voxels_coarse": self.nerf_tree_base.index_voxels_coarse,
            "voxels_fine": self.nerf_tree_base.voxels_fine,
            "xyz_min": self.nerf_tree_base.xyz_min,
            "xyz_max": self.nerf_tree_base.xyz_max,
            "grid_coarse": self.nerf_tree_base.grid_coarse,
            "grid_fine": self.nerf_tree_base.grid_fine,
            "xyz_scope": self.nerf_tree_base.xyz_max - self.nerf_tree_base.xyz_min,
        }
        results = self.forward(rays, params, tree_data, global_step)

        log = {}
        log["val_loss"] = self.loss_fn(results, rgbs)
        typ = "fine" if "rgb_fine" in results else "coarse"

        W, H = self.config.data_loading.img_wh  # (200, 200)
        # Map batch_idx to image index (0 to 19)
        img_idx = batch_idx // ((W * H // 4096) + 1)  # 10 batches per image
        batch_within_image = batch_idx % ((W * H // 4096) + 1)  # 0 to 9
        self.validation_step_outputs.append(
            {
                "img": results[f"rgb_{typ}"],  # (4096, 3)
                "gt": rgbs,  # (4096, 3)
                "idx": img_idx,  # Image index (0 to 19)
                "batch_idx": batch_within_image,  # Batch within image (0 to 9)
            }
        )

        mse = jnp.mean((results[f"rgb_{typ}"] - rgbs) ** 2)
        log["val_psnr"] = -10.0 * jnp.log10(mse)

        self.validation_step_outputs.append(log)
        return log

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            print(f"Step {self.global_step} - No validation data")
            return

        mean_loss = jnp.mean(jnp.array([x["val_loss"] for x in self.validation_step_outputs if "val_loss" in x]))
        mean_psnr = jnp.mean(jnp.array([x["val_psnr"] for x in self.validation_step_outputs if "val_psnr" in x]))
        num_voxels_coarse = jnp.sum(
            jnp.logical_and(
                self.nerf_tree_base.sigma_voxels_coarse > 0,
                self.nerf_tree_base.sigma_voxels_coarse != self.sigma_init,
            )
        )

        img_dict = {}
        for output in self.validation_step_outputs:
            if "img" in output:
                idx = output["idx"]
                if idx not in img_dict:
                    img_dict[idx] = {"pred": [], "gt": [], "batch_indices": []}
                img_dict[idx]["pred"].append(output["img"])
                img_dict[idx]["gt"].append(output["gt"])
                img_dict[idx]["batch_indices"].append(output["batch_idx"])

        W, H = self.config.data_loading.img_wh
        for idx in img_dict:
            sorted_indices = jnp.argsort(jnp.array(img_dict[idx]["batch_indices"]))
            pred_batches = [img_dict[idx]["pred"][i] for i in sorted_indices]
            gt_batches = [img_dict[idx]["gt"][i] for i in sorted_indices]

            pred_rays = jnp.concatenate(pred_batches, axis=0)[: W * H]
            gt_rays = jnp.concatenate(gt_batches, axis=0)[: W * H]

            # Just reshape to verify it works, no saving
            img = pred_rays.reshape(H, W, 3)
            img_gt = gt_rays.reshape(H, W, 3)

            # Log only the first image (idx=0) to Wandb
            if idx == 0:
                img_np = np.array(img.clip(0, 1) * 255, dtype=np.uint8)
                img_gt_np = np.array(img_gt.clip(0, 1) * 255, dtype=np.uint8)
                wandb.log(
                    {
                        "val/gt_pred": [
                            wandb.Image(img_gt_np, caption="Ground Truth"),
                            wandb.Image(img_np, caption="Prediction"),
                        ]
                    },
                    step=self.global_step,
                )

        # Log mean loss and PSNR for all images
        wandb.log({"val/loss": float(mean_loss), "val/psnr": float(mean_psnr)}, step=self.global_step)

        print(
            f"Step {self.global_step} - val/loss: {mean_loss:.4f}, val/psnr: {mean_psnr:.4f}, num_voxels_coarse: {num_voxels_coarse}"
        )
        self.validation_step_outputs = []


def train_step(system, state, batch, step):
    state, loss, results, grads = system.training_step(state, batch, step)
    system.global_step = step
    print(f"Step {step}, Loss: {loss}")
    log_dict = {"train/loss": float(loss)}

    # Helper function to recursively log weights
    def log_weights(params, prefix=""):
        for key, value in params.items():
            current_path = f"{prefix}/{key}" if prefix else key
            if isinstance(value, dict):
                # Recursively handle nested dictionaries
                log_weights(value, current_path)
            else:
                # Leaf node: assume it's a JAX array and log it
                param_np = np.array(value.flatten())
                log_dict[f"train/weights_{current_path}"] = wandb.Histogram(
                    np_histogram=np.histogram(param_np, bins=50)
                )

    # Log all weights for coarse and fine models
    # log_weights(state.state_coarse.params, "coarse")
    # log_weights(state.state_fine.params, "fine")

    # Compute and log training PSNR
    rgbs = batch["rgbs"]  # Ground truth RGB values
    pred_rgb = results.get("rgb_fine", results["rgb_coarse"])  # Use fine if available, else coarse
    mse = jnp.mean((pred_rgb - rgbs) ** 2)  # Mean Squared Error
    psnr = -10.0 * jnp.log10(mse)  # PSNR formula
    log_dict["train/psnr"] = float(psnr)

    # Flatten and log gradients (unchanged)
    coarse_grads_flat = jax.tree_util.tree_leaves(grads["nerf_coarse"])
    fine_grads_flat = jax.tree_util.tree_leaves(grads["nerf_fine"])
    coarse_grads_mean = jnp.mean(jnp.concatenate([jnp.ravel(g) for g in coarse_grads_flat]))
    fine_grads_mean = jnp.mean(jnp.concatenate([jnp.ravel(g) for g in fine_grads_flat]))
    # log_dict["train/grads_coarse_mean"] = float(coarse_grads_mean)
    # log_dict["train/grads_fine_mean"] = float(fine_grads_mean)
    # log_dict["train/grads_coarse_hist"] = wandb.Histogram(np_histogram=np.histogram(
    #    np.concatenate([g.flatten() for g in coarse_grads_flat]), bins=50))
    # log_dict["train/grads_fine_hist"] = wandb.Histogram(np_histogram=np.histogram(
    #    np.concatenate([g.flatten() for g in fine_grads_flat]), bins=50))

    wandb.log(log_dict, step=step)
    return state


class SystemState:
    def __init__(self, state_coarse, state_fine):
        self.state_coarse = state_coarse
        self.state_fine = state_fine


def main(config: NerfConfig):

    wandb.init(project="jax-nerf", config=config.__dict__)

    system = EfficientNeRFSystem(config)
    system.prepare_data()

    state = SystemState(system.state_coarse, system.state_fine)

    train_iter = iter(system.train_dataloader)
    total_steps = config.training.epochs * system.train_steps_per_epoch

    for epoch in range(config.training.epochs):
        system.current_epoch = epoch
        for step in range(system.train_steps_per_epoch):
            global_step = epoch * system.train_steps_per_epoch + step
            batch = next(train_iter)
            # print(f"Batch shape: rays={batch['rays'].shape}, rgbs={batch['rgbs'].shape}")
            state = train_step(system, state, batch, global_step)

            if global_step % config.training.log_every == 45:
                val_iter = iter(system.val_dataloader)  # Reset iterator each validation
                system.validation_step_outputs = []
                # print(system.val_steps_per_epoch)
                for batch_idx in range(system.val_steps_per_epoch):  # 195 steps
                    # print(f"Validation batch {batch_idx} of {system.val_steps_per_epoch}")
                    batch = next(val_iter)
                    system.validation_step(batch, batch_idx, global_step)
                system.on_validation_epoch_end()

    # Final validation
    val_iter = iter(system.val_dataloader)
    system.validation_step_outputs = []
    for batch_idx in range(system.val_steps_per_epoch):
        batch = next(val_iter)
        system.validation_step(batch, batch_idx, total_steps)
    system.on_validation_epoch_end()


if __name__ == "__main__":
    config_file_path = os.path.join(os.path.dirname(__file__), "test_nerf.yaml")
    config = load_config(config_file_path)
    main(config)
