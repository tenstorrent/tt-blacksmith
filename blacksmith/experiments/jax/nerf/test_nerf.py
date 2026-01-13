# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from configs import NerfConfig
from flax import serialization
from flax.serialization import from_state_dict
from flax.training import train_state
from jax import random
from nerf_rendering import generate_ray_samples, render_rays
from PIL import Image

from blacksmith.datasets.jax.nerf.blender import (
    BlenderDataset,
    create_dataloader,
    create_dataloader_val,
)
from blacksmith.experiments.jax.nerf.utils.optimizers import get_optimizer
from blacksmith.models.jax.nerf.nerf import Embedding, NeRF
from blacksmith.models.jax.nerf.nerftree import NerfTree
from blacksmith.tools.cli import generate_config, parse_cli_options


class EfficientNeRFSystem:
    def __init__(self, config: NerfConfig, rng_key):
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
            deg=self.deg,
        )
        self.nerf_fine = NeRF(
            depth=config.model.fine.depth,
            width=config.model.fine.width,
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

        self.optimizer = get_optimizer(config, [self.nerf_coarse, self.nerf_fine])
        self._init_state(rng_key)
        self.global_step = 0
        self.current_epoch = 0

    def _init_state(self, rng_key):
        rng_key, subkey = random.split(rng_key)
        dummy_input = jnp.ones((1, self.in_channels_xyz))
        params_coarse = self.nerf_coarse.init(subkey, dummy_input)
        params_fine = self.nerf_fine.init(subkey, dummy_input)
        self.params = {
            "nerf_coarse": params_coarse["params"],
            "nerf_fine": params_fine["params"],
        }
        self.state_coarse = train_state.TrainState.create(
            apply_fn=self.nerf_coarse.apply, params=self.params["nerf_coarse"], tx=self.optimizer
        )
        self.state_fine = train_state.TrainState.create(
            apply_fn=self.nerf_fine.apply, params=self.params["nerf_fine"], tx=self.optimizer
        )

    def prepare_data(self):
        dataset_kwargs = {
            "dataset_name": self.config.data_loading.dataset_name,
            "img_wh": tuple(self.config.data_loading.img_wh),
        }
        self.train_dataset = BlenderDataset(split="train", **dataset_kwargs)
        self.val_dataset = BlenderDataset(split="test", **dataset_kwargs)
        self.near = self.train_dataset.near
        self.far = self.train_dataset.far

        self.train_dataloader, self.train_steps_per_epoch = create_dataloader(
            self.train_dataset, self.config.data_loading.batch_size
        )
        self.val_dataloader, self.val_steps_per_epoch = create_dataloader_val(
            self.val_dataset, self.config.data_loading.batch_size
        )

    def preprocess_ray_chunk(self, rays, rgbs, ray_idx, batch_size, num_rays, rng_key):
        rays_chunk = rays[ray_idx : ray_idx + batch_size]
        rgbs_chunk = rgbs[ray_idx : ray_idx + batch_size]
        num_rays_in_chunk = rays_chunk.shape[0]
        num_padding_needed = batch_size - num_rays_in_chunk

        if num_padding_needed > 0:
            rng_key, subkey = random.split(rng_key)
            random_indices = random.randint(subkey, (num_padding_needed,), 0, num_rays)
            padding_rays = rays[random_indices]
            rays_chunk = jnp.concatenate([rays_chunk, padding_rays], axis=0)
            padding_rgbs = rgbs[random_indices]
            rgbs_chunk = jnp.concatenate([rgbs, padding_rgbs], axis=0)

        xyz_coarse, deltas_coarse = generate_ray_samples(rays_chunk, config.model.coarse.samples, self.near, self.far)
        xyz_fine, deltas_fine = generate_ray_samples(
            rays_chunk, config.model.coarse.samples * config.model.fine.samples, self.near, self.far
        )
        rays_o, rays_d = rays_chunk[:, 0:3], rays_chunk[:, 3:6]
        return (
            rays_chunk,
            rgbs_chunk,
            num_padding_needed,
            xyz_coarse,
            deltas_coarse,
            xyz_fine,
            deltas_fine,
            rays_o,
            rays_d,
        )

    def forward(self, rays, rays_data, params, tree_data, global_step, rng_key):
        if rays.ndim != 2 or rays.shape[1] != 6:
            raise ValueError(f"Expected rays shape (batch_size, 6), got {rays.shape}")

        rays_chunk = rays_data["rays_chunk"]
        num_padding_needed = rays_data["num_padding_needed"]
        xyz_coarse = rays_data["xyz_coarse"]
        deltas_coarse = rays_data["deltas_coarse"]
        xyz_fine = rays_data["xyz_fine"]
        deltas_fine = rays_data["deltas_fine"]
        rays_o = rays_data["rays_o"]
        rays_d = rays_data["rays_d"]

        chunk_results = render_rays(
            config=self.config,
            rays=rays_chunk,
            embedding_xyz=self.embedding_xyz,
            tree_data=tree_data,
            near=self.near,
            far=self.far,
            global_step=global_step,
            model_coarse=self.nerf_coarse,
            params_coarse=params["nerf_coarse"],
            model_fine=self.nerf_fine,
            params_fine=params["nerf_fine"],
            xyz_coarse=xyz_coarse,
            deltas_coarse=deltas_coarse,
            xyz_fine=xyz_fine,
            deltas_fine=deltas_fine,
            rays_origin=rays_o,
            rays_direction=rays_d,
        )
        return chunk_results

    def manage_padding(self, chunk_results, tree_data, num_padding_needed):
        with jax.default_device(jax.devices("cpu")[0]):
            results = {}
            sigma_voxels_coarse = None

            for keyyy, value in chunk_results.items():
                if keyyy != "sigma_voxels_coarse":
                    if num_padding_needed > 0:
                        trimmed_value = value[:-num_padding_needed]
                    else:
                        trimmed_value = value
                    results[keyyy] = results.get(keyyy, []) + [trimmed_value]
                else:
                    sigma_voxels_coarse = value

            for keyyy, value_list in results.items():
                if isinstance(value_list, Callable):
                    continue
                if isinstance(value_list, (list, tuple)) and all(isinstance(v, jnp.ndarray) for v in value_list):
                    results[keyyy] = jnp.concatenate(value_list, axis=0)
                else:
                    print(f"Skipping key '{keyyy}': value is not a list of arrays")

            if sigma_voxels_coarse is not None:
                results["sigma_voxels_coarse"] = sigma_voxels_coarse
            else:
                results["sigma_voxels_coarse"] = tree_data["sigma_voxels_coarse"]

            return results

    def loss_fn(self, results, rgbs):
        with jax.default_device(jax.devices("cpu")[0]):
            rgbs_valid_idx = results.get("rgb_valid", None)
            coarse_loss = jnp.mean((results["rgb_coarse"][rgbs_valid_idx] - rgbs[rgbs_valid_idx]) ** 2)
            total_loss = coarse_loss
            if "rgb_fine" in results:
                fine_loss = jnp.mean((results["rgb_fine"] - rgbs) ** 2)
                total_loss = total_loss + fine_loss
            return total_loss

    def training_step(self, batch, global_step, rng_key):

        with jax.default_device(jax.devices("cpu")[0]):
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
            if self.config.training.cache_voxels_fine and extract_time and nerf_tree.voxels_fine is None:
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

            ray_idx = 0
            batch_size = config.data_loading.batch_size
            num_rays = rays.shape[0]

            (
                rays_chunk,
                rgbs_chunk,
                num_padding_needed,
                xyz_coarse,
                deltas_coarse,
                xyz_fine,
                deltas_fine,
                rays_o,
                rays_d,
            ) = self.preprocess_ray_chunk(rays, rgbs, ray_idx, batch_size, num_rays, rng_key)
            rays_data = {
                "rays_chunk": rays_chunk,
                "rgbs_chunk": rgbs_chunk,
                "num_padding_needed": num_padding_needed,
                "xyz_coarse": xyz_coarse,
                "deltas_coarse": deltas_coarse,
                "xyz_fine": xyz_fine,
                "deltas_fine": deltas_fine,
                "rays_o": rays_o,
                "rays_d": rays_d,
            }

            # Forward pass
            params = {"nerf_coarse": self.state_coarse.params, "nerf_fine": self.state_fine.params}
            results = self.forward(rays, rays_data, params, tree_data, global_step, rng_key)
            loss = self.loss_fn(results, rays_data["rgbs_chunk"])

            print("Loss: ", loss)

            nn_backward_coarse = results.pop("nn_backward_coarse", None)
            nn_backward_fine = results.pop("nn_backward_fine", None)
            idx_render_coarse = results.get("idx_render_coarse")
            idx_render_fine = results.get("idx_render_fine")

            rgb_valid_idx = results.get("rgb_valid", None)

            # Helper function to recompute rgb from sigma and sh
            def compute_rgb_from_sigma_sh(
                sigma, sh, deltas, rays_d, idx_render, sigma_default, non_minus_one_mask, chunk_size, sigma_key
            ):
                batch_size, sample_size = deltas.shape[:2]
                real_chunk_size = idx_render.shape[0]

                model = self.nerf_coarse if "coarse" in sigma_key else self.nerf_fine

                view_dir = jnp.expand_dims(rays_d, 1).repeat(sample_size, axis=1)
                view_dir_flat = view_dir[idx_render[:, 0], idx_render[:, 1]]

                sigma = sigma.reshape(-1, 1)
                sh = sh.reshape(-1, 27)

                sigma_out, rgb, sh_out = model.sh2rgb(sigma, sh, model.deg, view_dir_flat)

                sigma = sigma_out[:real_chunk_size]
                rgb = rgb[:real_chunk_size]
                sh = sh_out[:real_chunk_size]

                out_rgb = jnp.ones((batch_size, sample_size, 3))
                out_sigma = jnp.full((batch_size, sample_size, 1), sigma_default)
                out_sh = jnp.zeros((batch_size, sample_size, 27))

                out_sigma = out_sigma.at[idx_render[:, 0], idx_render[:, 1]].set(sigma)
                out_rgb = out_rgb.at[idx_render[:, 0], idx_render[:, 1]].set(rgb)
                out_sh = out_sh.at[idx_render[:, 0], idx_render[:, 1]].set(sh)

                non_minus_one_mask = jnp.expand_dims(non_minus_one_mask, axis=-1)
                out_sigma = out_sigma * non_minus_one_mask

                weights, _ = model.sigma2weights(deltas, out_sigma, non_minus_one_mask)
                weights_sum = weights.sum(axis=1)
                rgb_final = jnp.sum(weights[..., None] * out_rgb, axis=-2)
                rgb_final = rgb_final + (1 - weights_sum[..., None])

                return rgb_final

            def loss_from_sigma_sh(sigma_key, sh_key, results, rgbs):
                deltas = rays_data["deltas_coarse"] if sigma_key == "sigma_coarse" else rays_data["deltas_fine"]
                rays_d = rays_data["rays_d"]
                idx_render = results.get("idx_render_coarse" if sigma_key == "sigma_coarse" else "idx_render_fine")

                sigma_immediate_key = (
                    "sigma_coarse_immediate" if sigma_key == "sigma_coarse" else "sigma_fine_immediate"
                )
                sh_immediate_key = "sh_coarse_immediate" if sigma_key == "sigma_coarse" else "sh_fine_immediate"
                sigma = results[sigma_immediate_key]
                sh = results[sh_immediate_key]

                batch_size, sample_size = deltas.shape[:2]
                non_minus_one_mask = jnp.ones((batch_size, sample_size))
                non_one_idx = idx_render * (idx_render == -1)
                non_minus_one_mask = non_minus_one_mask.at[non_one_idx].set(0)

                rgb = compute_rgb_from_sigma_sh(
                    sigma=sigma,
                    sh=sh,
                    deltas=deltas,
                    rays_d=rays_d,
                    idx_render=idx_render,
                    sigma_default=config.model.sigma_default,
                    non_minus_one_mask=non_minus_one_mask,
                    chunk_size=1024,
                    sigma_key=sigma_key,
                )

                new_results = {**results}
                if sigma_key == "sigma_coarse":
                    new_results["rgb_coarse"] = rgb
                else:
                    new_results["rgb_fine"] = rgb
                return self.loss_fn(new_results, rgbs)

            sigma_grad_coarse_full = jax.grad(
                lambda s: loss_from_sigma_sh(
                    "sigma_coarse", "sh_coarse", {**results, "sigma_coarse_immediate": s}, rgbs_chunk
                )
            )(results["sigma_coarse_immediate"])
            sh_grad_coarse_full = jax.grad(
                lambda sh: loss_from_sigma_sh(
                    "sigma_coarse", "sh_coarse", {**results, "sh_coarse_immediate": sh}, rgbs_chunk
                )
            )(results["sh_coarse_immediate"])
            sigma_grad_fine_full = jax.grad(
                lambda s: loss_from_sigma_sh(
                    "sigma_fine", "sh_fine", {**results, "sigma_fine_immediate": s}, rgbs_chunk
                )
            )(results["sigma_fine_immediate"])
            sh_grad_fine_full = jax.grad(
                lambda sh: loss_from_sigma_sh("sigma_fine", "sh_fine", {**results, "sh_fine_immediate": sh}, rgbs_chunk)
            )(results["sh_fine_immediate"])

            if idx_render_coarse is not None:
                sigma_grad_coarse = sigma_grad_coarse_full[idx_render_coarse[:, 0], idx_render_coarse[:, 1]]
                sh_grad_coarse = sh_grad_coarse_full[idx_render_coarse[:, 0], idx_render_coarse[:, 1]]
                sigma_grad_coarse = sigma_grad_coarse_full.reshape(-1, 1)
                sh_grad_coarse = sh_grad_coarse_full.reshape(-1, 27)
            else:
                sigma_grad_coarse = sigma_grad_coarse_full.reshape(-1, 1)
                sh_grad_coarse = sh_grad_coarse_full.reshape(-1, 27)

            if idx_render_fine is not None:
                sigma_grad_fine = sigma_grad_fine_full[idx_render_fine[:, 0], idx_render_fine[:, 1]]
                sh_grad_fine = sh_grad_fine_full[idx_render_fine[:, 0], idx_render_fine[:, 1]]
                sigma_grad_fine = sigma_grad_fine_full.reshape(-1, 1)
                sh_grad_fine = sh_grad_fine_full.reshape(-1, 27)
            else:
                sigma_grad_fine = sigma_grad_fine_full.reshape(-1, 1)
                sh_grad_fine = sh_grad_fine_full.reshape(-1, 27)

            grads = {}
            if nn_backward_coarse is None or nn_backward_fine is None:
                raise ValueError(
                    "One or both nn_backward functions are None: coarse=%s, fine=%s"
                    % (nn_backward_coarse is None, nn_backward_fine is None)
                )
            coarse_grads = nn_backward_coarse(sigma_grad_coarse, sh_grad_coarse)
            fine_grads = nn_backward_fine(sigma_grad_fine, sh_grad_fine)
            grads["nerf_coarse"] = coarse_grads
            grads["nerf_fine"] = fine_grads

            print("Global step: ", global_step)
            print("Loss: ", loss)

            if "sigma_voxels_coarse" in results:
                self.nerf_tree_base.sigma_voxels_coarse = results["sigma_voxels_coarse"]
            if "index_voxels_coarse" in results:
                self.nerf_tree_base.index_voxels_coarse = results["index_voxels_coarse"]
            if "voxels_fine" in results:
                self.nerf_tree_base.voxels_fine = results["voxels_fine"]

            self.state_coarse = self.state_coarse.apply_gradients(grads=grads["nerf_coarse"])
            self.state_fine = self.state_fine.apply_gradients(grads=grads["nerf_fine"])

        return loss, results, grads, rays_chunk, rgbs_chunk

    def validation_step(self, batch, batch_idx, global_step, key):
        with jax.default_device(jax.devices("cpu")[0]):
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

            ray_idx = 0
            batch_size = config.data_loading.batch_size
            num_rays = rays.shape[0]
            (
                rays_chunk,
                rgbs_chunk,
                num_padding_needed,
                xyz_coarse,
                deltas_coarse,
                xyz_fine,
                deltas_fine,
                rays_o,
                rays_d,
            ) = self.preprocess_ray_chunk(rays, rgbs, ray_idx, batch_size, num_rays, key)
            rays_data = {
                "rays_chunk": rays_chunk,
                "num_padding_needed": num_padding_needed,
                "xyz_coarse": xyz_coarse,
                "deltas_coarse": deltas_coarse,
                "xyz_fine": xyz_fine,
                "deltas_fine": deltas_fine,
                "rays_o": rays_o,
                "rays_d": rays_d,
            }

            results = self.forward(rays, rays_data, params, tree_data, global_step, key)
            results = self.manage_padding(results, tree_data, num_padding_needed)

            log = {}
            log["val_loss"] = self.loss_fn(results, rgbs)
            typ = "fine" if "rgb_fine" in results else "coarse"

            W, H = self.config.data_loading.img_wh
            img_idx = batch_idx // ((W * H // 4096) + 1)
            batch_within_image = batch_idx % ((W * H // 4096) + 1)
            self.validation_step_outputs.append(
                {
                    "img": results[f"rgb_{typ}"],
                    "gt": rgbs,
                    "idx": img_idx,
                    "batch_idx": batch_within_image,
                }
            )

            mse = jnp.mean((results[f"rgb_{typ}"] - rgbs) ** 2)
            log["val_psnr"] = -10.0 * jnp.log10(mse)

            self.validation_step_outputs.append(log)
            return log

    def on_validation_epoch_end(self):
        with jax.default_device(jax.devices("cpu")[0]):
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

                img = pred_rays.reshape(H, W, 3)
                img_gt = gt_rays.reshape(H, W, 3)

                if idx == 0:
                    img_np = np.array(img)
                    img_np = np.clip(img_np, 0.0, 1.0)
                    img_gt_np = np.array(img_gt)
                    if config.training.log_on_wandb:
                        wandb.log(
                            {
                                "val/gt_pred": [
                                    wandb.Image(img_gt_np, caption="Ground Truth"),
                                    wandb.Image(img_np, caption="Prediction"),
                                ]
                            },
                            step=self.global_step,
                        )

            # Convert JAX arrays to numpy for Wandb logging
            sigma_voxels_coarse_np = np.array(self.nerf_tree_base.sigma_voxels_coarse)
            index_voxels_coarse_np = np.array(self.nerf_tree_base.index_voxels_coarse)
            voxels_fine_np = np.array(self.nerf_tree_base.voxels_fine)

            if config.training.log_on_wandb:
                wandb_log_dict = {
                    "val/loss": float(mean_loss),
                    "val/psnr": float(mean_psnr),
                }

            if config.training.log_on_wandb:
                wandb.log(wandb_log_dict, step=self.global_step)

            print(
                f"Step {self.global_step} - val/loss: {mean_loss:.4f}, val/psnr: {mean_psnr:.4f}, "
                f"num_voxels_coarse: {num_voxels_coarse}, "
                f"sigma_voxels_mean: {float(np.mean(sigma_voxels_coarse_np)):.4f}"
            )
            self.validation_step_outputs = []


def train_step(system, batch, step, rng_key):
    with jax.default_device(jax.devices("cpu")[0]):
        loss, results, grads, rays_chunk, rgbs_chunk = system.training_step(batch, step, rng_key)
        system.global_step = step
        print(f"Step {step}, Loss: {loss}")
        log_dict = {"train/loss": float(loss)}

        def log_weights(params, prefix=""):
            for keyyy, value in params.items():
                current_path = f"{prefix}/{keyyy}" if prefix else keyyy
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
        if config.training.log_on_wandb:
            log_weights(system.state_coarse.params, "coarse")
            log_weights(system.state_fine.params, "fine")

        rgbs = batch["rgbs"]
        pred_rgb = results.get("rgb_fine", results["rgb_coarse"])
        mse = jnp.mean((pred_rgb - rgbs_chunk) ** 2)
        psnr = -10.0 * jnp.log10(mse)
        log_dict["train/psnr"] = float(psnr)

        coarse_grads_flat = jax.tree_util.tree_leaves(grads["nerf_coarse"])
        fine_grads_flat = jax.tree_util.tree_leaves(grads["nerf_fine"])

        log_dict["train/grads_coarse_hist"] = wandb.Histogram(
            np_histogram=np.histogram(np.concatenate([g.flatten() for g in coarse_grads_flat]), bins=50)
        )
        log_dict["train/grads_fine_hist"] = wandb.Histogram(
            np_histogram=np.histogram(np.concatenate([g.flatten() for g in fine_grads_flat]), bins=50)
        )

        if config.training.log_on_wandb:
            wandb.log(log_dict, step=step)
        return system


def save_checkpoint(
    system: EfficientNeRFSystem, global_step: int, rng_key: jnp.ndarray, checkpoint_dir: str, keep_last_n: int
):
    checkpoint_data = {
        "system": {
            "state_coarse": system.state_coarse,
            "state_fine": system.state_fine,
            "nerf_tree_base": {
                "sigma_voxels_coarse": system.nerf_tree_base.sigma_voxels_coarse,
                "index_voxels_coarse": system.nerf_tree_base.index_voxels_coarse,
                "voxels_fine": system.nerf_tree_base.voxels_fine,
                "xyz_min": jnp.array(system.nerf_tree_base.xyz_min, dtype=jnp.float32),
                "xyz_max": jnp.array(system.nerf_tree_base.xyz_max, dtype=jnp.float32),
                "grid_coarse": system.nerf_tree_base.grid_coarse,
                "grid_fine": system.nerf_tree_base.grid_fine,
                "deg": system.nerf_tree_base.deg,
                "sigma_init": system.nerf_tree_base.sigma_init,
                "sigma_default": system.nerf_tree_base.sigma_default,
            },
            "global_step": system.global_step,
            "current_epoch": system.current_epoch,
            # "config": system.config,
            "in_channels_xyz": system.in_channels_xyz,
            "in_channels_dir": system.in_channels_dir,
            "deg": system.deg,
            "dim_sh": system.dim_sh,
            "sigma_init": system.sigma_init,
            "sigma_default": system.sigma_default,
            "near": system.near,
            "far": system.far,
        },
        "global_step": global_step,
        "rng_key": rng_key,
    }
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{global_step}.flax")
    with jax.default_device(jax.devices("cpu")[0]):
        serialized_data = serialization.to_bytes(checkpoint_data)
        with open(checkpoint_path, "wb") as f:
            f.write(serialized_data)

    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".flax")]
    if len(checkpoint_files) > keep_last_n:
        steps = sorted([int(f.split("_")[1].split(".")[0]) for f in checkpoint_files])
        oldest_step = steps[0]
        oldest_checkpoint = os.path.join(checkpoint_dir, f"checkpoint_{oldest_step}.flax")
        os.remove(oldest_checkpoint)
        print(f"Deleted oldest checkpoint: checkpoint_{oldest_step}.flax")


def load_latest_checkpoint(
    checkpoint_dir: str, config: NerfConfig, rng_key: jnp.ndarray
) -> tuple[EfficientNeRFSystem, int, jnp.ndarray] | None:
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".flax")]
    if not checkpoint_files:
        return None

    steps = [int(f.split("_")[1].split(".")[0]) for f in checkpoint_files]
    latest_step = max(steps)
    latest_checkpoint = os.path.join(checkpoint_dir, f"checkpoint_{latest_step}.flax")

    print(f"Loading checkpoint: {latest_checkpoint}")

    try:
        with jax.default_device(jax.devices("cpu")[0]):
            with open(latest_checkpoint, "rb") as f:
                restored = serialization.from_bytes(None, f.read())

        system = EfficientNeRFSystem(config, rng_key)

        system.state_coarse = train_state.TrainState(
            step=restored["system"]["state_coarse"]["step"],
            apply_fn=system.nerf_coarse.apply,
            params=restored["system"]["state_coarse"]["params"],
            tx=system.state_coarse.tx,
            opt_state=from_state_dict(system.state_coarse.opt_state, restored["system"]["state_coarse"]["opt_state"]),
            # opt_state=restored["system"]["state_coarse"]["opt_state"],
        )
        system.state_fine = train_state.TrainState(
            step=restored["system"]["state_fine"]["step"],
            apply_fn=system.nerf_fine.apply,
            params=restored["system"]["state_fine"]["params"],
            tx=system.state_fine.tx,
            opt_state=from_state_dict(system.state_fine.opt_state, restored["system"]["state_fine"]["opt_state"]),
            # opt_state=restored["system"]["state_fine"]["opt_state"],
        )

        # Ensure xyz_min/max are JAX arrays with shape (1, 3)
        xyz_min = jnp.array(restored["system"]["nerf_tree_base"]["xyz_min"], dtype=jnp.float32)
        xyz_max = jnp.array(restored["system"]["nerf_tree_base"]["xyz_max"], dtype=jnp.float32)
        # Reshape to (1, 3) if necessary
        if xyz_min.ndim == 2 and xyz_min.shape[0] == 1:
            xyz_min = xyz_min
        elif xyz_min.ndim == 1:
            xyz_min = xyz_min.reshape(1, 3)
        else:
            raise ValueError(f"Unexpected xyz_min shape: {xyz_min.shape}")
        if xyz_max.ndim == 2 and xyz_max.shape[0] == 1:
            xyz_max = xyz_max
        elif xyz_max.ndim == 1:
            xyz_max = xyz_max.reshape(1, 3)
        else:
            raise ValueError(f"Unexpected xyz_max shape: {xyz_max.shape}")

        system.nerf_tree_base = NerfTree(
            xyz_min=xyz_min,
            xyz_max=xyz_max,
            grid_coarse=restored["system"]["nerf_tree_base"]["grid_coarse"],
            grid_fine=restored["system"]["nerf_tree_base"]["grid_fine"],
            deg=restored["system"]["nerf_tree_base"]["deg"],
            sigma_init=restored["system"]["nerf_tree_base"]["sigma_init"],
            sigma_default=restored["system"]["nerf_tree_base"]["sigma_default"],
        )
        system.nerf_tree_base.sigma_voxels_coarse = (
            jnp.array(restored["system"]["nerf_tree_base"]["sigma_voxels_coarse"])
            if restored["system"]["nerf_tree_base"]["sigma_voxels_coarse"] is not None
            else None
        )
        system.nerf_tree_base.index_voxels_coarse = (
            jnp.array(restored["system"]["nerf_tree_base"]["index_voxels_coarse"])
            if restored["system"]["nerf_tree_base"]["index_voxels_coarse"] is not None
            else None
        )
        system.nerf_tree_base.voxels_fine = (
            jnp.array(restored["system"]["nerf_tree_base"]["voxels_fine"])
            if restored["system"]["nerf_tree_base"]["voxels_fine"] is not None
            else None
        )

        system.global_step = restored["system"]["global_step"]
        system.current_epoch = restored["system"]["current_epoch"]
        system.in_channels_xyz = restored["system"]["in_channels_xyz"]
        system.in_channels_dir = restored["system"]["in_channels_dir"]
        system.deg = restored["system"]["deg"]
        system.dim_sh = restored["system"]["dim_sh"]
        system.sigma_init = restored["system"]["sigma_init"]
        system.sigma_default = restored["system"]["sigma_default"]
        system.near = restored["system"]["near"]
        system.far = restored["system"]["far"]

        system.prepare_data()

        return system, restored["global_step"], restored["rng_key"]

    except Exception as e:
        print(f"Failed to load checkpoint {latest_checkpoint}: {e}")
        return None


def main(config: NerfConfig):
    with jax.default_device(jax.devices("cpu")[0]):
        rng_key = random.PRNGKey(0)

    if config.training.log_on_wandb:
        wandb.init(project=config.project_name, config=config.__dict__, name=config.experiment_name)

    try:

        checkpoint_dir = os.path.join(config.checkpoint.save_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        with jax.default_device(jax.devices("cpu")[0]):
            if config.training.resume:
                checkpoint_data = load_latest_checkpoint(checkpoint_dir, config, rng_key)
                if checkpoint_data:
                    system, start_step, rng_key = checkpoint_data
                    print(f"Resuming from checkpoint at step {start_step}")
                    val_iter = iter(system.val_dataloader)
                    system.validation_step_outputs = []
                    for batch_idx in range(system.val_steps_per_epoch):
                        print(f"Validation batch {batch_idx} of {system.val_steps_per_epoch}")
                        batch = next(val_iter)
                        rng_key, subkey = random.split(rng_key)
                        system.validation_step(batch, batch_idx, start_step, subkey)
                    system.on_validation_epoch_end()
                else:
                    print("Resume requested but no checkpoint found, starting from scratch")
                    system = EfficientNeRFSystem(config, rng_key)
                    system.prepare_data()
                    start_step = 0
            else:
                system = EfficientNeRFSystem(config, rng_key)
                system.prepare_data()
                start_step = 0

            train_iter = iter(system.train_dataloader)
            total_steps = config.training.epochs * system.train_steps_per_epoch

            for epoch in range(config.training.epochs):
                system.current_epoch = epoch
                for step in range(system.train_steps_per_epoch):
                    global_step = epoch * system.train_steps_per_epoch + step

                    if global_step < start_step:
                        batch = next(train_iter)
                        continue

                    batch = next(train_iter)
                    rng_key, subkey = random.split(rng_key)

                    system = train_step(system, batch, global_step, subkey)

                    if (global_step + 1) % config.checkpoint.save_every == 0:
                        save_checkpoint(system, global_step, rng_key, checkpoint_dir, config.checkpoint.keep_last)
                        print(f"Saved checkpoint at step {global_step}")

                    if (global_step + 1) % config.training.log_every == 0:
                        val_iter = iter(system.val_dataloader)
                        system.validation_step_outputs = []
                        for batch_idx in range(system.val_steps_per_epoch):
                            print(f"Validation batch {batch_idx} of {system.val_steps_per_epoch}")
                            batch = next(val_iter)
                            rng_key, subkey = random.split(rng_key)
                            system.validation_step(batch, batch_idx, global_step, subkey)
                        system.on_validation_epoch_end()

            val_iter = iter(system.val_dataloader)
            system.validation_step_outputs = []
            for batch_idx in range(system.val_steps_per_epoch):
                batch = next(val_iter)
                rng_key, subkey = random.split(rng_key)
                system.validation_step(batch, batch_idx, total_steps, subkey)
            system.on_validation_epoch_end()
            save_checkpoint(system, total_steps, rng_key, checkpoint_dir, config.checkpoint.keep_last)
            print(f"Saved final checkpoint at step {total_steps}")

    finally:
        if config.training.log_on_wandb:
            wandb.finish()


def render(config: NerfConfig):
    with jax.default_device(jax.devices("cpu")[0]):
        rng_key = random.PRNGKey(0)

        render_output_dir = os.path.join(config.checkpoint.render_dir, "renders")
        os.makedirs(render_output_dir, exist_ok=True)

        # Load latest checkpoint
        checkpoint_dir = os.path.join(config.checkpoint.save_dir, "checkpoints")
        checkpoint_data = load_latest_checkpoint(checkpoint_dir, config, rng_key)

        if checkpoint_data:
            system, loaded_step, rng_key = checkpoint_data
            print(f"Loaded checkpoint from step {loaded_step}")

            val_iter = iter(system.val_dataloader)
            system.validation_step_outputs = []

            for batch_idx in range(system.val_steps_per_epoch):
                print(f"Rendering batch {batch_idx+1}/{system.val_steps_per_epoch}")
                batch = next(val_iter)
                rng_key, subkey = random.split(rng_key)
                system.validation_step(batch, batch_idx, loaded_step, subkey)

            save_rendered_images(system, render_output_dir)

            print(f"Rendering complete. Images saved to {render_output_dir}")
        else:
            print("Error: No checkpoint found to render from")


def save_rendered_images(system, output_dir):
    with jax.default_device(jax.devices("cpu")[0]):
        if not system.validation_step_outputs:
            print("No validation data to render")
            return

        img_dict = {}
        for output in system.validation_step_outputs:
            if "img" in output:
                idx = output["idx"]
                if idx not in img_dict:
                    img_dict[idx] = {"pred": [], "batch_indices": []}
                img_dict[idx]["pred"].append(output["img"])
                img_dict[idx]["batch_indices"].append(output["batch_idx"])

        W, H = system.config.data_loading.img_wh
        for idx in img_dict:
            sorted_indices = jnp.argsort(jnp.array(img_dict[idx]["batch_indices"]))
            pred_batches = [img_dict[idx]["pred"][i] for i in sorted_indices]

            pred_rays = jnp.concatenate(pred_batches, axis=0)[: W * H]
            img = pred_rays.reshape(H, W, 3)

            img_np = np.array(img)
            img_np = np.clip(img_np, 0.0, 1.0)

            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            img_pil.save(os.path.join(output_dir, f"render_{idx:03d}.png"))

        system.validation_step_outputs = []


if __name__ == "__main__":
    default_config = Path(__file__).parent / "test_nerf.yaml"
    args = parse_cli_options(default_config=default_config)
    config: NerfConfig = generate_config(NerfConfig, args.config)
    if config.training.render:
        render(config)
    else:
        main(config)
