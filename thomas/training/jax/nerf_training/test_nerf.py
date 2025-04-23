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
from jax import lax
from PIL import Image

from blender import BlenderDataset, create_dataloader, create_dataloader_val
from nerf_rendering import render_rays, generate_ray_samples
from nerf import Embedding, NeRF
from nerftree import NerfTree
from configs import NerfConfig, load_config
from optimizers import get_optimizer

import wandb

import time
from utils import init_device

from functools import partial

import orbax.checkpoint as ocp
from flax import serialization
from typing import Callable


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
            "root_dir": self.config.data_loading.input_dir,
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
            key, subkey = random.split(rng_key)
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

            # Add sigma_voxels_coarse to results (should be (384, 384, 384), not batch-dependent)
            if sigma_voxels_coarse is not None:
                results["sigma_voxels_coarse"] = sigma_voxels_coarse
            else:
                # Fallback to tree_data if not updated (unlikely, but for safety)
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

    def training_step(self, state, batch, global_step, rng_key):

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
            params = {"nerf_coarse": state.state_coarse.params, "nerf_fine": state.state_fine.params}
            results = self.forward(rays, rays_data, params, tree_data, global_step, rng_key)
            loss = self.loss_fn(results, rays_data["rgbs_chunk"])

            print("Loss: ", loss)

            # Extract neural network backward functions (fix duplicate pop)
            nn_backward_coarse = results.pop("nn_backward_coarse", None)
            nn_backward_fine = results.pop("nn_backward_fine", None)
            idx_render_coarse = results.get("idx_render_coarse")  # Use get to avoid KeyError
            idx_render_fine = results.get("idx_render_fine")

            # time.sleep(1000)

            rgb_valid_idx = results.get("rgb_valid", None)

            # Helper function to recompute rgb from sigma and sh
            def compute_rgb_from_sigma_sh(
                sigma, sh, deltas, rays_d, idx_render, sigma_default, non_minus_one_mask, chunk_size, sigma_key
            ):
                batch_size, sample_size = deltas.shape[:2]  # e.g., (4096, 64)
                real_chunk_size = idx_render.shape[0]  # Number of valid samples (e.g., 262144)

                # Select model based on sigma_key
                model = self.nerf_coarse if "coarse" in sigma_key else self.nerf_fine

                # Expand ray directions to match sample size
                view_dir = jnp.expand_dims(rays_d, 1).repeat(sample_size, axis=1)  # (batch_size, sample_size, 3)
                view_dir_flat = view_dir[idx_render[:, 0], idx_render[:, 1]]  # (real_chunk_size, 3)

                # Pad inputs to chunk size

                sigma = sigma.reshape(-1, 1)
                sh = sh.reshape(-1, 27)

                # sigma_padded = jnp.concatenate([sigma, jnp.zeros((chunk_size - real_chunk_size, 1))], axis=0)
                # sh_padded = jnp.concatenate([sh, jnp.zeros((chunk_size - real_chunk_size, 27))], axis=0)
                # view_dir_padded = jnp.concatenate([view_dir_flat, jnp.zeros((chunk_size - real_chunk_size, 3))], axis=0)

                # Compute RGB from sigma and sh using the selected model's sh2rgb method
                sigma_out, rgb, sh_out = model.sh2rgb(sigma, sh, model.deg, view_dir_flat)

                # Trim padded outputs back to real_chunk_size
                sigma = sigma_out[:real_chunk_size]
                rgb = rgb[:real_chunk_size]
                sh = sh_out[:real_chunk_size]

                # Initialize output arrays with defaults
                out_rgb = jnp.ones((batch_size, sample_size, 3))  # Default white
                out_sigma = jnp.full((batch_size, sample_size, 1), sigma_default)  # Default sigma
                out_sh = jnp.zeros((batch_size, sample_size, 27))  # Default zeros

                # Scatter computed values into output arrays using idx_render
                out_sigma = out_sigma.at[idx_render[:, 0], idx_render[:, 1]].set(sigma)
                out_rgb = out_rgb.at[idx_render[:, 0], idx_render[:, 1]].set(rgb)
                out_sh = out_sh.at[idx_render[:, 0], idx_render[:, 1]].set(sh)

                # Apply mask to sigma
                non_minus_one_mask = jnp.expand_dims(non_minus_one_mask, axis=-1)  # (batch_size, sample_size, 1)
                out_sigma = out_sigma * non_minus_one_mask

                # Volume rendering with the selected model
                weights, _ = model.sigma2weights(deltas, out_sigma, non_minus_one_mask)  # (batch_size, sample_size)
                weights_sum = weights.sum(axis=1)  # (batch_size,)
                rgb_final = jnp.sum(weights[..., None] * out_rgb, axis=-2)  # (batch_size, 3)
                rgb_final = rgb_final + (1 - weights_sum[..., None])  # Add white background, (batch_size, 3)

                return rgb_final

            def loss_from_sigma_sh(sigma_key, sh_key, results, rgbs):
                # print keys in results
                # for key in results.keys():
                #    print(key)
                # Select coarse or fine data
                deltas = rays_data["deltas_coarse"] if sigma_key == "sigma_coarse" else rays_data["deltas_fine"]
                rays_d = rays_data["rays_d"]
                idx_render = results.get("idx_render_coarse" if sigma_key == "sigma_coarse" else "idx_render_fine")

                # Access immediate sigma and sh from intermediates
                # intermediates = results["intermediates"]
                sigma_immediate_key = (
                    "sigma_coarse_immediate" if sigma_key == "sigma_coarse" else "sigma_fine_immediate"
                )
                sh_immediate_key = "sh_coarse_immediate" if sigma_key == "sigma_coarse" else "sh_fine_immediate"
                sigma = results[sigma_immediate_key]
                sh = results[sh_immediate_key]

                # Compute non_minus_one_mask
                batch_size, sample_size = deltas.shape[:2]
                non_minus_one_mask = jnp.ones((batch_size, sample_size))
                non_one_idx = idx_render * (idx_render == -1)
                non_minus_one_mask = non_minus_one_mask.at[non_one_idx].set(0)

                # Compute rgb using immediate sigma and sh
                rgb = compute_rgb_from_sigma_sh(
                    sigma=sigma,
                    sh=sh,
                    deltas=deltas,
                    rays_d=rays_d,
                    idx_render=idx_render,
                    sigma_default=config.model.sigma_default,  # Assuming this is in scope
                    non_minus_one_mask=non_minus_one_mask,
                    chunk_size=1024,
                    sigma_key=sigma_key,
                )

                new_results = {**results}
                if sigma_key == "sigma_coarse":
                    new_results["rgb_coarse"] = rgb
                else:
                    new_results["rgb_fine"] = rgb
                return self.loss_fn(new_results, rgbs)  # Assuming loss_fn is in scope

            # Compute gradients w.r.t. sigma and sh with rendering included
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

            # if idx_render_coarse is not None:
            #    # Index with explicit column selection to keep shape
            #    sigma_grad_coarse = sigma_grad_coarse_full[idx_render_coarse[:, 0], idx_render_coarse[:, 1], :]
            # else:
            #    sigma_grad_coarse = sigma_grad_coarse_full.reshape(-1, 1)

            # Apply to all gradients
            if idx_render_coarse is not None:
                sigma_grad_coarse = sigma_grad_coarse_full[idx_render_coarse[:, 0], idx_render_coarse[:, 1]]
                sh_grad_coarse = sh_grad_coarse_full[idx_render_coarse[:, 0], idx_render_coarse[:, 1]]
                sigma_grad_coarse = sigma_grad_coarse_full.reshape(-1, 1)
                sh_grad_coarse = sh_grad_coarse_full.reshape(-1, 27)
                # print('s ta ')
                # print(sigma_grad_coarse.shape)
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

            # print('SIGMA_GRAD_FINE SHAPE:', sigma_grad_fine.shape)
            # print('SH_GRAD_FINE SHAPE:', sh_grad_fine.shape)

            # expected_size = 262144
            # if sigma_grad_fine.shape[0] < expected_size:
            #    sigma_grad_fine = jnp.pad(sigma_grad_fine, ((0, expected_size - sigma_grad_fine.shape[0]), (0, 0)), mode='constant')
            #    sh_grad_fine = jnp.pad(sh_grad_fine, ((0, expected_size - sh_grad_fine.shape[0]), (0, 0)), mode='constant')
            # fa
            # full_sigma_grad_fine = jnp.zeros((batch_size, config.model.coarse.samples * config.model.fine.samples, 1))
            # full_sh_grad_fine = jnp.zeros((batch_size, config.model.coarse.samples * config.model.fine.samples, 27))

            # print("full_sigma_grad_fine shape:", full_sigma_grad_fine.shape)
            # print("full_sh_grad_fine shape:", full_sh_grad_fine.shape)

            # Insert the computed gradients at the correct positions
            # full_sigma_grad_fine = full_sigma_grad_fine.at[idx_render_fine[:, 0], idx_render_fine[:, 1]].set(sigma_grad_fine)
            # full_sh_grad_fine = full_sh_grad_fine.at[idx_render_fine[:, 0], idx_render_fine[:, 1]].set(sh_grad_fine)

            # print("full_sigma_grad_fine shape after set:", full_sigma_grad_fine.shape)
            # print("full_sh_grad_fine shape after set:", full_sh_grad_fine.shape)

            # Then reshape if needed
            # sigma_grad_fine = full_sigma_grad_fine.reshape(-1, 1)
            # sh_grad_fine = full_sh_grad_fine.reshape(-1, 27)

            # print("sigma_grad_fine shape after reshape:", sigma_grad_fine.shape)
            # print("sh_grad_fine shape after reshape:", sh_grad_fine.shape)

            # Compute neural network gradients on "tt" device
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
            # print("coarse_grads sample:", coarse_grads)  # Adjust based on your grad structure

            # Update state
            if "sigma_voxels_coarse" in results:
                self.nerf_tree_base.sigma_voxels_coarse = results["sigma_voxels_coarse"]
            if "index_voxels_coarse" in results:
                self.nerf_tree_base.index_voxels_coarse = results["index_voxels_coarse"]
            if "voxels_fine" in results:
                self.nerf_tree_base.voxels_fine = results["voxels_fine"]

            state_coarse = state.state_coarse.apply_gradients(grads=grads["nerf_coarse"])
            state_fine = state.state_fine.apply_gradients(grads=grads["nerf_fine"])

            new_state = SystemState(state_coarse, state_fine)

            # self.state_coarse.params = new_state.state_coarse.params
            # self.state_fine.params = new_state.state_fine.params

        return new_state, loss, results, grads, rays_chunk, rgbs_chunk

    def validation_step(self, state, batch, batch_idx, global_step, key):
        with jax.default_device(jax.devices("cpu")[0]):
            rays, rgbs = batch["rays"], batch["rgbs"]
            rays = rays.squeeze()  # (4096, 6)
            rgbs = rgbs.squeeze()  # (4096, 3)

            params = {"nerf_coarse": state.state_coarse.params, "nerf_fine": state.state_fine.params}
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

                # Just reshape to verify it works, no saving
                img = pred_rays.reshape(H, W, 3)
                img_gt = gt_rays.reshape(H, W, 3)

                # Log only the first image (idx=0) to Wandb
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
            # print(self.nerf_tree_base.voxels_fine)
            voxels_fine_np = np.array(self.nerf_tree_base.voxels_fine)

            # Create histograms or summary statistics since these might be large arrays
            if config.training.log_on_wandb:
                wandb_log_dict = {
                    "val/loss": float(mean_loss),
                    "val/psnr": float(mean_psnr),
                    # "val/num_voxels_coarse": int(num_voxels_coarse),
                    # "val/sigma_voxels_coarse_hist": wandb.Histogram(sigma_voxels_coarse_np.flatten()),
                    # "val/index_voxels_coarse_hist": wandb.Histogram(index_voxels_coarse_np.flatten()),
                    # "val/voxels_fine_hist": wandb.Histogram(voxels_fine_np.flatten()),
                    # "val/sigma_voxels_coarse_mean": float(np.mean(sigma_voxels_coarse_np)),
                    # "val/voxels_fine_mean": float(np.mean(voxels_fine_np))
                }

            # Log all metrics to Wandb
            if config.training.log_on_wandb:
                wandb.log(wandb_log_dict, step=self.global_step)

            print(
                f"Step {self.global_step} - val/loss: {mean_loss:.4f}, val/psnr: {mean_psnr:.4f}, "
                f"num_voxels_coarse: {num_voxels_coarse}, "
                f"sigma_voxels_mean: {float(np.mean(sigma_voxels_coarse_np)):.4f}"
            )
            self.validation_step_outputs = []


def train_step(system, state, batch, step, rng_key):
    with jax.default_device(jax.devices("cpu")[0]):
        state, loss, results, grads, rays_chunk, rgbs_chunk = system.training_step(state, batch, step, rng_key)
        system.global_step = step
        print(f"Step {step}, Loss: {loss}")
        log_dict = {"train/loss": float(loss)}
        # print(weights_coarse.shape)
        # print(weights_fine.shape)

        # Helper function to recursively log weights
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
            log_weights(state.state_coarse.params, "coarse")
            log_weights(state.state_fine.params, "fine")

        # Compute and log training PSNR
        rgbs = batch["rgbs"]  # Ground truth RGB values
        pred_rgb = results.get("rgb_fine", results["rgb_coarse"])  # Use fine if available, else coarse
        mse = jnp.mean((pred_rgb - rgbs_chunk) ** 2)  # Mean Squared Error
        psnr = -10.0 * jnp.log10(mse)  # PSNR formula
        log_dict["train/psnr"] = float(psnr)

        # Flatten and log gradients (unchanged)
        coarse_grads_flat = jax.tree_util.tree_leaves(grads["nerf_coarse"])
        fine_grads_flat = jax.tree_util.tree_leaves(grads["nerf_fine"])
        # coarse_grads_mean = jnp.mean(jnp.concatenate([jnp.ravel(g) for g in coarse_grads_flat]))
        # fine_grads_mean = jnp.mean(jnp.concatenate([jnp.ravel(g) for g in fine_grads_flat]))
        # log_dict["train/grads_coarse_mean"] = float(coarse_grads_mean)
        # log_dict["train/grads_fine_mean"] = float(fine_grads_mean)
        log_dict["train/grads_coarse_hist"] = wandb.Histogram(
            np_histogram=np.histogram(np.concatenate([g.flatten() for g in coarse_grads_flat]), bins=50)
        )
        log_dict["train/grads_fine_hist"] = wandb.Histogram(
            np_histogram=np.histogram(np.concatenate([g.flatten() for g in fine_grads_flat]), bins=50)
        )

        if config.training.log_on_wandb:
            wandb.log(log_dict, step=step)
        return state


class SystemState:
    def __init__(self, state_coarse, state_fine):
        self.state_coarse = state_coarse
        self.state_fine = state_fine


import orbax.checkpoint as ocp
import shutil

import flax.serialization as flax_serialization
from flax.serialization import from_state_dict, to_state_dict
import jax
import jax.numpy as jnp
import os
import shutil


def save_checkpoint(state: SystemState, global_step: int, rng_key: jnp.ndarray, checkpoint_dir: str, keep_last_n: int):
    """Save SystemState, global_step, and rng_key, keeping only the last three checkpoints."""
    checkpoint_data = {
        "state_coarse": state.state_coarse,
        "state_fine": state.state_fine,
        "global_step": global_step,
        "rng_key": rng_key,
    }
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{global_step}.flax")
    with jax.default_device(jax.devices("cpu")[0]):
        serialized_data = flax_serialization.to_bytes(checkpoint_data)
        with open(checkpoint_path, "wb") as f:
            f.write(serialized_data)

    # Keep only the last three checkpoints
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".flax")]
    if len(checkpoint_files) > keep_last_n:
        steps = sorted([int(f.split("_")[1].split(".")[0]) for f in checkpoint_files])
        oldest_step = steps[0]
        oldest_checkpoint = os.path.join(checkpoint_dir, f"checkpoint_{oldest_step}.flax")
        os.remove(oldest_checkpoint)
        print(f"Deleted oldest checkpoint: checkpoint_{oldest_step}.flax")


def load_latest_checkpoint(system, checkpoint_dir: str) -> tuple[SystemState, int, jnp.ndarray] | None:
    """Load the latest checkpoint, return (SystemState, global_step, rng_key) or None."""
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".flax")]
    if not checkpoint_files:
        return None

    steps = [int(f.split("_")[1].split(".")[0]) for f in checkpoint_files]
    latest_step = max(steps)
    latest_checkpoint = os.path.join(checkpoint_dir, f"checkpoint_{latest_step}.flax")

    try:
        with jax.default_device(jax.devices("cpu")[0]):
            with open(latest_checkpoint, "rb") as f:
                restored = flax_serialization.from_bytes(None, f.read())

        # Convert restored dictionaries to TrainState
        state_coarse = train_state.TrainState(
            step=restored["state_coarse"]["step"],
            apply_fn=system.nerf_coarse.apply,
            params=restored["state_coarse"]["params"],
            tx=system.state_coarse.tx,
            opt_state=from_state_dict(system.state_coarse.opt_state, restored["state_coarse"]["opt_state"]),
        )
        state_fine = train_state.TrainState(
            step=restored["state_fine"]["step"],
            apply_fn=system.nerf_fine.apply,
            params=restored["state_fine"]["params"],
            tx=system.state_fine.tx,
            opt_state=from_state_dict(system.state_fine.opt_state, restored["state_fine"]["opt_state"]),
        )

        state = SystemState(
            state_coarse=state_coarse,
            state_fine=state_fine,
        )
        return state, restored["global_step"], restored["rng_key"]
    except Exception as e:
        print(f"Failed to load checkpoint {latest_checkpoint}: {e}")
        return None


def main(config: NerfConfig):

    with jax.default_device(jax.devices("cpu")[0]):
        rng_key = random.PRNGKey(0)

    if config.training.log_on_wandb:
        wandb.init(project="jax-nerf", config=config.__dict__, name="EfficientNeRF_400_fwd_bwd_device_lr8e-4")

    checkpoint_dir = os.path.join(config.checkpoint.save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    with jax.default_device(jax.devices("cpu")[0]):
        system = EfficientNeRFSystem(config, rng_key)
        system.prepare_data()

        # Initialize state and step
        if config.training.resume:
            checkpoint_data = load_latest_checkpoint(system, checkpoint_dir)
            if checkpoint_data:
                state, start_step, rng_key = checkpoint_data
                # Run validation to log metrics
                val_iter = iter(system.val_dataloader)
                system.validation_step_outputs = []
                for batch_idx in range(system.val_steps_per_epoch):
                    print(f"Validation batch {batch_idx} of {system.val_steps_per_epoch}")
                    batch = next(val_iter)
                    rng_key, subkey = random.split(rng_key)
                    system.validation_step(state, batch, batch_idx, start_step, subkey)
                system.on_validation_epoch_end()
            else:
                print("Resume requested but no checkpoint found, starting from scratch")
                state = SystemState(system.state_coarse, system.state_fine)
                start_step = 0
        else:
            state = SystemState(system.state_coarse, system.state_fine)
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

                state = train_step(system, state, batch, global_step, subkey)

                if global_step % config.checkpoint.save_every == 0:
                    save_checkpoint(state, global_step, rng_key, checkpoint_dir, config.checkpoint.keep_last)
                    print(f"Saved checkpoint at step {global_step}")

                if global_step % config.training.log_every == 499:
                    val_iter = iter(system.val_dataloader)
                    system.validation_step_outputs = []
                    for batch_idx in range(system.val_steps_per_epoch):
                        print(f"Validation batch {batch_idx} of {system.val_steps_per_epoch}")
                        batch = next(val_iter)
                        rng_key, subkey = random.split(rng_key)
                        system.validation_step(state, batch, batch_idx, global_step, subkey)
                    system.on_validation_epoch_end()

        # Final validation and checkpoint
        val_iter = iter(system.val_dataloader)
        system.validation_step_outputs = []
        for batch_idx in range(system.val_steps_per_epoch):
            batch = next(val_iter)
            rng_key, subkey = random.split(rng_key)
            system.validation_step(state, batch, batch_idx, total_steps, subkey)
        system.on_validation_epoch_end()

        save_checkpoint(state, total_steps, rng_key, checkpoint_dir)
        print(f"Saved final checkpoint at step {total_steps}")


if __name__ == "__main__":

    init_device()

    config_file_path = os.path.join(os.path.dirname(__file__), "test_nerf.yaml")
    config = load_config(config_file_path)
    # print(config.checkpoint.save_dir)
    main(config)
