# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
from jax import random

from blacksmith.models.jax.nerf.nerf import inference
from blacksmith.models.jax.nerf.nerftree import (  # Import the standalone functions
    query_coarse_out,
    update_coarse_out,
    update_fine_out,
)


def render_rays(
    config,
    rays: jnp.ndarray,
    embedding_xyz,
    tree_data,
    near: float,
    far: float,
    global_step: int,
    model_coarse,
    params_coarse,
    model_fine,
    params_fine,
    xyz_coarse,
    deltas_coarse,
    xyz_fine,
    deltas_fine,
    rays_origin,
    rays_direction,
) -> dict:

    coarse_results = calculate_coarse_rendering(
        config=config,
        model_coarse=model_coarse,
        params_coarse=params_coarse,
        tree_data=tree_data,
        embedding_xyz=embedding_xyz,
        rays_directions=rays_direction,
        xyz_coarse=xyz_coarse,
        deltas_coarse=deltas_coarse,
        global_step=global_step,
        callee="coarse",
    )

    weights_coarse = coarse_results["weights_coarse"]

    fine_results = calculate_fine_rendering(
        config=config,
        model_fine=model_fine,
        params_fine=params_fine,
        tree_data=tree_data,
        embedding_xyz=embedding_xyz,
        rays_directions=rays_direction,
        xyz_fine=xyz_fine,
        deltas_fine=deltas_fine,
        weights_coarse=weights_coarse,
        callee="fine",
    )

    combined_results = {**coarse_results, **fine_results}
    return combined_results


def calculate_coarse_rendering(
    config,
    model_coarse,
    params_coarse,
    tree_data,
    embedding_xyz,
    rays_directions,
    xyz_coarse,
    deltas_coarse,
    global_step,
    callee,
):
    result = {}
    num_rays = rays_directions.shape[0]
    samples_per_ray = config.model.coarse.samples
    chunk_size = config.data_loading.batch_size * samples_per_ray

    sigmas = query_coarse_out(xyz_coarse.reshape(-1, 3), tree_data, type="sigma").reshape(num_rays, samples_per_ray)

    if tree_data["voxels_fine"] is None:
        max_samples = num_rays * samples_per_ray
        if config.model.warmup_step > 0 and global_step <= config.model.warmup_step:
            valid_sample_indices = jnp.nonzero(sigmas >= -1e10, size=max_samples, fill_value=-1)
            valid_sample_indices = jnp.stack([valid_sample_indices[0], valid_sample_indices[1]], axis=-1)
        else:
            valid_sample_indices = jnp.nonzero(sigmas > 0.0, size=max_samples, fill_value=-1)
            valid_sample_indices = jnp.stack([valid_sample_indices[0], valid_sample_indices[1]], axis=-1)

        rgb_values, updated_sigmas, spherical_harmonics, extras = inference(
            model=model_coarse,
            params=params_coarse,
            embedding_xyz=embedding_xyz,
            xyzs=xyz_coarse,
            dirs=rays_directions,
            deltas=deltas_coarse,
            idx_render=valid_sample_indices,
            sigma_default=config.model.sigma_default,
            chunk=chunk_size,
            callee="coarse",
        )

        result["rgb_coarse"] = rgb_values
        result["sigma_coarse"] = updated_sigmas
        result["sh_coarse"] = spherical_harmonics
        result["rgb_valid"] = valid_sample_indices
        result["nn_backward_coarse"] = extras["nn_backward"]  # Store the backward function
        result.update(extras["intermediates"])  # Include intermediates

        valid_mask = valid_sample_indices[:, 0] >= 0
        sample_positions = jnp.where(
            valid_mask[:, None], xyz_coarse[valid_sample_indices[:, 0], valid_sample_indices[:, 1]], 0.0
        )
        sample_densities = jnp.where(
            valid_mask, updated_sigmas[valid_sample_indices[:, 0], valid_sample_indices[:, 1]].squeeze(-1), 0.0
        )
        updated_sigma_voxels = update_coarse_out(
            tree_data["sigma_voxels_coarse"], sample_positions, sample_densities, config.model.beta, tree_data
        )
        result["sigma_voxels_coarse"] = updated_sigma_voxels

    sigmas = jnp.expand_dims(sigmas, axis=-1)
    weights, _ = model_coarse.sigma2weights(deltas_coarse, sigmas)
    result["weights_coarse"] = weights

    return result


def calculate_fine_rendering(
    config,
    model_fine,
    params_fine,
    tree_data,
    embedding_xyz,
    rays_directions,
    xyz_fine,
    deltas_fine,
    weights_coarse,
    callee,
):
    result = {}
    num_rays = rays_directions.shape[0]
    fine_samples_per_coarse = config.model.fine.samples
    chunk_size = config.data_loading.batch_size * config.model.coarse.samples

    # Pick top-k weights instead of thresholding
    k = chunk_size // fine_samples_per_coarse
    flat_weights = weights_coarse.reshape(-1)
    sorted_indices = jnp.argsort(flat_weights)[::-1]
    top_k_indices = sorted_indices[:k]

    important_samples = jnp.stack(
        [top_k_indices // weights_coarse.shape[1], top_k_indices % weights_coarse.shape[1]], axis=-1
    )

    expanded_indices = jnp.expand_dims(important_samples, 1).repeat(fine_samples_per_coarse, axis=1)
    fine_indices = expanded_indices.copy()
    fine_indices = fine_indices.at[..., 1].set(
        expanded_indices[..., 1] * fine_samples_per_coarse
        + jnp.arange(fine_samples_per_coarse).reshape(1, fine_samples_per_coarse)
    )
    fine_indices = fine_indices.reshape(-1, 2)

    assert (
        fine_indices.shape[0] <= chunk_size
    ), f"fine_indices size {fine_indices.shape[0]} exceeds chunk_size {chunk_size}"

    # Compute RGB values, densities, and spherical harmonics with params, including nn_backward
    rgb_values, sigma_values, spherical_harmonics, extras = inference(
        model=model_fine,
        params=params_fine,
        embedding_xyz=embedding_xyz,
        xyzs=xyz_fine,
        dirs=rays_directions,
        deltas=deltas_fine,
        idx_render=fine_indices,
        sigma_default=config.model.sigma_default,
        chunk=chunk_size,
        callee="fine",
    )

    # Update fine voxel grid using tree_data and update_fine_out
    if tree_data["voxels_fine"] is not None:
        sample_positions = xyz_fine[fine_indices[:, 0], fine_indices[:, 1]]
        sample_densities = sigma_values[fine_indices[:, 0], fine_indices[:, 1]]
        sample_harmonics = spherical_harmonics[fine_indices[:, 0], fine_indices[:, 1]]
        updated_voxels_fine = update_fine_out(
            tree_data["voxels_fine"], sample_positions, sample_densities, sample_harmonics, tree_data
        )
        result["voxels_fine"] = updated_voxels_fine

    # Store results
    result["rgb_fine"] = rgb_values
    result["sigma_fine"] = sigma_values
    result["sh_fine"] = spherical_harmonics
    result["num_samples_fine"] = jnp.array([fine_indices.shape[0] / num_rays])
    result["nn_backward_fine"] = extras["nn_backward"]  # Store the neural network backward function
    result.update(extras["intermediates"])  # Include intermediates like sigma, sh, weights, alphas

    return result


def generate_ray_samples(rays, num_samples, near, far):
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]
    N_rays = rays_o.shape[0]
    distance = far - near

    # Sample points along the z-axis
    z_vals = jnp.linspace(0, 1, num_samples)
    z_vals = near * (1 - z_vals) + far * z_vals
    z_vals = jnp.expand_dims(z_vals, 0)

    z_vals = jnp.repeat(z_vals, N_rays, axis=0)
    key = random.PRNGKey(0)
    delta_z_vals = random.uniform(key, z_vals.shape) * (distance / num_samples)

    z_vals = z_vals + delta_z_vals
    xyz_sampled = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[:, :, None]

    deltas = z_vals[:, 1:] - z_vals[:, :-1]
    delta_inf = 10_000 * jnp.ones_like(deltas[:, :1])
    deltas = jnp.concatenate([deltas, delta_inf], axis=-1)

    return xyz_sampled, deltas
