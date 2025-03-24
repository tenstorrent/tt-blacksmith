# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
from jax import random
from nerf import inference  # Importing from nerf.py
from nerftree import (  # Import the standalone functions
    query_coarse_out,
    update_coarse_out,
    calc_index_coarse_out,
    create_voxels_fine_out,
    calc_index_fine_out,
    update_fine_out,
    query_fine_out,
)
from nerftree import NerfTree


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
) -> dict:
    n_samples_coarse = config.model.coarse.samples
    n_samples_fine = n_samples_coarse * config.model.fine.samples

    rays_origin, rays_direction = rays[:, 0:3], rays[:, 3:6]
    xyz_coarse, deltas_coarse = generate_ray_samples(rays, n_samples_coarse, near, far)
    xyz_fine, deltas_fine = generate_ray_samples(rays, n_samples_fine, near, far)

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
    )

    weights_coarse = coarse_results["weights_coarse"]

    fine_results = calculate_fine_rendering(  # Assuming this exists
        config=config,
        model_fine=model_fine,
        params_fine=params_fine,
        tree_data=tree_data,
        embedding_xyz=embedding_xyz,
        rays_directions=rays_direction,
        xyz_fine=xyz_fine,
        deltas_fine=deltas_fine,
        weights_coarse=weights_coarse,
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
):
    result = {}
    num_rays = rays_directions.shape[0]
    samples_per_ray = config.model.coarse.samples
    chunk_size = config.data_loading.batch_size * samples_per_ray

    # Use standalone query_coarse_out instead of NerfTree.query_coarse
    sigmas = query_coarse_out(xyz_coarse.reshape(-1, 3), tree_data, type="sigma").reshape(num_rays, samples_per_ray)

    # import pdb; pdb.set_trace()

    if tree_data["voxels_fine"] is None:
        key = random.PRNGKey(0)
        uniform_mask = random.uniform(key, sigmas[:, 0].shape) < config.model.uniform_ratio
        sigmas = jnp.where(uniform_mask[:, None], config.model.sigma_init, sigmas)

        max_samples = num_rays * samples_per_ray
        if config.model.warmup_step > 0 and global_step <= config.model.warmup_step:
            valid_sample_indices = jnp.nonzero(sigmas >= -1e10, size=max_samples, fill_value=-1)
            valid_sample_indices = jnp.stack([valid_sample_indices[0], valid_sample_indices[1]], axis=-1)
        else:
            valid_sample_indices = jnp.nonzero(sigmas > 0.0, size=max_samples, fill_value=-1)
            valid_sample_indices = jnp.stack([valid_sample_indices[0], valid_sample_indices[1]], axis=-1)

        rgb_values, updated_sigmas, spherical_harmonics = inference(
            model=model_coarse,
            params=params_coarse,
            embedding_xyz=embedding_xyz,
            xyzs=xyz_coarse,
            dirs=rays_directions,
            deltas=deltas_coarse,
            idx_render=valid_sample_indices,
            sigma_default=config.model.sigma_default,
            chunk=chunk_size,
        )

        result["rgb_coarse"] = rgb_values
        result["sigma_coarse"] = updated_sigmas
        result["rgb_valid"] = valid_sample_indices

        valid_mask = valid_sample_indices[:, 0] >= 0
        sample_positions = jnp.where(
            valid_mask[:, None], xyz_coarse[valid_sample_indices[:, 0], valid_sample_indices[:, 1]], 0.0
        )
        sample_densities = jnp.where(
            valid_mask, updated_sigmas[valid_sample_indices[:, 0], valid_sample_indices[:, 1]].squeeze(-1), 0.0
        )
        # Use standalone update_coarse_out instead of NerfTree.update_coarse
        updated_sigma_voxels = update_coarse_out(
            tree_data["sigma_voxels_coarse"], sample_positions, sample_densities, config.model.beta, tree_data
        )
        result["sigma_voxels_coarse"] = updated_sigma_voxels

    sigmas = jnp.expand_dims(sigmas, axis=-1)
    weights, _ = model_coarse.sigma2weights(deltas_coarse, sigmas)
    result["weights_coarse"] = weights

    return result


import jax
import jax.numpy as jnp
from jax import random
from nerf import inference
from nerftree import update_fine_out


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
):
    """
    Calculates the fine rendering of NeRF using importance sampling based on top-k coarse weights.
    """
    result = {}
    num_rays = rays_directions.shape[0]
    fine_samples_per_coarse = config.model.fine.samples
    chunk_size = config.data_loading.batch_size * config.model.coarse.samples

    # Pick top-k weights instead of thresholding
    k = chunk_size // fine_samples_per_coarse  # Ensure we don’t exceed chunk_size after expansion
    # import pdb; pdb.set_trace()
    # print(k)
    flat_weights = weights_coarse.reshape(-1)  # Flatten to 1D for top-k
    top_k_indices = jnp.argpartition(flat_weights, -k)[-k:]  # Get indices of top k weights
    # import pdb; pdb.set_trace()
    important_samples = jnp.stack(
        [top_k_indices // weights_coarse.shape[1], top_k_indices % weights_coarse.shape[1]], axis=-1
    )

    # Expand indices for fine sampling
    expanded_indices = jnp.expand_dims(important_samples, 1).repeat(fine_samples_per_coarse, axis=1)
    fine_indices = expanded_indices.copy()
    fine_indices = fine_indices.at[..., 1].set(
        expanded_indices[..., 1] * fine_samples_per_coarse
        + jnp.arange(fine_samples_per_coarse).reshape(1, fine_samples_per_coarse)
    )
    fine_indices = fine_indices.reshape(-1, 2)

    # No need to limit samples since top-k already ensures static size <= chunk_size
    assert (
        fine_indices.shape[0] <= chunk_size
    ), f"fine_indices size {fine_indices.shape[0]} exceeds chunk_size {chunk_size}"

    # Compute RGB values and densities with params
    rgb_values, sigma_values, spherical_harmonics = inference(
        model=model_fine,
        params=params_fine,
        embedding_xyz=embedding_xyz,
        xyzs=xyz_fine,
        dirs=rays_directions,
        deltas=deltas_fine,
        idx_render=fine_indices,
        sigma_default=config.model.sigma_default,
        chunk=chunk_size,
    )

    # Update fine voxel grid using tree_data and update_fine_out
    if tree_data["voxels_fine"] is not None:
        sample_positions = xyz_fine[fine_indices[:, 0], fine_indices[:, 1]]
        sample_densities = sigma_values[fine_indices[:, 0], fine_indices[:, 1]]
        sample_harmonics = spherical_harmonics[fine_indices[:, 0], fine_indices[:, 1]]
        updated_voxels_fine = update_fine_out(
            tree_data["voxels_fine"], sample_positions, sample_densities, sample_harmonics, tree_data
        )
        result["voxels_fine"] = updated_voxels_fine  # Return the updated voxels

    result["rgb_fine"] = rgb_values
    result["num_samples_fine"] = jnp.array([fine_indices.shape[0] / num_rays])

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
    key = random.PRNGKey(0)  # Fixed seed for reproducibility
    delta_z_vals = random.uniform(key, z_vals.shape) * (distance / num_samples)
    z_vals = z_vals + delta_z_vals
    xyz_sampled = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[:, :, None]

    # Compute deltas
    deltas = z_vals[:, 1:] - z_vals[:, :-1]
    delta_inf = 10_000 * jnp.ones_like(deltas[:, :1])
    deltas = jnp.concatenate([deltas, delta_inf], axis=-1)

    return xyz_sampled, deltas


# Dummy Config class for testing (replace with actual config in practice)
class Config:
    class Model:
        class coarse:
            samples = 4

        class fine:
            samples = 2

        uniform_ratio = 0.1
        warmup_step = 100
        sigma_init = 0.0
        sigma_default = -1.0
        beta = 0.1
        weight_threshold = 0.01

    class DataLoading:
        batch_size = 2

    model = Model()
    data_loading = DataLoading()


if __name__ == "__main__":
    # Test setup
    key = random.PRNGKey(0)
    config = Config()
    rays = jnp.ones((2, 6))  # Dummy rays: [batch_size, 6] (origins + directions)
    near, far = 0.0, 1.0
    global_step = 50  # During warmup phase

    from nerf import NeRF, Embedding

    model_coarse = NeRF()
    model_fine = NeRF()
    embedding_xyz = Embedding(in_channels=3, num_freqs=10)
    variables_coarse = model_coarse.init(key, jnp.ones((4, 63)))
    variables_fine = model_fine.init(key, jnp.ones((4, 63)))
    params_coarse = variables_coarse["params"]
    params_fine = variables_fine["params"]

    nerf_tree = NerfTree(
        xyz_min=[[-1, -1, -1]],
        xyz_max=[[1, 1, 1]],
        grid_coarse=4,
        grid_fine=2,
        deg=2,
        sigma_init=config.model.sigma_init,
        sigma_default=config.model.sigma_default,
    )

    # Run rendering
    results = render_rays(
        config=config,
        rays=rays,
        embedding_xyz=embedding_xyz,
        nerf_tree=nerf_tree,
        near=near,
        far=far,
        global_step=global_step,
        model_coarse=model_coarse,
        params_coarse=params_coarse,  # Added
        model_fine=model_fine,
        params_fine=params_fine,  # Added
    )
    print("Coarse RGB shape:", results["rgb_coarse"].shape)
    print("Fine RGB shape:", results["rgb_fine"].shape)
