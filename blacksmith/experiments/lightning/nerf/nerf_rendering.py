# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch

from blacksmith.models.torch.nerf import inference


def render_rays(
    config,
    rays: torch.Tensor,
    embedding_xyz,
    nerf_tree,
    near: float,
    far: float,
    global_step: int,
    model_coarse,
    model_fine,
) -> dict:
    """
    Render a batch of rays using hierarchical sampling (coarse-to-fine).

    Args:
        rays: Tensor of shape [batch_size, 6] containing ray origins and directions
        embedding_xyz: Positional embedding for 3D coordinates
        model_coarse: Coarse NeRF model
        model_fine: Fine NeRF model

    Returns:
        dict: Combined rendering results from both coarse and fine models
    """
    # Sample configuration
    n_samples_coarse = config.model.coarse.samples
    n_samples_fine = n_samples_coarse * config.model.fine.samples

    # Decompose rays into origins and directions
    rays_origin, rays_direction = rays[:, 0:3], rays[:, 3:6]

    # Generate sample points along rays
    xyz_coarse, deltas_coarse = generate_ray_samples(rays, n_samples_coarse, near, far)

    xyz_fine, deltas_fine = generate_ray_samples(rays, n_samples_fine, near, far)

    # Process coarse samples through the network
    coarse_results = calculate_coarse_rendering(
        config=config,
        model_coarse=model_coarse,
        nerf_tree=nerf_tree,
        embedding_xyz=embedding_xyz,
        rays_directions=rays_direction,
        xyz_coarse=xyz_coarse,
        deltas_coarse=deltas_coarse,
        global_step=global_step,
    )

    # Extract weights for importance sampling in fine model
    weights_coarse = coarse_results["weights_coarse"]

    # Process fine samples through the network, using coarse weights for guidance
    fine_results = calculate_fine_rendering(
        config=config,
        model_fine=model_fine,
        nerf_tree=nerf_tree,
        embedding_xyz=embedding_xyz,
        rays_directions=rays_direction,
        xyz_fine=xyz_fine,
        deltas_fine=deltas_fine,
        weights_coarse=weights_coarse,
    )

    # Combine results from both rendering passes
    combined_results = {**coarse_results, **fine_results}

    return combined_results


def calculate_coarse_rendering(
    config,
    model_coarse,
    nerf_tree,
    embedding_xyz,
    rays_directions,
    xyz_coarse,
    deltas_coarse,
    global_step,
):
    """
    Calculates the coarse rendering of NeRF using the voxel grid and tree structure.

    Args:
        config: Configuration object containing model parameters
        model_coarse: The coarse NeRF model
        nerf_tree: Tree data structure for accelerated rendering
        embedding_xyz: Position embedding for coordinates
        rays_directions: Tensor of ray directions
        xyz_coarse: Tensor of 3D coordinates for coarse samples
        deltas_coarse: Tensor of delta values for coarse samples
        global_step: Current training step

    Returns:
        dict: Results of the coarse rendering pass including colors and weights
    """
    result = {}

    num_rays = rays_directions.shape[0]
    samples_per_ray = config.model.coarse.samples
    chunk_size = config.data_loading.batch_size * samples_per_ray

    # Query density values from voxel grid
    sigmas = nerf_tree.query_coarse(xyz_coarse.reshape(-1, 3), type="sigma").reshape(num_rays, samples_per_ray)

    # Handle tree updates during coarse training phase
    if nerf_tree.voxels_fine is None:
        with torch.no_grad():
            # Apply uniform sampling
            torch.manual_seed(0)
            uniform_mask = torch.rand_like(sigmas[:, 0]) < config.model.uniform_ratio
            sigmas[uniform_mask] = config.model.sigma_init

            if config.model.warmup_step > 0 and global_step <= config.model.warmup_step:
                # During warmup phase, consider all points valid
                valid_sample_indices = torch.nonzero(sigmas >= -1e10).detach()
            else:
                # After warmup, only consider points with positive density
                valid_sample_indices = torch.nonzero(sigmas > 0.0).detach()

        # Compute RGB values and updated densities for valid samples
        rgb_values, updated_sigmas, spherical_harmonics = inference(
            model_coarse,
            embedding_xyz,
            xyz_coarse,
            rays_directions,
            deltas_coarse,
            valid_sample_indices,
            config.model.sigma_default,
            chunk=chunk_size,
        )

        result["rgb_coarse"] = rgb_values
        result["sigma_coarse"] = updated_sigmas

        # Update tree
        sample_positions = xyz_coarse[valid_sample_indices[:, 0], valid_sample_indices[:, 1]]
        sample_densities = (
            updated_sigmas.detach().squeeze().clone()[valid_sample_indices[:, 0], valid_sample_indices[:, 1]]
        )
        nerf_tree.update_coarse(sample_positions, sample_densities, config.model.beta)

    # Calculate weights
    with torch.no_grad():
        weights, _ = model_coarse.sigma2weights(deltas_coarse, sigmas)
        weights = weights.detach()
        result["weights_coarse"] = weights

    return result


def calculate_fine_rendering(
    config,
    model_fine,
    nerf_tree,
    embedding_xyz,
    rays_directions,
    xyz_fine,
    deltas_fine,
    weights_coarse,
):
    """
    Calculates the fine rendering of NeRF using importance sampling based on coarse weights.

    Args:
        config: Configuration object containing model parameters
        model_fine: The fine NeRF model
        nerf_tree: Tree data structure for accelerated rendering
        embedding_xyz: Position embedding for coordinates
        rays_directions: Tensor of ray directions
        xyz_sampled_fine: Tensor of 3D coordinates for fine samples
        deltas_fine: Tensor of delta values for fine samples
        weights_coarse: Weights from coarse rendering for importance sampling

    Returns:
        dict: Results of the fine rendering pass including colors and sample counts
    """
    result = {}

    num_rays = rays_directions.shape[0]
    fine_samples_per_coarse = config.model.fine.samples
    chunk_size = config.data_loading.batch_size * config.model.coarse.samples

    # Find important samples based on coarse weights
    important_samples = torch.nonzero(weights_coarse >= config.model.weight_threshold)

    # Expand indices to generate multiple fine samples per important coarse sample
    expanded_indices = important_samples.unsqueeze(1).expand(-1, fine_samples_per_coarse, -1)  # (B, scale, 2)

    # Create fine sampling indices by offsetting within each coarse sample
    fine_indices = expanded_indices.clone()
    fine_indices[..., 1] = expanded_indices[..., 1] * fine_samples_per_coarse + (
        torch.arange(fine_samples_per_coarse)
    ).reshape(1, fine_samples_per_coarse)
    fine_indices = fine_indices.reshape(-1, 2)

    # Limit the number of samples to avoid memory issues
    if fine_indices.shape[0] > chunk_size:
        torch.manual_seed(0)
        subsample_indices = torch.randperm(fine_indices.shape[0])[:chunk_size]
        fine_indices = fine_indices[subsample_indices]

    # Compute RGB values and densities for selected fine samples
    rgb_values, sigma_values, spherical_harmonics = inference(
        model_fine,
        embedding_xyz,
        xyz_fine,
        rays_directions,
        deltas_fine,
        fine_indices,
        config.model.sigma_default,
        chunk=chunk_size,
    )

    # Update fine voxel grid
    if nerf_tree.voxels_fine is not None:
        with torch.no_grad():
            # Extract positions, densities and SH coefficients for selected samples
            sample_positions = xyz_fine[fine_indices[:, 0], fine_indices[:, 1]]
            sample_densities = sigma_values.detach()[fine_indices[:, 0], fine_indices[:, 1]].unsqueeze(-1)
            sample_harmonics = spherical_harmonics.detach()[fine_indices[:, 0], fine_indices[:, 1]]

            # Update the fine-level voxel grid
            nerf_tree.update_fine(sample_positions, sample_densities, sample_harmonics)

    result["rgb_fine"] = rgb_values
    result["num_samples_fine"] = torch.FloatTensor([fine_indices.shape[0] / num_rays])

    return result


def generate_ray_samples(rays, num_samples, near, far):
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]
    N_rays = rays_o.shape[0]
    distance = far - near

    # Sample points along the z-axis
    z_vals = torch.linspace(0, 1, num_samples)
    z_vals = near * (1 - z_vals) + far * z_vals
    z_vals = z_vals.unsqueeze(0)

    z_vals = z_vals.expand(N_rays, -1)
    delta_z_vals = torch.rand_like(z_vals) * (distance / num_samples)
    z_vals = z_vals + delta_z_vals
    xyz_sampled = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)

    # Compute deltas to approximate the integral of sigma
    deltas = z_vals[:, 1:] - z_vals[:, :-1]
    delta_inf = 10_000 * torch.ones_like(deltas[:, :1])
    deltas = torch.cat([deltas, delta_inf], -1)

    return xyz_sampled, deltas
