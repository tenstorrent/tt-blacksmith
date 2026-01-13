# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from blacksmith.models.jax.nerf.sh import eval_sh


class Embedding(nn.Module):
    in_channels: int
    num_freqs: int
    logscale: bool = True

    @nn.compact
    def __call__(self, x):
        if self.logscale:
            freq_bands = 2 ** jnp.linspace(0, self.num_freqs - 1, self.num_freqs)
        else:
            freq_bands = jnp.linspace(1, 2 ** (self.num_freqs - 1), self.num_freqs)

        out = [x]
        for freq in freq_bands:
            out.append(jnp.sin(freq * x))
            out.append(jnp.cos(freq * x))

        return jnp.concatenate(out, axis=-1)


class NeRFHead(nn.Module):
    W: int
    out_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.W)(x)
        x = nn.relu(x)
        x = nn.Dense(self.out_dim)(x)
        return x


class NeRFEncoding(nn.Module):
    W: int
    out_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.W)(x)
        x = nn.relu(x)
        x = nn.Dense(self.out_dim)(x)
        return x


class NeRF(nn.Module):
    depth: int = 8
    width: int = 256
    deg: int = 2

    @nn.compact
    def __call__(self, x):
        xyz_ = x
        for i in range(self.depth):
            if i == 0:
                xyz_ = NeRFEncoding(self.width, self.width)(xyz_)
            else:
                xyz_ = NeRFEncoding(self.width, self.width)(xyz_)

        sigma = NeRFHead(self.width, 1)(xyz_)
        sh = NeRFHead(self.width, 27)(xyz_)
        return sigma, sh

    def sh2rgb(self, sigma, sh, deg, dirs):
        """Converts spherical harmonics to RGB."""
        sh = sh[:, :27]
        rgb = eval_sh(deg=deg, sh=sh.reshape(-1, 3, (self.deg + 1) ** 2), dirs=dirs)
        rgb = jax.nn.sigmoid(rgb)
        return sigma, rgb, sh

    def sigma2weights(self, deltas, sigma_values, mask=None):
        sigmas = sigma_values.squeeze(-1)

        alphas = 1 - jnp.exp(-deltas * jax.nn.softplus(sigmas))
        if mask is not None:
            mask = mask.squeeze(-1)
            alphas = alphas * mask + 1 - mask
        alphas_shifted = jnp.concatenate([jnp.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], axis=-1)
        weights = alphas * jnp.cumprod(alphas_shifted, axis=-1)[:, :-1]
        return weights, alphas


def inference(
    model: NeRF,
    params: Any,
    embedding_xyz: Embedding,
    xyzs: jnp.ndarray,
    dirs: jnp.ndarray,
    deltas: jnp.ndarray,
    idx_render: jnp.ndarray,
    sigma_default: float,
    chunk: int = 1024,
    callee: str = "coarse",
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Perform NeRF inference with gradient computation support.

    Args:
        model: NeRF neural network model
        params: Model parameters
        embedding_xyz: Positional embedding function for 3D coordinates
        xyzs: 3D sample points [batch_size, sample_size, 3]
        dirs: View directions [batch_size, 3]
        deltas: Ray segment lengths for volume rendering
        idx_render: Indices of points to render [N, 2]
        sigma_default: Default density value for non-rendered points
        chunk: Batch size for processing (memory optimization)
        callee: "coarse" or "fine" to track rendering stage

    Returns:
        rgb_final: Final rendered RGB colors
        out_sigma: Density values for all sample points
        out_sh: Spherical harmonics coefficients
        dict: Contains intermediates and backward function
    """
    batch_size, num_samples_per_ray = xyzs.shape[0], xyzs.shape[1]

    # Extract only the points that need to be rendered (sparse sampling optimization)
    points_to_render = xyzs[idx_render[:, 0], idx_render[:, 1]].reshape(-1, 3)

    # Expand view directions to match sample points
    view_dirs_expanded = jnp.expand_dims(dirs, 1).repeat(num_samples_per_ray, axis=1)
    view_dirs_for_points = view_dirs_expanded[idx_render[:, 0], idx_render[:, 1]]

    # Pad to chunk size for efficient batch processing
    actual_points_count = points_to_render.shape[0]

    # Zero-pad to fill the chunk for vectorized processing
    points_padded = jnp.concatenate([points_to_render, jnp.zeros((chunk - actual_points_count, 3))], axis=0)
    view_dirs_padded = jnp.concatenate([view_dirs_for_points, jnp.zeros((chunk - actual_points_count, 3))], axis=0)

    # Apply positional encoding to 3D coordinates
    embedded_points = embedding_xyz.apply({}, points_padded)

    # VJP (Vector-Jacobian Product) Explanation:
    # VJP computes gradients efficiently by doing forward pass + preparing backward pass
    # - Forward: f(x) -> y (normal function evaluation)
    # - VJP returns: (output, vjp_fn) where vjp_fn(dy) -> dx (gradients w.r.t. inputs)

    # Define neural network forward pass
    @jax.jit
    def network_forward(model_params, embedded_coordinates):
        """Forward pass through NeRF network"""
        density, spherical_harmonics = model.apply({"params": model_params}, embedded_coordinates)
        return density, spherical_harmonics

    # Execute forward pass on accelerator device with VJP for gradient computation
    with jax.default_device(jax.devices("tt")[0]):
        params_on_device = jax.device_put(params, jax.devices("tt")[0])
        embedded_points_on_device = jax.device_put(embedded_points, jax.devices("tt")[0])

        # VJP returns both forward pass results and a function for computing gradients
        (density_on_device, sh_on_device), vjp_backward_fn = jax.vjp(
            network_forward, params_on_device, embedded_points_on_device
        )

    # Define backward function for gradient computation during training
    def compute_gradients(gradient_seed_density, gradient_seed_sh):
        """
        Compute gradients w.r.t. model parameters using VJP.

        Args:
            gradient_seed_density: Upstream gradients w.r.t. density
            gradient_seed_sh: Upstream gradients w.r.t. spherical harmonics

        Returns:
            Gradients w.r.t. model parameters
        """
        with jax.default_device(jax.devices("tt")[0]):
            grad_density_on_device = jax.device_put(gradient_seed_density, jax.devices("tt")[0])
            grad_sh_on_device = jax.device_put(gradient_seed_sh, jax.devices("tt")[0])

            # VJP function computes gradients w.r.t. all inputs (params and embedded_points)
            param_gradients_on_device, _ = vjp_backward_fn((grad_density_on_device, grad_sh_on_device))

            # Return only parameter gradients, moved back to CPU
            return jax.device_put(param_gradients_on_device, jax.devices("cpu")[0])

    # Move network outputs back to CPU for downstream processing
    density_values = jax.device_put(density_on_device, jax.devices("cpu")[0])
    sh_coefficients = jax.device_put(sh_on_device, jax.devices("cpu")[0])

    # Convert spherical harmonics to RGB colors based on view direction
    density_processed, rgb_colors, sh_processed = model.sh2rgb(
        density_values, sh_coefficients, model.deg, view_dirs_padded
    )

    # Remove padding to get actual results
    density_processed = density_processed[:actual_points_count]
    rgb_colors = rgb_colors[:actual_points_count]
    sh_processed = sh_processed[:actual_points_count]

    # Initialize output arrays with default values
    final_rgb = jnp.ones((batch_size, num_samples_per_ray, 3))  # Default white
    final_density = jnp.full((batch_size, num_samples_per_ray, 1), sigma_default)  # Default density
    final_sh = jnp.zeros((batch_size, num_samples_per_ray, 27))  # Default SH coefficients

    # Scatter processed values back to their original positions
    final_density = final_density.at[idx_render[:, 0], idx_render[:, 1]].set(density_processed)
    final_rgb = final_rgb.at[idx_render[:, 0], idx_render[:, 1]].set(rgb_colors)
    final_sh = final_sh.at[idx_render[:, 0], idx_render[:, 1]].set(sh_processed)

    # Create mask for valid (non-empty) sample points
    valid_points_mask = jnp.ones((batch_size, num_samples_per_ray))
    empty_point_indices = idx_render * (idx_render == -1)  # Find empty points marked with -1
    valid_points_mask = valid_points_mask.at[empty_point_indices].set(0)

    # Apply mask to density values (zero out invalid points)
    valid_points_mask_expanded = jnp.expand_dims(valid_points_mask, axis=-1)
    final_density = final_density * valid_points_mask_expanded

    # Perform volume rendering: convert density to weights and compute final RGB
    rendering_weights, alpha_values = model.sigma2weights(deltas, final_density, valid_points_mask_expanded)
    total_weight_per_ray = rendering_weights.sum(axis=1)

    # Weighted sum of RGB values along each ray
    rendered_rgb = jnp.sum(rendering_weights[..., None] * final_rgb, axis=-2)

    # Add white background for areas with low accumulated weight
    rendered_rgb = rendered_rgb + (1 - total_weight_per_ray[..., None])

    if callee == "coarse":
        intermediate_results = {
            "sigma_coarse_immediate": density_processed,
            "sh_coarse_immediate": sh_processed,
            "weights": rendering_weights,
            "alphas": alpha_values,
            "idx_render_coarse": idx_render,
        }
    else:
        intermediate_results = {
            "sigma_fine_immediate": density_processed,
            "sh_fine_immediate": sh_processed,
            "weights": rendering_weights,
            "alphas": alpha_values,
            "idx_render_fine": idx_render,
        }

    return (
        rendered_rgb,
        final_density,
        final_sh,
        {"intermediates": intermediate_results, "nn_backward": compute_gradients},
    )
