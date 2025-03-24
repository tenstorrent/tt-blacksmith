# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from typing import Callable, Tuple, Any

from sh import eval_sh


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
    in_dim: int
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
    in_channels_xyz: int = 63
    in_channels_dir: int = 27
    deg: int = 2

    @nn.compact
    def __call__(self, x):
        xyz_ = x
        for i in range(self.depth):
            if i == 0:
                xyz_ = NeRFEncoding(self.in_channels_xyz, self.width, self.width)(xyz_)
            else:
                xyz_ = NeRFEncoding(self.width, self.width, self.width)(xyz_)

        sigma = NeRFHead(self.width, 1)(xyz_)
        sh = NeRFHead(self.width, 32)(xyz_)
        return sigma, sh

    def sh2rgb(self, sigma, sh, deg, dirs):
        """Converts spherical harmonics to RGB."""
        sh = sh[:, :27]
        rgb = eval_sh(deg=deg, sh=sh.reshape(-1, 3, (self.deg + 1) ** 2), dirs=dirs)
        rgb = jax.nn.sigmoid(rgb)
        return sigma, rgb, sh

    def sigma2weights(self, deltas, sigmas, mask=None):
        """Compute weights and alphas from sigmas and deltas."""
        sigmas2 = sigmas.squeeze(-1)
        noise = jnp.zeros(sigmas.shape[:2])
        sigmas2 = sigmas2 + noise

        alphas = 1 - jnp.exp(-deltas * jax.nn.softplus(sigmas2))
        if mask is not None:
            mask = mask.squeeze(-1)
            # jax.debug.print("Sum of mask: {}", (mask.sum(),), ordered = True)
            # jax.debug.print("Size of mask: {}", (mask.shape,), ordered = True)
            alphas = alphas * mask + 1 - mask
        alphas_shifted = jnp.concatenate([jnp.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], axis=-1)
        weights = alphas * jnp.cumprod(alphas_shifted, axis=-1)[:, :-1]
        # norm = jax.numpy.linalg.norm(weights)
        # print(norm)
        # jax.debug.print("Norm of weights: {}", (jax.numpy.linalg.norm(weights),), ordered = True)
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
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    batch_size, sample_size = xyzs.shape[0], xyzs.shape[1]

    non_minus_one_mask = jnp.ones((batch_size, sample_size))
    non_one_idx = idx_render * (idx_render == -1)
    non_minus_one_mask = non_minus_one_mask.at[non_one_idx].set(0)

    xyzs_flat = xyzs[idx_render[:, 0], idx_render[:, 1]].reshape(-1, 3)
    view_dir = jnp.expand_dims(dirs, 1).repeat(sample_size, axis=1)
    view_dir_flat = view_dir[idx_render[:, 0], idx_render[:, 1]]

    # Pad to chunk size
    real_chunk_size = xyzs_flat.shape[0]
    xyz_to_process = jnp.concatenate([xyzs_flat, jnp.zeros((chunk - real_chunk_size, 3))], axis=0)
    view_dir_to_process = jnp.concatenate([view_dir_flat, jnp.zeros((chunk - real_chunk_size, 3))], axis=0)

    # Apply embedding without parameters (stateless)
    embedded_xyz = embedding_xyz.apply({}, xyz_to_process)  # No 'params' needed
    # import pdb; pdb.set_trace()
    # print(embedded_xyz)
    # Apply model with parameters
    sigma, sh = model.apply({"params": params}, embedded_xyz)
    sigma, rgb, sh = model.sh2rgb(sigma, sh, model.deg, view_dir_to_process)
    sigma = sigma[:real_chunk_size]
    rgb = rgb[:real_chunk_size]
    sh = sh[:real_chunk_size]

    # Initialize outputs
    out_rgb = jnp.ones((batch_size, sample_size, 3))
    out_sigma = jnp.full((batch_size, sample_size, 1), sigma_default)
    out_sh = jnp.zeros((batch_size, sample_size, 27))

    # Update outputs with computed values
    out_sigma = out_sigma.at[idx_render[:, 0], idx_render[:, 1]].set(sigma)
    out_rgb = out_rgb.at[idx_render[:, 0], idx_render[:, 1]].set(rgb)
    out_sh = out_sh.at[idx_render[:, 0], idx_render[:, 1]].set(sh)

    non_minus_one_mask = jnp.expand_dims(non_minus_one_mask, axis=-1)
    out_sigma = out_sigma * non_minus_one_mask

    weights, alphas = model.sigma2weights(deltas, out_sigma, non_minus_one_mask)
    weights_sum = weights.sum(axis=1)

    # Compute final weighted outputs
    rgb_final = jnp.sum(weights[..., None] * out_rgb, axis=-2)
    rgb_final = rgb_final + (1 - weights_sum[..., None])  # White background
    return rgb_final, out_sigma, out_sh


# Example usage:
if __name__ == "__main__":
    key = random.PRNGKey(0)
    model = NeRF()
    embedding_xyz = Embedding(in_channels=3, num_freqs=10)

    # Dummy inputs
    xyzs = jnp.ones((2, 4, 3))
    dirs = jnp.ones((2, 3))
    deltas = jnp.ones((2, 4))
    idx_render = jnp.array([[0, 0], [0, 1], [1, 2], [1, 3]])

    # Initialize parameters for NeRF (which has trainable params)
    variables = model.init(key, jnp.ones((4, 63)))
    params = variables["params"]  # Extract 'params' for NeRF

    # No need to initialize params for Embedding since it’s stateless
    rgb_final, out_sigma, out_sh = inference(
        model, params, embedding_xyz, xyzs, dirs, deltas, idx_render, sigma_default=0.0
    )
    print(rgb_final.shape, out_sigma.shape, out_sh.shape)
