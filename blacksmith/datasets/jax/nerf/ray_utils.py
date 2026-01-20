# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Tuple

import jax
import jax.numpy as jnp


# JAX equivalent of kornia.create_meshgrid
def create_meshgrid(height: int, width: int, normalized_coordinates: bool = False) -> jnp.ndarray:
    """Generate a coordinate grid for an image of size (height, width)."""
    if normalized_coordinates:
        xs = jnp.linspace(-1, 1, width)
        ys = jnp.linspace(-1, 1, height)
    else:
        xs = jnp.arange(width, dtype=jnp.float32)
        ys = jnp.arange(height, dtype=jnp.float32)
    i, j = jnp.meshgrid(xs, ys, indexing="xy")
    return jnp.stack([i, j], axis=-1)[None]  # (1, H, W, 2)


def get_ray_directions(H: int, W: int, focal: float) -> jnp.ndarray:
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal: image height, width and focal length

    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]  # (H, W, 2)
    i, j = grid[..., 0], grid[..., 1]  # Split into x and y coordinates

    # Compute directions without +0.5 pixel centering (per NeRF convention)
    directions = jnp.stack([(i - W / 2) / focal, -(j - H / 2) / focal, -jnp.ones_like(i)], axis=-1)  # (H, W, 3)

    return directions


def get_rays(directions: jnp.ndarray, c2w: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Get ray origin and directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the direction of the rays in world coordinate
    """
    # Rotate ray directions from camera to world coordinates
    rays_d = directions @ c2w[:3, :3].T  # (H, W, 3)
    # Normalization is handled in BlenderDataset, so omitted here (as in original)

    # The origin of all rays is the camera origin in world coordinates
    rays_o = jnp.broadcast_to(c2w[:3, 3], rays_d.shape)  # (H, W, 3)

    # Flatten to (H*W, 3)
    rays_d = rays_d.reshape(-1, 3)
    rays_o = rays_o.reshape(-1, 3)

    return rays_o, rays_d


def get_ndc_rays(
    H: int, W: int, focal: float, near: float, rays_o: jnp.ndarray, rays_d: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Transform rays from world coordinate to NDC (Normalized Device Coordinates).
    Inputs:
        H, W, focal: image height, width, and focal length
        near: near plane distance
        rays_o, rays_d: (H*W, 3) ray origins and directions in world coordinate
    Outputs:
        rays_o, rays_d: (H*W, 3) ray origins and directions in NDC
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection to NDC
    o0 = -1.0 / (W / (2.0 * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1.0 / (H / (2.0 * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1.0 + 2.0 * near / rays_o[..., 2]

    d0 = -1.0 / (W / (2.0 * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1.0 / (H / (2.0 * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2.0 * near / rays_o[..., 2]

    rays_o = jnp.stack([o0, o1, o2], axis=-1)
    rays_d = jnp.stack([d0, d1, d2], axis=-1)

    return rays_o, rays_d
