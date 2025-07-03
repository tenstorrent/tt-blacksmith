# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp


class NerfTree:
    def __init__(self, xyz_min, xyz_max, grid_coarse, grid_fine, deg, sigma_init, sigma_default, device=None):
        """
        xyz_min: list (3,) or (1, 3)
        xyz_max: list (3,) or (1, 3)
        """
        self.sigma_init = sigma_init
        self.sigma_default = sigma_default
        self.sigma_voxels_coarse = jnp.full((grid_coarse, grid_coarse, grid_coarse), sigma_init)
        self.index_voxels_coarse = jnp.full((grid_coarse, grid_coarse, grid_coarse), 0, dtype=jnp.int32)
        self.voxels_fine = None

        self.deg = deg
        self.xyz_min = jnp.array(xyz_min[0])
        self.xyz_max = jnp.array(xyz_max[0])
        self.xyz_scope = self.xyz_max - self.xyz_min
        self.grid_coarse = grid_coarse
        self.grid_fine = grid_fine
        self.res_coarse = grid_coarse
        self.res_fine = grid_coarse * grid_fine
        self.dim_sh = 3 * (deg + 1) ** 2
        self.device = device


# Standalone functions using tree_data
def calc_index_coarse_out(xyz, tree_data):
    ijk_coarse = (
        ((xyz - tree_data["xyz_min"]) / tree_data["xyz_scope"] * tree_data["grid_coarse"])
        .astype(jnp.int32)
        .clip(min=0, max=tree_data["grid_coarse"] - 1)
    )
    return ijk_coarse


def update_coarse_out(sigma_voxels_coarse, xyz, sigma, beta, tree_data):
    ijk_coarse = calc_index_coarse_out(xyz, tree_data)
    updated_sigma_voxels = sigma_voxels_coarse.at[ijk_coarse[:, 0], ijk_coarse[:, 1], ijk_coarse[:, 2]].set(
        (1 - beta) * sigma_voxels_coarse[ijk_coarse[:, 0], ijk_coarse[:, 1], ijk_coarse[:, 2]] + beta * sigma
    )
    return updated_sigma_voxels


def create_voxels_fine_out(sigma_voxels_coarse, index_voxels_coarse, tree_data):
    ijk_coarse = jnp.stack(
        jnp.nonzero(jnp.logical_and(sigma_voxels_coarse > 0, sigma_voxels_coarse != tree_data["sigma_init"])), axis=-1
    )
    num_valid = ijk_coarse.shape[0] + 1

    index = jnp.arange(1, num_valid, dtype=jnp.int32)
    updated_index_voxels = index_voxels_coarse.at[ijk_coarse[:, 0], ijk_coarse[:, 1], ijk_coarse[:, 2]].set(index)

    voxels_fine = jnp.zeros(
        (num_valid, tree_data["grid_fine"], tree_data["grid_fine"], tree_data["grid_fine"], tree_data["dim_sh"] + 1),
        dtype=jnp.float32,
    )
    voxels_fine = voxels_fine.at[..., 0].set(tree_data["sigma_default"])
    voxels_fine = voxels_fine.at[..., 1:].set(0.0)

    return updated_index_voxels, voxels_fine


def calc_index_fine_out(xyz, tree_data):
    xyz_norm = (xyz - tree_data["xyz_min"]) / tree_data["xyz_scope"]
    xyz_fine = (xyz_norm * tree_data["res_fine"]).astype(jnp.int32)
    index_fine = xyz_fine % tree_data["grid_fine"]
    return index_fine


def update_fine_out(voxels_fine, xyz, sigma, sh, tree_data):
    index_coarse = query_coarse_out(xyz, tree_data, "index")
    nonzero_index_coarse = jnp.nonzero(index_coarse)[0]
    index_coarse_filtered = index_coarse[nonzero_index_coarse]

    ijk_fine = calc_index_fine_out(xyz[nonzero_index_coarse], tree_data)
    feat = jnp.concatenate([sigma, sh], axis=-1)[nonzero_index_coarse]

    updated_voxels_fine = voxels_fine.at[index_coarse_filtered, ijk_fine[:, 0], ijk_fine[:, 1], ijk_fine[:, 2]].set(
        feat
    )
    return updated_voxels_fine


def query_coarse_out(xyz, tree_data, type="sigma"):
    ijk_coarse = calc_index_coarse_out(xyz, tree_data)
    if type == "sigma":
        # print(ijk_coarse.shape)
        # print(tree_data["sigma_voxels_coarse"].shape)
        return tree_data["sigma_voxels_coarse"][ijk_coarse[:, 0], ijk_coarse[:, 1], ijk_coarse[:, 2]]
    else:
        return tree_data["index_voxels_coarse"][ijk_coarse[:, 0], ijk_coarse[:, 1], ijk_coarse[:, 2]]
