# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch


class NerfTree(object):  # This is only based on Pytorch implementation
    def __init__(self, xyz_min, xyz_max, grid_coarse, grid_fine, deg, sigma_init, sigma_default, device):
        """
        xyz_min: list (3,) or (1, 3)
        scope: float
        """
        super().__init__()
        self.sigma_init = sigma_init
        self.sigma_default = sigma_default
        self.sigma_voxels_coarse = torch.full((grid_coarse, grid_coarse, grid_coarse), self.sigma_init, device=device)
        self.index_voxels_coarse = torch.full(
            (grid_coarse, grid_coarse, grid_coarse), 0, dtype=torch.long, device=device
        )
        self.voxels_fine = None

        self.xyz_min = xyz_min[0]
        self.xyz_max = xyz_max[0]
        self.xyz_scope = self.xyz_max - self.xyz_min
        self.grid_coarse = grid_coarse
        self.grid_fine = grid_fine
        self.res_coarse = grid_coarse
        self.res_fine = grid_coarse * grid_fine
        self.dim_sh = 3 * (deg + 1) ** 2
        self.device = device

    def calc_index_coarse(self, xyz):
        ijk_coarse = (
            ((xyz - self.xyz_min) / self.xyz_scope * self.grid_coarse).long().clamp(min=0, max=self.grid_coarse - 1)
        )
        return ijk_coarse

    def update_coarse(self, xyz, sigma, beta):
        """
        xyz: (N, 3)
        sigma: (N,)
        """
        ijk_coarse = self.calc_index_coarse(xyz)
        ijk_coarse = ijk_coarse.to(xyz.device)

        self.sigma_voxels_coarse[ijk_coarse[:, 0], ijk_coarse[:, 1], ijk_coarse[:, 2]] = (
            1 - beta
        ) * self.sigma_voxels_coarse[ijk_coarse[:, 0], ijk_coarse[:, 1], ijk_coarse[:, 2]] + beta * sigma

    def create_voxels_fine(self):
        ijk_coarse = (
            torch.logical_and(self.sigma_voxels_coarse > 0, self.sigma_voxels_coarse != self.sigma_init)
            .nonzero()
            .squeeze(1)
        )  # (N, 3)
        num_valid = ijk_coarse.shape[0] + 1

        index = torch.arange(1, num_valid, dtype=torch.long, device=ijk_coarse.device)
        self.index_voxels_coarse[ijk_coarse[:, 0], ijk_coarse[:, 1], ijk_coarse[:, 2]] = index

        self.voxels_fine = torch.zeros(
            num_valid, self.grid_fine, self.grid_fine, self.grid_fine, self.dim_sh + 1, device=self.device
        )
        self.voxels_fine[..., 0] = self.sigma_default
        self.voxels_fine[..., 1:] = 0.0

    def calc_index_fine(self, xyz):
        xyz_norm = (xyz - self.xyz_min) / self.xyz_scope
        xyz_fine = (xyz_norm * self.res_fine).long()
        index_fine = xyz_fine % self.grid_fine
        return index_fine

    def update_fine(self, xyz, sigma, sh):
        # calc ijk_coarse
        index_coarse = self.query_coarse(xyz, "index")
        nonzero_index_coarse = torch.nonzero(index_coarse).squeeze(1)
        index_coarse = index_coarse[nonzero_index_coarse]

        # calc index_fine
        ijk_fine = self.calc_index_fine(xyz[nonzero_index_coarse])

        # feat
        feat = torch.cat([sigma, sh], dim=-1)

        self.voxels_fine[index_coarse, ijk_fine[:, 0], ijk_fine[:, 1], ijk_fine[:, 2]] = feat[nonzero_index_coarse]

    def query_coarse(self, xyz, type="sigma"):
        ijk_coarse = self.calc_index_coarse(xyz)
        ijk_coarse = ijk_coarse.to(self.sigma_voxels_coarse.device)

        if type == "sigma":
            out = self.sigma_voxels_coarse[ijk_coarse[:, 0], ijk_coarse[:, 1], ijk_coarse[:, 2]]
        else:
            out = self.index_voxels_coarse[ijk_coarse[:, 0], ijk_coarse[:, 1], ijk_coarse[:, 2]]
        return out

    def query_fine(self, xyz):
        # calc index_coarse
        index_coarse = self.query_coarse(xyz, "index")

        # calc index_fine
        ijk_fine = self.calc_index_fine(xyz)

        return self.voxels_fine[index_coarse, ijk_fine[:, 0], ijk_fine[:, 1], ijk_fine[:, 2]]
