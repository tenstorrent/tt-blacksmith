# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn

from blacksmith.models.torch.nerf.sh import eval_sh


class Embedding(nn.Module):
    def __init__(self, in_channels, num_freqs, logscale=True):
        super(Embedding, self).__init__()
        self.num_freqs = num_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * num_freqs + 1)

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, num_freqs - 1, num_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (num_freqs - 1), num_freqs)

    def forward(self, x):
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]

        return torch.cat(out, -1)


class NeRFHead(nn.Module):
    def __init__(self, W, out_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer1 = nn.Linear(W, W)
        self.relu1 = nn.ReLU(False)
        self.layer2 = nn.Linear(W, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        return x


class NeRFEncoding(nn.Module):
    def __init__(self, in_dim, W, out_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer1 = nn.Linear(in_dim, W)
        self.relu1 = nn.ReLU(False)
        self.layer2 = nn.Linear(W, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        return x


class NeRF(nn.Module):
    def __init__(self, depth=4, width=128, in_channels_xyz=63, in_channels_dir=32, deg=2):
        super(NeRF, self).__init__()
        self.depth = depth
        self.width = width
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.deg = deg

        for i in range(depth):
            if i == 0:
                layer = NeRFEncoding(in_channels_xyz, width, width)
            else:
                layer = NeRFEncoding(width, width, width)
            setattr(self, f"xyz_encoding_{i+1}", layer)

        self.sigma = NeRFHead(width, 1)
        self.sh = NeRFHead(width, 32)
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        input_xyz = x

        xyz_ = input_xyz
        for i in range(self.depth):
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        sh = self.sh(xyz_)
        return sigma, sh

    def sh2rgb(self, sigma, sh, deg, dirs):
        """
        Converts spherical harmonics to RGB.
        """
        sh = sh[:, :27]
        rgb = eval_sh(deg=deg, sh=sh.reshape(-1, 3, (self.deg + 1) ** 2), dirs=dirs)  # sh: [..., x , (deg + 1) ** 2]
        rgb = torch.sigmoid(rgb)
        return sigma, rgb, sh

    def sigma2weights(self, deltas, sigmas):
        """
        Compute weights and alphas from sigmas and deltas.
        """
        sigmas2 = sigmas.squeeze(-1)
        # Noise can be added here make the training more robust
        # noise =  torch.randn(sigmas.shape[:2], device=sigmas.device)
        # sigmas2 = sigmas2 + noise

        alphas = 1 - torch.exp(-deltas * self.softplus(sigmas2))  # (N_rays, N_samples_)
        alphas_shifted = torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1)  # [1, a1, a2, ...]
        weights = alphas * torch.cumprod(alphas_shifted, -1)[:, :-1]  # (N_rays, N_samples_)
        return weights, alphas


def inference(
    model,
    embedding_xyz,
    xyzs,
    dirs,
    deltas,
    idx_render,
    sigma_default,
    chunk=1024,
):
    batch_size = xyzs.shape[0]
    sample_size = xyzs.shape[1]
    device = xyzs.device

    xyzs = xyzs[idx_render[:, 0], idx_render[:, 1]].view(-1, 3)
    view_dir = dirs.unsqueeze(1).expand(-1, sample_size, -1)
    view_dir = view_dir[idx_render[:, 0], idx_render[:, 1]]

    # Pad to chunk size
    real_chunk_size = xyzs.shape[0]
    xyz_to_process = torch.cat([xyzs, torch.zeros(chunk - real_chunk_size, 3, device=xyzs.device)], dim=0)
    view_dir_to_process = torch.cat(
        [view_dir, torch.zeros(chunk - real_chunk_size, 3, device=view_dir.device)],
        dim=0,
    )

    if not hasattr(model, "out_sigma"):
        out_rgb = torch.ones((batch_size, sample_size, 3), device=device).requires_grad_()  # Only use ones where needed
        out_sigma = torch.full((batch_size, sample_size, 1), sigma_default, device=device).requires_grad_()
        out_sh = torch.zeros((batch_size, sample_size, 27), device=device).requires_grad_()
        model.out_sigma = out_sigma
        model.out_rgb = out_rgb
        model.out_sh = out_sh

    embedded_xyz = embedding_xyz(xyz_to_process)
    sigma, sh = model(embedded_xyz)

    sigma.retain_grad()
    sh.retain_grad()

    sigma, rgb, sh = model.sh2rgb(sigma, sh, model.deg, view_dir_to_process)
    sigma = sigma[:real_chunk_size]
    rgb = rgb[:real_chunk_size]
    sh = sh[:real_chunk_size]

    out_sigma = model.out_sigma.detach()
    out_rgb = model.out_rgb.detach()
    out_sh = model.out_sh.detach()

    out_rgb.fill_(1.0)
    out_sigma.fill_(sigma_default)
    out_sh.zero_()

    out_sigma.index_put_((idx_render[:, 0], idx_render[:, 1]), sigma)
    out_rgb.index_put_((idx_render[:, 0], idx_render[:, 1]), rgb)
    out_sh.index_put_((idx_render[:, 0], idx_render[:, 1]), sh)

    weights, alphas = model.sigma2weights(deltas, out_sigma)
    weights_sum = weights.sum(1)  # (N_rays), the accumulated opacity along the rays
    # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically

    # compute final weighted outputs
    rgb_final = torch.sum(weights.unsqueeze(-1) * out_rgb, -2)  # (N_rays, 3)
    # white background
    rgb_final = rgb_final + 1 - weights_sum.unsqueeze(-1)
    return rgb_final, out_sigma, out_sh
