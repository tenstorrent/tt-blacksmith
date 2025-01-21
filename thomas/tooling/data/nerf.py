# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2

from thomas.tooling.utils.nerf.nerf import get_rays_np


trans_t = lambda t: torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor(
    [[1, 0, 0, 0], [0, np.cos(phi), -np.sin(phi), 0], [0, np.sin(phi), np.cos(phi), 0], [0, 0, 0, 1]]
).float()

rot_theta = lambda th: torch.Tensor(
    [[np.cos(th), 0, -np.sin(th), 0], [0, 1, 0, 0], [np.sin(th), 0, np.cos(th), 0], [0, 0, 0, 1]]
).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ["train", "val", "test"]
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, "transforms_{}.json".format(s)), "r") as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s == "train" or testskip == 0:
            skip = 1
        else:
            skip = testskip

        for frame in meta["frames"][::skip]:
            fname = os.path.join(basedir, frame["file_path"] + ".png")
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame["transform_matrix"]))
        imgs = (np.array(imgs) / 255.0).astype(np.float32)  # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)

    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.0

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res

    return imgs, poses, render_poses, [H, W, focal], i_split


class NeRFDataset(torch.utils.data.Dataset):
    def __init__(self, path, half_res=False, testskip=1):
        self.images, self.poses, self.render_poses, self.hwf, self.splits = load_blender_data(path, half_res, testskip)
        self.H, self.W, self.F = self.hwf
        self.K = np.array([[self.F, 0, self.W / 2], [0, self.F, self.H / 2], [0, 0, 1]])
        self.train_split, self.val_split, self.test_split = self.splits
        self.images = self.images[..., :3] * self.images[..., -1:] + (1.0 - self.images[..., -1:])
        self.rays = np.stack(
            [get_rays_np(self.H, self.W, self.K, p) for p in self.poses[:, :3, :4]], 0
        )  # [N, ro+rd, H, W, 3]
        self.rays_rgb = np.concatenate([self.rays, self.images[:, None]], 1)  # [N, ro+rd+rgb, H, W, 3]
        self.rays_rgb = np.transpose(self.rays_rgb, [0, 2, 3, 1, 4])  # [N, H, W, ro+rd+rgb, 3]
        self.rays_rgb = np.stack([self.rays_rgb[i] for i in self.train_split], 0)  # train images only
        self.rays_rgb = np.reshape(self.rays_rgb, [-1, 3, 3])  # [(N-1)*H*W, ro+rd+rgb, 3]
        self.rays_rgb = self.rays_rgb.astype(np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.rays_rgb[idx]
