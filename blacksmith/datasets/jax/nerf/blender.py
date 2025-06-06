# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
import json
import numpy as np
import os
from PIL import Image
from typing import Tuple, Dict, Any
from functools import partial

# Assuming ray_utils.py will be translated separately
from blacksmith.datasets.jax.nerf.ray_utils import get_ray_directions, get_rays

# Transformation matrix functions
def trans_t(t: float) -> jnp.ndarray:
    return jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]], dtype=jnp.float32)


def rot_phi(phi: float) -> jnp.ndarray:
    return jnp.array(
        [[1, 0, 0, 0], [0, jnp.cos(phi), -jnp.sin(phi), 0], [0, jnp.sin(phi), jnp.cos(phi), 0], [0, 0, 0, 1]],
        dtype=jnp.float32,
    )


def rot_theta(th: float) -> jnp.ndarray:
    return jnp.array(
        [[jnp.cos(th), 0, -jnp.sin(th), 0], [0, 1, 0, 0], [jnp.sin(th), 0, jnp.cos(th), 0], [0, 0, 0, 1]],
        dtype=jnp.float32,
    )


def pose_spherical(theta: float, phi: float, radius: float) -> jnp.ndarray:
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * jnp.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * jnp.pi) @ c2w
    c2w = jnp.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=jnp.float32) @ c2w
    return c2w


class BlenderDataset:
    def __init__(self, root_dir: str, split: str = "train", img_wh: Tuple[int, int] = (400, 400)):
        self.root_dir = root_dir
        self.split = split
        if img_wh[0] != img_wh[1]:
            raise ValueError("image width must equal image height!")
        self.img_wh = img_wh
        self.white_back = True

        # Load metadata to get focal length first
        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), "r") as f:
            self.meta = json.load(f)

        # Compute focal length before directions
        w, h = self.img_wh
        self.focal = 0.5 * 400 / jnp.tan(0.5 * self.meta["camera_angle_x"])
        self.focal *= self.img_wh[0] / 400

        # Precompute directions
        directions = get_ray_directions(h, w, self.focal)  # (h, w, 3)
        self.directions = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True)

        # Finish remaining metadata setup
        self.read_meta()

    def read_meta(self):
        # Focal length already set in __init__, no need to recompute
        w, h = self.img_wh
        self.near = 2.0
        self.far = 6.0
        self.bounds = jnp.array([self.near, self.far])

        if self.split == "train":
            self._prepare_train_data()
        elif self.split == "test":
            self.meta["frames"] = self.meta["frames"][0:1]
        else:  # val
            angles = jnp.linspace(-180, 180, 1001)[:-1]
            self.pose_vis = jax.vmap(pose_spherical, in_axes=(0, None, None))(angles, -30.0, 4.0)

    def _prepare_train_data(self):
        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []

        def process_frame(frame):
            pose = jnp.array(frame["transform_matrix"], dtype=jnp.float32)[:3, :4]
            image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")

            img = Image.open(image_path)
            if self.img_wh[0] != img.size[0]:
                img = img.resize(self.img_wh, Image.Resampling.LANCZOS)
            img = jnp.array(img, dtype=jnp.float32) / 255.0  # (h, w, 4)

            # Blend RGBA to RGB with white background
            rgb = img[..., :3] * img[..., -1:] + (1 - img[..., -1:])
            rgb = rgb.reshape(-1, 3)  # (h*w, 3)

            rays_o, rays_d = get_rays(self.directions, pose)
            rays_d = rays_d / jnp.linalg.norm(rays_d, axis=-1, keepdims=True)
            rays = jnp.concatenate([rays_o, rays_d], axis=-1)  # (h*w, 6)

            return image_path, pose, rays, rgb

        results = [process_frame(frame) for frame in self.meta["frames"]]
        self.image_paths = [r[0] for r in results]
        self.poses = jnp.stack([r[1] for r in results])
        self.all_rays = jnp.concatenate([r[2] for r in results])
        self.all_rgbs = jnp.concatenate([r[3] for r in results])

    def __len__(self) -> int:
        if self.split == "train":
            return self.all_rays.shape[0]
        elif self.split == "test":
            return len(self.meta["frames"])
        elif self.split == "val":
            return self.pose_vis.shape[0]
        return 0

    def __getitem__(self, idx: int) -> Dict[str, jnp.ndarray]:
        if self.split == "train":
            return {"rays": self.all_rays[idx], "rgbs": self.all_rgbs[idx]}
        elif self.split == "test":
            frame = self.meta["frames"][idx]
            c2w = jnp.array(frame["transform_matrix"], dtype=jnp.float32)[:3, :4]

            img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
            if self.img_wh[0] != img.size[0]:
                img = img.resize(self.img_wh, Image.Resampling.LANCZOS)
            img = jnp.array(img, dtype=jnp.float32) / 255.0  # (h, w, 4)

            valid_mask = (img[..., -1] > 0).flatten()
            rgb = img[..., :3] * img[..., -1:] + (1 - img[..., -1:])
            rgb = rgb.reshape(-1, 3)

            rays_o, rays_d = get_rays(self.directions, c2w)
            rays = jnp.concatenate([rays_o, rays_d], axis=-1)

            return {"rays": rays, "rgbs": rgb, "c2w": c2w, "valid_mask": valid_mask}
        elif self.split == "val":
            c2w = self.pose_vis[idx][:3, :4]
            rays_o, rays_d = get_rays(self.directions, c2w)
            rays_d = rays_d / jnp.linalg.norm(rays_d, axis=-1, keepdims=True)
            rays = jnp.concatenate([rays_o, rays_d], axis=-1)

            return {"rays": rays}

        raise ValueError(f"Unknown split: {self.split}")


def create_dataloader(dataset: BlenderDataset, batch_size: int, seed: int = 0):
    num_samples = len(dataset)
    steps_per_epoch = num_samples // batch_size
    rng = jax.random.PRNGKey(seed)

    def data_generator():
        nonlocal rng  # Allow updating the random key
        start_idx = 0
        while True:
            # At the start of an epoch, shuffle indices
            if start_idx == 0:
                rng, subkey = jax.random.split(rng)
                indices = jax.random.permutation(subkey, jnp.arange(num_samples))

            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            batch = jax.vmap(dataset.__getitem__)(batch_indices)
            yield batch

            start_idx = end_idx
            if start_idx >= num_samples:
                start_idx = 0

    return data_generator(), steps_per_epoch


def create_dataloader_val(dataset: BlenderDataset, batch_size: int):
    if dataset.split != "test":
        raise ValueError("create_dataloader_val is only for test split")

    num_images = len(dataset)
    rays_per_image = dataset.img_wh[0] * dataset.img_wh[1]
    batches_per_image = (rays_per_image + batch_size - 1) // batch_size
    steps_per_epoch = num_images * batches_per_image

    def data_generator():
        while True:
            for img_idx in range(num_images):

                item = dataset[img_idx]
                rays = item["rays"]
                rgbs = item["rgbs"]

                for start_idx in range(0, rays_per_image + batch_size, batch_size):
                    end_idx = min(start_idx + batch_size, rays_per_image)
                    if start_idx >= rays_per_image:
                        break
                    batch_rays = rays[start_idx:end_idx]
                    batch_rgbs = rgbs[start_idx:end_idx]

                    # Pad if necessary
                    if end_idx - start_idx < batch_size:
                        pad_size = batch_size - (end_idx - start_idx)
                        batch_rays = jnp.pad(batch_rays, ((0, pad_size), (0, 0)), mode="edge")
                        batch_rgbs = jnp.pad(batch_rgbs, ((0, pad_size), (0, 0)), mode="edge")

                    yield {"rays": batch_rays, "rgbs": batch_rgbs}  # (batch_size, 6)  # (batch_size, 3)

    return data_generator(), steps_per_epoch
