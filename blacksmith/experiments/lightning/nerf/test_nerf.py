# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
from collections import defaultdict

import matplotlib
import torch
import wandb
from nerf_rendering import render_rays
from pytorch_lightning import LightningModule, Trainer, seed_everything

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from torchvision import transforms

from blacksmith.datasets.torch.torch_dataset import get_dataset
from blacksmith.experiments.lightning.nerf.configs import NerfConfig, load_config
from blacksmith.experiments.lightning.nerf.utils import *
from blacksmith.experiments.lightning.nerf.utils.log import (
    log_gradients,
    log_training_metrics,
)
from blacksmith.experiments.lightning.nerf.utils.losses import loss_dict
from blacksmith.experiments.lightning.nerf.utils.metrics import *
from blacksmith.models.torch.nerf import Embedding, NeRF
from blacksmith.models.torch.nerf.nerftree import NerfTree

matplotlib.use("Agg")
import logging

import numpy as np

logging.getLogger("lightning").setLevel(logging.DEBUG)
from loguru import logger

logger.disable("")

torch.manual_seed(0)


def copy_model_attributes(model, forge_model):
    forge_model.sh2rgb = model.sh2rgb
    forge_model.sigma2weights = model.sigma2weights
    forge_model.depth = model.depth
    forge_model.width = model.width
    forge_model.in_channels_xyz = model.in_channels_xyz
    forge_model.in_channels_dir = model.in_channels_dir
    forge_model.deg = model.deg


class EfficientNeRFSystem(LightningModule):
    def __init__(self, config: NerfConfig):
        super(EfficientNeRFSystem, self).__init__()
        self.save_hyperparameters(config.model_dump(exclude_none=True))
        self.config: NerfConfig = config

        self.experiment_log_dir = os.path.join(config.training.log_dir, config.experiment_name)

        self.loss = loss_dict[config.training.loss]()

        self.batch_size = self.config.data_loading.batch_size

        self.in_channels_xyz = 3 + config.model.num_freqs * 2 * 3
        self.in_channels_dir = config.model.in_channels_dir
        self.embedding_xyz = Embedding(3, config.model.num_freqs)

        self.deg = config.model.deg
        self.dim_sh = 3 * (self.deg + 1) ** 2
        self.depth_coarse = config.model.coarse.depth
        self.width_coarse = config.model.coarse.width

        self.depth_fine = config.model.fine.depth
        self.width_fine = config.model.fine.width

        self.sigma_init = config.model.sigma_init
        self.sigma_default = config.model.sigma_default

        self.nerf_coarse = NeRF(
            depth=self.depth_coarse,
            width=self.width_coarse,
            in_channels_xyz=self.in_channels_xyz,
            in_channels_dir=self.in_channels_dir,
            deg=self.deg,
        )

        self.nerf_fine = NeRF(
            depth=self.depth_fine,
            width=self.width_fine,
            in_channels_xyz=self.in_channels_xyz,
            in_channels_dir=self.in_channels_dir,
            deg=self.deg,
        )

        self.framework_models = [self.nerf_coarse, self.nerf_fine]

        self.optimizer = get_optimizer(config, self.framework_models)

        if config.training.use_forge:
            import forge

            self.loss_coarse_op = forge.op.loss.MSELoss(name="mse_loss_coarse")
            self.loss_fine_op = forge.op.loss.MSELoss(name="mse_loss_fine")

            self.loss_coarse_forge = forge.compile(
                self.loss_coarse_op,
                sample_inputs=forge.tensor.to_forge_tensors(
                    [
                        torch.randn(self.batch_size, 3).requires_grad_(True),
                        torch.randn(self.batch_size, 3).requires_grad_(True),
                    ]
                ),
                training=True,
            )

            self.loss_fine_forge = forge.compile(
                self.loss_fine_op,
                sample_inputs=forge.tensor.to_forge_tensors(
                    [
                        torch.randn(self.batch_size, 3).requires_grad_(True),
                        torch.randn(self.batch_size, 3).requires_grad_(True),
                    ]
                ),
                training=True,
            )

            max_input = self.batch_size * config.model.coarse.samples
            self.nerf_coarse_forge = forge.compile(
                self.nerf_coarse,
                sample_inputs=[torch.randn(max_input, self.in_channels_xyz)],
                optimizer=self.optimizer,
                training=True,
                module_name="nerf_coarse_sigma",
            )

            copy_model_attributes(self.nerf_coarse, self.nerf_coarse_forge)

            self.nerf_fine_forge = forge.compile(
                self.nerf_fine,
                sample_inputs=[torch.randn(max_input, self.in_channels_xyz)],
                optimizer=self.optimizer,
                training=True,
                module_name="nerf_fine_sigma",
            )

            copy_model_attributes(self.nerf_fine, self.nerf_fine_forge)

        # forge messes up the random state, so we need to reset it
        torch.manual_seed(0)
        seed_everything(0, workers=True)
        np.random.seed(0)

        coord_scope = config.model.coord_scope
        self.nerf_tree = NerfTree(
            xyz_min=[-coord_scope, -coord_scope, -coord_scope],
            xyz_max=[coord_scope, coord_scope, coord_scope],
            grid_coarse=384,
            grid_fine=3,
            deg=self.deg,
            sigma_init=self.sigma_init,
            sigma_default=self.sigma_default,
            device=config.training.device,
        )

        self.xyz_min = self.nerf_tree.xyz_min
        self.xyz_max = self.nerf_tree.xyz_max
        self.xyz_scope = self.nerf_tree.xyz_scope
        self.grid_coarse = self.nerf_tree.grid_coarse
        self.grid_fine = self.nerf_tree.grid_fine
        self.res_coarse = self.nerf_tree.res_coarse
        self.res_fine = self.nerf_tree.res_fine
        self.validation_step_outputs = []

    def forward(self, rays: torch.Tensor) -> dict:
        """
        Process a batch of rays through the NeRF models, processing in chunks to manage memory.

        Args:
            rays: Tensor of shape [num_rays, 6] containing ray origins and directions

        Returns:
            dict: Aggregated rendering results for all rays
        """
        results = defaultdict(list)

        models = (
            [self.nerf_coarse_forge, self.nerf_fine_forge] if self.config.training.use_forge else self.framework_models
        )

        for ray_idx in range(0, rays.shape[0], self.batch_size):
            rays_chunk = rays[ray_idx : ray_idx + self.batch_size]

            # Handle partial batches by padding with random rays
            num_rays_in_chunk = len(rays_chunk)
            num_padding_needed = self.batch_size - num_rays_in_chunk

            if num_padding_needed > 0:
                # Select random rays to pad the batch
                random_ray_indices = torch.randint(0, rays.shape[0], (num_padding_needed,))
                padding_rays = rays[random_ray_indices]
                rays_chunk = torch.cat([rays_chunk, padding_rays], dim=0)

            # Render the rays in this chunk
            chunk_results = render_rays(
                config=self.config,
                rays=rays_chunk,
                embedding_xyz=self.embedding_xyz,
                nerf_tree=self.nerf_tree,
                near=self.near,
                far=self.far,
                global_step=self.global_step,
                model_coarse=models[0],
                model_fine=models[1],
            )

            # Remove padding from results before storing
            for key, value in chunk_results.items():
                if num_padding_needed > 0:
                    trimmed_value = value[:-num_padding_needed]
                else:
                    trimmed_value = value
                results[key].append(trimmed_value)

        # Concatenate chunk results
        for key, value_list in results.items():
            results[key] = torch.cat(value_list, dim=0)

        return results

    def prepare_data(self):
        self.train_dataset = get_dataset(self.config, split="train")
        self.val_dataset = get_dataset(self.config, split="test")

        self.near = self.train_dataset.near
        self.far = self.train_dataset.far

    def train_dataloader(self):
        return self.train_dataset.get_dataloader()

    def val_dataloader(self):
        return self.val_dataset.get_dataloader()

    def training_step(self, batch: int, batch_idx: int) -> torch.Tensor:
        self.log("train/lr", get_learning_rate(self.optimizer), on_step=True, prog_bar=True)
        rays, rgbs = batch["rays"], batch["rgbs"]
        extract_time = self.current_epoch >= (self.config.training.epochs - 1)

        if extract_time and self.nerf_tree.voxels_fine == None:
            self.nerf_tree.create_voxels_fine()
        results = self(rays)

        if self.config.training.use_forge:
            loss_total = 0.0
            if "rgb_coarse" in results:
                rgb_coarse = results["rgb_coarse"]
                rgb_coarse.requires_grad_(True)
                # pad with zeros to match (batch_size, 3)
                rgb_coarse = torch.cat(
                    [rgb_coarse, torch.zeros((abs(self.batch_size - rgb_coarse.shape[0]), 3))], dim=0
                )
                results["rgb_coarse"] = rgb_coarse
                rgbs = torch.cat([rgbs, torch.zeros((abs(self.batch_size - rgbs.shape[0]), 3))], dim=0)
                loss_coarse = self.loss_coarse_forge(rgb_coarse, rgbs)[0]
                loss_total += loss_coarse
            if "rgb_fine" in results:
                rgb_fine = results["rgb_fine"]
                rgb_fine.requires_grad_(True)
                # pad with zeros to match (batch_size, 3)
                rgb_fine = torch.cat([rgb_fine, torch.zeros((abs(self.batch_size - rgb_fine.shape[0]), 3))], dim=0)
                results["rgb_fine"] = rgb_fine
                rgbs = torch.cat([rgbs, torch.zeros((abs(self.batch_size - rgbs.shape[0]), 3))], dim=0)
                loss_fine = self.loss_fine_forge(rgb_fine, rgbs)[0]
                loss_total += loss_fine
            loss_rgb = loss_total
        else:
            loss_total = loss_rgb = self.loss(results, rgbs)
        log_training_metrics(self.log, rgbs, results, loss_rgb, loss_total)

        self.results = results
        if self.device.type.startswith("cuda"):
            torch.cuda.empty_cache()
        return loss_total

    def backward(self, loss, *args, **kwargs):
        if self.config.training.use_forge:
            if "rgb_coarse" in self.results:
                self.loss_coarse_forge.backward()
                # do autograd on results of postprocessing
                # forward pass: input -> neural nets (device) -> postprocessing (host) -> loss (device)
                # backward pass: loss backward (device) -> autograd of postprocessing (host) -> neural nets (device)
                torch.autograd.backward(
                    tensors=[self.results["rgb_coarse"]],
                    # gradient_outputs contains grads for predictions and for labels, first being predictions
                    grad_tensors=[-self.loss_coarse_forge.gradient_outputs[0].to_torch()],
                )
                self.nerf_coarse_forge.backward()
            if "rgb_fine" in self.results:
                self.loss_fine_forge.backward()
                torch.autograd.backward(
                    tensors=[self.results["rgb_fine"]],
                    grad_tensors=[-self.loss_fine_forge.gradient_outputs[0].to_torch()],
                )
                self.nerf_fine_forge.backward()
        else:
            loss.backward(*args, **kwargs)

        del self.results
        log_gradients(self.log, self.nerf_coarse.sigma, self.nerf_fine.sigma)

    def configure_optimizers(self):
        return self.optimizer

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        self.optimizer.step(optimizer_closure)
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

    def validation_step(self, batch, batch_idx):
        rays, rgbs = batch["rays"], batch["rgbs"]
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)

        results = self(rays)
        log = {}
        log["val_loss"] = self.loss(results, rgbs)
        typ = "fine" if "rgb_fine" in results else "coarse"

        W, H = self.config.data_loading.img_wh
        img = results[f"rgb_{typ}"].view(H, W, 3).cpu()
        img = img.permute(2, 0, 1)  # (3, H, W)
        img_path = os.path.join(self.experiment_log_dir, "video", "%06d.png" % batch_idx)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        transforms.ToPILImage()(img).convert("RGB").save(img_path)

        if batch_idx == 0:
            W, H = self.config.data_loading.img_wh
            img = results[f"rgb_{typ}"].view(H, W, 3).cpu()
            img = img.permute(2, 0, 1)  # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
            stack = torch.stack([img_gt, img])  # (3, 3, H, W)
            wandb.log(
                {"val/gt_pred": [wandb.Image(img) for img in stack]},
                step=self.global_step,
            )

            img_path = os.path.join(self.experiment_log_dir, f"epoch_{self.current_epoch}.png")
            transforms.ToPILImage()(img).convert("RGB").save(img_path)

        log["val_psnr"] = psnr(results[f"rgb_{typ}"], rgbs)

        self.validation_step_outputs.append(log)

        return log

    def on_validation_epoch_end(self):
        mean_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
        mean_psnr = torch.stack([x["val_psnr"] for x in self.validation_step_outputs]).mean()
        num_voxels_coarse = (
            torch.logical_and(
                self.nerf_tree.sigma_voxels_coarse > 0,
                self.nerf_tree.sigma_voxels_coarse != self.sigma_init,
            )
            .nonzero()
            .shape[0]
        )
        self.log("val/loss", mean_loss)
        self.log("val/psnr", mean_psnr, prog_bar=True)
        self.log("val/num_voxels_coarse", num_voxels_coarse)

    def state_dict(self):
        """Save custom state along with model parameters"""
        state = super().state_dict()

        # Clean and save voxel data
        sigma_voxels_coarse_clean = self.nerf_tree.sigma_voxels_coarse.clone()
        sigma_voxels_coarse_clean[sigma_voxels_coarse_clean == self.sigma_init] = self.sigma_default

        # Add voxel data to state
        state["nerf_tree"] = {
            "sigma_voxels_coarse": sigma_voxels_coarse_clean,
            "index_voxels_coarse": self.nerf_tree.index_voxels_coarse,
            "voxels_fine": self.nerf_tree.voxels_fine,
        }

        use_forge = self.config.training.use_forge
        state["nerf_coarse"] = (
            self.nerf_coarse.state_dict()
            if not use_forge
            else self.nerf_coarse_forge.framework_module.module.state_dict()
        )
        state["nerf_fine"] = (
            self.nerf_fine.state_dict() if not use_forge else self.nerf_fine_forge.framework_module.module.state_dict()
        )

        return state

    def load_state_dict(self, state_dict, strict=False, assign=False):
        # Load voxel data
        nerf_tree_state = state_dict.pop("nerf_tree")
        to_device = lambda x: (x.to(self.config.training.device) if x is not None else None)
        self.nerf_tree.sigma_voxels_coarse = to_device(nerf_tree_state["sigma_voxels_coarse"])
        self.nerf_tree.index_voxels_coarse = to_device(nerf_tree_state["index_voxels_coarse"])
        self.nerf_tree.voxels_fine = to_device(nerf_tree_state["voxels_fine"])

        # Load models
        if self.config.training.use_forge:
            self.nerf_coarse_forge.framework_module.module.load_state_dict(state_dict.pop("nerf_coarse"))
            self.nerf_fine_forge.framework_module.module.load_state_dict(state_dict.pop("nerf_fine"))
        else:
            self.nerf_coarse.load_state_dict(state_dict.pop("nerf_coarse"))
            self.nerf_fine.load_state_dict(state_dict.pop("nerf_fine"))

        super().load_state_dict(state_dict, strict, assign)


def main(config: NerfConfig):
    device = config.training.device
    use_forge = config.training.use_forge

    system = EfficientNeRFSystem(config).to(device)
    checkpoint_dir = os.path.join(config.training.log_dir, config.experiment_name, "ckpts")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="nerf-epoch-{epoch}-step-{step:06d}-val_psnr-{val/psnr:.2f}",
        monitor="val/psnr",
        mode="max",
        every_n_train_steps=500,
        save_top_k=5,
        save_on_train_epoch_end=True,
        save_last=True,
        auto_insert_metric_name=False,
    )

    wandb_id = str(hash(str(config.model_dump())))[-10:]

    experiment_name_suffix = "_forge" if use_forge else ("_cpu" if device == "cpu" else "_gpu")
    wandb.init(
        project="blacksmith-nerf",
        config=config,
        name=config.experiment_name + experiment_name_suffix,
        mode="online" if not config.training.val_only else "disabled",
        id=wandb_id,
        resume="auto",
    )

    logger = WandbLogger(
        save_dir=config.training.log_dir,
        name=config.experiment_name,
    )
    logger.log_hyperparams(config.model_dump())

    trainer = Trainer(
        max_epochs=config.training.epochs,
        logger=logger,
        num_sanity_val_steps=0,
        accelerator=device,
        callbacks=[checkpoint_callback],
        log_every_n_steps=config.training.log_every,
        deterministic=False,
    )

    if config.training.val_only:
        trainer.validate(system, ckpt_path=config.training.ckpt_path)
        return
    trainer.fit(system, ckpt_path=config.training.ckpt_path)


if __name__ == "__main__":
    import os

    config_file_path = os.path.join(os.path.dirname(__file__), "test_nerf.yaml")
    config = load_config(config_file_path)
    main(config)
