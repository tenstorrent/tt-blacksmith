# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
import imageio
import numpy as np
from pydantic import BaseModel
import torch
import torch.utils
import torch.utils
from tqdm import tqdm, trange
from thomas.models.torch.nerf import *
from thomas.tooling.cli import generate_config
from thomas.tooling.forge_tooling import disable_forge_logger
from thomas.tooling.utils.nerf import *

from torch import nn, optim
from thomas.tooling.cli import generate_config, print_trainable_params
from thomas.tooling.utils.nerf.render import render_path, render_scene
from thomas.tooling.data.nerf import NeRFDataset


class NerfDataLoadingConfig(BaseModel):
    basedir: str
    datadir: str
    lrate_decay: int
    N_samples: int
    N_importance: int
    N_rand: int
    precrop_iters: int
    precrop_frac: float
    half_res: bool
    chunk: int = 1024
    netchunk: int = 8192
    testskip: int = 8
    perturb: float = 1.0
    render_factor: int = 4


class ModelConfig(BaseModel):
    path: Optional[str] = None
    render_only: bool
    multires: int = 10
    multires_views: int = 4
    netdepth: int = 8
    netwidth: int = 256
    netdepth_fine: int = 8
    netwidth_fine: int = 256


class TrainingConfig(BaseModel):
    lrate: float = 5e-4
    tt: bool = False


class ExperimentConfig(BaseModel):
    experiment_name: str
    tags: List[str]
    model: ModelConfig
    training_config: TrainingConfig
    data_loading: NerfDataLoadingConfig


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    target_rows = embedded.shape[0]
    outputs_flat = outputs_flat[:target_rows, :]
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def create_network_functions(
    model: nn.Module,
    training_config: TrainingConfig,
    netchunk: int,
) -> Tuple[Callable, Optional[Callable]]:
    if training_config.tt:
        import forge

        sample_inputs = [torch.randn((netchunk, 84))]
        tt_model = forge.compile(model, sample_inputs=sample_inputs, training=True)
        model_fn = lambda x: tt_model(x)[0]
    else:
        model_fn = model

    return model_fn


def create_render_kwargs(
    network_query_fn: Callable,
    model_fn: Callable,
    data_config: NerfDataLoadingConfig,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Create render kwargs for training and testing."""
    render_kwargs_train = {
        "network_fn": model_fn,
        "network_query_fn": network_query_fn,
        "perturb": data_config.perturb,
        "N_importance": data_config.N_importance,
        "N_samples": data_config.N_samples,
        "lindisp": True,
    }

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test["perturb"] = False

    return render_kwargs_train, render_kwargs_test


def save_checkpoint(basedir, expname, global_step, render_kwargs_train, optimizer, tt=False):
    path = os.path.join(basedir, expname, "nerf.tar")
    network_state_dict = render_kwargs_train["model"].state_dict()
    torch.save(
        {
            "global_step": global_step,
            "network_fn_state_dict": network_state_dict,
            "network_fine_state_dict": network_fine_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    print("Saved checkpoints at", path)


def init_nerf_components(
    args: ExperimentConfig,
) -> Tuple[Dict[str, Any], Dict[str, Any], int, List[nn.Parameter], optim.Optimizer]:
    """
    Initialize all components required for NeRF training.
    The components include:
    - Embedding functions for positions and view directions
    - Coarse and fine NeRF models
    - Optimizer
    - Network functions
    - Render kwargs for training and testing
    """

    # Initialize embedders
    embed_fn, embeddirs_fn, input_ch, input_ch_views = initialize_embedders(
        args.model.multires,
        args.model.multires_views,
    )

    # Create models
    model = create_models(
        depth=args.model.netdepth,
        width=args.model.netwidth,
        input_ch=input_ch,
        input_ch_views=input_ch_views,
        depth_fine=args.model.netdepth_fine,
        width_fine=args.model.netwidth_fine,
    )

    # Create optimizer
    grad_vars = list(model.parameters())

    optimizer = torch.optim.Adam(params=grad_vars, lr=args.training_config.lrate, betas=(0.9, 0.999))

    # Load checkpoint if provided
    start = 0
    if args.model.path is not None and args.model.path != "None":
        ckpts = [args.model.path]
        if len(ckpts) > 0:
            start = load_checkpoint(ckpts[-1], model, optimizer)

    # Create network functions
    model_fn = create_network_functions(model, args.training_config, args.data_loading.netchunk)

    # Create network query function
    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(
        inputs,
        viewdirs,
        network_fn,
        embed_fn=embed_fn,
        embeddirs_fn=embeddirs_fn,
        netchunk=args.data_loading.netchunk,
    )

    # Create render kwargs
    render_kwargs_train, render_kwargs_test = create_render_kwargs(network_query_fn, model_fn, args.data_loading)

    render_kwargs_train["model"] = model
    render_kwargs_test["model"] = model

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def train():
    config: ExperimentConfig = generate_config(ExperimentConfig, "thomas/experiments/test_nerf.yaml")

    near = 2.0
    far = 6.0

    basedir = config.data_loading.basedir
    expname = config.experiment_name

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = init_nerf_components(config)

    bds_dict = {
        "near": near,
        "far": far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    batch_size = config.data_loading.N_rand

    dataset = NeRFDataset(
        config.data_loading.datadir,
        config.data_loading.half_res,
        config.data_loading.testskip,
    )
    epochs = (200_000 + 1) // batch_size

    start = 0
    global_step = 0
    print("Starting training")
    disable_forge_logger()
    for epoch in tqdm(range(epochs)):
        data = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for batch in data:
            optimizer.zero_grad()
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            # print shapes
            print("batch_rays shape", batch_rays.shape)
            print("target_s shape", target_s.shape)

            rgb, disp, acc, extras = render_scene(
                dataset.H,
                dataset.W,
                dataset.K,
                chunk=config.data_loading.chunk,
                rays=batch_rays,
                retraw=True,
                **render_kwargs_train,
            )

            loss = torch.mean(torch.sum((rgb - target_s) ** 2, -1))

            if "rgb0" in extras:
                loss += img2mse(extras["rgb0"], target_s)

            loss.backward()
            optimizer.step()
            print("Loss", loss.item())

            decay_rate = 0.1
            decay_steps = config.data_loading.lrate_decay * 1000
            new_lrate = config.training_config.lrate * (decay_rate ** (global_step / decay_steps))
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lrate
            global_step += 1

    import pdb

    pdb.set_trace()

    # save_checkpoint(basedir, expname, global_step, render_kwargs_train, optimizer, config.training_config.tt)

    # ovaj deo je dobar
    with torch.no_grad():
        print("USAO SAM U RENDER TEST")
        images = dataset.images[dataset.test_split]

        testsavedir = os.path.join(basedir, expname, "renderonly")
        os.makedirs(testsavedir, exist_ok=True)
        print("test poses shape", dataset.render_poses.shape)

        rgbs, _ = render_path(
            dataset.render_poses,
            dataset.hwf,
            dataset.K,
            config.data_loading.chunk,
            render_kwargs_test,
            savedir=testsavedir,
            render_factor=config.data_loading.render_factor,
            debug=False,
        )
        print("Done rendering", testsavedir)
        imageio.mimwrite(os.path.join(testsavedir, "video.mp4"), to8b(rgbs), fps=30, quality=8)


if __name__ == "__main__":
    train()
