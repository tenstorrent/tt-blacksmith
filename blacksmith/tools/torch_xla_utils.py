# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Tuple

from blacksmith.experiments.torch.mnist.configs import TrainingConfig
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
import numpy as np
import os


def setup_tt_environment(config: TrainingConfig):
    # Setup for single device
    xr.set_device_type("TT")
    os.environ["PJRT_DEVICE"] = "TT"
    os.environ["XLA_STABLEHLO_COMPILE"] = "1"

    # Additional setup for multichip
    if config.parallelism != "single":
        os.environ["XLA_ALWAYS_ALLREDUCE"] = "1"
        os.environ["MESH_SHAPE"] = config.mesh_shape
        os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
        os.environ["DISABLE_NUMERIC_CC_TOKEN"] = "1"
        xr.use_spmd()


def get_mesh(config: TrainingConfig) -> xs.Mesh:
    # TODO: Extend this for other multichip setups once we have them.
    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh_shape = None
    axis_names = None

    if config.parallelism == "data":
        mesh_shape = (num_devices, 1)
        axis_names = ("data", "model")
    elif config.parallelism == "tensor":
        mesh_shape = (num_devices,)
        axis_names = ("model",)
    else:
        raise ValueError(f"Invalid parallelism: {config.parallelism}")

    mesh = xs.Mesh(device_ids=device_ids, mesh_shape=mesh_shape, axis_names=axis_names)
    return mesh
