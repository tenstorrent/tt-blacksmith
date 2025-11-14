# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import List
from pydantic import BaseModel, Field

from blacksmith.tools.logging.configs import LoggerConfig, get_default_logger_config
from torch_xla.experimental import plugins
import os
import torch_xla
import sys

os.environ["PJRT_DEVICE"] = "TT"
os.environ["XLA_STABLEHLO_COMPILE"] = "1"

DEFAULT_PJRT_PATH = "third_party/tt-xla/build/src/tt/pjrt_plugin_tt.so"


class TTPjrtPlugin(plugins.DevicePlugin):
    def __init__(self, plugin_path: str) -> None:
        self._plugin_path = plugin_path
        super().__init__()

    def library_path(self):
        return self._plugin_path


def init_device(plugin_path: str = DEFAULT_PJRT_PATH):
    backend = "TT"
    path = os.path.join(os.getcwd(), plugin_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find tt_pjrt C API plugin at {path}")

    plugin = TTPjrtPlugin(path)
    plugins.register_plugin(backend, plugin)
    print("Loaded", file=sys.stderr)


def setup_multi_chip_environment(config):
    import torch_xla.runtime as xr
    os.environ["PJRT_DEVICE"] = "TT"
    os.environ["XLA_STABLEHLO_COMPILE"] = "1"
    os.environ["XLA_ALWAYS_ALLREDUCE"] = "1"
    os.environ["MESH_SHAPE"] = config.mesh_shape
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    os.environ["DISABLE_NUMERIC_CC_TOKEN"] = "1"
    xr.set_device_type("TT")
    xr.use_spmd()


def get_mesh(config):
    # TODO: Extend this for other multichip setups once we have them.
    import torch_xla.runtime as xr
    from torch_xla.distributed.spmd import Mesh
    import numpy as np

    if config.parallelism != "single":
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (num_devices, 1)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ('data', 'model'))
    else:
        mesh = None
    return mesh

