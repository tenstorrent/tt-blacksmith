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


def init_device(plugin_path: str):
    backend = "TT"
    path = os.path.join(os.getcwd(), plugin_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find tt_pjrt C API plugin at {path}")

    plugin = plugins.DevicePlugin(library_path=path)
    plugins.register_plugin(backend, plugin)
    print("Loaded", file=sys.stderr)