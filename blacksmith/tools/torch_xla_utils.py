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
