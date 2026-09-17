# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

# The experiment config only exists once that experiment is ported to this tree; it is
# only needed here as a type hint.
if TYPE_CHECKING:
    from blacksmith.experiments.torch.wan2_2.configs import TrainingConfig

from blacksmith.tools.device_manager import DeviceManager

# tt-crank compile options come from the config (`DeviceManager.compile_options()`); the tt-xla
# dynamo knobs (`tt_legacy_compile`, `tt_enable_composite_ops`, ...) and the string-valued XLA
# custom options have no tt-crank counterpart.


class WanDeviceManager(DeviceManager):
    """DeviceManager extended with cached `torch.compile(backend="tt")` wrappers.
    Mesh/device/optimizer_step and the regex-based `shard_model` come from the
    shared base; sharding for every Wan component (UMT5, VAE, DiT) is driven by
    `model_sharding_patterns`/`param_sharding_patterns` in the YAML.
    """

    def __init__(self, config: "TrainingConfig"):
        super().__init__(config)
        self._compile_cache: dict = {}

    def to_device(self, module_or_tensor):
        return module_or_tensor.to(self.device)

    def compile(self, module: nn.Module):
        # Cached on id(module); callers must keep wrappers alive across calls.
        if not self.config.use_tt:
            return module
        cached = self._compile_cache.get(id(module))
        if cached is None:
            cached = torch.compile(module, backend="tt", options=self.compile_options())
            self._compile_cache[id(module)] = cached
        return cached

    def sync(self) -> None:
        # tt-crank executes eagerly; kept so the Wan entry point ports unchanged.
        return
