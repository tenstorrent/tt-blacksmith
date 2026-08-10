# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn

from blacksmith.experiments.torch.wan2_2.configs import TrainingConfig
from blacksmith.tools.device_manager import DeviceManager

# TT backend (dynamo) compile knobs. Accept Python types (not the XLA options).
_TORCH_COMPILE_OPTIONS = {
    "tt_enable_torch_fx_fusion_pass": False,
    "tt_legacy_compile": True,
    "tt_enable_composite_ops": True,
    "tt_use_aot_autograd": False,
}

# XLA custom compile options. Values must be strings (the API does not coerce bool/int).
_XLA_COMPILE_OPTIONS = {
    "optimization_level": "0",
    "fp32_dest_acc_en": "true",
    "math_fidelity": "hifi4",
    "experimental-enable-dram-space-saving-optimization": "true",
}


class WanDeviceManager(DeviceManager):
    """DeviceManager extended with cached `torch.compile(backend="tt")` wrappers.
    Mesh/device/optimizer_step and the regex-based `shard_model` come from the
    shared base; sharding for every Wan component (UMT5, VAE, DiT) is driven by
    `model_sharding_patterns`/`param_sharding_patterns` in the YAML.
    """

    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self._compile_cache: dict = {}
        # Passed to torch_xla.set_custom_compile_options(...) in the entry point.
        self.xla_compile_options: dict = dict(_XLA_COMPILE_OPTIONS)

    def to_device(self, module_or_tensor):
        return module_or_tensor.to(self.device)

    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Backend-specific graph rewrites before a model is moved/sharded.

        Nothing to do on tt-xla: the patches this backend needs are class-level and are
        applied by `model_overrides.apply_generality_overrides()` at start-up. The hook
        exists so the shared code (`generate.py`) can support backends that must rewrite
        module instances -- tt-kurbla retypes the VAE's Conv3d/ZeroPad2d here.
        """
        return model

    def compile(self, module: nn.Module):
        # Cached on id(module); callers must keep wrappers alive across calls.
        if not self.config.use_tt:
            return module
        cached = self._compile_cache.get(id(module))
        if cached is None:
            cached = torch.compile(module, backend="tt", options=_TORCH_COMPILE_OPTIONS)
            self._compile_cache[id(module)] = cached
        return cached

    def sync(self) -> None:
        if not self.config.use_tt:
            return
        import torch_xla

        torch_xla.sync(wait=True)
