# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict

import ttml

from blacksmith.configs.trainer import TrainerConfig
from blacksmith.tools.device_manager import DeviceManager
from blacksmith.tools.ttml.ttml_helpers import to_ttml


class TtmlDeviceManager(DeviceManager):
    """Owns the ttml device: a single chip, opened here and released by ``close``."""

    def __init__(self, config: TrainerConfig):
        super().__init__(config)
        if not config.use_tt:
            raise ValueError("ttml has no host backend; set `use_tt: True`.")
        multichip = {
            name: getattr(config, name)
            for name in ("mesh_shape", "mesh_axis_names", "input_sharding_dim", "model_sharding_patterns")
            if getattr(config, name)
        }
        if multichip:
            raise NotImplementedError(f"The ttml trainer runs on a single chip; unset {sorted(multichip)}.")

        self._closed = False
        # Single chip: no fabric. Multichip later opens ttml.Mesh(shape, ("dp", "tp", ...)) here.
        ttml.open_device_mesh(ttml.Mesh((1, 1), ("_0", "_1")))
        try:
            self.mesh = ttml.mesh()
            self.device = ttml.autograd.AutoContext.get_instance().get_device()
        except BaseException:
            # Never leave a half-opened device behind: the trainer has no handle on it yet.
            self.close()
            raise

    def shard_model(self, model: Any) -> Any:
        return model

    def prepare_batch(self, batch: Dict[str, Any], skip_keys: tuple[str, ...] = ()) -> Dict[str, Any]:
        """Upload a host (numpy) batch from ``collate_causal_lm`` as ttml tensors."""
        return {key: value if key in skip_keys else to_ttml(key, value) for key, value in batch.items()}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.mesh = None
        self.device = None
        ttml.autograd.AutoContext.get_instance().reset_graph()
        # Closes the MeshDevice, disables fabric and resets ttml's global mesh; safe if nothing is open.
        ttml.close_device_mesh()
