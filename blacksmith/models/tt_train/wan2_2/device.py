# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from typing import Any

import ttml

MGD_WARNING = """
================================================================================
  WARNING: TT_MESH_GRAPH_DESC_PATH is NOT set!

  Distributed mode requires a Mesh Graph Descriptor (MGD) file.
  enable_fabric() will attempt automatic selection for 8 or 32 devices,
  but this may not match your hardware topology.

  For reliable operation, set the environment variable explicitly:

      export TT_MESH_GRAPH_DESC_PATH="/path/to/your/mesh_graph_descriptor.textproto"

  Common MGD files (relative to $TT_METAL_HOME):
      tt-train/configs/mgd/bh_galaxy_4_8_line_line.textproto
      tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_1x8_mesh_graph_descriptor.textproto

  See: https://github.com/tenstorrent/tt-metal/blob/main/tt-train/docs/DISTRIBUTED_TRAINING.md#setting-mgd-files-via-environment-variable
================================================================================
"""


def setup_device(dp_size: int, tp_size: int, seed: int = 42, logger: Any = None) -> tuple[Any, Any]:
    distributed = dp_size > 1 or tp_size > 1
    if distributed and logger is not None:
        if "TT_MESH_GRAPH_DESC_PATH" not in os.environ:
            logger.warning(MGD_WARNING)
        logger.info(
            f"Enabling distributed mode: DP={dp_size}, TP={tp_size} "
            f"({dp_size * tp_size} devices, mesh [{dp_size}, {tp_size}])"
        )

    ttml.open_device_mesh(ttml.Mesh((dp_size, tp_size), ("dp", "tp")))

    ctx = ttml.autograd.AutoContext.get_instance()
    if distributed:
        ctx.initialize_parallelism_context(
            ttml.autograd.DistributedConfig(enable_ddp=dp_size > 1, enable_tp=tp_size > 1)
        )
    ctx.set_seed(seed)
    return ctx, ctx.get_device()
