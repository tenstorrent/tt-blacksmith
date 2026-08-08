# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from pydantic import ValidationError

from blacksmith.experiments.torch.BOUNTIES.graphsage_reddit.configs import (
    GraphSAGEConfig,
)


def test_tt_execution_requires_spmm_and_static_shapes() -> None:
    with pytest.raises(ValidationError, match="TT execution requires"):
        GraphSAGEConfig(use_tt=True, use_spmm=False, static_shapes=True)

    with pytest.raises(ValidationError, match="TT execution requires"):
        GraphSAGEConfig(use_tt=True, use_spmm=True, static_shapes=False)


def test_tt_execution_accepts_fixed_shape_spmm() -> None:
    config = GraphSAGEConfig(use_tt=True, use_spmm=True, static_shapes=True)
    assert config.num_neighbors == [25, 10]
