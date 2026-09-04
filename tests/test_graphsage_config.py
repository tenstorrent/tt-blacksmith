# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from blacksmith.experiments.torch.BOUNTIES.graphsage_reddit.configs import (
    GraphSAGEConfig,
)

pytestmark = [
    pytest.mark.push,
    pytest.mark.n300,
    pytest.mark.torch,
    pytest.mark.single_chip,
    pytest.mark.pyg,
]

CONFIG_DIR = Path("blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/single_chip")


def _load_config(filename: str) -> GraphSAGEConfig:
    with (CONFIG_DIR / filename).open() as config_file:
        return GraphSAGEConfig.model_validate(yaml.safe_load(config_file))


def test_tt_execution_requires_spmm_and_static_shapes() -> None:
    with pytest.raises(ValidationError, match="TT execution requires"):
        GraphSAGEConfig(use_tt=True, use_spmm=False, static_shapes=True)

    with pytest.raises(ValidationError, match="TT execution requires"):
        GraphSAGEConfig(use_tt=True, use_spmm=True, static_shapes=False)


def test_tt_execution_uses_validated_static_shape_defaults() -> None:
    config = GraphSAGEConfig(use_tt=True, use_spmm=True, static_shapes=True)
    assert config.batch_size == 32
    assert config.val_batch_size == 32
    assert config.num_neighbors == [5, 3]


def test_run_configs_use_isolated_project_directories() -> None:
    configs = {
        "graphsage_reddit.yaml": _load_config("graphsage_reddit.yaml"),
        "graphsage_reddit_spmm_cpu.yaml": _load_config("graphsage_reddit_spmm_cpu.yaml"),
        "graphsage_reddit_tt.yaml": _load_config("graphsage_reddit_tt.yaml"),
    }

    assert {config.project_dir for config in configs.values()} == {
        "blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/runs/stock_cpu",
        "blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/runs/matched_cpu",
        "blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/runs/tt",
    }

    stock_cpu = configs["graphsage_reddit.yaml"]
    assert stock_cpu.batch_size == 512
    assert stock_cpu.val_batch_size == 4096
    assert stock_cpu.num_neighbors == [25, 10]
