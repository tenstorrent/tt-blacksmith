# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import importlib
import importlib.util
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytestmark = [
    pytest.mark.push,
    pytest.mark.n300,
    pytest.mark.torch,
    pytest.mark.single_chip,
    pytest.mark.pyg,
]


@pytest.fixture(scope="module")
def train_module():
    """Import the training module without requiring torch-xla on CPU-only hosts."""
    if importlib.util.find_spec("torch_xla") is not None:
        yield importlib.import_module("blacksmith.experiments.torch.BOUNTIES.graphsage_reddit.train")
        return

    torch_xla = ModuleType("torch_xla")
    torch_xla.__path__ = []
    torch_xla.sync = Mock()
    torch_xla.device = Mock(return_value=torch.device("cpu"))

    core = ModuleType("torch_xla.core")
    core.__path__ = []
    xla_model = ModuleType("torch_xla.core.xla_model")
    xla_model.optimizer_step = Mock()
    distributed = ModuleType("torch_xla.distributed")
    distributed.__path__ = []
    spmd = ModuleType("torch_xla.distributed.spmd")
    spmd.Mesh = object
    runtime = ModuleType("torch_xla.runtime")
    experimental = ModuleType("torch_xla.experimental")
    experimental.__path__ = []
    fsdp = ModuleType("torch_xla.experimental.spmd_fully_sharded_data_parallel")
    fsdp.SpmdFullyShardedDataParallel = object

    stubs = {
        "torch_xla": torch_xla,
        "torch_xla.core": core,
        "torch_xla.core.xla_model": xla_model,
        "torch_xla.distributed": distributed,
        "torch_xla.distributed.spmd": spmd,
        "torch_xla.runtime": runtime,
        "torch_xla.experimental": experimental,
        "torch_xla.experimental.spmd_fully_sharded_data_parallel": fsdp,
    }
    imported_with_stubs = (
        "blacksmith.experiments.torch.BOUNTIES.graphsage_reddit.train",
        "blacksmith.tools.checkpoints_manager",
        "blacksmith.tools.device_manager",
        "blacksmith.tools.workaround_utils",
    )
    previous_modules = {name: sys.modules.get(name) for name in (*stubs, *imported_with_stubs)}
    sys.modules.update(stubs)

    try:
        yield importlib.import_module("blacksmith.experiments.torch.BOUNTIES.graphsage_reddit.train")
    finally:
        for name in (*imported_with_stubs, *stubs):
            previous = previous_modules[name]
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def test_resume_training_state_restores_checkpoint_progress(train_module) -> None:
    config = SimpleNamespace(resume_from_checkpoint=True)
    checkpoint_manager = Mock()
    checkpoint_manager.load_checkpoint.return_value = {
        "step": 120,
        "epoch": 3,
        "metrics": {},
    }
    model = Mock()
    optimizer = Mock()
    logger = Mock()

    progress = train_module._resume_training_state(config, checkpoint_manager, model, optimizer, logger)

    assert progress == (120, 3)
    checkpoint_manager.load_checkpoint.assert_called_once_with(model, optimizer)
    logger.info.assert_called_once_with("Resuming after epoch 3 at step 120")


def test_resume_training_state_starts_fresh_when_checkpoint_is_missing(
    train_module,
) -> None:
    config = SimpleNamespace(resume_from_checkpoint=True)
    checkpoint_manager = Mock()
    checkpoint_manager.load_checkpoint.return_value = None

    progress = train_module._resume_training_state(config, checkpoint_manager, Mock(), Mock(), Mock())

    assert progress == (0, 0)


def test_train_continues_after_completed_checkpoint_epoch(train_module, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDataset:
        num_nodes = 10
        num_edges = 20
        num_features = 4
        num_classes = 2
        train_nodes = 6
        val_nodes = 2
        test_nodes = 2

        def get_neighbour_loader(self, split: str) -> list:
            return []

    class FakeModel:
        def to(self, device: torch.device):
            return self

        def parameters(self) -> list:
            return []

    config = SimpleNamespace(
        static_shapes=False,
        hidden_channels=8,
        dropout=0.0,
        use_spmm=True,
        learning_rate=0.001,
        weight_decay=0.0,
        use_tt=False,
        resume_from_checkpoint=True,
        num_epochs=5,
        val_batch_size=2,
        save_strategy="none",
    )
    checkpoint_manager = Mock()
    checkpoint_manager.load_checkpoint.return_value = {
        "step": 120,
        "epoch": 3,
        "metrics": {},
    }
    checkpoint_manager.should_save_checkpoint.return_value = False
    observed_progress = []

    def fake_train_epoch(*args):
        epoch = args[-2]
        global_step = args[-1]
        observed_progress.append((epoch, global_step))
        return global_step + 10, 0.5

    monkeypatch.setattr(train_module, "RedditDataset", lambda _: FakeDataset())
    monkeypatch.setattr(train_module, "GraphSAGE", lambda **_: FakeModel())
    monkeypatch.setattr(train_module.torch.optim, "Adam", lambda *_, **__: Mock())
    monkeypatch.setattr(train_module, "evaluate", lambda *_, **__: (0.25, 0.75))
    monkeypatch.setattr(train_module, "train_epoch", fake_train_epoch)

    train_module.train(
        config,
        SimpleNamespace(device=torch.device("cpu")),
        Mock(),
        checkpoint_manager,
    )

    assert observed_progress == [(4, 120), (5, 130)]


def test_evaluate_rejects_an_empty_loader(train_module) -> None:
    model = Mock()

    with pytest.raises(ValueError, match="evaluation loader yielded no seed nodes"):
        train_module.evaluate(
            model,
            [],
            SimpleNamespace(device=torch.device("cpu")),
            SimpleNamespace(),
            seed_capacity=1,
        )

    model.eval.assert_called_once_with()
