# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import importlib
import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch_geometric.data import Data

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

    def fake_train_epoch(*args, **kwargs):
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


@pytest.fixture
def tiny_training(train_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Exercise real training and checkpoint I/O without downloading Reddit."""
    batch = Data(
        x=torch.tensor([[1.0, 0.0, 0.5, -0.5], [0.0, 1.0, -0.5, 0.5]]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        y=torch.tensor([0, 1]),
        batch_size=2,
    )
    dataset = SimpleNamespace(
        num_nodes=2,
        num_edges=2,
        num_features=4,
        num_classes=2,
        train_nodes=2,
        val_nodes=2,
        test_nodes=2,
        get_neighbour_loader=lambda _: [batch] * 3,
    )
    config = train_module.GraphSAGEConfig(
        hidden_channels=4,
        dropout=0.0,
        use_tt=False,
        use_spmm=True,
        batch_size=2,
        val_batch_size=2,
        num_neighbors=[1, 1],
        num_epochs=1,
        steps_freq=2,
        epoch_freq=1,
        save_optim=True,
        keep_last_n=20,
        project_dir=str(tmp_path),
    )
    model_factory = train_module.GraphSAGE
    optimizer_factory = torch.optim.Adam
    models, optimizers = [], []

    def make_model(**kwargs):
        model = model_factory(**kwargs)
        models.append(model)
        return model

    def make_optimizer(*args, **kwargs):
        optimizer = optimizer_factory(*args, **kwargs)
        optimizers.append(optimizer)
        return optimizer

    monkeypatch.setattr(train_module, "RedditDataset", lambda _: dataset)
    monkeypatch.setattr(train_module, "GraphSAGE", make_model)
    monkeypatch.setattr(train_module.torch.optim, "Adam", make_optimizer)
    device_manager = SimpleNamespace(device=torch.device("cpu"), optimizer_step=lambda optimizer: optimizer.step())
    manager = train_module.CheckpointManager(config, Mock(), device_manager.device)
    return SimpleNamespace(
        config=config,
        device_manager=device_manager,
        manager=manager,
        batch=batch,
        models=models,
        optimizers=optimizers,
        make_model=lambda: model_factory(4, 4, 2, dropout=0.0, use_spmm=True),
        make_optimizer=lambda model: optimizer_factory(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        ),
    )


@pytest.mark.parametrize("save_optim", [True, False])
def test_final_checkpoint_round_trip(
    train_module, tiny_training, monkeypatch: pytest.MonkeyPatch, save_optim: bool
) -> None:
    run = tiny_training
    run.config.save_optim = save_optim
    run.config.keep_last_n = 1
    run.config.keep_best_n = 0
    # Force epoch and final saves into the same timestamp to exercise retention
    # even on a slow runner; their filenames must remain distinct.
    checkpoint_module = importlib.import_module(type(run.manager).__module__)
    frozen_datetime = Mock()
    frozen_datetime.now.return_value = datetime(2026, 1, 1)
    monkeypatch.setattr(checkpoint_module, "datetime", frozen_datetime)
    train_module.train(run.config, run.device_manager, Mock(), run.manager)

    checkpoint_path = run.manager.checkpoint_history["checkpoints"][-1]["path"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert ("optimizer_state_dict" in checkpoint) is save_optim

    restored_model = run.make_model()
    restored_optimizer = run.make_optimizer(restored_model)
    run.config.resume_from_checkpoint = True
    restored_manager = train_module.CheckpointManager(run.config, Mock(), run.device_manager.device)
    assert train_module._resume_training_state(
        run.config, restored_manager, restored_model, restored_optimizer, Mock()
    ) == (3, 1)

    for expected, restored in zip(run.models[0].parameters(), restored_model.parameters()):
        torch.testing.assert_close(restored, expected, rtol=0, atol=0)

    if not save_optim:
        assert not restored_optimizer.state
        return

    # A real next Adam update must agree with uninterrupted training, including
    # the moment estimates and step counters, not only the restored weights.
    for model, optimizer in ((run.models[0], run.optimizers[0]), (restored_model, restored_optimizer)):
        model.train()
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(model(run.batch.x, run.batch.edge_index), run.batch.y).backward()
        optimizer.step()
    for expected, restored in zip(run.models[0].parameters(), restored_model.parameters()):
        torch.testing.assert_close(restored, expected, rtol=0, atol=0)
    assert all(state["step"].item() == 4 for state in restored_optimizer.state.values())


@pytest.mark.parametrize(
    ("strategy", "expected_steps", "expected_epochs"),
    [
        ("step", [2, 4, 6, 6], [0, 1, 1, 2]),
        ("epoch", [3, 6, 6], [1, 2, 2]),
        ("none", [], []),
    ],
)
def test_checkpoint_strategy_schedule(
    train_module, tiny_training, monkeypatch: pytest.MonkeyPatch, strategy, expected_steps, expected_epochs
) -> None:
    run = tiny_training
    run.config.save_strategy = strategy
    run.config.num_epochs = 2
    saved = []
    real_save = run.manager.save_checkpoint

    def record_save(*args, **kwargs):
        path = real_save(*args, **kwargs)
        saved.append(torch.load(path, map_location="cpu", weights_only=True))
        return path

    monkeypatch.setattr(run.manager, "save_checkpoint", record_save)
    train_module.train(run.config, run.device_manager, Mock(), run.manager)

    assert [checkpoint["step"] for checkpoint in saved] == expected_steps
    assert [checkpoint["epoch"] for checkpoint in saved] == expected_epochs
    for checkpoint in saved:
        assert all(
            state["step"].item() == checkpoint["step"] for state in checkpoint["optimizer_state_dict"]["state"].values()
        )
    if strategy == "none":
        assert not list(Path(run.manager.checkpoint_dir).glob("*.pt"))


def test_step_checkpoint_resume_restarts_unfinished_epoch(train_module, tiny_training) -> None:
    run = tiny_training
    run.config.save_strategy = "step"
    train_module.train(run.config, run.device_manager, Mock(), run.manager)
    intermediate = next(cp for cp in run.manager.checkpoint_history["checkpoints"] if cp["step"] == 2)

    run.config.resume_from_checkpoint = True
    run.config.resume_option = "path"
    run.config.checkpoint_path = intermediate["path"]
    train_module.train(run.config, run.device_manager, Mock(), run.manager)

    # The interrupted epoch was not completed. Re-run it from the saved model
    # and Adam state rather than incorrectly skipping it as an epoch checkpoint.
    final = run.manager.checkpoint_history["checkpoints"][-1]
    assert (final["step"], final["epoch"]) == (5, 1)
    checkpoint = torch.load(final["path"], map_location="cpu", weights_only=True)
    assert all(state["step"].item() == 5 for state in checkpoint["optimizer_state_dict"]["state"].values())


@pytest.mark.parametrize("checkpoint_path", [None, "saved-model.pt"])
def test_cli_forwards_checkpoint_path(train_module, monkeypatch: pytest.MonkeyPatch, checkpoint_path) -> None:
    config_path = Path(train_module.__file__).parent / "single_chip" / "graphsage_reddit_spmm_cpu.yaml"
    argv = ["train.py", "--config", str(config_path)]
    if checkpoint_path is not None:
        argv.extend(["--test-checkpoint-path", checkpoint_path])
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(train_module, "ReproducibilityManager", Mock())
    monkeypatch.setattr(train_module, "TrainingLogger", Mock())
    monkeypatch.setattr(train_module, "DeviceManager", Mock())
    monkeypatch.setattr(train_module, "CheckpointManager", Mock())
    training = Mock()
    monkeypatch.setattr(train_module, "train", training)

    train_module.main()

    config = training.call_args.args[0]
    assert config.resume_from_checkpoint is (checkpoint_path is not None)
    assert config.checkpoint_path == (checkpoint_path or "")
    assert config.resume_option == ("path" if checkpoint_path else "last")
