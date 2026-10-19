# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from blacksmith.configs import CheckpointConfig
from blacksmith.tools.crank.checkpoint_manager import CrankCheckpointManager
from blacksmith.tools.trainer.callbacks.checkpoint_callback import CheckpointCallback


def make_config(project_dir, **overrides) -> CheckpointConfig:
    fields = dict(
        steps_freq=5,
        epoch_freq=1,
        save_strategy="step",
        project_dir=str(project_dir),
        final_checkpoint_name="final_model.pth",
        save_optim=True,
        keep_last_n=10,
        keep_best_n=10,
        checkpoint_metric="val/loss",
        checkpoint_metric_mode="min",
        init_from_checkpoint=False,
        init_option="path",
        checkpoint_path="",
    )
    return CheckpointConfig(**(fields | overrides))


def make_model(seed: int, trainable_head: bool = False) -> torch.nn.Module:
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    model[0].requires_grad_(False)
    model[2].requires_grad_(trainable_head)
    return model


def make_trainer(model: torch.nn.Module, config: CheckpointConfig) -> SimpleNamespace:
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return SimpleNamespace(
        config=SimpleNamespace(checkpoint=config),
        logger=None,
        model=model,
        optimizer=torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.0),
        create_checkpoint_manager=lambda config, logger=None: CrankCheckpointManager(config, logger=logger),
    )


def save_source_checkpoint(project_dir) -> tuple[str, torch.nn.Module]:
    """Save a checkpoint with optimizer state from settings that differ from `make_trainer`'s."""
    model = make_model(seed=0)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.01)
    model(torch.randn(2, 4)).sum().backward()
    optimizer.step()
    path = CrankCheckpointManager(make_config(project_dir)).save_checkpoint(model, step=1, optimizer=optimizer)
    return path, model


def test_init_loads_trainable_weights_only(tmp_path):
    path, source = save_source_checkpoint(tmp_path / "source")
    config = make_config(tmp_path / "run", init_from_checkpoint=True, checkpoint_path=path)
    trainer = make_trainer(make_model(seed=1), config)

    CheckpointCallback().on_train_start(trainer)

    assert torch.equal(trainer.model[1].weight, source[1].weight)
    assert not torch.equal(trainer.model[0].weight, source[0].weight)
    assert not trainer.optimizer.state
    assert trainer.optimizer.param_groups[0]["weight_decay"] == 0.0


def test_init_rejects_checkpoint_missing_trainable_params(tmp_path):
    path, _ = save_source_checkpoint(tmp_path / "source")
    config = make_config(tmp_path / "run", init_from_checkpoint=True, checkpoint_path=path)
    trainer = make_trainer(make_model(seed=1, trainable_head=True), config)

    with pytest.raises(AssertionError, match=r"missing trainable parameters: \['2.bias', '2.weight'\]"):
        CheckpointCallback().on_train_start(trainer)
