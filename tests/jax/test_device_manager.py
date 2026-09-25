# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import logging
import os
from types import SimpleNamespace

from blacksmith.tools.jax.device_manager import JaxDeviceManager


def make_device_manager(use_tt: bool) -> JaxDeviceManager:
    """Build a JaxDeviceManager without running __init__ (no device or mesh needed)."""
    device_manager = object.__new__(JaxDeviceManager)
    device_manager.config = SimpleNamespace(use_tt=use_tt, num_devices=1)
    return device_manager


def test_setup_env_sets_pjrt_device_when_unset(monkeypatch):
    monkeypatch.delenv("PJRT_DEVICE", raising=False)

    make_device_manager(use_tt=True)._setup_env()

    assert os.environ["PJRT_DEVICE"] == "TT"


def test_setup_env_overrides_stale_pjrt_device(monkeypatch, caplog):
    # `env/activate --gpu` exports PJRT_DEVICE=CUDA, which used to leak into TT runs.
    monkeypatch.setenv("PJRT_DEVICE", "CUDA")

    with caplog.at_level(logging.WARNING):
        make_device_manager(use_tt=True)._setup_env()

    assert os.environ["PJRT_DEVICE"] == "TT"
    assert "PJRT_DEVICE" in caplog.text
    assert "CUDA" in caplog.text


def test_setup_env_does_not_warn_when_already_tt(monkeypatch, caplog):
    monkeypatch.setenv("PJRT_DEVICE", "TT")

    with caplog.at_level(logging.WARNING):
        make_device_manager(use_tt=True)._setup_env()

    assert os.environ["PJRT_DEVICE"] == "TT"
    assert "PJRT_DEVICE" not in caplog.text


def test_setup_env_leaves_env_untouched_without_use_tt(monkeypatch):
    monkeypatch.setenv("PJRT_DEVICE", "CUDA")

    make_device_manager(use_tt=False)._setup_env()

    assert os.environ["PJRT_DEVICE"] == "CUDA"
