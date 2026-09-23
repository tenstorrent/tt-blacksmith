# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from blacksmith.tools.logging_manager import TEST_LOGS_DIR

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO_ROOT / "blacksmith/experiments/torch/llama/crank/train.py"
TRAINER_SCRIPT = REPO_ROOT / "blacksmith/tools/trainer/examples/lora_llm/train_crank.py"
TEST_CONFIG = REPO_ROOT / "tests/configs/tt-crank-llama-sst2.yaml"
TRAINER_TEST_CONFIG = REPO_ROOT / "tests/configs/tt-crank-trainer-llama-sst2.yaml"


def _run(config: str, log_prefix: str, timeout: float, script=TRAIN_SCRIPT, test_config=TEST_CONFIG) -> pd.DataFrame:
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(REPO_ROOT / config),
            "--test-config",
            str(test_config),
            "--test-log-filename-prefix",
            log_prefix,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, f"training failed:\n{result.stdout[-4000:]}\n{result.stderr[-4000:]}"

    train_csv = REPO_ROOT / TEST_LOGS_DIR / f"{log_prefix}_train.csv"
    assert train_csv.exists(), f"no train log at {train_csv}"
    return pd.read_csv(train_csv)


def _assert_loss_decreases(df: pd.DataFrame) -> None:
    assert len(df) >= 2, f"need at least two logged train losses, got {len(df)}"
    first, last = df["train/loss"].iloc[0], df["train/loss"].iloc[-1]
    assert last < first, f"training loss did not decrease: {first:.4f} -> {last:.4f}"


@pytest.mark.push
@pytest.mark.torch
@pytest.mark.single_chip
@pytest.mark.n150
def test_llama_3_2_1b_sst2_single_chip():
    df = _run(
        "blacksmith/experiments/torch/llama/xla/lora/single_chip/llama_3_2_1b_sst2.yaml",
        "tt-crank-llama_3_2_1b-sst2-single_chip",
        timeout=1800,
    )
    _assert_loss_decreases(df)


@pytest.mark.uplift
@pytest.mark.torch
@pytest.mark.tensor_parallel
@pytest.mark.n300_llmbox
def test_llama_3_1_8b_sst2_multichip():
    df = _run(
        "blacksmith/experiments/torch/llama/xla/lora/quietbox/llama_3_1_8b_sst2.yaml",
        "tt-crank-llama_3_1_8b-sst2-multichip",
        timeout=3600,
    )
    _assert_loss_decreases(df)


@pytest.mark.push
@pytest.mark.torch
@pytest.mark.single_chip
@pytest.mark.n150
def test_trainer_lora_llm_3_2_1b_sst2_single_chip():
    df = _run(
        "blacksmith/tools/trainer/examples/lora_llm/single_chip/llama_3_2_1b_sst2.yaml",
        "tt-crank-trainer-llama_3_2_1b-sst2-single_chip",
        timeout=1800,
        script=TRAINER_SCRIPT,
        test_config=TRAINER_TEST_CONFIG,
    )
    _assert_loss_decreases(df)
