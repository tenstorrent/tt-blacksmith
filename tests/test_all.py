# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from training_test_cases import TRAINING_TEST_CASES

from blacksmith.tools.logging_manager import GOLDEN_LOGS_DIR, TEST_LOGS_DIR


@pytest.fixture
def config(request):
    return request.config


def assert_loss_with_tolerance(log_file: str, golden_file: str, tolerance: float):
    log_df = pd.read_csv(log_file)
    golden_df = pd.read_csv(golden_file)
    pd.testing.assert_frame_equal(golden_df, log_df, rtol=tolerance)


def get_log_files(log_filename_prefix: str) -> tuple[Path, Path]:
    train_log_file = Path(f"{log_filename_prefix}_train.csv")
    val_log_file = Path(f"{log_filename_prefix}_val.csv")

    return train_log_file, val_log_file


@pytest.mark.parametrize("setup_dict", TRAINING_TEST_CASES)
def test_training_script(
    setup_dict: dict,
    request: pytest.FixtureRequest,
):
    """
    Test that training script runs successfully with test configuration.

    Spawns subprocess to execute training script, verifies exit code 0.

    Args:
        setup_dict: Dictionary containing the test setup:
            - test_script: Path to the training script.
            - experiment_config: Path to the experiment configuration.
            - test_config: Path to the test configuration.
            - tolerance: Tolerance for loss and accuracy metrics.
            - timeout: Timeout in seconds.
            - skip_loss_checks: Whether to skip the loss checks.
        request: pytest request object.
    """

    default_setup_dict = {
        "test_script": None,
        "experiment_config": None,
        "test_config": "tests/configs/test_training_fast.yaml",
        "tolerance": 0.3,
        "timeout": 800.0,
        "skip_loss_checks": False,
    }

    setup_dict = default_setup_dict | setup_dict

    assert setup_dict["test_script"] is not None, "`test_script` is required."
    assert setup_dict["experiment_config"] is not None, "`experiment_config` is required."

    test_id = request.node.callspec.id

    assert Path(setup_dict["test_script"]).exists(), f"Script not found: {setup_dict['test_script']}"
    assert Path(setup_dict["test_config"]).exists(), f"Config not found: {setup_dict['test_config']}"

    TEST_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    GOLDEN_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(setup_dict["test_script"]), "--test-config", str(setup_dict["test_config"])]
    if setup_dict["experiment_config"] is not None:
        cmd.append("--config")
        cmd.append(str(setup_dict["experiment_config"]))
    cmd.append("--test-log-filename-prefix")
    cmd.append(test_id)

    try:
        result = subprocess.run(
            cmd,
            cwd=str(Path.cwd()),
            timeout=setup_dict["timeout"],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            print(f"\n{'='*60}")
            print(f"FAILED: {test_id}")
            print(f"Exit code: {result.returncode}")
            print(f"\nSTDOUT:\n{result.stdout}")
            print(f"\nSTDERR:\n{result.stderr}")
            print(f"{'='*60}\n")
            pytest.fail(f"Training script exited with code {result.returncode}")

    except subprocess.TimeoutExpired:
        pytest.fail(f"Training script timed out after {setup_dict['timeout']} seconds")

    if setup_dict["skip_loss_checks"]:
        return  # If a test does not support golden files yet.

    train_log_file, val_log_file = get_log_files(test_id)

    if request.config.getoption("--generate-golden-files"):
        # Reference run, move the log files to golden_files.
        (TEST_LOGS_DIR / train_log_file).rename(GOLDEN_LOGS_DIR / train_log_file)
        (TEST_LOGS_DIR / val_log_file).rename(GOLDEN_LOGS_DIR / val_log_file)
        return

    # Test run, compare the train and val log files in training_logs with those in golden_files.
    assert_loss_with_tolerance(
        TEST_LOGS_DIR / train_log_file,
        GOLDEN_LOGS_DIR / train_log_file,
        tolerance=setup_dict["tolerance"],
    )
    assert_loss_with_tolerance(
        TEST_LOGS_DIR / val_log_file,
        GOLDEN_LOGS_DIR / val_log_file,
        tolerance=setup_dict["tolerance"],
    )
