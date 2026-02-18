# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml
from training_test_cases import TRAINING_TEST_CASES

LOG_DIR = Path("tests/training_logs")
GOLDEN_DIR = Path("tests/golden_files")


def assert_loss_with_tolerance(log_file: str, golden_file: str, tolerance: float):
    log_df = pd.read_csv(log_file)
    golden_df = pd.read_csv(golden_file)
    pd.testing.assert_frame_equal(log_df, golden_df, rtol=tolerance)


def get_run_config(config_path: Path) -> dict:
    with config_path.open() as file:
        config_data = yaml.safe_load(file)

    return config_data


def get_log_files(run_name: str) -> tuple[Path, Path]:
    """
    Get the names of the train and val log files for the given run name.

    Returns:
        train_log_file: Name of the train log file.
        val_log_file: Name of the val log file.
    """

    train_log_file = Path(f"{run_name}_train.csv")
    val_log_file = Path(f"{run_name}_val.csv")

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
        request: pytest request object.
    """

    default_setup_dict = {
        "test_script": None,
        "experiment_config": None,
        "test_config": "tests/configs/test_training_fast.yaml",
        "tolerance": 0.5,
        "timeout": 800.0,
    }

    setup_dict = default_setup_dict | setup_dict

    assert setup_dict["test_script"] is not None, "`test_script` is required."
    assert setup_dict["experiment_config"] is not None, "`experiment_config` is required."

    test_id = request.node.callspec.id

    assert Path(setup_dict["test_script"]).exists(), f"Script not found: {setup_dict['test_script']}"
    assert Path(setup_dict["test_config"]).exists(), f"Config not found: {setup_dict['test_config']}"

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(setup_dict["test_script"]), "--test-config", str(setup_dict["test_config"])]
    if setup_dict["experiment_config"] is not None:
        cmd.append("--config")
        cmd.append(str(setup_dict["experiment_config"]))

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

    run_config = get_run_config(Path(setup_dict["experiment_config"]))
    test_config = get_run_config(Path(setup_dict["test_config"]))

    run_name = run_config["wandb_run_name"] if "wandb_run_name" in run_config else None
    if run_name is None:
        return  # If a test does not support golden files yet.

    train_log_file, val_log_file = get_log_files(run_name)

    if not (LOG_DIR / train_log_file).exists() or not (LOG_DIR / val_log_file).exists():
        return  # If a test does not support golden files yet.

    if not test_config["use_tt"] and not (GOLDEN_DIR / train_log_file).exists():
        # Reference run, move the log files to golden_files.
        (LOG_DIR / train_log_file).rename(GOLDEN_DIR / train_log_file)
        (LOG_DIR / val_log_file).rename(GOLDEN_DIR / val_log_file)
    else:
        # Test run, compare the train and val log files in training_logs with those in golden_files.
        assert_loss_with_tolerance(
            LOG_DIR / train_log_file,
            GOLDEN_DIR / train_log_file,
            tolerance=setup_dict["tolerance"],
        )
        assert_loss_with_tolerance(
            LOG_DIR / val_log_file,
            GOLDEN_DIR / val_log_file,
            tolerance=setup_dict["tolerance"],
        )

        (LOG_DIR / train_log_file).unlink()
        (LOG_DIR / val_log_file).unlink()
