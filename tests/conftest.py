"""Pytest fixtures for test suite."""
import pytest
import os
from pathlib import Path


@pytest.fixture(scope="session")
def repo_root():
    """
    Get the repository root directory.

    Uses TT_BLACKSMITH_HOME environment variable set by env/activate script.
    """
    blacksmith_home = os.environ.get('TT_BLACKSMITH_HOME')
    assert blacksmith_home is not None, (
        "TT_BLACKSMITH_HOME environment variable not set. "
        "Please activate the environment using 'source env/activate'"
    )
    return Path(blacksmith_home)


@pytest.fixture(scope="session")
def test_configs_dir(repo_root):
    """
    Get the test configurations directory.

    Returns the directory containing test YAML configuration files.
    """
    return repo_root / "tests" / "configs"
