import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--debug-experiment", action="store_true", default=False, help="For debugging purposes, show stdout and stderr of the tests. Meant to be used with pytest -s."
    )
