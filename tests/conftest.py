# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
def pytest_addoption(parser):
    parser.addoption(
        "--debug-experiment",
        action="store_true",
        default=False,
        help="For debugging purposes, show stdout and stderr of the tests. Meant to be used with pytest -s.",
    )


def pytest_addoption(parser):
    parser.addoption("--generate-golden-files", action="store_true", default=False, help="Generate golden files.")
