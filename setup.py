# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from setuptools import find_packages, setup

setup(
    name="blacksmith",
    version="0.1",
    description="Tenstorrent Python Blacksmith",
    packages=find_packages(include=["blacksmith*"]),
)
