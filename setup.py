# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
import re
from setuptools import find_packages, setup
import os

# Frontends can be tt-forge-fe or tt-xla
FRONTEND = os.environ.get("TT_BLACKSMITH_FRONTEND", "")
if not FRONTEND:
    print("Error: TT_BLACKSMITH_FRONTEND environment variable not set.")
    exit(1)


exclude_keywords = defaultdict(
    list,
    {
        "tt-forge-fe": ["jax", "torch_xla"],
        "tt-xla": ["forge"],
    },
)


all_packages = find_packages(include=["blacksmith*"])
excluded_packages = []
for pkg in all_packages:
    for keyword in exclude_keywords[FRONTEND]:
        if re.search(keyword, pkg):
            excluded_packages.append(pkg)
            break

setup(
    name="blacksmith",
    version="0.1",
    description="Tenstorrent Python Blacksmith",
    packages=[pkg for pkg in all_packages if pkg not in excluded_packages],
    package_dir={"blacksmith": "blacksmith"},
)
