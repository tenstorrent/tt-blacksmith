# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
import re
from setuptools import find_packages, setup
import os

# Frontends can be tt-forge-fe or tt-xla
FRONTEND = os.environ.get("TT_THOMAS_FRONTEND", "")
if not FRONTEND:
    print("Warning: TT_THOMAS_FRONTEND environment variable not set. All packages will be included.")


exclude_keywords = defaultdict(list)
exclude_keywords.update(
    {
        "tt-forge-fe": ["jax"],
        "tt-xla": ["forge", "torch", "lightning", "torchvision"],
    }
)

all_packages = find_packages(include=["thomas*"])
excluded_packages = [
    pkg for pkg in all_packages if any([re.search(keyword, pkg) for keyword in exclude_keywords[FRONTEND]])
]

setup(
    name="thomas",
    version="0.1",
    description="Tenstorrent Python Thomas",
    packages=[pkg for pkg in all_packages if pkg not in excluded_packages],
    package_dir={"thomas": "thomas"},
)
