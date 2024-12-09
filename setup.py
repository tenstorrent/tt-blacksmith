# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import re
from setuptools import find_packages, setup


# frontends can be forge or jax. Exclude the other frontend from the package
FRONTEND = "forge"

exclude_keywords = {
    "forge": ["jax"],
    "jax": ["forge", "torch", "lightning", "torchvision"],
}

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
