# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""tt-crank counterparts of the tt-xla tools in `blacksmith/tools`.

Every experiment keeps its tt-xla `train.py`; the tt-crank port lives next to it as
`train_crank.py` and reads the *same* YAML. Only the pieces whose implementation
differs between the two stacks live here (device/mesh handling, checkpoint
host-transfer); everything else (datasets, models, logger, CLI, configs) is shared.
"""
