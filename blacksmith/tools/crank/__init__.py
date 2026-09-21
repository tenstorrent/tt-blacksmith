# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""tt-crank counterparts of the tt-xla tools in `blacksmith/tools`.

Every experiment keeps its tt-xla `train.py`; the tt-crank port lives next to it as
`train_crank.py` and reads the *same* YAML. Nothing in this package imports tt-xla code:
whatever a tt-crank script needs from the tt-xla tools (checkpointing, loss / label helpers,
the HF model loader, the Trainer pipeline) is copied here and stripped of its tt-xla parts,
so the tt-xla tree can be deprecated without touching the tt-crank one. Backend-neutral
infrastructure with no tt-xla code in it (datasets, logger, CLI, reproducibility, the
pydantic configs) stays shared.
"""
