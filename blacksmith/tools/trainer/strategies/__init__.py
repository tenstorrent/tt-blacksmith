# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import importlib

# Strategies import their framework (torch, ttml) at module level, so each one is loaded only
# when it is asked for: importing one strategy never pulls in another framework.
_STRATEGIES = {
    "CrankLLMTrainer": "blacksmith.tools.trainer.strategies.crank_llm_trainer",
    "TtmlLLMTrainer": "blacksmith.tools.trainer.strategies.ttml_llm_trainer",
}

__all__ = list(_STRATEGIES)


def __getattr__(name):
    if name in _STRATEGIES:
        return getattr(importlib.import_module(_STRATEGIES[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
