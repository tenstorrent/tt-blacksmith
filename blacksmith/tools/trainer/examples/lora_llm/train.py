# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from blacksmith.tools.cli import generate_config, parse_cli_options
from blacksmith.tools.trainer import CheckpointCallback, LoggingCallback
from blacksmith.tools.trainer.configs import LoraLLMConfig
from blacksmith.tools.trainer.strategies import LoraLLMTrainer

if __name__ == "__main__":
    default_config = Path(__file__).parent / "single_chip" / "llama_3_2_1b_sst2.yaml"
    args = parse_cli_options(default_config=default_config)
    config: LoraLLMConfig = generate_config(LoraLLMConfig, args.config)

    # Logging/checkpointing are opt-in: pass the callbacks you want.
    trainer = LoraLLMTrainer(callbacks=[LoggingCallback(), CheckpointCallback()])
    trainer.setup(config)
    trainer.train()
