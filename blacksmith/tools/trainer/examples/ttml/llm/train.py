# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from blacksmith.configs import LLMTrainerConfig
from blacksmith.tools.cli import generate_config, parse_cli_options
from blacksmith.tools.trainer import MetricsCallback
from blacksmith.tools.trainer.strategies import TtmlLLMTrainer

if __name__ == "__main__":
    default_config = Path(__file__).parent / "lora" / "single_chip" / "llama_3_1_8b_sst2.yaml"
    args = parse_cli_options(default_config=default_config)
    config: LLMTrainerConfig = generate_config(
        LLMTrainerConfig,
        args.config,
        args.test_config,
        args.test_checkpoint_path,
    )

    # Metrics are opt-in: pass the callbacks you want.
    trainer = TtmlLLMTrainer(callbacks=[MetricsCallback()])
    try:
        trainer.setup(config, test_log_filename_prefix=args.test_log_filename_prefix)
        trainer.train()
    finally:
        # Releases the device also when setup or training fails.
        trainer.cleanup()
