# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Example runner for :class:`LoraLLMTrainer`.

Runs the LoRA LLM trainer from a YAML config so you can check that the shared
``Trainer`` works across device setups (single chip / tensor parallel / data
parallel). Pick the config with ``--config``; it defaults to the single-chip one.

    python blacksmith/tools/trainer/examples/lora_llm/train.py --config blacksmith/tools/trainer/examples/lora_llm/tensor_parallel/llama_3_2_1b_sst2.yaml
"""
from pathlib import Path

from blacksmith.tools.cli import generate_config, parse_cli_options
from blacksmith.tools.trainer.configs import LoraLLMConfig
from blacksmith.tools.trainer.strategies import LoraLLMTrainer

if __name__ == "__main__":
    default_config = Path(__file__).parent / "single_chip" / "llama_3_2_1b_sst2.yaml"
    args = parse_cli_options(default_config=default_config)
    config: LoraLLMConfig = generate_config(LoraLLMConfig, args.config)

    trainer = LoraLLMTrainer()
    trainer.setup(config)
    trainer.train()
