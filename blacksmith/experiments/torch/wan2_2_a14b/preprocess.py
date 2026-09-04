# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from blacksmith.datasets.torch.omniconsistency_lego.omniconsistency_lego_dataset import (
    download_style_subset,
)
from blacksmith.experiments.torch.wan2_2_a14b.configs import TrainingConfig
from blacksmith.tools.cli import generate_config, parse_cli_options
from blacksmith.tools.logging_manager import TrainingLogger

DEFAULT_CONFIG = Path(__file__).parent / "kurbla" / "lora" / "galaxy" / "wan2_2_t2v_a14b_lego.yaml"


def preprocess(config: TrainingConfig, logger: TrainingLogger) -> Path:
    logger.info(f"downloading {config.style}/tar + {config.style}/caption from {config.dataset_id} ...")
    out, kept, skipped = download_style_subset(config.dataset_id, config.style, config.data_dir)
    logger.info(f"wrote {kept} (image, caption) pairs -> {out.resolve()} (skipped {skipped} without a caption)")
    return out


if __name__ == "__main__":
    args = parse_cli_options(default_config=DEFAULT_CONFIG)
    config: TrainingConfig = generate_config(
        TrainingConfig, args.config, args.test_config, overrides=args.overrides
    )

    # A download is not a training run: force wandb off so it never opens one.
    logger = TrainingLogger(config.model_copy(update={"use_wandb": False}), args.test_log_filename_prefix)
    preprocess(config, logger)
