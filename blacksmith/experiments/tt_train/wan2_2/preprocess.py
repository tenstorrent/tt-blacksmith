# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import time
from pathlib import Path

from blacksmith.datasets.tt_train.omniconsistency_lego.omniconsistency_lego_dataset import (
    download_style_subset,
)
from blacksmith.experiments.tt_train.wan2_2.configs import TrainingConfig
from blacksmith.experiments.tt_train.wan2_2.timing import phase, set_sink, summary
from blacksmith.tools.cli import generate_config, parse_cli_options
from blacksmith.tools.tt_train.logging_manager import TrainingLogger

DEFAULT_CONFIG = Path(__file__).parent / "lora" / "galaxy" / "wan2_2_t2v_a14b_lego.yaml"


def preprocess(config: TrainingConfig, logger: TrainingLogger) -> Path:
    logger.info(f"downloading {config.style}/tar + {config.style}/caption from {config.dataset_id} ...")
    with phase("hf download + copy"):
        out, kept, skipped = download_style_subset(config.dataset_id, config.style, config.data_dir)
    logger.info(f"wrote {kept} (image, caption) pairs -> {out.resolve()} (skipped {skipped} without a caption)")
    return out


if __name__ == "__main__":
    args = parse_cli_options(default_config=DEFAULT_CONFIG)
    config: TrainingConfig = generate_config(TrainingConfig, args.config, args.test_config, overrides=args.overrides)

    logger = TrainingLogger(config.model_copy(update={"use_wandb": False}), args.test_log_filename_prefix)
    set_sink(logger.info)

    started = time.perf_counter()
    try:
        preprocess(config, logger)
    finally:
        summary("preprocess", time.perf_counter() - started)
        logger.finish()
