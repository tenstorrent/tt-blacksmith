# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from blacksmith.experiments.torch.wan2_2.configs import TrainingConfig
from blacksmith.experiments.torch.wan2_2.kurbla.device_manager import WanDeviceManager
from blacksmith.experiments.torch.wan2_2.train import infer, train
from blacksmith.models.torch.wan2_2.model_overrides import (
    apply_generality_overrides,
    apply_perf_overrides,
)
from blacksmith.tools.checkpoints_manager import CheckpointManager
from blacksmith.tools.cli import generate_config, parse_cli_options
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.reproducibility_manager import ReproducibilityManager

if __name__ == "__main__":
    default_config = Path(__file__).parent / "lora" / "single_chip" / "wan2_2_ti2v_5b_diffusiondb.yaml"
    args = parse_cli_options(default_config=default_config)
    config: TrainingConfig = generate_config(TrainingConfig, args.config, args.test_config, args.test_checkpoint_path)

    ReproducibilityManager(config).setup()

    apply_generality_overrides()
    apply_perf_overrides()

    logger = TrainingLogger(config, args.test_log_filename_prefix)
    device_manager = WanDeviceManager(config)
    logger.info(f"Using device: {device_manager.device} (mesh: {device_manager.mesh})")

    checkpoint_manager = CheckpointManager(config, logger, device_manager.device)
    if config.mode == "infer":
        infer(config, device_manager, logger, checkpoint_manager)
    else:
        train(config, device_manager, logger, checkpoint_manager)
