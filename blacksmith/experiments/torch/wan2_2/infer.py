# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from blacksmith.experiments.torch.wan2_2.configs import TrainingConfig
from blacksmith.experiments.torch.wan2_2.generate import build_pipeline_for_validation, generate_wan_video
from blacksmith.models.torch.wan2_2.device import WanDeviceManager
from blacksmith.models.torch.wan2_2.model_overrides import build_lora_transformer
from blacksmith.tools.checkpoints_manager import CheckpointManager
from blacksmith.tools.logging_manager import TrainingLogger


@torch.no_grad()
def infer(
    config: TrainingConfig,
    device_manager: WanDeviceManager,
    logger: TrainingLogger,
    checkpoint_manager: CheckpointManager,
):
    from diffusers.utils import export_to_video

    transformer = build_lora_transformer(config, device_manager)
    checkpoint_manager.load_checkpoint(transformer)
    transformer.eval()

    pipe = build_pipeline_for_validation(transformer, config, device_manager)
    compiled_transformer = device_manager.compile(transformer)

    logger.info(
        f"Generating {config.infer_frames} frames @ {config.infer_h}x{config.infer_w} in {config.infer_steps} steps."
    )
    t0 = time.time()
    gen = torch.Generator(device="cpu").manual_seed(config.seed)
    video = generate_wan_video(
        pipe,
        compiled_transformer,
        config,
        device_manager,
        prompt=config.trigger + config.val_prompt,
        negative_prompt=config.neg_prompt or None,
        height=config.infer_h,
        width=config.infer_w,
        num_frames=config.infer_frames,
        num_inference_steps=config.infer_steps,
        guidance_scale=config.infer_guidance,
        generator=gen,
        output_type="pil",
    )
    frames = video[0]
    logger.info(f"Generated in {(time.time() - t0) / 60.0:.1f} min; frames={len(frames)}")

    out_path = config.infer_output
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, out_path, fps=config.infer_fps)
    logger.info(f"Saved video -> {out_path}")

    np_frames = np.stack([np.asarray(f) for f in frames], axis=0).astype(np.uint8)
    logger.log_video("infer/video", np_frames, fps=config.infer_fps)
    logger.finish()


if __name__ == "__main__":
    import torch_xla

    from blacksmith.models.torch.wan2_2.device import wan_xla_compile_options
    from blacksmith.models.torch.wan2_2.model_overrides import apply_generality_overrides, apply_perf_overrides
    from blacksmith.tools.cli import generate_config, parse_cli_options
    from blacksmith.tools.reproducibility_manager import ReproducibilityManager

    default_config = Path(__file__).parent / "lora" / "quietbox" / "wan2_2_ti2v_5b_diffusiondb.yaml"
    args = parse_cli_options(default_config=default_config)
    config: TrainingConfig = generate_config(TrainingConfig, args.config, args.test_config, args.test_checkpoint_path)

    ReproducibilityManager(config).setup()

    apply_generality_overrides()
    apply_perf_overrides()

    logger = TrainingLogger(config, args.test_log_filename_prefix)
    device_manager = WanDeviceManager(config)
    logger.info(f"Using device: {device_manager.device}")
    if config.use_tt:
        torch_xla.set_custom_compile_options(wan_xla_compile_options())

    checkpoint_manager = CheckpointManager(config, logger, device_manager.device)
    infer(config, device_manager, logger, checkpoint_manager)
