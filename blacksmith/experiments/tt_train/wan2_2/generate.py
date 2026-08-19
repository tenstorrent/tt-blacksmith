# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import time
from pathlib import Path

import ttnn

from blacksmith.experiments.tt_train.wan2_2.configs import TrainingConfig
from blacksmith.experiments.tt_train.wan2_2.timing import phase, set_sink, summary
from blacksmith.models.tt_train.wan2_2.encoders import close_mesh, open_mesh
from blacksmith.tools.cli import generate_config, parse_cli_options
from blacksmith.tools.tt_train.logging_manager import TrainingLogger

DEFAULT_CONFIG = Path(__file__).parent / "lora" / "galaxy" / "wan2_2_t2v_a14b_lego.yaml"


def generate(config: TrainingConfig, logger: TrainingLogger) -> None:
    from models.tt_dit.experimental.pipelines.pipeline_wan_runtime_lora import (
        WanPipelineRuntimeLoRA,
    )
    from models.tt_dit.pipelines.wan.pipeline_wan import WanPipelineConfig

    inf = config.inference
    if inf.infer_frames <= 5:
        logger.warning(
            f"{inf.infer_frames} frames = {(inf.infer_frames - 1) // 4 + 1} latent frame(s); very short "
            f"clips can decode to black. Use >= 13 for reliable video."
        )

    logger.info(f"opening mesh device {tuple(config.mesh_shape)} ...")
    with phase("open mesh"):
        mesh_device = open_mesh(config.mesh_shape)
    try:
        pipeline_config = WanPipelineConfig.default(
            mesh_shape=mesh_device.shape,
            checkpoint_name=config.model_id,
            height=inf.infer_h,
            width=inf.infer_w,
            num_frames=inf.infer_frames,
            max_sequence_length=config.max_seq,
            topology=ttnn.Topology.Linear,
        )
        logger.info(f"topology={pipeline_config.topology} num_links={pipeline_config.num_links}")
        with phase("build pipeline"):
            pipe = WanPipelineRuntimeLoRA(device=mesh_device, config=pipeline_config)

        if inf.infer_no_lora:
            logger.info("infer_no_lora: running the BASE model (no adapter bound)")
        else:
            high_p = inf.infer_high_lora or config.expert_path("high")
            low_p = inf.infer_low_lora or config.expert_path("low")
            high_p = high_p if Path(high_p).exists() else None
            low_p = low_p if Path(low_p).exists() else None
            if not high_p and not low_p:
                raise FileNotFoundError(
                    f"no LoRA files found ({config.expert_path('high')!r}, {config.expert_path('low')!r}) — "
                    f"train first, or set inference.infer_no_lora: true to run the base model"
                )
            logger.info(f"registering LoRA (high={high_p}, low={low_p}, scale={inf.lora_scale})")
            with phase("register LoRA"):
                handle = pipe.register_lora("style", high_path=high_p, low_path=low_p, scale=inf.lora_scale)
                pipe.set_active_lora(handle)

        logger.info(f"generating {inf.infer_frames}f @ {inf.infer_h}x{inf.infer_w}, {inf.infer_steps} steps ...")
        t0 = time.time()
        with phase("denoise + VAE decode"):
            frames = pipe(
                prompts=[config.trigger + inf.val_prompt],
                negative_prompts=[inf.neg_prompt] if inf.neg_prompt else None,
                num_inference_steps=inf.infer_steps,
                guidance_scale=inf.infer_guidance,
                guidance_scale_2=inf.infer_guidance_2,
                flow_shift=inf.infer_flow_shift,
                boundary_ratio=config.boundary_ratio,
                seed=config.seed,
            )
        elapsed = time.time() - t0
        logger.info(
            f"done in {elapsed / 60:.1f} min "
            f"({elapsed / max(inf.infer_steps, 1):.1f}s/step over {inf.infer_steps} steps)"
        )
    finally:
        with phase("close mesh"):
            close_mesh(mesh_device)

    with phase("write output"):
        _write_output(frames, inf.infer_output, inf.infer_fps, logger)


def _write_output(frames, out_path: str, fps: int, logger: TrainingLogger) -> None:
    import numpy as np
    from PIL import Image

    if not isinstance(frames[0], Image.Image):
        frames = np.asarray(frames)
        frames = frames.reshape(-1, *frames.shape[-3:])

    frames_dir = Path(out_path).with_suffix("")
    frames_dir.mkdir(parents=True, exist_ok=True)
    pil_frames = [fr if isinstance(fr, Image.Image) else Image.fromarray(fr) for fr in frames]
    for i, fr in enumerate(pil_frames):
        fr.save(frames_dir / f"frame_{i:03d}.png")
    logger.info(f"saved {len(pil_frames)} PNG frames -> {frames_dir}/")

    try:
        from diffusers.utils import export_to_video

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        export_to_video(pil_frames, out_path, fps=fps)
        logger.info(f"saved video -> {out_path}")
    except Exception as e:
        logger.warning(
            f"mp4 export skipped ({type(e).__name__}: {e}); PNG frames are in {frames_dir}/. "
            f"Install imageio + imageio-ffmpeg and re-run for the mp4."
        )


if __name__ == "__main__":
    args = parse_cli_options(default_config=DEFAULT_CONFIG)
    config: TrainingConfig = generate_config(TrainingConfig, args.config, args.test_config, overrides=args.overrides)

    logger = TrainingLogger(config.model_copy(update={"use_wandb": False}), args.test_log_filename_prefix)
    set_sink(logger.info)

    started = time.perf_counter()
    try:
        generate(config, logger)
    finally:
        summary("infer", time.perf_counter() - started)
        logger.finish()
