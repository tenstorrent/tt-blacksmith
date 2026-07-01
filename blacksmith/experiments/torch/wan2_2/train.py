# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_xla
from torch.utils.data import DataLoader

from blacksmith.datasets.torch.diffusiondb_pixelart.diffusiondb_pixelart_dataset import (
    PixelArtLatentDataset,
    make_collate_fn,
)
from blacksmith.experiments.torch.wan2_2.configs import TrainingConfig
from blacksmith.experiments.torch.wan2_2.generate import (
    build_pipeline_for_validation,
    generate_validation_sample,
    generate_wan_video,
)
from blacksmith.models.torch.wan2_2.device import WanDeviceManager
from blacksmith.models.torch.wan2_2.model_overrides import (
    apply_generality_overrides,
    apply_perf_overrides,
    build_lora_transformer,
)
from blacksmith.tools.checkpoints_manager import CheckpointManager
from blacksmith.tools.cli import generate_config, parse_cli_options
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.reproducibility_manager import ReproducibilityManager


def _sample_timesteps(batch_size: int, config: TrainingConfig, generator: torch.Generator | None = None):
    # SD3-style logit-normal + flow shift; returns t in [0, 1] of shape (B,) on CPU.
    if generator is None:
        u = torch.randn(batch_size) * config.lognorm_std + config.lognorm_mean
    else:
        u = torch.randn(batch_size, generator=generator) * config.lognorm_std + config.lognorm_mean
    u = torch.sigmoid(u)
    shift = config.train_flow_shift
    return shift * u / (1.0 + (shift - 1.0) * u)


def flow_matching_step(
    transformer,
    batch: dict,
    config: TrainingConfig,
    device_manager: WanDeviceManager,
    *,
    fixed_t=None,
    fixed_noise=None,
):
    dtype = config.torch_dtype()
    x0 = batch["latent"].to(dtype)
    text_embed = batch["text_embed"].to(dtype)
    B = x0.shape[0]

    t = _sample_timesteps(B, config) if fixed_t is None else fixed_t
    noise = torch.randn(x0.shape, dtype=x0.dtype) if fixed_noise is None else fixed_noise.to(x0.dtype)

    x0 = device_manager.to_device(x0)
    text_embed = device_manager.to_device(text_embed)
    t = device_manager.to_device(t.to(dtype))
    noise = device_manager.to_device(noise)

    sigma = t.view(B, 1, 1, 1, 1)
    timestep = (t * 1000.0).long()

    x_t = (1.0 - sigma) * x0 + sigma * noise
    pred = transformer(hidden_states=x_t, timestep=timestep, encoder_hidden_states=text_embed, return_dict=True).sample
    target = noise - x0
    return F.mse_loss(pred.float(), target.float())


def validate(transformer, config: TrainingConfig, device_manager: WanDeviceManager, logger: TrainingLogger, step: int):
    logger.info(f"Generating validation sample at step {step} ...")
    img, video_np = generate_validation_sample(transformer, config, device_manager, step)
    caption = f"step={step} prompt={config.trigger + config.inference.val_prompt!r}"
    logger.log_image("val/sample", img, step=step, caption=caption)
    if video_np is not None:
        logger.log_video("val/sample_video", video_np, fps=config.inference.infer_fps, step=step)


def train(
    config: TrainingConfig,
    device_manager: WanDeviceManager,
    logger: TrainingLogger,
    checkpoint_manager: CheckpointManager,
):
    logger.info("Starting training...")

    cache = Path(config.cache_dir)
    samples_dir = cache / "samples"
    embeds_path = cache / "embeds.pt"
    if not embeds_path.exists() or not samples_dir.exists():
        raise FileNotFoundError(f"missing cache at {cache} — run precompute.py first.")

    metadata = json.loads((cache / "metadata.json").read_text())
    all_indices = sorted(m["idx"] for m in metadata)
    if len(all_indices) <= config.val_holdout:
        raise RuntimeError(f"need > {config.val_holdout} cached samples; got {len(all_indices)}")
    val_indices = all_indices[-config.val_holdout :] if config.val_holdout > 0 else []
    train_indices = all_indices[: -config.val_holdout] if config.val_holdout > 0 else all_indices
    logger.info(f"{len(train_indices)} train / {len(val_indices)} val cached samples")

    embeds = torch.load(embeds_path, weights_only=False)
    train_ds = PixelArtLatentDataset(config.cache_dir, train_indices)
    train_collate = make_collate_fn(embeds, p_drop=config.text_drop_prob, seed=config.seed)
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=train_collate, num_workers=0, drop_last=True
    )

    transformer = build_lora_transformer(config, device_manager)
    transformer.train()
    logger.info(f"Trainable parameters: {sum(p.numel() for p in transformer.parameters() if p.requires_grad)}")
    compiled_transformer = device_manager.compile(transformer)

    trainable = [p for p in transformer.parameters() if p.requires_grad]
    # Note: we don't use capturable = True, as it results in collective_permute op
    # which we don't support yet. (https://github.com/tenstorrent/tt-mlir/issues/3370)
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay, betas=(0.9, 0.999)
    )

    if config.resume_from_checkpoint:
        checkpoint_manager.load_checkpoint(transformer, optimizer)

    global_step = 0
    avg_loss = float("nan")
    accum_loss = 0.0
    accum_count = 0
    micro_step = 0
    step_start = time.time()
    data_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)

    try:
        while global_step < config.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            loss = flow_matching_step(compiled_transformer, batch, config, device_manager)
            (loss / config.gradient_accumulation_steps).backward()
            device_manager.sync()
            accum_loss += loss.item()
            accum_count += 1
            micro_step += 1

            if micro_step % config.gradient_accumulation_steps == 0:
                device_manager.optimizer_step(optimizer)
                optimizer.zero_grad()
                device_manager.sync()
                global_step += 1

                avg_loss = accum_loss / accum_count
                accum_loss = 0.0
                accum_count = 0

                if global_step % config.steps_freq == 0:
                    step_time = time.time() - step_start
                    step_start = time.time()
                    logger.log_metrics(
                        {
                            "train/loss": avg_loss,
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "train/step_time_s": step_time,
                        },
                        step=global_step,
                    )

                if global_step % config.val_steps_freq == 0:
                    device_manager.sync()
                    checkpoint_manager.save_checkpoint(
                        transformer, global_step, 0, optimizer, metrics={"train/loss": avg_loss}
                    )
                    validate(transformer, config, device_manager, logger, global_step)

        device_manager.sync()
        final_path = checkpoint_manager.save_checkpoint(
            transformer, global_step, 0, optimizer, metrics={"train/loss": avg_loss}, checkpoint_name="final_model.pt"
        )
        logger.log_artifact(final_path, artifact_type="model", name="final_model.pt")
        logger.info(f"Training done at step {global_step}.")
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", traceback.format_exc())
        raise
    finally:
        logger.finish()


@torch.no_grad()
def infer(
    config: TrainingConfig,
    device_manager: WanDeviceManager,
    logger: TrainingLogger,
    checkpoint_manager: CheckpointManager,
):
    from diffusers.utils import export_to_video

    inf = config.inference
    transformer = build_lora_transformer(config, device_manager)
    checkpoint_manager.load_checkpoint(transformer)
    transformer.eval()

    pipe = build_pipeline_for_validation(transformer, config, device_manager)
    compiled_transformer = device_manager.compile(transformer)

    logger.info(f"Generating {inf.infer_frames} frames @ {inf.infer_h}x{inf.infer_w} in {inf.infer_steps} steps.")
    t0 = time.time()
    gen = torch.Generator(device="cpu").manual_seed(config.seed)
    video = generate_wan_video(
        pipe,
        compiled_transformer,
        config,
        device_manager,
        prompt=config.trigger + inf.val_prompt,
        negative_prompt=inf.neg_prompt or None,
        height=inf.infer_h,
        width=inf.infer_w,
        num_frames=inf.infer_frames,
        num_inference_steps=inf.infer_steps,
        guidance_scale=inf.infer_guidance,
        generator=gen,
        output_type="pil",
    )
    frames = video[0]
    logger.info(f"Generated in {(time.time() - t0) / 60.0:.1f} min; frames={len(frames)}")

    out_path = inf.infer_output
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, out_path, fps=inf.infer_fps)
    logger.info(f"Saved video -> {out_path}")

    np_frames = np.stack([np.asarray(f) for f in frames], axis=0).astype(np.uint8)
    logger.log_video("infer/video", np_frames, fps=inf.infer_fps)
    logger.finish()


if __name__ == "__main__":
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
        torch_xla.set_custom_compile_options(device_manager.xla_compile_options)

    checkpoint_manager = CheckpointManager(config, logger, device_manager.device)
    if config.mode == "infer":
        infer(config, device_manager, logger, checkpoint_manager)
    else:
        train(config, device_manager, logger, checkpoint_manager)
