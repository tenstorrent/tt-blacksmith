# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import ttml
import ttnn
from ttml.datasets import InMemoryDataloader

from blacksmith.datasets.tt_train.omniconsistency_lego.omniconsistency_lego_dataset import (
    LatentEmbedDataset,
    TextEmbeds,
    make_collate_fn,
)
from blacksmith.experiments.tt_train.wan2_2.configs import SUBFOLDER, TrainingConfig
from blacksmith.experiments.tt_train.wan2_2.timing import (
    fmt,
    phase,
    record,
    set_sink,
    summary,
)
from blacksmith.models.tt_train.wan2_2 import (
    WanConfig,
    WanTransformer3D,
    build_rope_params,
    load_expert_from_safetensors,
    patchify,
    patchify_output_order,
)
from blacksmith.models.tt_train.wan2_2.device import setup_device
from blacksmith.models.tt_train.wan2_2.lora import init_lora_A_gaussian, load_all, resolve_targets, save_all
from blacksmith.tools.cli import generate_config, parse_cli_options
from blacksmith.tools.tt_train.logging_manager import TrainingLogger

DEFAULT_CONFIG = Path(__file__).parent / "lora" / "galaxy" / "wan2_2_t2v_a14b_lego.yaml"


def _to_ttml(arr: np.ndarray, dtype=ttnn.bfloat16, mapper=None):
    return ttml.autograd.Tensor.from_numpy(np.ascontiguousarray(arr, dtype=np.float32), ttnn.Layout.TILE, dtype, mapper)


def _loss_value(loss) -> float:
    mesh = ttml.maybe_mesh()
    if mesh is None or mesh.num_devices() == 1:
        return float(np.asarray(loss.to_numpy()).reshape(-1)[0])

    device = ttml.autograd.AutoContext.get_instance().get_device()
    composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)
    return float(np.asarray(loss.to_numpy(composer=composer)).mean())


def latent_shape(config: TrainingConfig) -> tuple:
    return (
        config.batch_size,
        WanConfig.in_channels,
        (config.train_frames - 1) // 4 + 1,
        config.train_h // 8,
        config.train_w // 8,
    )


def model_config_for(config: TrainingConfig, *, init_weights: bool = True) -> WanConfig:
    _, tp_size = config.mesh_shape
    return WanConfig(
        runner_type=(
            ttml.models.RunnerType.MemoryEfficient if config.gradient_checkpointing else ttml.models.RunnerType.Default
        ),
        init_weights=init_weights,
        use_tp=tp_size > 1,
    )


def build_lora_expert(role: str, config: TrainingConfig, logger: TrainingLogger) -> ttml.modules.LoraModel:
    sub = SUBFOLDER[role]
    logger.info(f"loading {role}-noise expert ({sub}) from {config.model_id} ...")

    model_config = model_config_for(config, init_weights=False)
    model = WanTransformer3D(model_config)
    load_expert_from_safetensors(model, config.model_id, subfolder=sub)

    lora_config = ttml.modules.LoraConfig(
        rank=config.lora_rank,
        alpha=float(config.lora_alpha),
        target_modules=resolve_targets(config.lora_target_set),
        lora_dropout=0.0,
        use_rslora=False,
        verbose=True,
    )
    lora_model = ttml.modules.LoraModel(model, lora_config)

    if config.lora_a_init == "gaussian":
        n = init_lora_A_gaussian(lora_model, config.lora_rank, config.mesh_shape, seed=config.seed)
        logger.info(f"{role}: re-initialized {n} lora_A ~ N(0, 1/{config.lora_rank})")

    all_params = lora_model.parameters()
    trainable = {name: p for name, p in all_params.items() if "lora" in name}
    logger.info(f"{role}: {len(trainable)} LoRA params trainable, {len(all_params) - len(trainable)} frozen")
    if not trainable:
        raise RuntimeError("LoRA injection produced no trainable parameters")
    return lora_model


def _range_for(config: TrainingConfig) -> tuple[float, float]:
    if config.train_experts == "low":
        return 0.0, config.boundary_ratio
    if config.train_experts == "high":
        return config.boundary_ratio, 1.0
    return 0.0, 1.0


def _sample_timestep(config: TrainingConfig, lo: float, hi: float, rng: np.random.Generator) -> float:
    shift = config.train_flow_shift
    while True:
        z = rng.standard_normal() * config.lognorm_std + config.lognorm_mean
        u = 1.0 / (1.0 + np.exp(-z))
        t = shift * u / (1.0 + (shift - 1.0) * u)
        if lo <= t < hi:
            return float(t)


def _route(t: float, experts: dict, config: TrainingConfig):
    if len(experts) == 1:
        return next(iter(experts.values()))
    return experts["high"] if t >= config.boundary_ratio else experts["low"]


def flow_matching_step(
    model,
    batch,
    t: float,
    rope_params,
    patch_size: tuple,
    rng: np.random.Generator,
    fixed_noise: np.ndarray | None = None,
    dp_mapper=None,
):
    x0 = np.asarray(batch["latent"], dtype=np.float32)
    noise = (
        rng.standard_normal(x0.shape, dtype=np.float32)
        if fixed_noise is None
        else np.asarray(fixed_noise, dtype=np.float32)
    )
    x_t = (1.0 - t) * x0 + t * noise
    target = noise - x0

    tokens = patchify(x_t, patch_size)
    target_tokens = patchify_output_order(target, patch_size)

    text_embed = np.asarray(batch["text_embed"], dtype=np.float32)
    text_embed = text_embed.reshape(text_embed.shape[0], 1, *text_embed.shape[-2:])

    timesteps = [t * 1000.0]

    pred = model(_to_ttml(tokens, mapper=dp_mapper), timesteps, _to_ttml(text_embed, mapper=dp_mapper), rope_params)
    return ttml.ops.loss.mse_loss(pred, _to_ttml(target_tokens, mapper=dp_mapper), reduce=ttml.ops.ReduceType.MEAN)


def validation_loss(experts, val_loader, config: TrainingConfig, ctx, rope_params, patch_size: tuple) -> float:
    for m in experts.values():
        m.eval()
    ctx.set_gradient_mode(ttml.autograd.GradMode.DISABLED)
    lo, hi = _range_for(config)
    losses = []
    try:
        for batch in val_loader:
            idx = int(batch["idx"][0])
            g = np.random.default_rng(config.seed + idx)
            t = _sample_timestep(config, lo, hi, g)
            noise = g.standard_normal(batch["latent"].shape, dtype=np.float32)
            model = _route(t, experts, config)
            losses.append(
                _loss_value(flow_matching_step(model, batch, t, rope_params, patch_size, g, fixed_noise=noise))
            )
            ctx.reset_graph()
    finally:
        ctx.set_gradient_mode(ttml.autograd.GradMode.ENABLED)
        for m in experts.values():
            m.train()
    return float(np.mean(losses)) if losses else float("nan")


def train(config: TrainingConfig, logger: TrainingLogger) -> None:
    random.seed(config.seed + config.resume_step)
    rng = np.random.default_rng(config.seed + config.resume_step)

    cache = Path(config.cache_dir)
    if not (cache / "embeds.npy").exists() or not (cache / "samples").exists():
        raise FileNotFoundError(f"missing cache at {cache} — run precompute.py first.")

    metadata = json.loads((cache / "metadata.json").read_text())
    all_idx = sorted(m["idx"] for m in metadata)
    if len(all_idx) <= config.val_holdout:
        raise RuntimeError(f"need > {config.val_holdout} samples; got {len(all_idx)}")
    val_idx, train_idx = all_idx[-config.val_holdout :], all_idx[: -config.val_holdout]
    logger.info(f"{len(train_idx)} train / {len(val_idx)} val | experts={config.train_experts}")

    dp_size, tp_size = config.mesh_shape

    with phase("open device"):
        ctx, _device = setup_device(dp_size, tp_size, seed=config.seed, logger=logger)

    with phase("load cache"):
        embeds = TextEmbeds(config.cache_dir)
        train_ds = LatentEmbedDataset(config.cache_dir, train_idx)
        val_ds = LatentEmbedDataset(config.cache_dir, val_idx)
    train_collate = make_collate_fn(embeds, config.text_drop_prob, config.seed)
    val_collate = make_collate_fn(embeds, 0.0, config.seed + 1)

    global_batch = config.batch_size * dp_size
    dp_mapper = ttml.mesh().axis_mapper("dp", tdim=0) if dp_size > 1 else None
    logger.info(
        f"batch: {config.batch_size}/device x dp={dp_size} = {global_batch} global, "
        f"accum={config.gradient_accumulation_steps} -> effective "
        f"{global_batch * config.gradient_accumulation_steps}"
    )

    train_loader = InMemoryDataloader(
        train_ds,
        train_collate,
        batch_size=global_batch,
        shuffle=True,
        drop_last=True,
        seed=config.seed + config.resume_step,
    )
    val_loader = InMemoryDataloader(
        val_ds, val_collate, batch_size=1, shuffle=False, drop_last=False, seed=config.seed + 1
    )

    with phase("load experts + inject LoRA"):
        experts = {role: build_lora_expert(role, config, logger) for role in config.experts_to_load()}
    if config.resume_step:
        with phase(f"resume from step {config.resume_step}"):
            load_all(experts, config, suffix=f"_step{config.resume_step:05d}", logger=logger)
    for m in experts.values():
        m.train()

    trainable: dict[str, Any] = {}
    for role, model in experts.items():
        for name, param in model.parameters().items():
            if "lora" in name:
                trainable[f"{role}/{name}"] = param
    adamw_config = ttml.optimizers.AdamWConfig.make(config.learning_rate, 0.9, 0.999, 1e-8, config.weight_decay)
    optimizer = ttml.optimizers.AdamW(trainable, adamw_config)
    logger.info(f"constant lr={config.learning_rate}, {len(trainable)} trainable tensors")

    patch_size = model_config_for(config).patch_size
    shape = latent_shape(config)
    with phase("build RoPE tables"):
        rope_params = build_rope_params(
            head_dim=model_config_for(config).head_dim,
            patch_size=patch_size,
            latent_shape=shape,
            max_seq_len=model_config_for(config).rope_max_seq_len,
        )
    logger.info(f"latent {shape} patch {patch_size} -> {rope_params.sequence_length} tokens")

    lo, hi = _range_for(config)

    global_step, micro = config.resume_step, 0
    accum_loss, accum_n = 0.0, 0
    ema = None
    step_times: list[float] = []
    loop_start = step_start = time.time()
    data_iter = iter(train_loader)
    optimizer.zero_grad()
    logger.info(
        f"loop: step {global_step} -> max_steps={config.max_steps} "
        f"accum={config.gradient_accumulation_steps} t-range=[{lo:.3f},{hi:.3f})"
    )

    while global_step < config.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        t = _sample_timestep(config, lo, hi, rng)
        model = _route(t, experts, config)
        loss = flow_matching_step(model, batch, t, rope_params, patch_size, rng, dp_mapper=dp_mapper)
        accum_loss += _loss_value(loss)
        accum_n += 1

        if config.gradient_accumulation_steps > 1:
            loss = loss * (1.0 / float(config.gradient_accumulation_steps))
        loss.backward(False)
        ctx.reset_graph()
        micro += 1

        if micro % config.gradient_accumulation_steps == 0:
            if dp_size > 1:
                ttml.sync_gradients(trainable, axis_names=("dp",))
            if config.grad_clip > 0.0:
                ttml.core.clip_grad_norm(trainable, config.grad_clip, 2.0, False)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            avg = accum_loss / accum_n
            ema = avg if ema is None else 0.9 * ema + 0.1 * avg
            dt = time.time() - step_start
            step_start = time.time()
            step_times.append(dt)
            if global_step == 1:
                logger.info(f"[time] first step (includes kernel compile): {fmt(dt)}")
            logger.log_metrics(
                {"train/loss": avg, "train/loss_ema": ema, "train/step_time_s": dt},
                step=global_step,
            )
            accum_loss, accum_n = 0.0, 0

            if config.val_loss_every and global_step % config.val_loss_every == 0:
                with phase(f"val @ step {global_step}"):
                    vloss = validation_loss(experts, val_loader, config, ctx, rope_params, patch_size)
                logger.log_metrics({"val/loss": vloss}, step=global_step)
                step_start = time.time()

            if config.ckpt_every and global_step % config.ckpt_every == 0:
                with phase(f"checkpoint @ step {global_step}"):
                    save_all(experts, config, suffix=f"_step{global_step:05d}", logger=logger)
                step_start = time.time()

    record("train loop", time.time() - loop_start)
    if len(step_times) > 1:
        steady = step_times[1:]
        mean = sum(steady) / len(steady)
        logger.info(
            f"[time] steady-state step: {fmt(mean)} mean over {len(steady)} steps "
            f"(min {fmt(min(steady))}, max {fmt(max(steady))}) — "
            f"{config.gradient_accumulation_steps} micro-steps each"
        )

    with phase("save final LoRA"):
        save_all(experts, config, logger=logger)
    logger.info(f"done at step {global_step}. LoRA(s): {', '.join(config.expert_path(r) for r in experts)}")


if __name__ == "__main__":
    args = parse_cli_options(default_config=DEFAULT_CONFIG)
    config: TrainingConfig = generate_config(TrainingConfig, args.config, args.test_config, overrides=args.overrides)

    logger = TrainingLogger(config, args.test_log_filename_prefix)
    set_sink(logger.info)

    started = time.perf_counter()
    try:
        if config.mode == "infer":
            from blacksmith.experiments.tt_train.wan2_2.generate import generate

            generate(config, logger)
        else:
            train(config, logger)
    finally:
        summary(config.mode, time.perf_counter() - started)
        logger.finish()
