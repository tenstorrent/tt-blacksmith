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
from torch.utils.data import DataLoader

from blacksmith.datasets.torch.omniconsistency_lego.omniconsistency_lego_dataset import (
    LatentEmbedDataset,
    make_collate_fn,
)
from blacksmith.experiments.torch.wan2_2_a14b.configs import TrainingConfig
from blacksmith.models.torch.wan2_2_a14b.lora import build_lora_expert, expert_suffix_path, save_lora
from blacksmith.tools.kurbla.device_manager import DeviceManager

EMA_ALPHA = 0.1


def _timestep_range(config: TrainingConfig) -> tuple[float, float]:
    if config.train_experts == "low":
        return 0.0, config.boundary_ratio
    if config.train_experts == "high":
        return config.boundary_ratio, 1.0
    return 0.0, 1.0


def _sample_timestep(config: TrainingConfig, lo: float, hi: float, rng: np.random.Generator) -> float:
    """Shifted logit-normal, rejection-sampled into the expert's range."""
    shift = config.train_flow_shift
    while True:
        z = rng.standard_normal() * config.lognorm_std + config.lognorm_mean
        u = 1.0 / (1.0 + np.exp(-z))
        t = shift * u / (1.0 + (shift - 1.0) * u)
        if lo <= t < hi:
            return float(t)


def _route(t: float, experts: dict, config: TrainingConfig) -> str:
    if len(experts) == 1:
        return next(iter(experts))
    return "high" if t >= config.boundary_ratio else "low"


def flow_matching_step(
    model,
    batch: dict,
    t: float,
    config: TrainingConfig,
    device_manager: DeviceManager,
    *,
    fixed_noise: torch.Tensor | None = None,
    shard_batch: bool = True,
) -> torch.Tensor:
    dtype = config.torch_dtype()
    x0 = batch["latent"].to(dtype)
    noise = torch.randn(x0.shape, dtype=dtype) if fixed_noise is None else fixed_noise.to(dtype)

    inputs = {
        "latent": (1.0 - t) * x0 + t * noise,
        "target": noise - x0,
        "text_embed": batch["text_embed"].to(dtype),
        # fp32: bf16 would round t*1000 to the nearest integer.
        "timestep": torch.full((x0.shape[0],), t * 1000.0, dtype=torch.float32),
    }
    if shard_batch:
        inputs = device_manager.prepare_batch(inputs)
    else:
        inputs = {k: device_manager.to_device(v) for k, v in inputs.items()}

    pred = model(
        hidden_states=inputs["latent"],
        timestep=inputs["timestep"],
        encoder_hidden_states=inputs["text_embed"],
        return_dict=True,
    ).sample

    # sum/numel, not mse_loss: numel is global, so the mean is exact at any placement.
    target = inputs["target"]
    return (pred.float() - target.float()).pow(2).sum() / target.numel()


@torch.no_grad()
def validation_loss(experts, compiled, val_loader, config, device_manager) -> float:
    for model in experts.values():
        model.eval()
    lo, hi = _timestep_range(config)
    losses = []
    try:
        for batch in val_loader:
            idx = int(batch["idx"][0])
            rng = np.random.default_rng(config.seed + idx)
            t = _sample_timestep(config, lo, hi, rng)
            noise = torch.from_numpy(rng.standard_normal(tuple(batch["latent"].shape), dtype=np.float32))
            loss = flow_matching_step(
                compiled[_route(t, experts, config)],
                batch,
                t,
                config,
                device_manager,
                fixed_noise=noise,
                shard_batch=False,
            )
            losses.append(float(device_manager.gather(loss).item()))
    finally:
        for model in experts.values():
            model.train()
    return float(np.mean(losses)) if losses else float("nan")


def _build_loaders(config: TrainingConfig, logger):
    cache = Path(config.cache_dir)
    metadata_path = cache / "metadata.json"
    embeds_path = cache / "embeds.pt"
    if not metadata_path.exists() or not embeds_path.exists():
        raise FileNotFoundError(f"{cache}: run preprocess.py then precompute.py first")

    metadata = json.loads(metadata_path.read_text())
    all_indices = sorted(m["idx"] for m in metadata)
    val_indices = all_indices[-config.val_holdout :] if config.val_holdout else []
    train_indices = all_indices[: -config.val_holdout] if config.val_holdout else all_indices
    embeds = torch.load(embeds_path, weights_only=False)
    logger.info(f"{len(train_indices)} train / {len(val_indices)} val samples, {len(embeds)} embeds")

    train_loader = DataLoader(
        LatentEmbedDataset(config.cache_dir, train_indices),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=make_collate_fn(embeds, p_drop=config.text_drop_prob, seed=config.seed),
    )
    val_loader = DataLoader(
        LatentEmbedDataset(config.cache_dir, val_indices),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=make_collate_fn(embeds, p_drop=0.0, seed=config.seed + 1),
    )
    return train_loader, val_loader


def train(config: TrainingConfig, device_manager: DeviceManager, logger) -> None:
    try:
        train_loader, val_loader = _build_loaders(config, logger)

        experts = {
            role: build_lora_expert(role, config, device_manager, logger)
            for role in config.experts_to_load()
        }
        compiled = {role: device_manager.compile(model) for role, model in experts.items()}
        optimizers = {
            role: torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=config.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=config.weight_decay,
            )
            for role, model in experts.items()
        }
        for model in experts.values():
            model.train()

        lo, hi = _timestep_range(config)
        rng = np.random.default_rng(config.seed)
        data_iter = iter(train_loader)
        global_step = config.resume_step
        micro_step = 0
        accum_loss = 0.0
        accum_count = 0
        ema_loss = None
        step_times: list[float] = []
        step_started = time.perf_counter()

        logger.info(
            f"experts={list(experts)} global batch {config.batch_size} x accum "
            f"{config.gradient_accumulation_steps} -> effective "
            f"{config.batch_size * config.gradient_accumulation_steps}, t in [{lo}, {hi})"
        )

        while global_step < config.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            t = _sample_timestep(config, lo, hi, rng)
            role = _route(t, experts, config)
            loss = flow_matching_step(compiled[role], batch, t, config, device_manager)
            (loss / config.gradient_accumulation_steps).backward()
            device_manager.sync()

            accum_loss += float(device_manager.gather(loss.detach()).item())
            accum_count += 1
            micro_step += 1
            if micro_step % config.gradient_accumulation_steps != 0:
                continue

            for optimizer_role, optimizer in optimizers.items():
                if any(p.grad is not None for p in optimizer.param_groups[0]["params"]):
                    device_manager.optimizer_step(optimizer)
                optimizer.zero_grad(set_to_none=True)
            device_manager.sync()
            global_step += 1

            avg_loss = accum_loss / accum_count
            ema_loss = avg_loss if ema_loss is None else (1 - EMA_ALPHA) * ema_loss + EMA_ALPHA * avg_loss
            step_time = time.perf_counter() - step_started
            step_started = time.perf_counter()
            step_times.append(step_time)
            accum_loss = 0.0
            accum_count = 0

            logger.info(f"step {global_step} loss {avg_loss:.4f} ema {ema_loss:.4f} took {step_time:.2f}s")
            logger.log_metrics(
                {"train/loss": avg_loss, "train/loss_ema": ema_loss, "train/step_time_s": step_time},
                step=global_step,
                commit=False,
            )

            if config.val_loss_every and global_step % config.val_loss_every == 0:
                val_loss = validation_loss(experts, compiled, val_loader, config, device_manager)
                logger.info(f"step {global_step} val/loss {val_loss:.4f}")
                logger.log_metrics({"val/loss": val_loss}, step=global_step, commit=False)
                step_started = time.perf_counter()

            if config.ckpt_every and global_step % config.ckpt_every == 0:
                for save_role, model in experts.items():
                    save_lora(model, expert_suffix_path(config, save_role, global_step), device_manager, logger)
                step_started = time.perf_counter()

            logger.log_metrics({}, step=global_step, commit=True)

        if step_times:
            steady = step_times[1:] or step_times
            logger.info(
                f"timing: first step {step_times[0]:.1f}s (includes compile), steady mean "
                f"{np.mean(steady):.1f}s over {len(steady)} steps, min {np.min(steady):.1f}s, "
                f"per micro-step {np.mean(steady) / config.gradient_accumulation_steps:.1f}s"
            )

        for save_role, model in experts.items():
            save_lora(model, expert_suffix_path(config, save_role), device_manager, logger)
    except Exception as error:
        logger.error(str(error), traceback.format_exc())
        raise
    finally:
        logger.finish()


if __name__ == "__main__":
    from blacksmith.experiments.torch.wan2_2_a14b.kurbla.model_overrides import apply_generality_overrides
    from blacksmith.tools.cli import generate_config, parse_cli_options
    from blacksmith.tools.logging_manager import TrainingLogger
    from blacksmith.tools.reproducibility_manager import ReproducibilityManager

    DEFAULT_CONFIG = Path(__file__).parent / "kurbla" / "lora" / "galaxy" / "wan2_2_t2v_a14b_lego.yaml"
    args = parse_cli_options(default_config=DEFAULT_CONFIG)
    config: TrainingConfig = generate_config(TrainingConfig, args.config, args.test_config, overrides=args.overrides)

    ReproducibilityManager(config).setup()
    apply_generality_overrides()

    logger = TrainingLogger(config, args.test_log_filename_prefix)
    device_manager = DeviceManager(config)
    logger.info(f"device={device_manager.device} mesh={device_manager.mesh}")

    train(config, device_manager, logger)
