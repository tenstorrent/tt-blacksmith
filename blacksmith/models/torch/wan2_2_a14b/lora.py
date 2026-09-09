# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import time
from pathlib import Path

import torch
import torch.nn as nn

from blacksmith.experiments.torch.wan2_2_a14b.configs import SUBFOLDER, TrainingConfig
from blacksmith.tools.kurbla.device_manager import DeviceManager

_ATTN_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]
_FFN_TARGETS = ["ffn.net.0.proj", "ffn.net.2"]

_PEFT_PARAM_RE = re.compile(r"^(?P<base>.+)\.(?P<slot>lora_A|lora_B)\.default\.weight$")


def lora_targets(config: TrainingConfig) -> list[str]:
    return _ATTN_TARGETS + (_FFN_TARGETS if config.lora_target_set == "attn+ffn" else [])


def build_lora_expert(
    role: str, config: TrainingConfig, device_manager: DeviceManager, logger=None
) -> nn.Module:
    """Load one expert, freeze it, inject LoRA, and distribute it over the mesh."""
    from diffusers import WanTransformer3DModel
    from peft import LoraConfig

    def log(message: str) -> None:
        logger.info(message) if logger else print(message, flush=True)

    started = time.perf_counter()
    transformer = WanTransformer3DModel.from_pretrained(
        config.model_id,
        subfolder=SUBFOLDER[role],
        torch_dtype=config.torch_dtype(),
        low_cpu_mem_usage=True,
    )
    log(f"[lora] {role}: loaded {SUBFOLDER[role]} in {time.perf_counter() - started:.1f}s")

    if config.dit_layers is not None and config.dit_layers < len(transformer.blocks):
        log(f"[lora] {role}: truncating to {config.dit_layers} of {len(transformer.blocks)} blocks")
        transformer.blocks = transformer.blocks[: config.dit_layers]

    transformer = device_manager.prepare_model(transformer)
    for param in transformer.parameters():
        param.requires_grad_(False)
    if config.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    transformer.add_adapter(
        LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=lora_targets(config),
            lora_dropout=0.0,
            init_lora_weights="gaussian" if config.lora_a_init == "gaussian" else True,
        )
    )

    # Sharded last so LoRA tensors become DTensors; to_device first would stage 14B/chip.
    started = time.perf_counter()
    if device_manager.mesh is None:
        transformer = device_manager.to_device(transformer)
    else:
        transformer = device_manager.shard_model(transformer)
    log(f"[lora] {role}: distributed in {time.perf_counter() - started:.1f}s")

    total = sum(p.numel() for p in transformer.parameters())
    trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    assert trainable > 0, f"{role}: no trainable LoRA params; check lora_target_set"
    assert trainable < total // 20, f"{role}: {trainable} trainable of {total}; LoRA is not isolated"
    log(f"[lora] {role}: {trainable} trainable of {total} params")
    return transformer


def lora_state_dict(transformer: nn.Module, device_manager: DeviceManager) -> dict[str, torch.Tensor]:
    """LoRA weights keyed the way diffusers loads them, gathered to full fp32 host tensors."""
    tensors = {}
    for name, param in transformer.named_parameters():
        match = _PEFT_PARAM_RE.match(name)
        if match is None:
            continue
        key = f"transformer.{match['base']}.{match['slot']}.weight"
        tensors[key] = device_manager.gather(param.detach()).float().cpu()
    return tensors


def save_lora(
    transformer: nn.Module, path: str | Path, device_manager: DeviceManager, logger=None
) -> int:
    from safetensors.torch import save_file

    tensors = lora_state_dict(transformer, device_manager)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))
    if logger:
        logger.info(f"[lora] wrote {len(tensors)} tensors -> {path}")
    return len(tensors)


def load_lora(transformer: nn.Module, path: str | Path, logger=None) -> int:
    from safetensors.torch import load_file

    tensors = load_file(str(path))
    loaded = 0
    with torch.no_grad():
        for name, param in transformer.named_parameters():
            match = _PEFT_PARAM_RE.match(name)
            if match is None:
                continue
            key = f"transformer.{match['base']}.{match['slot']}.weight"
            if key not in tensors:
                raise KeyError(f"{path}: missing {key}")
            param.copy_(tensors[key].to(param.dtype))
            loaded += 1
    if logger:
        logger.info(f"[lora] loaded {loaded} tensors <- {path}")
    return loaded


def expert_suffix_path(config: TrainingConfig, role: str, step: int | None = None) -> Path:
    path = Path(config.expert_path(role))
    if step is None:
        return path
    return path.with_name(f"{path.stem}_step{step:05d}{path.suffix}")
