# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from peft import LoraConfig, get_peft_model
from transformers import Gemma4ForConditionalGeneration

from blacksmith.tools.device_manager import DeviceManager


# TODO(umales): Align get_model signatures between hf_models.py and this function.
def get_model(config, device_manager: DeviceManager, shard_model=False):
    """
    Load Gemma 4 E2B (text-only view) with multimodal towers stripped,
    `embed_tokens_per_layer` split workaround applied, LoRA, and compilation.
    """
    dtype = eval(config.dtype)

    base = Gemma4ForConditionalGeneration.from_pretrained(config.model_name, torch_dtype=dtype)
    _strip_multimodal_towers(base)
    _patch_embed_tokens_per_layer_split(base)

    if config.training_type == "lora":
        lora_cfg = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            task_type=config.lora_task_type,
        )
        model = get_peft_model(base, lora_cfg)
    else:
        raise ValueError(
            f"Only training_type='lora' is supported for Gemma 4 E2B, got '{config.training_type}'."
        )

    model.to(dtype)
    model.to(device_manager.device)

    if shard_model:
        model = device_manager.shard_model(model)

    if config.use_tt:
        compile_options = {
            "tt_enable_torch_fx_fusion_pass": False,
            "tt_legacy_compile": True,
            "tt_use_aot_autograd": False,
        }
        model = torch.compile(model, backend="tt", options=compile_options)

    return model


def get_vocab_size(model: torch.nn.Module) -> int:
    m = model
    while hasattr(m, "model") and not hasattr(m, "config"):
        m = m.model
    cfg = m.config
    return getattr(cfg, "vocab_size", None) or cfg.text_config.vocab_size


def _strip_multimodal_towers(model: Gemma4ForConditionalGeneration) -> None:
    # Drop vision/audio towers + embedders so PEFT only attaches to text attn.
    inner = model.model
    for attr in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
        if hasattr(inner, attr):
            delattr(inner, attr)


def _patch_embed_tokens_per_layer_split(model: torch.nn.Module) -> int:
    # Workaround ttnn.embedding silent-corruption at HIDDEN > 256 col-tiles:
    # Gemma-4 `embed_tokens_per_layer` is (V, 8960). Split lookup along hidden
    # dim into 2 halves, concat, then apply `embed_scale`.
    n = 0
    for mod_name, mod in model.named_modules():
        if not mod_name.endswith("embed_tokens_per_layer"):
            continue
        if not isinstance(mod, torch.nn.Embedding):
            continue

        embed_scale = getattr(mod, "embed_scale", None)

        def _make_split_forward(embed_mod, scale):
            def _forward(input_ids):
                chunks = embed_mod.weight.chunk(2, dim=-1)
                outs = [torch.nn.functional.embedding(input_ids, w.contiguous()) for w in chunks]
                out = torch.cat(outs, dim=-1)
                if scale is not None:
                    out = out * scale
                return out

            return _forward

        mod.forward = _make_split_forward(mod, embed_scale)
        n += 1
    return n
