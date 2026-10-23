# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""LoRA training of HF Llama checkpoints on ttml's Python ``Llama``.

What ttml's Llama can and can't train, and why ``check`` rejects the rest:

- k_proj and v_proj are fused into one ``kv_linear``, so an adapter targets both or neither.
- Without tensor parallelism, the MLP runs a fused swiglu on the raw ``w1`` / ``w2`` / ``w3``
  weights, so LoRA on gate / up / down projections would silently get no gradients.
- Sequence lengths are multiples of the 32-row tile; weights and adapters are bf16.
"""
import os
from typing import Any, Dict, List, Optional

import numpy as np

from blacksmith.configs import LoraConfig, TrainerConfig
from blacksmith.models.ttml.llama_peft import (
    hf_target_modules,
    peft_adapter_config,
    peft_to_ttml,
    ttml_to_peft,
)
from blacksmith.models.ttml.provider import AdapterState, UnsupportedTrainingConfig

SUPPORTED_DTYPES = ("bfloat16", "torch.bfloat16")
TILE = 32
# ttml's RMSNorm epsilon is fixed.
RMS_NORM_EPS = 1e-5

ATTENTION_TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj"}
MLP_TARGETS = {"gate_proj", "up_proj", "down_proj"}
# Regexes on the ttml module names `LoraModel` matches (with `re.search`), e.g. "blocks.3.attention.q_linear".
Q_PATTERN = r"\.attention\.q_linear$"
KV_PATTERN = r"\.attention\.kv_linear$"
O_PATTERN = r"\.attention\.out_linear$"


def _fail(field: str, reason: str):
    raise UnsupportedTrainingConfig(f"{field}: {reason}")


def _rope_scaling(hf_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The HF rope scaling block, or None for unscaled RoPE (older and transformers-5 layouts)."""
    rope = hf_config.get("rope_scaling") or hf_config.get("rope_parameters")
    if not rope or rope.get("rope_type", rope.get("type", "default")) == "default":
        return None
    return rope


def _rope_theta(hf_config: Dict[str, Any]) -> float:
    if "rope_theta" in hf_config:
        return float(hf_config["rope_theta"])
    return float((hf_config.get("rope_parameters") or {}).get("rope_theta", 10000.0))


def llama_config_kwargs(hf_config: Dict[str, Any], max_length: int) -> Dict[str, Any]:
    """HF ``config.json`` -> keyword arguments of ``ttml.models.llama.LlamaConfig`` (as plain values)."""
    rope = _rope_scaling(hf_config)
    return {
        "hidden_size": hf_config["hidden_size"],
        "intermediate_size": hf_config["intermediate_size"],
        "num_hidden_layers": hf_config["num_hidden_layers"],
        "num_attention_heads": hf_config["num_attention_heads"],
        "num_key_value_heads": hf_config.get("num_key_value_heads", hf_config["num_attention_heads"]),
        "vocab_size": hf_config["vocab_size"],
        # Sizes the RoPE tables; training sequences are exactly max_length long.
        "max_position_embeddings": max_length,
        "rope_theta": _rope_theta(hf_config),
        "attention_bias": bool(hf_config.get("attention_bias", False)),
        "tie_word_embeddings": bool(hf_config.get("tie_word_embeddings", False)),
        "rope_scaling": (
            None
            if rope is None
            else {
                "scaling_factor": float(rope["factor"]),
                "high_freq_factor": float(rope["high_freq_factor"]),
                "low_freq_factor": float(rope["low_freq_factor"]),
                "original_context_length": int(rope["original_max_position_embeddings"]),
            }
        ),
    }


def ttml_projections(lora: LoraConfig) -> List[str]:
    """The ttml attention projections ("q", "kv", "o") the HF targets select."""
    targets = set(lora.target_modules)
    return [
        projection
        for projection, needed in (("q", {"q_proj"}), ("kv", {"k_proj", "v_proj"}), ("o", {"o_proj"}))
        if needed <= targets
    ]


def _lora_layers(model: Any) -> List[Dict[str, Any]]:
    """Per decoder block: ttml projection -> its ``LoraLinear``, for the projections LoRA wraps."""
    from ttml.modules import LoraLinear

    layers = []
    for block in model.model.blocks:
        attention = block.attention
        candidates = (("q", attention.q_linear), ("kv", attention.kv_linear), ("o", attention.out_linear))
        layers.append({name: layer for name, layer in candidates if isinstance(layer, LoraLinear)})
    return layers


def lora_config_kwargs(lora: LoraConfig) -> Dict[str, Any]:
    """blacksmith ``LoraConfig`` (HF target names) -> keyword arguments of ``ttml.modules.LoraConfig``."""
    patterns = {"q": Q_PATTERN, "kv": KV_PATTERN, "o": O_PATTERN}
    return {
        "rank": lora.r,
        # Scaling alpha / r, as in PEFT.
        "alpha": float(lora.alpha),
        "lora_dropout": lora.dropout,
        "target_modules": [patterns[projection] for projection in ttml_projections(lora)],
        "use_rslora": False,
        "is_bias_trainable": False,
    }


class LlamaProvider:
    """Provider for HF ``model_type: llama`` checkpoints (Llama 3.x)."""

    def check(self, config: TrainerConfig, hf_config: Dict[str, Any]) -> None:
        lora = config.lora
        if lora is None:
            _fail("lora", "the ttml Llama provider only trains LoRA adapters; full fine-tuning is not supported")
        if lora.task_type != "CAUSAL_LM":
            _fail("lora.task_type", f"only CAUSAL_LM is supported, got {lora.task_type!r}")

        targets = set(lora.target_modules)
        if not targets:
            _fail("lora.target_modules", "no target modules given")
        mlp = sorted(targets & MLP_TARGETS)
        if mlp:
            _fail(
                "lora.target_modules",
                f"{mlp} can't be trained: ttml's Llama MLP runs a fused swiglu on the raw weights, so adapters on "
                "gate/up/down projections would get no gradients. Use q_proj, k_proj + v_proj, o_proj.",
            )
        unknown = sorted(targets - ATTENTION_TARGETS)
        if unknown:
            _fail("lora.target_modules", f"{unknown} are not Llama projections; use q_proj, k_proj + v_proj, o_proj")
        if len(targets & {"k_proj", "v_proj"}) == 1:
            _fail(
                "lora.target_modules",
                "k_proj and v_proj are one fused kv projection in ttml's Llama; target both or neither",
            )

        if config.max_length % TILE != 0:
            _fail("max_length", f"must be a multiple of {TILE} (the ttml tile size), got {config.max_length}")
        if config.max_length > hf_config.get("max_position_embeddings", config.max_length):
            _fail("max_length", f"exceeds the model's max_position_embeddings ({hf_config['max_position_embeddings']})")
        if config.dtype not in SUPPORTED_DTYPES:
            _fail("dtype", f"ttml Llama weights and adapters are bf16; use 'bfloat16', got {config.dtype!r}")

        rope = _rope_scaling(hf_config)
        if rope is not None and rope.get("rope_type", rope.get("type")) != "llama3":
            _fail("model_name", f"RoPE scaling {rope.get('rope_type', rope.get('type'))!r} is not implemented in ttml")
        for flag in ("attention_bias", "mlp_bias"):
            if hf_config.get(flag, False):
                _fail("model_name", f"{flag} is not implemented in ttml's Llama")
        if hf_config.get("hidden_act", "silu") != "silu":
            _fail("model_name", f"activation {hf_config['hidden_act']!r} is not implemented in ttml's Llama")
        if hf_config.get("rms_norm_eps", RMS_NORM_EPS) != RMS_NORM_EPS:
            _fail("model_name", f"rms_norm_eps {hf_config['rms_norm_eps']} differs from ttml's fixed {RMS_NORM_EPS}")
        head_dim = hf_config["hidden_size"] // hf_config["num_attention_heads"]
        if hf_config.get("head_dim", head_dim) != head_dim:
            _fail("model_name", f"head_dim {hf_config['head_dim']} != hidden_size / num_attention_heads ({head_dim})")

    def build(self, config: TrainerConfig, hf_config: Dict[str, Any]) -> Any:
        from huggingface_hub import snapshot_download
        from ttml.models import WeightTyingType
        from ttml.models.llama import (
            Llama,
            LlamaConfig,
            LlamaRopeScalingConfig,
            load_from_safetensors,
        )
        from ttml.modules import LoraConfig as TtmlLoraConfig
        from ttml.modules import LoraModel

        kwargs = llama_config_kwargs(hf_config, config.max_length)
        rope = kwargs.pop("rope_scaling")
        tied = kwargs.pop("tie_word_embeddings")
        try:
            llama_config = LlamaConfig(
                **kwargs,
                rope_scaling=LlamaRopeScalingConfig(**rope) if rope else LlamaRopeScalingConfig(),
                weight_tying=WeightTyingType.Enabled if tied else WeightTyingType.Disabled,
            )
        except ValueError as e:
            raise UnsupportedTrainingConfig(f"model_name: ttml rejects this Llama config: {e}") from e

        if os.path.isdir(config.model_name):
            weights_dir = config.model_name
        else:
            weights_dir = snapshot_download(config.model_name, allow_patterns=["*.safetensors", "config.json"])

        model = Llama(llama_config)
        load_from_safetensors(model, weights_dir, llama_config)
        try:
            return LoraModel(model, TtmlLoraConfig(**lora_config_kwargs(config.lora)))
        except (ValueError, RuntimeError) as e:
            raise UnsupportedTrainingConfig(f"lora.target_modules: {e}") from e

    def export_adapter(self, model: Any, config: TrainerConfig) -> AdapterState:
        layers = []
        for lora_layers in _lora_layers(model):
            # Stored [1, 1, r, in] / [1, 1, out, r] in bf16; read as float32, which is lossless.
            layers.append(
                {
                    name: (layer.lora_A.tensor.to_numpy()[0, 0], layer.lora_B.tensor.to_numpy()[0, 0])
                    for name, layer in lora_layers.items()
                }
            )
        llama_config = model.model.config
        lora = config.lora
        return AdapterState(
            tensors=ttml_to_peft(layers, llama_config.num_attention_heads, llama_config.num_key_value_heads),
            config=peft_adapter_config(
                config.model_name,
                r=lora.r,
                alpha=lora.alpha,
                dropout=lora.dropout,
                target_modules=hf_target_modules(ttml_projections(lora)),
                task_type=lora.task_type,
            ),
        )

    def import_adapter(self, model: Any, state: AdapterState, config: TrainerConfig) -> None:
        lora = config.lora
        projections = ttml_projections(lora)
        adapter = state.config
        expected = {
            "peft_type": "LORA",
            "r": lora.r,
            "lora_alpha": lora.alpha,
            "use_rslora": False,
            "target_modules": hf_target_modules(projections),
        }
        found = {key: adapter.get(key) for key in expected}
        if isinstance(found["target_modules"], list):
            found["target_modules"] = sorted(found["target_modules"])
        mismatched = {key: (found[key], value) for key, value in expected.items() if found[key] != value}
        if mismatched:
            details = ", ".join(f"{key}: adapter {got!r} vs run {want!r}" for key, (got, want) in mismatched.items())
            raise UnsupportedTrainingConfig(f"checkpoint: adapter doesn't match this run's LoRA ({details})")

        import ttml

        llama_config = model.model.config
        adapters = peft_to_ttml(
            state.tensors,
            num_layers=llama_config.num_hidden_layers,
            num_heads=llama_config.num_attention_heads,
            num_kv_heads=llama_config.num_key_value_heads,
            projections=projections,
        )
        for lora_layers, layer_adapters in zip(_lora_layers(model), adapters):
            for name, layer in lora_layers.items():
                a, b = layer_adapters[name]
                for parameter, value in ((layer.lora_A, a), (layer.lora_B, b)):
                    shape = tuple(parameter.tensor.shape()[2:])
                    if value.shape != shape:
                        raise UnsupportedTrainingConfig(
                            f"checkpoint: adapter tensor of shape {value.shape} for a {shape} LoRA weight"
                        )
                    # Keeps the parameter's dtype (bf16) and layout.
                    ttml.autograd.assign_numpy(parameter.tensor, value.reshape(1, 1, *shape).astype(np.float32))
