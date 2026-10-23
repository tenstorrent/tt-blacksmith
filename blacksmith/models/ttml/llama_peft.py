# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Llama LoRA adapters between ttml's layout and HF PEFT's. Pure numpy: no ttml import.

ttml's Llama differs from HF's in two ways that adapters see, and both convert exactly:

- q / k rows are permuted per head (HF's half-split RoPE layout -> interleaved pairs):
  ``W_ttml = P @ W_hf``. Since ``P @ W + s * B @ A == P @ (W + s * P^-1 @ B @ A)``, the HF adapter
  is ``(A, P^-1 @ B)``: ``A`` is unchanged, ``B``'s rows are permuted back.
- k and v are one ``kv_linear`` with rows ``[k; v]``. Its adapter ``(A, [B_k; B_v])`` is the two
  PEFT adapters ``(A, B_k)`` and ``(A, B_v)``, which share ``A`` (and r / alpha, so the scaling).
  The reverse needs ``A_k == A_v``: independent k / v adapters have no rank-r fused equivalent.

``o`` is not permuted (v isn't, so o's inputs are in HF head order).
"""
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from blacksmith.models.ttml.provider import UnsupportedTrainingConfig

# (A [r, in], B [out, r]) of one LoRA layer.
LoraPair = Tuple[np.ndarray, np.ndarray]
# Per decoder layer: ttml projection ("q", "kv", "o") -> its adapter, for the targeted ones.
LayerAdapters = Dict[str, LoraPair]

PEFT_PREFIX = "base_model.model."
PEFT_KEY = "model.layers.{layer}.self_attn.{proj}.lora_{part}.weight"
# HF target module(s) of each ttml projection.
HF_PROJECTIONS = {"q": ("q_proj",), "kv": ("k_proj", "v_proj"), "o": ("o_proj",)}


def _interleave_index(rows: int, n_heads: int) -> np.ndarray:
    """Row index taking HF's per-head [first half, second half] to ttml's interleaved pairs."""
    head_dim = rows // n_heads
    half = head_dim // 2
    index = np.empty(rows, dtype=np.int64)
    for head in range(n_heads):
        base = head * head_dim
        index[base : base + head_dim : 2] = base + np.arange(half)
        index[base + 1 : base + head_dim : 2] = base + half + np.arange(half)
    return index


def to_ttml_rows(w: np.ndarray, n_heads: int) -> np.ndarray:
    """HF q/k rows -> ttml's (what ttml's safetensors loader does to the weights)."""
    return w[_interleave_index(w.shape[0], n_heads)]


def to_hf_rows(w: np.ndarray, n_heads: int) -> np.ndarray:
    """ttml q/k rows -> HF's; the inverse of ``to_ttml_rows``."""
    return w[np.argsort(_interleave_index(w.shape[0], n_heads))]


def hf_target_modules(ttml_projections: Sequence[str]) -> List[str]:
    return sorted(name for projection in ttml_projections for name in HF_PROJECTIONS[projection])


def ttml_to_peft(layers: Sequence[LayerAdapters], num_heads: int, num_kv_heads: int) -> Dict[str, np.ndarray]:
    """ttml adapters per layer -> PEFT tensors (keys as ``save_pretrained`` writes them)."""
    tensors = {}

    def put(layer, proj, a, b):
        tensors[PEFT_PREFIX + PEFT_KEY.format(layer=layer, proj=proj, part="A")] = np.ascontiguousarray(a)
        tensors[PEFT_PREFIX + PEFT_KEY.format(layer=layer, proj=proj, part="B")] = np.ascontiguousarray(b)

    for index, adapters in enumerate(layers):
        if "q" in adapters:
            a, b = adapters["q"]
            put(index, "q_proj", a, to_hf_rows(b, num_heads))
        if "kv" in adapters:
            a, b = adapters["kv"]
            k_rows = b.shape[0] // 2
            put(index, "k_proj", a, to_hf_rows(b[:k_rows], num_kv_heads))
            put(index, "v_proj", a, b[k_rows:])
        if "o" in adapters:
            put(index, "o_proj", *adapters["o"])
    return tensors


def _strip_prefix(tensors: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {key[len(PEFT_PREFIX) :] if key.startswith(PEFT_PREFIX) else key: value for key, value in tensors.items()}


def peft_to_ttml(
    tensors: Dict[str, np.ndarray],
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    projections: Sequence[str],
) -> List[LayerAdapters]:
    """PEFT tensors -> ttml adapters per layer for ``projections`` (e.g. ``["q", "kv"]``).

    Raises ``UnsupportedTrainingConfig`` when the tensors are not exactly those adapters, or when
    k and v don't share ``A`` (see the module docstring).
    """
    tensors = _strip_prefix(tensors)
    expected = {
        PEFT_KEY.format(layer=layer, proj=proj, part=part)
        for layer in range(num_layers)
        for projection in projections
        for proj in HF_PROJECTIONS[projection]
        for part in ("A", "B")
    }
    missing, unexpected = sorted(expected - tensors.keys()), sorted(tensors.keys() - expected)
    if missing or unexpected:
        raise UnsupportedTrainingConfig(
            "checkpoint: adapter tensors don't match the model's LoRA layers "
            f"(missing {missing[:4]}{'...' if len(missing) > 4 else ''}, "
            f"unexpected {unexpected[:4]}{'...' if len(unexpected) > 4 else ''})"
        )

    def get(layer, proj) -> LoraPair:
        key = PEFT_KEY.format(layer=layer, proj=proj, part="{}")
        return tensors[key.format("A")], tensors[key.format("B")]

    layers = []
    for layer in range(num_layers):
        adapters: LayerAdapters = {}
        if "q" in projections:
            a, b = get(layer, "q_proj")
            adapters["q"] = (a, to_ttml_rows(b, num_heads))
        if "kv" in projections:
            (a_k, b_k), (a_v, b_v) = get(layer, "k_proj"), get(layer, "v_proj")
            if a_k.shape != a_v.shape or not np.allclose(a_k, a_v):
                raise UnsupportedTrainingConfig(
                    f"checkpoint: layer {layer} has different lora_A for k_proj and v_proj; ttml's fused kv "
                    "projection can only load k / v adapters that share lora_A (as ttml saves them)"
                )
            adapters["kv"] = (a_k, np.concatenate([to_ttml_rows(b_k, num_kv_heads), b_v]))
        if "o" in projections:
            adapters["o"] = get(layer, "o_proj")
        layers.append(adapters)
    return layers


def peft_adapter_config(
    model_name: str,
    r: int,
    alpha: float,
    dropout: float,
    target_modules: Sequence[str],
    task_type: str,
    base_model_revision: Optional[str] = None,
) -> Dict:
    """The ``adapter_config.json`` PEFT writes for this LoRA."""
    return {
        "peft_type": "LORA",
        "task_type": task_type,
        "base_model_name_or_path": model_name,
        "revision": base_model_revision,
        "r": r,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "target_modules": sorted(target_modules),
        "bias": "none",
        "fan_in_fan_out": False,
        "use_rslora": False,
        "use_dora": False,
        "modules_to_save": None,
        "init_lora_weights": True,
        "layers_to_transform": None,
        "layers_pattern": None,
        "rank_pattern": {},
        "alpha_pattern": {},
        "inference_mode": True,
    }
