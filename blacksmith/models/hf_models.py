# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model
from transformers import AttentionInterface, AutoModelForCausalLM
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask

from blacksmith.configs import TrainerConfig

TORCH_DTYPES = {
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
}


def torch_dtype(dtype: str) -> torch.dtype:
    """Map a config dtype string (e.g. ``"torch.bfloat16"``) to a ``torch.dtype``."""
    try:
        return TORCH_DTYPES[dtype]
    except KeyError as e:
        raise ValueError(f"Unsupported dtype {dtype!r}; expected one of {sorted(TORCH_DTYPES)}") from e


# HF's SDPA, except a forward without a padding mask or KV cache gets no mask, so sdpa runs `is_causal` on tt-crank's
# fused sdpa_fw/sdpa_bw kernels. HF skips the mask only outside torch.compile. Under compile it also clears
# `allow_is_causal_skip` if position_ids are passed, so `use_vmap` (or/and overlays) and `allow_is_bidirectional_skip`
# are checked instead; packed sequences and `block_sequence_ids` set neither and need an explicit mask.
CAUSAL_SDPA = "sdpa_causal"


def _causal_sdpa_mask(**kwargs):
    q_offset, kv_offset = kwargs.get("q_offset", 0), kwargs.get("kv_offset", 0)
    without_cache = isinstance(q_offset, int) and isinstance(kv_offset, int) and q_offset == kv_offset == 0
    plain_causal = kwargs.get("allow_is_causal_skip", True) or (
        torch.compiler.is_compiling() and not kwargs.get("use_vmap") and "allow_is_bidirectional_skip" not in kwargs
    )
    if (
        plain_causal
        and kwargs.get("attention_mask") is None
        and kwargs.get("local_size") is None
        and without_cache
        and kwargs["q_length"] == kwargs["kv_length"]
    ):
        return None
    return sdpa_mask(**kwargs)


AttentionInterface.register(CAUSAL_SDPA, sdpa_attention_forward)
ALL_MASK_ATTENTION_FUNCTIONS.register(CAUSAL_SDPA, _causal_sdpa_mask)


# tt-crank lowers `aten.linear` as is, and tt-mlir runs a linear on [batch, seq, hidden] activations at about
# half the speed of the same linear on [batch * seq, hidden] once batch > 1.
class _FlatLinear(torch.nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input.reshape(-1, input.shape[-1]))
        return output.reshape(*input.shape[:-1], self.out_features)


def _flatten_linear_inputs(model: torch.nn.Module) -> None:
    for module in model.modules():
        if type(module) is torch.nn.Linear:
            module.__class__ = _FlatLinear


def get_model(config: TrainerConfig, device: torch.device):
    # Before loading, so an unsupported dtype fails without loading any weights.
    dtype = torch_dtype(config.dtype)

    # Load a model
    load_kwargs = {
        "use_cache": False,
        "low_cpu_mem_usage": True,
        "attn_implementation": CAUSAL_SDPA,
    }
    if device.type == "cuda":
        # Stream weights directly to GPU to avoid CPU RAM spikes during load.
        load_kwargs["device_map"] = device

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **load_kwargs)

    # Apply LoRA when configured; otherwise full fine-tuning (all params trainable).
    if config.lora is not None:
        model = _apply_lora(model, config.lora)
    else:
        for param in model.parameters():
            param.requires_grad = True

    model.to(dtype)
    if config.use_tt:
        _flatten_linear_inputs(model)
        model.to(device)

    return model


def _apply_lora(model, lora_config):
    peft_config = PeftLoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout,
        target_modules=lora_config.target_modules,
        task_type=lora_config.task_type,
    )

    return get_peft_model(model, peft_config)
