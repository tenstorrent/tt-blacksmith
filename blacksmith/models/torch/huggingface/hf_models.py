# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from blacksmith.tools.templates.configs import TrainingConfig


def _is_trainable_param(model: torch.nn.Module, param_path: str) -> bool:
    """Look up a parameter by its pre-parametrize dotted path and return whether it's trainable.

    After register_parametrization the original lives at
    `<path-without-.weight>.parametrizations.<weight-name>.original`, so we try
    that location first and fall back to the original path.
    """
    module_path, param_name = param_path.rsplit(".", 1)
    try:
        module = model.get_submodule(module_path)
    except AttributeError:
        return False
    param = None
    if hasattr(module, "parametrizations") and param_name in getattr(module.parametrizations, "_modules", {}):
        param = getattr(module.parametrizations[param_name], "original", None)
    if param is None:
        param = getattr(module, param_name, None)
    return isinstance(param, torch.nn.Parameter) and param.requires_grad


def get_model(config: TrainingConfig, device: torch.device):
    # This will be replaced with forge models loader, we should add adapter functions to modify the model as needed

    # Load a model
    model = AutoModelForCausalLM.from_pretrained(config.model_name, use_cache=config.gradient_checkpointing)

    # Apply training specific modifications
    # Apply LoRA if rank is specified
    if config.training_type == "lora":
        model = _apply_lora(model, config)
    elif config.training_type == "adapters":
        _apply_adapters(model, config)
    else:
        raise ValueError(f"Invalid training type: {config.training_type}")

    model.to(eval(config.dtype))
    model.to(device)

    # Per-tensor weight dtype overrides must be registered before torch.compile
    # so the custom_call appears in the traced graph.
    overrides = getattr(config, "weight_dtype_overrides", None)
    if config.use_tt and overrides:
        from tt_torch import apply_weight_dtype_overrides

        applied = apply_weight_dtype_overrides(model, overrides)

        # register_parametrization does `set_(original, original)` internally,
        # which freezes XLA storage. If the target is trainable, the optimizer
        # later fails with "cannot mutate tensors with frozen storage" on the
        # first in-place update. Fail loudly here with an actionable message.
        trainable_hits = [name for name, _ in applied if _is_trainable_param(model, name)]
        if trainable_hits:
            raise RuntimeError(
                "weight_dtype_overrides matched trainable parameters, which is "
                "unsupported during training on XLA (torch parametrize freezes "
                "their storage and optimizer.step() will fail). Restrict the "
                "config to frozen weights only.\nOffending parameters:\n  - " + "\n  - ".join(trainable_hits)
            )

    if config.use_tt:
        compile_options = {"tt_enable_torch_fx_fusion_pass": False, "tt_legacy_compile": True}
        model = torch.compile(model, backend="tt", options=compile_options)

    return model


def _apply_lora(model, config: TrainingConfig):
    # When unfreeze_embeddings is enabled, use modules_to_save to also train
    # the embedding layer alongside LoRA adapters. This is needed for models
    # like Falcon3 that have limited language coverage - unfreezing embeddings
    # allows the model to adapt token representations for unseen languages.
    modules_to_save = None
    if getattr(config, "unfreeze_embeddings", False):
        modules_to_save = ["embed_tokens"]

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        task_type=config.lora_task_type,
        modules_to_save=modules_to_save,
    )

    return get_peft_model(model, lora_config)


def _apply_adapters(model, config: TrainingConfig):
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Apply adapters
    if len(config.adapter_layers) == 0:
        adapter_layers = list(range(len(model.model.layers)))
    else:
        adapter_layers = config.adapter_layers

    for block_idx in adapter_layers:
        #### Insert first adapter
        original_layer_output = model.model.layers[block_idx].self_attn.o_proj
        adapted_layer = make_adapted_layer(original_layer_output, config)
        model.model.layers[block_idx].self_attn.o_proj = adapted_layer

        #### Insert second adapter
        original_layer_output = model.model.layers[block_idx].mlp.down_proj
        adapted_layer = make_adapted_layer(original_layer_output, config)
        model.model.layers[block_idx].mlp.down_proj = adapted_layer

    return model


def make_adapted_layer(linear, config: TrainingConfig):
    class ResidualAdapter(nn.Module):
        def __init__(self, linear, bottleneck_dim):
            super().__init__()
            self.linear = linear
            d = linear.out_features

            self.adapter = nn.Sequential(
                nn.Linear(d, bottleneck_dim),
                nn.GELU(),
                nn.Linear(bottleneck_dim, d),
            )

            # Start as identity
            nn.init.zeros_(self.adapter[-1].weight)
            nn.init.zeros_(self.adapter[-1].bias)

        def forward(self, x):
            y = self.linear(x)
            return y + self.adapter(y)

    return ResidualAdapter(linear, config.adapter_bottleneck_dim)
