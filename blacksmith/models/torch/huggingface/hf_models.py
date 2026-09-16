# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace causal-LM loader with LoRA applied, for tt-crank experiments."""
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from blacksmith.tools.configs import TrainingConfig


def get_model(config: TrainingConfig, device: torch.device) -> torch.nn.Module:
    """Load the model, apply LoRA, cast, and move it to `device`.

    Returns the plain module. Unlike the tt-xla path, nothing is wrapped in
    `torch.compile` here: the tt-crank training step compiles forward+loss as a
    single callable (see the experiment's `train.py`), which keeps the loss and
    its backward inside the compiled graph.
    """
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        use_cache=False,
        low_cpu_mem_usage=True,
    )

    model = get_peft_model(
        model,
        LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            task_type=config.lora_task_type,
        ),
    )

    model.to(config.torch_dtype())
    # Cast on the host first: moving a bf16 model is half the transfer of
    # moving fp32 weights and then casting on device.
    model.to(device)
    return model
