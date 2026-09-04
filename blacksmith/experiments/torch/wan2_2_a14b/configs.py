# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Self

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

SUBFOLDER = {"high": "transformer", "low": "transformer_2"}

_TORCH_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
}


class InferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    infer_h: int = Field(default=512, gt=0)
    infer_w: int = Field(default=512, gt=0)
    infer_frames: int = Field(default=49, gt=0)
    infer_fps: int = Field(default=16, gt=0)
    infer_steps: int = Field(default=40, gt=0)
    infer_guidance: float = Field(default=7.0)
    infer_guidance_2: float = Field(default=5.0)
    infer_flow_shift: float = Field(default=12.0)
    infer_output: str = Field(default="cache/wan22_14b_lego/lego_video.mp4")
    val_prompt: str = Field(default="a cat sitting on a wooden table")
    neg_prompt: str = Field(default="")
    infer_no_lora: bool = Field(default=False)
    lora_scale: float = Field(default=1.0)
    infer_high_lora: str = Field(default="")
    infer_low_lora: str = Field(default="")

    @model_validator(mode="after")
    def check_frames(self) -> Self:
        if self.infer_frames != 1 and (self.infer_frames - 1) % 4 != 0:
            raise ValueError(
                f"infer_frames={self.infer_frames} must be 1 or 4k+1 (1, 5, 9, ... 49); other values "
                f"decode to black video (VAE temporal stride 4)."
            )
        return self


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    mode: Literal["train", "infer"] = Field(default="train")

    model_id: str = Field(default="Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    boundary_ratio: float = Field(default=0.875, gt=0.0, lt=1.0)
    train_experts: Literal["low", "high", "both"] = Field(default="both")
    dtype: str = Field(default="bfloat16")
    vae_dtype: str = Field(default="bfloat16")
    gradient_checkpointing: bool = Field(default=True)
    conv3d_patch_embed: bool = Field(default=False)

    dataset_id: str = Field(default="showlab/OmniConsistency")
    style: str = Field(default="LEGO")
    data_dir: str = Field(default="data/lego")
    cache_dir: str = Field(default="cache/wan22_14b_lego")
    train_h: int = Field(default=512, gt=0)
    train_w: int = Field(default=512, gt=0)
    train_frames: int = Field(default=1, gt=0)
    trigger: str = Field(default="lg, ")
    strip_style_words: bool = Field(default=True)
    text_drop_prob: float = Field(default=0.10, ge=0.0, le=1.0)
    subset_size: int = Field(default=0, ge=0)
    max_seq: int = Field(default=512, gt=0)
    val_holdout: int = Field(default=4, ge=0)

    lora_rank: int = Field(default=32, gt=0)
    lora_alpha: int = Field(default=32, gt=0)
    lora_target_set: Literal["attn", "attn+ffn"] = Field(default="attn")
    lora_a_init: Literal["gaussian", "kaiming"] = Field(default="gaussian")
    lora_path: str = Field(default="cache/wan22_14b_lego/wan22_14b_lego_lora.safetensors")

    learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    grad_clip: float = Field(default=1.0, ge=0)
    batch_size: int = Field(default=1, gt=0)
    gradient_accumulation_steps: int = Field(default=4, gt=0)
    max_steps: int = Field(default=3000, gt=0)
    train_flow_shift: float = Field(default=3.0, gt=0)
    lognorm_mean: float = Field(default=0.0)
    lognorm_std: float = Field(default=1.0, gt=0)
    val_loss_every: int = Field(default=200, ge=0)
    ckpt_every: int = Field(default=500, ge=0)
    resume_step: int = Field(default=0, ge=0)

    inference: InferenceConfig = Field(default_factory=InferenceConfig)

    mesh_shape: Optional[list[int]] = Field(default_factory=lambda: [4, 8])
    mesh_axis_names: Optional[list[str]] = Field(default_factory=lambda: ["batch", "model"])
    input_sharding_dim: Optional[str] = Field(default=None)
    model_sharding_patterns: Optional[list] = Field(default=None)
    param_sharding_patterns: Optional[list] = Field(default=None)
    optimization_level: int = Field(default=0, ge=0)
    compile_optimizer: bool = Field(default=False)

    vae_parallel_shape: Optional[list[int]] = Field(default_factory=lambda: [4, 8])

    seed: int = Field(default=42)
    deterministic: bool = Field(default=False)

    log_level: str = Field(default="INFO")
    use_wandb: bool = Field(default=True)
    wandb_project: str = Field(default="wan22-14b-lego-lora")
    wandb_run_name: str = Field(default="tt-wan22-a14b-lego-galaxy")
    wandb_tags: list[str] = Field(default_factory=lambda: ["test"])

    framework: str = Field(default="pytorch")
    project_dir: str = Field(default="blacksmith/experiments/torch/wan2_2_a14b")

    test_config: Optional[dict] = Field(default=None)
    steps_freq: Optional[int] = Field(default=None)
    val_steps_freq: Optional[int] = Field(default=None)
    save_strategy: Optional[str] = Field(default=None)
    use_tt: bool = Field(default=True)
    log_on_wandb: Optional[bool] = Field(default=None)
    resume_from_checkpoint: bool = Field(default=False)
    resume_option: str = Field(default="last")
    checkpoint_path: str = Field(default="")

    @model_validator(mode="after")
    def check_mesh_shapes(self) -> Self:
        shapes = [
            (label, shape)
            for label, shape in [("mesh_shape", self.mesh_shape), ("vae_parallel_shape", self.vae_parallel_shape)]
            if shape is not None
        ]
        for label, shape in shapes:
            if len(shape) != 2:
                raise ValueError(f"{label} must have two entries, got {shape!r}")
            if any(d < 1 for d in shape):
                raise ValueError(f"{label} entries must be >= 1, got {shape!r}")

        if self.mesh_shape is not None:
            if self.mesh_axis_names is None or len(self.mesh_axis_names) != len(self.mesh_shape):
                raise ValueError(
                    f"mesh_axis_names must name every mesh_shape entry, got "
                    f"{self.mesh_axis_names!r} for {self.mesh_shape!r}"
                )
        elif self.input_sharding_dim is not None or self.model_sharding_patterns or self.param_sharding_patterns:
            raise ValueError("mesh_shape is null (single chip), so sharding cannot be configured")
        if self.input_sharding_dim is not None and self.input_sharding_dim not in (self.mesh_axis_names or []):
            raise ValueError(
                f"input_sharding_dim {self.input_sharding_dim!r} is not one of mesh_axis_names "
                f"{self.mesh_axis_names!r}"
            )
        return self

    @model_validator(mode="after")
    def check_train_frames(self) -> Self:
        if (self.train_frames - 1) % 4 != 0:
            raise ValueError(f"train_frames must be 4k+1 (VAE temporal stride 4), got {self.train_frames}")
        return self

    @model_validator(mode="after")
    def check_grad_clip_under_tp(self) -> Self:
        dp_size, tp_size = self.mesh_shape if self.mesh_shape is not None else (1, 1)
        if self.grad_clip > 0.0 and tp_size > 1:
            raise ValueError(
                f"grad_clip={self.grad_clip} is not supported at TP={tp_size}: the clip may "
                f"apply to shard-local norms instead of the global one. Set grad_clip: 0, "
                f"or use mesh_shape: [{dp_size}, 1]."
            )
        return self

    @model_validator(mode="after")
    def check_resume_step(self) -> Self:
        if self.resume_step and self.resume_step >= self.max_steps:
            raise ValueError(f"resume_step={self.resume_step} leaves nothing to train (max_steps={self.max_steps})")
        return self

    def torch_dtype(self):
        return _TORCH_DTYPES[self.dtype]

    def vae_torch_dtype(self):
        return _TORCH_DTYPES[self.vae_dtype]

    def experts_to_load(self) -> list[str]:
        return {"low": ["low"], "high": ["high"], "both": ["high", "low"]}[self.train_experts]

    def expert_path(self, role: str) -> str:
        stem = Path(self.lora_path)
        return str(stem.with_name(stem.stem + f"_{role}.safetensors"))
