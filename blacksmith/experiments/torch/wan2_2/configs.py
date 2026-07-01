# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional, Tuple

import torch
from pydantic import BaseModel, Field, model_validator

from blacksmith.tools.templates.configs import Framework

_TORCH_DTYPES = {
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float16": torch.float16,
}


class InferenceConfig(BaseModel):
    # Video generation (used by infer.py and by in-training validation).
    infer_h: int = Field(default=480)
    infer_w: int = Field(default=832)
    infer_frames: int = Field(default=65)
    infer_fps: int = Field(default=16)
    infer_steps: int = Field(default=40)
    infer_guidance: float = Field(default=5.0)
    infer_flow_shift: float = Field(default=5.0)
    infer_output: str = Field(default="cache/wan22_5b/pixelart_video.mp4")

    # Validation (image generation only)
    val_prompt: str = Field(default="a car driving through the desert with sunset in background")
    val_img_steps: int = Field(default=40)
    val_img_frames: int = Field(default=65)
    neg_prompt: str = Field(default="")

    @model_validator(mode="after")
    def _validate_shapes(self):
        """infer_h/w must be multiples of 32 (VAE spatial stride 16 * DiT patch size 2) and
        infer_frames must be 4k+1 (Wan VAE temporal stride is 4); otherwise patch_embedding /
        unpatchify round down and pred/target shapes mismatch."""
        for label, v in [("infer_h", self.infer_h), ("infer_w", self.infer_w)]:
            if v % 32 != 0:
                raise ValueError(f"{label}={v} must be a multiple of 32 (VAE_stride * patch_size).")
        if (self.infer_frames - 1) % 4 != 0:
            raise ValueError(f"infer_frames={self.infer_frames} must satisfy 4k+1 (Wan VAE temporal stride is 4).")
        return self


class TrainingConfig(BaseModel):
    # Entry mode dispatched by train.py's __main__ ("train" or "infer").
    mode: str = Field(default="train")

    # Model settings
    model_id: str = Field(default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    dtype: str = Field(default="torch.bfloat16")
    vae_precompute_dtype: str = Field(default="torch.bfloat16")
    gradient_checkpointing: bool = Field(default=False)

    # Dataset / cache settings
    dataset_id: str = Field(default="jainr3/diffusiondb-pixelart")
    cache_dir: str = Field(default="cache/wan22_5b")
    subset_size: int = Field(default=64, gt=0)
    val_holdout: int = Field(default=4, ge=0)

    # Train resolution (single image as 1-frame video).
    train_h: int = Field(default=480)
    train_w: int = Field(default=832)

    # Inference / validation params (decomposed from the training params).
    inference: InferenceConfig = Field(default_factory=InferenceConfig)

    # Style trigger + CFG dropout
    trigger: str = Field(default="pxa, ")
    text_drop_prob: float = Field(default=0.10, ge=0)

    # LoRA setup
    lora_rank: int = Field(default=32, ge=0)
    lora_alpha: int = Field(default=32, gt=0)
    lora_targets: List[str] = Field(
        default_factory=lambda: ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]
    )

    # Training hyperparameters
    learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    batch_size: int = Field(default=1, gt=0)
    gradient_accumulation_steps: int = Field(default=4, gt=0)
    max_steps: int = Field(default=3000, gt=0)

    # Flow-matching training
    train_flow_shift: float = Field(default=3.0)
    lognorm_mean: float = Field(default=0.0)
    lognorm_std: float = Field(default=1.0)

    # Logging settings
    log_level: str = Field(default="INFO")
    use_wandb: bool = Field(default=True)
    wandb_project: str = Field(default="wan22-pixelart-lora")
    wandb_run_name: str = Field(default="tt-wan22-5b-pxa")
    wandb_tags: list[str] = Field(default_factory=lambda: ["test"])
    wandb_watch_mode: str = Field(default="all")
    wandb_log_freq: int = Field(default=1000)
    model_to_wandb: bool = Field(default=False)
    steps_freq: int = Field(default=25)
    val_steps_freq: int = Field(default=300)
    epoch_freq: int = Field(default=1)

    # Checkpoint settings
    resume_from_checkpoint: bool = Field(default=False)
    resume_option: str = Field(default="last")
    checkpoint_path: str = Field(default="")
    checkpoint_metric: str = Field(default="train/loss")
    checkpoint_metric_mode: str = Field(default="min")
    keep_last_n: int = Field(default=3, ge=0)
    keep_best_n: int = Field(default=1, ge=0)
    save_strategy: str = Field(default="step")
    project_dir: str = Field(default="blacksmith/experiments/torch/wan2_2")
    save_optim: bool = Field(default=True)
    storage_backend: str = Field(default="local")
    sync_to_storage: bool = Field(default=False)
    load_from_storage: bool = Field(default=False)
    remote_path: str = Field(default="")

    # Reproducibility settings
    seed: int = Field(default=42)
    deterministic: bool = Field(default=False)

    # Device / sharding settings
    use_tt: bool = Field(default=True)
    mesh_shape: Optional[list[int]] = Field(default=None)
    mesh_axis_names: Optional[list[str]] = Field(default=None)
    input_sharding_dim: Optional[str] = Field(default=None)
    model_sharding_patterns: Optional[List[Tuple[str, Tuple[Optional[str], ...]]]] = Field(default=None)
    param_sharding_patterns: Optional[List[Tuple[str, Tuple[Optional[str], ...]]]] = Field(default=None)

    framework: Framework = Field(default=Framework.PYTORCH)

    def torch_dtype(self) -> torch.dtype:
        return _TORCH_DTYPES[self.dtype]

    def vae_precompute_torch_dtype(self) -> torch.dtype:
        return _TORCH_DTYPES[self.vae_precompute_dtype]

    @model_validator(mode="after")
    def _validate_shapes(self):
        """train_h/w must be multiples of 32 (VAE spatial stride 16 * DiT patch size 2);
        a non-multiple rounds down in patch_embedding / unpatchify and pred/target shapes mismatch.
        (infer_* shapes are validated on InferenceConfig.)"""
        for label, v in [("train_h", self.train_h), ("train_w", self.train_w)]:
            if v % 32 != 0:
                raise ValueError(f"{label}={v} must be a multiple of 32 (VAE_stride * patch_size).")
        return self
