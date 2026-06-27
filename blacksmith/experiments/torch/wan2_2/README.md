# Wan 2.2 TI2V-5B with LoRA Experiment in TT-XLA

This directory contains the code for LoRA fine-tuning of the Wan 2.2 TI2V-5B diffusion
transformer on the `jainr3/diffusiondb-pixelart` dataset in TT-XLA.

- Wan 2.2 TI2V-5B model specification can be found [here](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers).
- Original LoRA paper can be found [here](https://arxiv.org/pdf/2106.09685).

## Overview

The experiment teaches the Wan 2.2 DiT a pixel-art style (triggered by the `pxa, ` prompt
prefix) with a flow-matching objective. The four model components (UMT5 text encoder, Wan VAE
encoder/decoder, and the DiT) each run on TT as a sharded, `torch.compile(backend="tt")`
graph managed by `WanDeviceManager` (an extension of the shared `DeviceManager`).

The run is split into three config-driven stages:

1. **precompute** — VAE-encode the image subset into latents and UMT5-encode the captions,
   caching both to `cache_dir`. This is done once, before training.
2. **train** — LoRA fine-tune the DiT on the cached latents/embeds. Validation generates an
   image sample (logged to W&B) and a checkpoint at `val_steps_freq`.
3. **infer** — generate a video with the trained LoRA loaded from a checkpoint.

LoRA is applied to the DiT attention and FFN projections
(`to_q`, `to_k`, `to_v`, `to_out.0`, `ff.net.0.proj`, `ff.net.2`).

## Mesh and Sharding Configuration

Sharding is defined in the YAML with the same regex mechanism as the other experiments:
`model_sharding_patterns` matches module names and shards each matched `.weight`, while
`param_sharding_patterns` matches parameter names (biases, the `scale_shift_table` parameters)
and shards them directly. Each component (UMT5 text encoder, Wan VAE, DiT) is sharded by
`device_manager.shard_model(component)`, and the patterns are anchored per component so the one
shared list applies cleanly to all of them.

The layout is Megatron-style: QKV / FFN-up projections are column-parallel (`["model", "batch"]`),
the output / FFN-down projections are row-parallel (`["batch", "model"]`), and LoRA `A`/`B`
adapters keep the rank dim replicated so the fused AdamW step stays element-wise. The mesh is
driven by `mesh_shape`/`mesh_axis_names`.

| Hardware | mesh_shape | mesh_axis_names |
| --- | --- | --- |
| [WH QuietBox](lora/quietbox/wan2_2_ti2v_5b_diffusiondb.yaml) | `[2, 4]` | `["batch", "model"]` |

## Running

**Precompute (run once):**
```bash
python3 blacksmith/experiments/torch/wan2_2/precompute.py --config blacksmith/experiments/torch/wan2_2/lora/quietbox/wan2_2_ti2v_5b_diffusiondb.yaml
```

**Train:**
```bash
python3 blacksmith/experiments/torch/wan2_2/train.py --config blacksmith/experiments/torch/wan2_2/lora/quietbox/wan2_2_ti2v_5b_diffusiondb.yaml
```

**Infer (set `resume_option`/`checkpoint_path` to pick a checkpoint):**
```bash
python3 blacksmith/experiments/torch/wan2_2/infer.py --config blacksmith/experiments/torch/wan2_2/lora/quietbox/wan2_2_ti2v_5b_diffusiondb.yaml
```

## Data

`jainr3/diffusiondb-pixelart` is a pixel-art subset of DiffusionDB (prompt + image pairs).
The dataset metadata and image zips are pulled directly from the Hugging Face hub
(`datasets >= 4` dropped loader-script support), center-cropped/resized to the train
resolution, and VAE-encoded into latents during precompute.

Source: [Hugging Face Dataset Hub](https://huggingface.co/datasets/jainr3/diffusiondb-pixelart)

## Configuration

The experiment is configured with the YAML files under `lora/`. Resolutions must be a multiple
of 32 (VAE spatial stride 16 × DiT patch size 2) and `infer_frames`/`val_img_frames` must be
`4k+1` (VAE temporal stride 4); both are validated at config load.

### Configuration Parameters

| Parameter | Description | Default Value |
| --- | --- | --- |
| `model_id` | Hub id of the Wan 2.2 TI2V-5B diffusers model. | `"Wan-AI/Wan2.2-TI2V-5B-Diffusers"` |
| `dtype` | Data type used for the DiT/UMT5 forward. | `"torch.bfloat16"` |
| `vae_precompute_dtype` | Data type used for VAE encode during precompute. | `"torch.bfloat16"` |
| `gradient_checkpointing` | Enable DiT gradient checkpointing. | `False` |
| `dataset_id` | Dataset used for fine-tuning. | `"jainr3/diffusiondb-pixelart"` |
| `cache_dir` | Directory for precomputed latents/embeds. | `"cache/wan22_5b"` |
| `subset_size` | Number of images to use. | `64` |
| `val_holdout` | Number of held-out samples (not trained on). | `4` |
| `train_h` / `train_w` | Train resolution (multiple of 32). | `480` / `832` |
| `infer_h` / `infer_w` | Inference resolution (multiple of 32). | `480` / `832` |
| `infer_frames` | Frames to generate at inference (`4k+1`). | `65` |
| `infer_fps` | Output video fps. | `16` |
| `infer_steps` | Inference denoise steps. | `40` |
| `infer_guidance` | Classifier-free guidance scale. | `5.0` |
| `infer_flow_shift` | UniPC scheduler flow shift. | `5.0` |
| `infer_output` | Output mp4 path. | `"cache/wan22_5b/pixelart_video.mp4"` |
| `trigger` | Style trigger prepended to prompts. | `"pxa, "` |
| `text_drop_prob` | Probability of dropping the caption (CFG). | `0.10` |
| `lora_rank` | Rank of LoRA matrices. | `32` |
| `lora_alpha` | Scaling factor for LoRA updates. | `32` |
| `lora_targets` | Target modules for LoRA. | see YAML |
| `learning_rate` | Optimizer learning rate. | `1e-4` |
| `weight_decay` | AdamW weight decay. | `0.01` |
| `batch_size` | Samples per micro-batch. | `1` |
| `gradient_accumulation_steps` | Micro-batches per optimizer step. | `4` |
| `max_steps` | Total optimizer steps. | `3000` |
| `train_flow_shift` | Flow shift for training timestep sampling. | `3.0` |
| `lognorm_mean` / `lognorm_std` | Logit-normal timestep sampling params. | `0.0` / `1.0` |
| `val_prompt` | Prompt used for validation/inference. | see YAML |
| `val_img_steps` | Denoise steps for validation sample. | `40` |
| `val_img_frames` | Frames for validation sample (`4k+1`). | `65` |
| `neg_prompt` | Negative prompt (empty by default). | `""` |
| `log_level` | Logging verbosity. | `"INFO"` |
| `use_wandb` | Enable Weights & Biases logging. | `True` |
| `wandb_project` | W&B project name. | `"wan22-pixelart-lora"` |
| `steps_freq` | Frequency (steps) for train metric logging. | `25` |
| `val_steps_freq` | Frequency (steps) for validation + checkpoint. | `300` |
| `resume_from_checkpoint` | Resume training from a checkpoint. | `False` |
| `resume_option` | Resume method (`last`, `best`, `path`). | `"last"` |
| `checkpoint_path` | Checkpoint path if `resume_option="path"`. | `""` |
| `save_strategy` | Checkpoint save strategy. | `"step"` |
| `project_dir` | Directory for checkpoints/outputs. | `"blacksmith/experiments/torch/wan2_2"` |
| `save_optim` | Save optimizer state in checkpoints. | `True` |
| `seed` | Random seed. | `42` |
| `use_tt` | Run on TT device (or GPU/CPU otherwise). | `True` |
| `mesh_shape` | Mesh shape for SPMD sharding. | `[2, 4]` |
| `mesh_axis_names` | Axis names for the mesh. | `["batch", "model"]` |
| `input_sharding_dim` | Mesh dimension for input sharding. | `null` |
| `model_sharding_patterns` | Regex-based module sharding specs (shards `.weight`). | see YAML |
| `param_sharding_patterns` | Regex-based parameter sharding specs (biases, tables). | see YAML |
