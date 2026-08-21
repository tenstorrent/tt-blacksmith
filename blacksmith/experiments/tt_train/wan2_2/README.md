# Wan 2.2 T2V-A14B with LoRA Experiment in TT-Train

This directory contains the code for LoRA fine-tuning of the Wan 2.2 T2V-A14B diffusion
transformer on the `showlab/OmniConsistency` dataset (LEGO subset) in TT-Train.

- Wan 2.2 T2V-A14B model specification can be found [here](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers).
- Original LoRA paper can be found [here](https://arxiv.org/pdf/2106.09685).

## Overview

The experiment teaches the Wan 2.2 DiT a LEGO style (triggered by the `lg, ` prompt prefix)
with a flow-matching objective. Unlike the TT-XLA experiments, this one drives `ttml`/`ttnn`
directly against a locally built tt-metal — there is no `torch.compile` and no tt-xla. It is
unrelated to `blacksmith/experiments/torch/wan2_2/`, which trains the TI2V-5B model.

A14B is a 2-expert MoE with two-stage denoising. Only the expert selected by the sampled
timestep receives gradients on a given step:

| Expert | HF subfolder | Timestep range |
| --- | --- | --- |
| high-noise | `transformer` | `t >= boundary_ratio` (0.875) |
| low-noise | `transformer_2` | `t < boundary_ratio` |

The run is split into four config-driven stages:

1. **preprocess** — download the style subset and write images + captions to `data_dir`.
2. **precompute** — VAE-encode the images into latents and UMT5-encode the captions, caching
   both to `cache_dir`. This is done once, before training.
3. **train** — LoRA fine-tune the two DiT experts on the cached latents/embeds.
4. **infer** — generate a video with the trained LoRA through tt_dit's runtime-LoRA pipeline;
   run `train.py` with `mode: infer`, or `generate.py` directly.

LoRA is applied to the DiT attention projections (`to_q`, `to_k`, `to_v`, `to_out`), and
optionally the FFN (`ffn.ff1`, `ffn.ff2`) with `lora_target_set: attn+ffn`.

`train_experts: both` keeps both 14B experts resident — ~56 GB of bf16 weights, so it needs the
weights sharded across the mesh. `low` halves the resident footprint, and a style LoRA often
only needs the low-noise expert.

## Environment

This experiment does **not** use `env/activate`. It imports `ttml`, `ttnn`, and `models.tt_dit`
from a built tt-metal checkout, so it runs from tt-metal's own virtualenv:

```bash
source $TT_METAL_HOME/python_env/bin/activate
pip install -e /path/to/tt-blacksmith

export TT_METAL_HOME=/path/to/tt-metal
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt-train/configs/mgd/bh_galaxy_4_8_line_line.textproto
```

The `.pth` files in that virtualenv already put `ttml`, the compiled `_ttml` extension, `ttnn`,
and the tt-metal repo root on `sys.path`. `TT_MESH_GRAPH_DESC_PATH` must be set explicitly: on
a 32-chip Blackhole Galaxy the automatic descriptor selection can pick the wrong topology.

## Mesh and Sharding Configuration

Two separate mesh fields, because the two stages mean different things by a `[4, 8]` mesh:
`mesh_shape` is `[DP, TP]` for training, while `vae_parallel_shape` is the VAE height/width
parallel factor pair used during precompute.

Sharding is not regex-driven here. `ttml.modules.ColumnParallelLinear` / `RowParallelLinear`
declare it structurally, and `blacksmith/models/tt_train/wan2_2/weights.py` derives the load-time
shard plan from those module types: QKV / FFN-up projections are column-parallel, output /
FFN-down projections are row-parallel, and everything else is replicated. LoRA `A`/`B` adapters
keep the rank dim replicated. A prefix mismatch in that plan raises rather than silently loading
sharded weights as replicated.

Each stage runs as its own process — `precompute` and `generate` drive ttnn/tt_dit, `train`
drives ttml, and the two frameworks cannot hold the device at the same time.

| Architecture | mesh_shape | vae_parallel_shape | Dataset | Method |
| --- | --- | --- | --- | --- |
| [BH Galaxy](lora/galaxy/wan2_2_t2v_a14b_lego.yaml) | `[4, 8]` | `[4, 8]` | LEGO (OmniConsistency) | LoRA |

## Running

All paths are relative to the tt-blacksmith root, and `train.py` does not `chdir`, so run from
there.

**Preprocess (run once):**
```bash
python3 blacksmith/experiments/tt_train/wan2_2/preprocess.py --config blacksmith/experiments/tt_train/wan2_2/lora/galaxy/wan2_2_t2v_a14b_lego.yaml
```

**Precompute (run once):**
```bash
python3 blacksmith/experiments/tt_train/wan2_2/precompute.py --config blacksmith/experiments/tt_train/wan2_2/lora/galaxy/wan2_2_t2v_a14b_lego.yaml
```

**Train:**
```bash
python3 blacksmith/experiments/tt_train/wan2_2/train.py --config blacksmith/experiments/tt_train/wan2_2/lora/galaxy/wan2_2_t2v_a14b_lego.yaml
```

`run_train.sh` wraps this with the device env, trace logging and hang instrumentation;
`run_train_conv.sh` is the same with the conv3d patch embed and a separate run directory.

**Infer (set `mode: infer`, and `inference.infer_high_lora`/`infer_low_lora` to pick a checkpoint):**
```bash
python3 blacksmith/experiments/tt_train/wan2_2/train.py --config blacksmith/experiments/tt_train/wan2_2/lora/galaxy/wan2_2_t2v_a14b_lego.yaml \
    --set mode=infer \
    --set inference.infer_high_lora=cache/wan22_14b_lego/wan22_14b_lego_lora_high_step03000.safetensors \
    --set inference.infer_low_lora=cache/wan22_14b_lego/wan22_14b_lego_lora_low_step03000.safetensors
```

Leave both empty to derive them from `lora_path`. Note that an explicitly set path which does
not exist is silently discarded and falls back to the derived one.

Override individual values without editing the YAML using `--set KEY=VALUE` (repeatable, dotted
paths for nested sections). Unknown keys and out-of-range values are rejected at load time
rather than failing part-way into a run:

```bash
python3 blacksmith/experiments/tt_train/wan2_2/train.py --config $CONFIG \
    --set max_steps=2500 --set resume_step=1500 --set inference.infer_steps=10
```

## Data

`showlab/OmniConsistency` is a style-transfer dataset; this experiment uses its LEGO subset
(prompt + image pairs). Captions are cleaned with `strip_style_words` so the `lg, ` trigger
alone carries the style rather than the words "lego/blocky/minifigure". Stills are repeated into
a static clip of `train_frames` so the adapter stylizes every temporal position.

Latents encoded on the mesh land at ~0.995 PCC against the torch VAE, so a cache built here is
not bit-identical to a CUDA-built one.

Source: [Hugging Face Dataset Hub](https://huggingface.co/datasets/showlab/OmniConsistency)

Cache layout, relative to the tt-blacksmith root:

```
data/lego/                                      preprocess output (images + metadata.jsonl)
cache/wan22_14b_lego/samples/sample_%04d.npy    latents, (C, F, H, W)
cache/wan22_14b_lego/embeds.npy                 UMT5 embeddings, includes the "" CFG caption
cache/wan22_14b_lego/embeds_index.json          caption -> row
cache/wan22_14b_lego/metadata.json              [{"idx", "caption"}, ...]
cache/wan22_14b_lego/*_{high,low}*.safetensors   adapters
```

## Configuration

The experiment is configured with the YAML files under `lora/`. Inference/validation params are
grouped under a nested `inference:` block (a separate `InferenceConfig`); the training params
stay at the top level. `train_h`/`train_w` must be a multiple of 16 (VAE spatial stride 8 × DiT
patch size 2), `train_frames` and `inference.infer_frames` must be `4k+1` (VAE temporal stride
4), and `grad_clip` must be `0` whenever `mesh_shape` has TP > 1; all are validated at config
load.

### Configuration Parameters

| Parameter | Description | Default Value |
| --- | --- | --- |
| `mode` | Entry mode dispatched by `train.py` (`train` or `infer`). | `"train"` |
| `model_id` | Hub id of the Wan 2.2 T2V-A14B diffusers model. | `"Wan-AI/Wan2.2-T2V-A14B-Diffusers"` |
| `boundary_ratio` | Timestep split between the high- and low-noise experts. | `0.875` |
| `train_experts` | Which MoE expert(s) to adapt (`low`, `high`, `both`). | `"both"` |
| `dtype` | Data type used for the DiT forward. | `"bfloat16"` |
| `vae_dtype` | ttnn dtype for the on-device VAE encoder. | `"bfloat16"` |
| `gradient_checkpointing` | Use `RunnerType.MemoryEfficient` (activation recompute). | `True` |
| `conv3d_patch_embed` | Patch-embed the raw latent with `ttnn.experimental.conv3d` instead of host `patchify` + a linear. | `False` |
| `dataset_id` | Dataset used for fine-tuning. | `"showlab/OmniConsistency"` |
| `style` | Style subset of the dataset. | `"LEGO"` |
| `data_dir` | Directory for preprocessed images + captions. | `"data/lego"` |
| `cache_dir` | Directory for precomputed latents/embeds. | `"cache/wan22_14b_lego"` |
| `train_h` / `train_w` | Train resolution (multiple of 16). | `512` / `512` |
| `train_frames` | Frames per training clip (`4k+1`). | `13` |
| `trigger` | Style trigger prepended to prompts. | `"lg, "` |
| `strip_style_words` | Drop style words from captions so the trigger carries the style. | `True` |
| `text_drop_prob` | Probability of dropping the caption (CFG). | `0.10` |
| `subset_size` | Number of samples to use (`0` = all). | `0` |
| `max_seq` | Max UMT5 caption token length. | `512` |
| `val_holdout` | Number of held-out samples (not trained on). | `4` |
| `lora_rank` | Rank of LoRA matrices. | `32` |
| `lora_alpha` | Scaling factor for LoRA updates (`alpha/rank`, `use_rslora=False`). | `32` |
| `lora_target_set` | Target modules: `attn`, or `attn+ffn` to adapt the FFN too. | `"attn"` |
| `lora_a_init` | `gaussian` = PEFT's `N(0, 1/rank)`; `kaiming` = ttml's own init. | `"gaussian"` |
| `lora_path` | Adapter path; per-expert names are derived from it. | `"cache/wan22_14b_lego/wan22_14b_lego_lora.safetensors"` |
| `learning_rate` | Optimizer learning rate (constant). | `1e-4` |
| `weight_decay` | AdamW weight decay. | `0.01` |
| `grad_clip` | Gradient clip norm; must be `0` when TP > 1. | `0` |
| `batch_size` | Samples per micro-batch, per device. | `1` |
| `gradient_accumulation_steps` | Micro-batches per optimizer step. | `4` |
| `max_steps` | Total optimizer steps. | `3000` |
| `train_flow_shift` | Flow shift for training timestep sampling. | `3.0` |
| `lognorm_mean` / `lognorm_std` | Logit-normal timestep sampling params. | `0.0` / `1.0` |
| `val_loss_every` | Validation-loss frequency in steps (`0` = off). | `200` |
| `ckpt_every` | Checkpoint frequency in steps (`0` = off). | `500` |
| `resume_step` | Restore LoRA weights from a `_step<NNNNN>` checkpoint; optimizer state is not restored. | `0` |
| `inference.infer_h` / `inference.infer_w` | Inference resolution. | `512` / `512` |
| `inference.infer_frames` | Frames to generate at inference (`4k+1`). | `49` |
| `inference.infer_fps` | Output video fps. | `16` |
| `inference.infer_steps` | Inference denoise steps. | `40` |
| `inference.infer_guidance` / `infer_guidance_2` | CFG scale for the high- / low-noise expert. | `7.0` / `5.0` |
| `inference.infer_flow_shift` | Scheduler flow shift. | `12.0` |
| `inference.infer_output` | Output mp4 path. | `"cache/wan22_14b_lego/lego_video.mp4"` |
| `inference.val_prompt` | Prompt used for validation/inference. | `"a cat sitting on a wooden table"` |
| `inference.neg_prompt` | Negative prompt. | `""` |
| `inference.infer_no_lora` | Run the base model with no adapter bound (A/B reference). | `False` |
| `inference.lora_scale` | Adapter scale; 1.5-2.0 over-drives it to survive temporal dilution. | `1.0` |
| `inference.infer_high_lora` / `infer_low_lora` | Explicit adapter paths; empty derives both from `lora_path`. | `""` / `""` |
| `mesh_shape` | `[DP, TP]` mesh for training. | `[4, 8]` |
| `vae_parallel_shape` | VAE height/width parallel factors for precompute. | `[4, 8]` |
| `seed` | Random seed. | `42` |
| `log_level` | Logging verbosity. | `"INFO"` |
| `use_wandb` | Enable Weights & Biases logging. | `True` |
| `wandb_project` | W&B project name. | `"wan22-14b-lego-lora"` |
| `wandb_run_name` | W&B run name. | `"tt-wan22-a14b-lego-galaxy"` |
| `wandb_tags` | W&B run tags. | `["test"]` |
| `project_dir` | Directory for checkpoints/outputs. | `"blacksmith/experiments/tt_train/wan2_2"` |
