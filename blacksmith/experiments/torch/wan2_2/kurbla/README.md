# Wan 2.2 TI2V-5B on tt-kurbla

Pure-torch (`tt_kurbla`) variant of the [tt-xla Wan 2.2 experiment](../README.md). Same model,
same config schema, same three stages (precompute → train → infer) — the difference is the
backend: a torch-native `tt` device with DTensor sharding instead of `torch_xla` with GSPMD.

**Status:** the DiT, VAE encoder and VAE decoder all run on a single chip, eager and
`torch.compile(backend="tt")`, with no CPU fallbacks, at pcc ≥ 0.9995 vs CPU; compiled LoRA
training (forward + backward) works. Nothing has been validated against the real 5B weights
yet. Full detail: `/localdev/pglusac/handoff/wan2_2-kurbla-bringup.md`.

`train`, `infer`, `flow_matching_step` and `validate` are **reused unmodified** from
`../train.py`, and the denoise loop from `../generate.py` — those modules are backend-agnostic.
The only backend seam is `WanDeviceManager.prepare_model()`, a no-op on tt-xla, which applies
this directory's per-instance graph rewrites on tt-kurbla.

## Files

| file | role |
|---|---|
| `device_manager.py` | `WanDeviceManager` — same public surface as `blacksmith.tools.device_manager.DeviceManager`, but device/mesh/sharding are `torch.tt` + DTensor. Standalone rather than a subclass because the base imports `torch_xla` at module scope. |
| `model_overrides.py` | The tt-kurbla graph rewrites: `Conv3d` patch-embed as a matmul, the VAE's causal conv3d as a stack of conv2d, zero-padding via `cat`, and RoPE without a strided scatter. Each documents the backend gap it works around and should be deleted when that gap closes. |
| `precompute.py` | VAE-encode latents + UMT5-encode captions into `cache_dir`. |
| `train.py` | Entry point for `mode: train` / `mode: infer`; wires the kurbla manager into the shared training loop. |
| `bringup.py` | Single-device bring-up ladder — runs one component and reports pcc vs a CPU reference. |
| `lora/single_chip/*.yaml` | Single-chip config (`mesh_shape: null`, small resolution and step counts). |

## Running

```bash
# from the repo root, with tt-kurbla's interpreter (see the handoff doc for the env)
KPY=/localdev/pglusac/tt-kurbla/venv/bin/python
CFG=blacksmith/experiments/torch/wan2_2/kurbla/lora/single_chip/wan2_2_ti2v_5b_diffusiondb.yaml

$KPY -m blacksmith.experiments.torch.wan2_2.kurbla.precompute --config $CFG   # once, ~34 GB download
$KPY -m blacksmith.experiments.torch.wan2_2.kurbla.train      --config $CFG
```

### Bring-up ladder

`bringup.py` builds models from inlined copies of the TI2V-5B `transformer/` and `vae/`
configs, so it needs no network access and no weight download. `--stage` isolates one piece, so
a failure names an op instead of "the model".

```bash
P=blacksmith.experiments.torch.wan2_2.kurbla.bringup
$KPY -m $P --stage embed --strict       # patch_embedding + condition_embedder
$KPY -m $P --stage attn  --strict       # block 0 self-attention
$KPY -m $P --stage block --strict       # block 0 end to end
$KPY -m $P --stage dit   --strict       # the whole N-layer transformer (default N=1)
$KPY -m $P --stage vae-encode --strict  # VAE encoder (the precompute path)
$KPY -m $P --stage vae-decode --strict  # VAE decoder (the validation/infer path)

$KPY -m $P --stage dit --mode compile --h 480 --w 832 --text-len 512 --strict   # real train shape
$KPY -m $P --stage dit --mode compile --lora --backward --no-check --strict     # training step
$KPY -m $P --stage dit --layers 30 --pretrained --strict                        # real weights
```

**Always pass `--strict`.** It makes any CPU fallback raise, so a PASS means the graph really
ran on device; without it an unimplemented op silently runs on the host and still yields an
excellent pcc.

Each stage prints input shapes and a pcc against a CPU reference of the same module, exiting
non-zero below `--tolerance` (default 0.98). The comparison flags degenerate (constant) device
output instead of scoring it 1.0. Other flags: `--iters N` (exercises the compile cache),
`--no-overrides` / `--no-shared-overrides` (A/B whether a patch earns its place),
`--device cpu` (validate the harness itself), `--layers N`, `--dtype float32`.

## Configuration differences from the tt-xla path

The YAML schema is unchanged (`../configs.py`), but:

* `mesh_shape` must be `null` on a single-chip host. A non-null `mesh_shape` opens a runtime
  `MeshDevice` and converts every parameter to a `DTensor`; the quietbox config's `[2, 4]`
  needs 8 chips and asserts otherwise. Multi-chip is written but **untested**.
* `enable_trace` and the four `tt_*` dynamo flags have no tt-kurbla equivalent and are ignored.
  Compile knobs are `tt_kurbla.torch._compile.CompileOption` values on
  `WanDeviceManager.compile_options`, passed via `torch.compile(options=...)` — not
  `torch_xla.set_custom_compile_options`.
* `optimization_level` still applies, via `CompileOption.OPT_LEVEL`.
