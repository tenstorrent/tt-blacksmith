# Llama LoRA fine-tuning (tt-crank)

LoRA fine-tuning of Llama on Tenstorrent hardware through **tt-crank**, the
torch frontend in `tt-mlir`. One script covers both configs; the only
difference is whether the YAML has a `mesh:` block.

| Config | Model | Hardware |
|---|---|---|
| `single_chip/llama_3_2_1b_sst2.yaml` | Llama 3.2 1B | 1 chip |
| `multichip/llama_3_1_8b_sst2.yaml` | Llama 3.1 8B | 2x4 mesh (Wormhole QuietBox) |

## Setup

```bash
source env/activate --crank
```

`--crank` installs a pinned `tt-crank` wheel. Until tt-mlir CI publishes one,
point it at a local tt-mlir checkout and it builds the wheel once:

```bash
TT_MLIR_HOME=/path/to/tt-mlir source env/activate --crank
```

Llama weights are gated on HuggingFace: `huggingface-cli login` first.

## Run

```bash
python blacksmith/experiments/torch/llama/train.py \
    --config blacksmith/experiments/torch/llama/single_chip/llama_3_2_1b_sst2.yaml

python blacksmith/experiments/torch/llama/train.py \
    --config blacksmith/experiments/torch/llama/multichip/llama_3_1_8b_sst2.yaml
```

## Parallelism

Multichip uses torch DTensor over a `DeviceMesh` that `torch.tt.init_device_mesh`
opens on the runtime mesh:

- **Data parallel** (`data_axis`) shards the batch dim of every input tensor.
  Gradients come back `Partial` and are reduced when the optimizer reads them --
  there is no explicit all-reduce step.
- **Tensor parallel** (`tensor_axis`) applies Megatron column/row sharding to
  each decoder layer's attention and MLP projections. The rules are in
  `blacksmith/tools/device_manager.py`; `embed_tokens` / `lm_head` stay
  replicated because Llama ties them.

## Tooling parity with `blacksmith_xla/`

`blacksmith/tools`, `blacksmith/datasets/torch` and `blacksmith/models/torch`
carry the full tt-xla tool set (checkpoint resume / best-N / storage backends,
W&B watch / artifacts / summaries, decode helpers, DPO/GRPO utils, the
`Trainer` framework, every dataset in `dataset_utils.py`) with the same module
paths and call signatures, adapted for tt-crank where tt-xla leaked through.
Porting an `_xla` experiment is therefore an edit to its `train.py` only; the
config and YAML carry over (the `mesh:` block replaces the sharding quartet).

## Differences from the tt-xla experiment

The tt-xla version of this experiment is preserved under
`blacksmith/experiments/torch/llama/xla/`. What changed:

| tt-xla | tt-crank |
|---|---|
| `torch_xla.device()`, `PJRT_DEVICE=TT` | `torch.device("tt")`, registered by `import tt_crank.torch` |
| lazy execution, explicit `torch_xla.sync(wait=True)` fences | eager; no fences |
| `torch_xla.set_custom_compile_options({...})` set once, globally | per-callable `torch.compile(..., options={...})` |
| SPMD `xs.Mesh` + `mark_sharding`, re-applied every step | DTensor `DeviceMesh` + `distribute_tensor`, applied once |
| regex sharding patterns in the YAML | column/row-parallel rules in `DeviceManager` |
| grad/AdamW pre-materialization, `capturable=True` | not needed -- no lazy graph signature to keep stable |
