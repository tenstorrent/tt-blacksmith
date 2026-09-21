# TT-Blacksmith

Optimized ML training recipes for Tenstorrent hardware using TT-Forge compiler stack.

## Project Structure
- `blacksmith/models/` - Model implementations (vision, LLMs, NLP)
- `blacksmith/datasets/` - Dataset loaders and preprocessing
- `blacksmith/tools/` - Utilities (DeviceManager, TrainingLogger, CheckpointManager)
- `blacksmith/experiments/` - Training scripts for various models; most of the work happens here

## Setup & Commands
```bash
source env/activate --xla    # Activate environment (required before ANY work)
source env/activate --crank  # ... or the tt-crank environment, for the train_crank.py scripts
pre-commit install           # Install git hooks for linting
pre-commit run --all-files   # Lint code before commits
```

## Development Guidelines
- Follow `docs/src/coding-guidelines.md` for code style
- Keep the `docs/src/experiments.md` table up to date
- The `README.md` files in each experiment folder should reflect the actual config used
- Prefer using the same structure and patterns as in `blacksmith/models/`, `blacksmith/experiments/`, and `blacksmith/datasets/`
- Prefer editing existing files over creating new ones
- Use shared tools from `blacksmith/tools/` when possible

## tt-crank ports

The repo is migrating from tt-xla to **tt-crank** (the PyTorch frontend in `tt-mlir`), one
experiment at a time, inside the same tree:

- Every experiment keeps its tt-xla `train.py`. The tt-crank port lives next to it as
  `train_crank.py`, maps the experiment 1:1 and reads the *same* YAML (including the
  `mesh_shape` / `model_sharding_patterns` block).
- `blacksmith/tools/crank/` imports no tt-xla code. Whatever a tt-crank script needs from the
  tt-xla tools is a *copy* there with the tt-xla parts stripped (`checkpoints_manager`,
  `hf_models`, `loss_utils`, `torch_helpers`, the whole `trainer/` pipeline incl. configs),
  plus the tt-crank-only `DeviceManager`. Duplication is deliberate: tt-xla is being
  deprecated, so tt-crank must not depend on it. Backend-neutral infrastructure with no
  tt-xla code in it (datasets, logger, CLI, reproducibility, `tools/configs.py`) stays shared,
  and where it needs a trainer config it imports the tt-crank one
  (`blacksmith.tools.crank.trainer.configs`), never `blacksmith.tools.trainer`: that package's
  `__init__` pulls in `torch_xla`. Never add `try: import torch_xla` guards to tt-xla modules
  to make them importable from tt-crank -- copy instead.
- `env/activate --crank` installs a pinned `tt-crank` wheel from pypi.eng.aws.tenstorrent.com.
  Until that wheel is published, point at a local tt-mlir checkout and it is built once into
  `env/wheels/`: `TT_MLIR_HOME=/path/to/tt-mlir source env/activate --crank`.
- tt-crank notes: `import tt_crank.torch` registers the `tt` device, dynamo backend and c10d
  backend (no `PJRT_DEVICE` / `XLA_*` setup). Execution is eager: no `torch_xla.sync()`, none
  of the lazy-graph workarounds (grad / AdamW-state pre-materialization, `capturable=True`).
  Compile options are per `torch.compile(fn, backend="tt", options=...)` call, see
  `DeviceManager.compile_options()`. Multichip is torch DTensor over
  `torch.tt.init_device_mesh(...)`; the chips are one logical device (`torch.tt.num_chips()`).
- Ported so far: Llama LoRA (`blacksmith/experiments/torch/llama/xla/train_crank.py`) and the
  Trainer pipeline (`blacksmith/tools/crank/trainer/`: `Trainer`, `LoraLLMTrainer`,
  `MetricsCallback` / `CheckpointCallback`, `TrainerConfig` / `LoraLLMConfig`; entry point
  `tools/trainer/examples/lora_llm/train_crank.py`). Smoke tests in `tests/crank/`.
  Not ported: FSDP, `SFTLLMTrainer`.

## Debugging (tt-xla)
For debugging use following environment variables:
- TTXLA_LOGGER_LEVEL: DEBUG or VERBOSE.

If compilation fails, it is useful to use:
```python
torch_xla.set_custom_compile_options({
    "export_path": "./irs",
    "export_tensors": True
})
```
But only use this at start of the training, as doing this once is enough.
