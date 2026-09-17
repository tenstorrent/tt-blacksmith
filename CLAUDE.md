# TT-Blacksmith

Optimized ML training recipes for Tenstorrent hardware using the TT-Forge compiler stack.

## Two trees

The repo is mid-migration from tt-xla to **tt-crank** (the torch frontend in `tt-mlir`).

- `blacksmith/` — the tt-crank tree. New work goes here. `tools/`, `datasets/torch`
  and `models/torch` carry the full tt-xla tool set (same module paths and
  signatures, adapted for tt-crank), so porting an experiment is a `train.py`
  change only. Only ported experiments live under `experiments/`.
- `blacksmith_xla/` — the previous tt-xla / tt-forge-fe / GPU tree, unchanged
  apart from the rename. Kept until every experiment is ported, then deleted.

Both follow the same layout:

- `models/` - Model implementations (vision, LLMs, NLP)
- `datasets/` - Dataset loaders and preprocessing
- `tools/` - Utilities (DeviceManager, TrainingLogger, CheckpointManager)
- `experiments/` - Training scripts; most of the work happens here

## Setup & Commands

```bash
source env/activate --crank   # tt-crank (blacksmith/)
source env/activate --xla     # tt-xla   (blacksmith_xla/)
pre-commit install            # Install git hooks for linting
pre-commit run --all-files    # Lint code before commits
```

`--crank` installs a pinned `tt-crank` wheel from pypi.eng.aws.tenstorrent.com.
That wheel is not published yet; until it is, point at a local tt-mlir checkout:

```bash
TT_MLIR_HOME=/path/to/tt-mlir source env/activate --crank
```

It builds the wheel once into `env/wheels/` (~40 min cold) and reuses it after.

## Development Guidelines

- Follow `docs/src/coding-guidelines.md` for code style
- Keep the `docs/src/experiments.md` table up to date
- The `README.md` files in each experiment folder should reflect the actual config used
- Prefer using the same structure and patterns as the rest of the tree you are in
- Prefer editing existing files over creating new ones
- Use shared tools from `tools/` when possible

## tt-crank notes

- `import tt_crank.torch` registers the `tt` PrivateUse1 device, the `tt` dynamo
  backend and the `tt` c10d backend. There is no `PJRT_DEVICE` / `XLA_*` setup.
- Execution is eager — there is no `torch_xla.sync()` equivalent, and none of the
  lazy-graph workarounds (grad/optimizer-state pre-materialization,
  `capturable=True`) apply.
- Compile options are per-callable: `torch.compile(fn, backend="tt", options={...})`.
  See `DeviceManager.compile_options()`.
- Multichip is torch DTensor over `torch.tt.init_device_mesh(...)`, not SPMD
  `mark_sharding`.
- The chips are one logical device: `torch.tt.num_chips()` reports the mesh size,
  `torch.tt.device_count()` is always 1.

## Debugging (tt-xla tree only)

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
