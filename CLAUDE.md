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

## Debugging
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
