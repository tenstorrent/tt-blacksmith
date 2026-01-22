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
pre-commit run --all-files   # Lint code before commits
```

## Development Guidelines
- Follow `docs/src/coding-guidelines.md` for code style
- Keep the `docs/src/experiments.md` table up to date
- The `README.md` files in each experiment folder should reflect the actual config used
- Prefer using the same structure and patterns as in `blacksmith/models/` and `blacksmith/experiments/`
- Prefer editing existing files over creating new ones
- Use shared tools from `blacksmith/tools/` when possible
- Prefer pure torch over lightning if not specified otherwise
