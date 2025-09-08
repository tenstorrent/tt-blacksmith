# JAX LoRA Training for Llama 3.2-1B

This directory contains JAX-based LoRA (Low-Rank Adaptation) training implementations for Llama 3.2-1B model on both CPU and TT (Tenstorrent) devices.

## Overview

This implementation provides:
- **CPU Training** (`training_cpu.py`): LoRA fine-tuning on CPU using JAX
- **TT Device Training** (`training_tt.py`): LoRA fine-tuning on Tenstorrent devices using JAX
- **Custom LoRAx Implementation**: Located in the `lorax/` directory
- **SST-2 Dataset**: Sentiment classification fine-tuning task
- **Weights & Biases Integration**: Comprehensive experiment tracking

## Features

- ✅ **Modular Design**: Clean separation of concerns with helper functions
- ✅ **Type Hints**: Full type annotation for better code quality
- ✅ **Configurable Parameters**: Easy hyperparameter adjustment
- ✅ **Device Management**: Automatic CPU/TT device handling
- ✅ **Experiment Tracking**: Built-in wandb logging
- ✅ **Error Handling**: Robust exception handling and cleanup

## Setup Instructions

### 1. Environment Setup

Navigate to the TT-XLA directory and set up the environment:

```bash
cd tt-blacksmith/third_party/tt-xla
export TOOLCHAIN_DIR=$(pwd)
source venv/bin/activate
```

### 2. Export Blacksmith Path

Add the blacksmith directory to your Python path:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/../../blacksmith"
```

### 3. Hugging Face Authentication

Login to Hugging Face to access the model:

```bash
huggingface-cli login
```

Follow the prompts to enter your Hugging Face token.

## Usage

### CPU Training

Run LoRA training on CPU:

```bash
cd blacksmith/experiments/jax/llama_jax
python training_cpu.py
```

### TT Device Training

Run LoRA training on Tenstorrent device:

```bash
cd blacksmith/experiments/jax/llama_jax
python training_tt.py
```

## Configuration Options

Both scripts support the same configurable parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `"Erland/Llama-3.2-1B-JAX"` | HuggingFace model identifier |
| `dataset_id` | `"stanfordnlp/sst2"` | Dataset for fine-tuning |
| `max_length` | `128` | Maximum sequence length |
| `learning_rate` | `1e-4` | Learning rate for optimizer |
| `batch_size` | `4` | Training batch size |
| `num_epochs` | `5` | Number of training epochs |
| `lora_rank` | `4` | LoRA adaptation rank |
| `num_hidden_layers` | `16` | Number of transformer layers |

### Example with Custom Parameters

```python
# Modify parameters in the script or call main() directly
if __name__ == "__main__":
    main(
        learning_rate=2e-4,
        batch_size=8,
        num_epochs=3,
        lora_rank=8
    )
```

## Architecture

### Key Components

1. **Model Loading** (`load_model`): Configures and loads the Llama model
2. **Data Processing** (`load_data`): Handles SST-2 dataset preprocessing and batching
3. **LoRA Configuration** (`create_lora_decision_fn`): Defines which parameters receive LoRA adaptation
4. **Loss Function** (`create_loss_fn`): Implements causal language modeling loss
5. **Training Step** (`create_train_step`/`create_compute_grads_fn`): Handles forward/backward passes

### LoRA Target Modules

The implementation applies LoRA adaptation to MLP layers only:
- `mlp.gate_proj.kernel`
- `mlp.up_proj.kernel`
- `mlp.down_proj.kernel`

## Device-Specific Differences

### CPU Version (`training_cpu.py`)
- Uses standard JAX operations
- Single-device training
- Simpler device management

### TT Version (`training_tt.py`)
- Hybrid CPU/TT device execution
- Model parameters on TT device
- Optimizer operations on CPU
- Complex device transfer management



## File Structure

```
llama_jax/
├── README.md              # This file
├── training_cpu.py         # CPU training implementation
├── training_tt.py          # TT device training implementation
└── lorax/                  # Custom LoRAx implementation
    ├── __init__.py
    ├── constants.py
    ├── helpers.py
    └── transform.py
```

## Dependencies

- JAX with TT backend support
- Transformers (HuggingFace)
- Optax (JAX optimizers)
- Datasets (HuggingFace)
- Weights & Biases
- NumPy
