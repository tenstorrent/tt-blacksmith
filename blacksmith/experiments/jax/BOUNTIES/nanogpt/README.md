# NanoGPT Training (JAX)

This directory contains the JAX implementation of the NanoGPT model, trained from scratch on the Tiny Shakespeare dataset.
The original PyTorch NanoGPT repository by Andrej Karpathy can be found [here](https://github.com/karpathy/nanoGPT).

## Overview

This experiment trains a ~10.7M parameter autoregressive Transformer (GPT) using JAX/Flax. The pipeline is designed to run natively on Tenstorrent hardware (e.g., Wormhole/TT-N150) using the TT-XLA framework. It implements a fully unrolled training loop with Weights & Biases integration for experiment tracking.

### Hardware-Specific Adaptations (TT-XLA)

Because the TT-XLA compiler and Tenstorrent metal backend are under active development, this implementation includes several critical architectural adaptations to ensure mathematical stability and prevent hardware deadlocks:

* **MatMul Embeddings:** Standard `Gather`/`Scatter` operations for token embeddings are replaced with a dense matrix multiplication fallback (`use_matmul_embed=True`) to avoid `NaN` poisoning during the backward pass.
* **CPU-Offloaded Optimizer:** Due to known precision issues with power/exponent operations in the current `tt-metal` compiler (Issue #27072), the AdamW optimizer step is offloaded to the host CPU. Gradients are computed on the TT device, pulled to the host for the update, cast strictly back to `float32`, and pushed to the TT device.
* **Safe Softmax & Masking:** Causal masking uses a safe `-10000.0` penalty (instead of `-inf`), and attention/loss logits are manually shifted by their maximum value before exponentiation to prevent ALU underflow and `NaN` cascades.
* **Hardware-Safe GELU:** The MLP blocks enforce `approximate=True` for GELU activations to utilize hardware-friendly Tanh approximations instead of the exact error function (`erf`).

## Training

To launch the training pipeline, run:

```bash
python3 blacksmith/experiments/jax/BOUNTIES/nanogpt/train_sp.py
```

*Note: If running on a machine without a Tenstorrent device, the JAX runtime will automatically fall back to the Host CPU.*

## Data

The Tiny Shakespeare dataset contains approximately 1 million characters of Shakespearean text. The task is character-level (or sub-word level) causal language modeling—predicting the next token given the previous context.

Because the legacy `karpathy/tiny_shakespeare` execution script is deprecated, the dataset is loaded securely via Hugging Face's raw `text` builder.

Source: [Karpathy Char-RNN Dataset](https://huggingface.co/datasets/karpathy/tiny_shakespeare)

Example:
```text
First Citizen:
Soft! who comes here?

Second Citizen:
Worthy Menenius Agrippa; one that hath always loved
the people.

First Citizen:
He's one honest enough: would all the rest were so!

MENENIUS:
What work's, my countrymen, in hand? where go you
With bats and clubs? The matter? speak, I pray you.
```

## Configuration

The experiment is controlled via standard Blacksmith CLI tools. The configuration parameters dictate the model size, training hyperparameters, and Weights & Biases logging settings.

### Configuration Parameters

| Parameter | Description | Default Value |
| --- | --- | --- |
| `model_name` | Name assigned to the model run. | "NanoGPT-Shakespeare-JAX" |
| `dataset_id` | The dataset identifier used for training. | "tiny_shakespeare" |
| `block_size` | Maximum sequence length / context window. | 256 |
| `batch_size` | Number of sequences per micro-batch. | 64 |
| `learning_rate` | Peak learning rate for the AdamW optimizer. | 3e-4 |
| `num_epochs` | Total number of training epochs over the dataset. | 5 |
| `model_to_wandb` | Whether to enable Weights & Biases tracking. | True |
| `device` | Target accelerator framework. | "tt" |
