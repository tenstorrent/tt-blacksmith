# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from dataset import ShakespeareDataset
from flax.core import freeze, unfreeze
from model_jax import GPT, GPTConfig


# If tt-blacksmith has a generic TrainingConfig you must use, import it here.
# Otherwise, we define a structured config specific to NanoGPT.
@dataclass
class NanoTrainingConfig:
    model_name: str = "NanoGPT-Shakespeare-JAX"
    dataset_id: str = "tiny_shakespeare"
    block_size: int = 256
    batch_size: int = 64
    learning_rate: float = 3e-4
    num_epochs: int = 500
    model_to_wandb: bool = True


DEFAULT_EXPERIMENT_NAME = "NanoGPT-TT-Training"
DEFAULT_RUN_NAME = "nanogpt-shakespeare-tt"

WANDB_ENABLED = True


def setup_wandb(config: NanoTrainingConfig, enable: bool = False, device: str = "tt") -> Optional[Any]:
    """Optionally setup wandb for experiment tracking; returns run or None."""
    global WANDB_ENABLED
    WANDB_ENABLED = bool(enable and (wandb is not None))
    if not WANDB_ENABLED:
        return None

    wandb_run = wandb.init(
        project=DEFAULT_EXPERIMENT_NAME,
        name=DEFAULT_RUN_NAME,
        config={
            "model_name": config.model_name,
            "dataset_id": config.dataset_id,
            "block_size": config.block_size,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "device": device,
            "framework": "jax",
        },
    )
    print(f"Started wandb run: {wandb_run.name}")
    return wandb_run


def log_to_wandb(data_dict: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log data to wandb if enabled; otherwise no-op."""
    if WANDB_ENABLED and wandb is not None:
        wandb.log(data_dict, step=step)


def _select_preferred_device() -> Tuple[jax.Device, str]:
    """Prefer TT device if available, otherwise fall back to CPU."""
    cpu = jax.devices("cpu")[0]
    try:
        tt_devs = jax.devices("tt")
    except Exception:
        tt_devs = []
    if tt_devs:
        return tt_devs[0], "tt"
    return cpu, "cpu"


def create_batches(data: jnp.ndarray, block_size: int, batch_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Create deterministic training batches from sequential token data."""
    # Calculate how many full sequences we can extract
    num_sequences = (len(data) - 1) // block_size

    # Extract inputs (x) and targets (y)
    xs = data[: num_sequences * block_size].reshape(num_sequences, block_size)
    ys = data[1 : num_sequences * block_size + 1].reshape(num_sequences, block_size)

    # Drop remainder to fit perfect batches
    num_batches = num_sequences // batch_size
    xs = xs[: num_batches * batch_size].reshape(num_batches, batch_size, block_size)
    ys = ys[: num_batches * batch_size].reshape(num_batches, batch_size, block_size)

    return xs, ys


def load_data(config: NanoTrainingConfig) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Load and preprocess the dataset for training and validation."""
    print("Loading Tiny Shakespeare dataset...")
    # Dataset loads into host memory (CPU)
    dataset = ShakespeareDataset()
    train_data = dataset.get_data("train")
    val_data = dataset.get_data("val")

    train_x, train_y = create_batches(train_data, config.block_size, config.batch_size)
    val_x, val_y = create_batches(val_data, config.block_size, config.batch_size)

    return train_x, train_y, val_x, val_y


def load_model(config: NanoTrainingConfig) -> Tuple[GPT, Any, Any]:
    """Initialize the NanoGPT model and parameters."""
    gpt_config = GPTConfig(
        block_size=config.block_size,
        vocab_size=65,
        num_layers=6,
        num_heads=6,
        num_embeds=384,
        dropout_rate=0.2,
        dtype=jnp.float32,
        use_matmul_embed=True,  # Critical hardware fallback.
    )
    model = GPT(gpt_config)
    key = jax.random.PRNGKey(1337)
    _, init_key = jax.random.split(key)

    variables = model.init(init_key)
    variables = unfreeze(variables)
    cache = freeze({"cache": variables.pop("cache")})
    params = freeze(variables)

    return model, params, cache


def create_loss_fn(model: GPT) -> Any:
    """Create training loss function with hardware-stable softmax."""

    def loss_fn(params: Any, cache: Any, input_ids: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        vars = {"params": params["params"], **cache}
        logits = model.apply(vars, input_ids, deterministic=True)

        # Manually stabilized log_softmax to bypass compiler instabilities on TT.
        logits_max = jnp.max(logits, axis=-1, keepdims=True)
        shifted_logits = logits - jax.lax.stop_gradient(logits_max)

        log_normalizers = jnp.log(jnp.sum(jnp.exp(shifted_logits), axis=-1, keepdims=True))
        log_probs = shifted_logits - log_normalizers

        vocab_size = logits.shape[-1]
        one_hot = jax.nn.one_hot(labels, vocab_size)
        loss = -jnp.sum(one_hot * log_probs, axis=-1)

        return jnp.mean(loss)

    return loss_fn


def create_compute_grads_fn(loss_fn: Any) -> Any:
    """Create JIT-compiled gradient computation function."""

    @jax.jit
    def compute_grads_tt(
        params_tt: Any,
        cache_tt: Any,
        input_ids_batch: jnp.ndarray,
        labels_batch: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Any]:
        loss, grads = jax.value_and_grad(loss_fn, argnums=0)(params_tt, cache_tt, input_ids_batch, labels_batch)
        return loss, grads

    return compute_grads_tt


def main(config: NanoTrainingConfig) -> None:
    """Main training orchestrator."""
    cpu_device = jax.devices("cpu")[0]
    current_device, device_kind = _select_preferred_device()

    print(f"Loading NanoGPT model... Using device: {device_kind} -> {current_device}")

    # Initialize Model & Weights on CPU.
    with jax.default_device(cpu_device):
        model, params, cache = load_model(config)

    # Setup WandB.
    _ = setup_wandb(config, enable=config.model_to_wandb, device=device_kind)

    # Load Dataset Batches (Deterministic Epochs).
    train_x, train_y, val_x, val_y = load_data(config)

    # Setup Optimizer with Gradient Clipping (Crucial for hardware stability).
    with jax.default_device(cpu_device):
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=config.learning_rate, weight_decay=1e-1)
        )
        opt_state = optimizer.init(params)

    # Push initial weights to device.
    params = jax.tree_util.tree_map(lambda x: jax.device_put(x, current_device), params)
    cache = jax.tree_util.tree_map(lambda x: jax.device_put(x, current_device), cache)

    # Build Computation Graphs.
    loss_fn = create_loss_fn(model)
    compute_grads_tt = create_compute_grads_fn(loss_fn)

    print("Starting training on Tiny Shakespeare...")
    global_step = 0
    last_10_losses = []

    try:
        for epoch in range(config.num_epochs):
            epoch_losses = []
            num_batches = len(train_x)

            for batch_idx in range(num_batches):
                # Push micro-batch to TT L1 Cache.
                input_ids = jax.device_put(train_x[batch_idx], current_device)
                labels = jax.device_put(train_y[batch_idx], current_device)

                # Forward & Backward Pass on TT Device.
                loss, grads = compute_grads_tt(params, cache, input_ids, labels)

                # Perform optimizer step on CPU because of tt-metal #27072 (pow/exp accuracy).
                # Move grads/params to CPU, compute Adam update, then move updated params back to TT.
                # See: https://github.com/tenstorrent/tt-metal/issues/27072
                with jax.default_device(cpu_device):
                    grads_cpu = jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu_device), grads)
                    params_cpu = jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu_device), params)

                    updates, new_opt_state = optimizer.update(grads_cpu, opt_state, params_cpu)
                    new_params_cpu = optax.apply_updates(params_cpu, updates)

                # Strict FP32 Push to prevent silent FP64 recompilation hangs.
                params = jax.tree_util.tree_map(
                    lambda x: jax.device_put(x.astype(jnp.float32), current_device), new_params_cpu
                )
                opt_state = new_opt_state

                current_loss = float(loss)
                epoch_losses.append(current_loss)
                last_10_losses.append(current_loss)
                global_step += 1

                log_to_wandb(
                    {
                        "step_loss": current_loss,
                        "epoch": epoch + 1,
                        "batch": batch_idx + 1,
                    },
                    step=global_step,
                )

                if len(last_10_losses) == 10:
                    avg_10_loss = np.mean(last_10_losses)
                    log_to_wandb({"avg_10_loss": avg_10_loss}, step=global_step)
                    print(
                        f"Epoch {epoch + 1}, Batch {batch_idx + 1:2d}: Loss = {current_loss:.4f} | Avg 10 = {avg_10_loss:.4f}"
                    )
                    last_10_losses = []

            # Optional: Epoch Validation Logic can be inserted here following the same device_put pattern.
            avg_epoch_loss = np.mean(epoch_losses)
            print(f"--- Epoch {epoch + 1} Completed | Average Loss: {avg_epoch_loss:.4f} ---")

        log_to_wandb(
            {
                "training_completed": True,
                "total_steps": global_step,
            },
            step=global_step,
        )

        print("TRAINING COMPLETED - All metrics logged to wandb!")

    except Exception as e:
        print(f"Error during training: {e}")
        log_to_wandb({"error": str(e), "training_failed": True})
        raise

    finally:
        if WANDB_ENABLED and wandb is not None:
            wandb.finish()
            print("Finished wandb run")


if __name__ == "__main__":
    # If using Blacksmith CLI tools, replace this with parse_cli_options.
    config = NanoTrainingConfig()
    main(config)
