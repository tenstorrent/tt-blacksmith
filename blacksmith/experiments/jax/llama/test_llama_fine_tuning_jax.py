# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import warnings
import os
from typing import Tuple, Dict, Any, Optional

import jax
import jax.numpy as jnp
import optax
import numpy as np
from transformers import FlaxAutoModelForCausalLM, AutoTokenizer, AutoConfig

from datasets import load_dataset
import lorax
from lorax import LORA_FREEZE

from blacksmith.datasets.torch.llama.sst_dataset import SSTDataset
from blacksmith.experiments.torch.llama.configs import TrainingConfig
from blacksmith.tools.cli import generate_config
import wandb


MODEL_NAME = "Erland/Llama-3.2-1B-JAX"
DEFAULT_EXPERIMENT_NAME = "Llama-TT-LoRA-Training"
DEFAULT_RUN_NAME = "llama-3.2-1b-sst2-tt-lorax"


WANDB_ENABLED = False


def setup_wandb(training_config: TrainingConfig, enable: bool = False, device: str = "tt") -> Optional[Any]:
    """Optionally setup wandb for experiment tracking; returns run or None.

    device: one of {"tt", "cpu"}
    """
    global WANDB_ENABLED
    WANDB_ENABLED = bool(enable and (wandb is not None))
    if not WANDB_ENABLED:
        return None
    wandb_run = wandb.init(
        project=DEFAULT_EXPERIMENT_NAME,
        name=DEFAULT_RUN_NAME,
        config={
            "model_name": training_config.model_name,
            "dataset_id": training_config.dataset_id,
            "max_length": training_config.max_length,
            "learning_rate": training_config.learning_rate,
            "batch_size": training_config.batch_size,
            "num_epochs": training_config.num_epochs,
            "lora_rank": training_config.lora_r,
            "lora_target_modules": training_config.lora_target_modules,
            "device": device,
            "framework": "jax_lorax",
        },
    )
    print(f"Started wandb run: {wandb_run.name}")
    return wandb_run


def log_to_wandb(data_dict: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log data to wandb if enabled; otherwise no-op."""
    if WANDB_ENABLED and wandb is not None:
        wandb.log(data_dict, step=step)


def _select_preferred_device() -> Tuple[jax.Device, str]:
    """Prefer TT device if available, otherwise fall back to CPU.

    Returns (device, device_kind_str)
    """
    cpu = jax.devices("cpu")[0]
    try:
        tt_devs = jax.devices("tt")
    except Exception:
        tt_devs = []
    if tt_devs:
        return tt_devs[0], "tt"
    return cpu, "cpu"


def create_batches(data: jnp.ndarray, batch_size: int = 8) -> jnp.ndarray:
    """Create training batches from input data."""
    num_batches = len(data) // batch_size
    batched_data = data[: num_batches * batch_size].reshape(num_batches, batch_size, -1)
    return batched_data


def load_model(model_name: str) -> FlaxAutoModelForCausalLM:
    """Load and configure the Llama model for training."""
    config = AutoConfig.from_pretrained(model_name)
    config.use_cache = False
    config.dtype = jnp.float32
    return FlaxAutoModelForCausalLM.from_pretrained(model_name, config=config, from_pt=False, dtype=jnp.float32)


def load_data(training_config: TrainingConfig) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Load and preprocess the SST dataset for training."""
    dataset_loader = SSTDataset(training_config)
    train_dataset, _ = dataset_loader.load_tokenized_data()

    train_input_ids = []
    train_attention_mask = []
    train_labels = []

    for item in train_dataset:
        train_input_ids.append(np.array(item["input_ids"]))
        train_attention_mask.append(np.array(item["attention_mask"]))
        train_labels.append(np.array(item["labels"]))

    train_input_ids = create_batches(jnp.array(train_input_ids), training_config.batch_size)
    train_attention_mask = create_batches(jnp.array(train_attention_mask), training_config.batch_size)
    train_labels = create_batches(jnp.array(train_labels), training_config.batch_size)

    return train_input_ids, train_attention_mask, train_labels


def create_lora_decision_fn(lora_rank: int) -> Any:
    """Create LoRA decision function for parameter selection."""

    def decision_fn(path: Any, param: Any) -> int:
        path_str = ".".join(str(k.key) if hasattr(k, "key") else str(k) for k in path)
        if ".mlp." in path_str and (
            ".gate_proj.kernel" in path_str or ".up_proj.kernel" in path_str or ".down_proj.kernel" in path_str
        ):
            print(f"Applying LoRA rank {lora_rank} to: {path_str}")
            return lora_rank
        else:
            return LORA_FREEZE

    return decision_fn


def create_loss_fn(lora_model: Any) -> Any:
    """Create training loss function."""

    def loss_fn(
        trainable_params: Any,
        frozen_params: Any,
        input_ids_batch: jnp.ndarray,
        attention_mask_batch: jnp.ndarray,
        labels_batch: jnp.ndarray,
    ) -> jnp.ndarray:
        merged_params = lorax.merge_trainable_frozen(trainable_params, frozen_params)
        logits = lora_model(input_ids_batch, attention_mask=attention_mask_batch, params=merged_params).logits

        shift_logits = logits[:, :-1, :]
        shift_labels = labels_batch[:, 1:]

        logprobs = jax.nn.log_softmax(shift_logits, axis=-1)
        vocab_size = logprobs.shape[-1]

        valid_mask = shift_labels != -100
        masked_labels = jnp.where(valid_mask, shift_labels, 0)

        one_hot = jax.nn.one_hot(masked_labels, num_classes=vocab_size, dtype=logprobs.dtype)
        target_logprobs = jnp.sum(logprobs * one_hot, axis=-1)

        masked_loss = -(target_logprobs * valid_mask.astype(jnp.float32))
        loss = jnp.sum(masked_loss) / jnp.sum(valid_mask.astype(jnp.float32))

        return loss

    return loss_fn


def create_compute_grads_fn(loss_fn: Any) -> Any:
    """Create JIT-compiled gradient computation function."""

    @jax.jit
    def compute_grads_tt(
        trainable_params_tt: Any,
        frozen_params_tt: Any,
        input_ids_batch: jnp.ndarray,
        attention_mask_batch: jnp.ndarray,
        labels_batch: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Any]:
        # Compute gradients only with respect to argnums=0 (trainable_params).
        # Frozen parameters participate in the forward pass but are treated as
        # constants for differentiation and receive no gradients.
        loss, grads = jax.value_and_grad(loss_fn, argnums=0)(
            trainable_params_tt, frozen_params_tt, input_ids_batch, attention_mask_batch, labels_batch
        )
        return loss, grads

    return compute_grads_tt


def main() -> None:
    """Main training function with configurable parameters."""

    config_file_path = os.path.join(os.path.dirname(__file__), "test_llama_fine_tuning_jax.yaml")
    training_config = generate_config(TrainingConfig, config_file_path)

    cpu_device = jax.devices("cpu")[0]
    current_device, device_kind = _select_preferred_device()

    print(f"Loading Llama 3.2-1B model... Using device: {device_kind} -> {current_device}")

    # Initializing model parameters on CPU, since jax.random.normal
    # is currently not supported on device (https://github.com/tenstorrent/tt-xla/issues/1105).
    with jax.default_device(cpu_device):
        model = load_model(training_config.model_name)

    model.params = jax.tree_util.tree_map(lambda x: jax.device_put(x, current_device), model.params)

    wandb_run = setup_wandb(training_config, enable=training_config.model_to_wandb, device=device_kind)

    input_id_batches, attention_mask_batches, label_batches = load_data(training_config)

    decision_fn = create_lora_decision_fn(training_config.lora_r)
    lora_spec = lorax.simple_spec(model.params, decision_fn=decision_fn, tune_vectors=False)

    # Initializing model parameters on CPU, since jax.random.normal
    # is currently not supported on device (https://github.com/tenstorrent/tt-xla/issues/1105).
    with jax.default_device(cpu_device):
        lora_params = lorax.init_lora(model.params, lora_spec, jax.random.PRNGKey(42))

    lora_params = jax.tree_util.tree_map(lambda x: jax.device_put(x, current_device), lora_params)
    # Split parameters into trainable and frozen sets: only LoRA‑adapted weights
    # are optimized during training, while the base model weights remain fixed.
    trainable_params, frozen_params = lorax.split_trainable_frozen(lora_params, lora_spec)

    with jax.default_device(cpu_device):
        optimizer = optax.adamw(learning_rate=training_config.learning_rate, weight_decay=0.01)
        trainable_params_cpu = jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu_device), trainable_params)
        opt_state = optimizer.init(trainable_params_cpu)

    lora_model = lorax.lora(model)
    loss_fn = create_loss_fn(lora_model)
    compute_grads_tt = create_compute_grads_fn(loss_fn)

    print("Starting training on SST dataset...")
    global_step = 0
    last_10_losses = []

    try:
        for epoch in range(training_config.num_epochs):
            epoch_losses = []

            num_batches = len(input_id_batches)

            for batch_idx in range(num_batches):
                input_ids = jax.device_put(input_id_batches[batch_idx], current_device)
                attention_mask = jax.device_put(attention_mask_batches[batch_idx], current_device)
                labels = jax.device_put(label_batches[batch_idx], current_device)

                loss, grads = compute_grads_tt(trainable_params, frozen_params, input_ids, attention_mask, labels)

                # Perform optimizer step on CPU because of tt-metal #27072 (pow/exp accuracy).
                # Move grads/params to CPU, compute Adam update, then move updated params back to TT.
                # See: https://github.com/tenstorrent/tt-metal/issues/27072
                with jax.default_device(cpu_device):
                    grads_cpu = jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu_device), grads)
                    trainable_params_cpu = jax.tree_util.tree_map(
                        lambda x: jax.device_put(x, cpu_device), trainable_params
                    )
                    updates, new_opt_state = optimizer.update(grads_cpu, opt_state, trainable_params_cpu)
                    new_params_cpu = optax.apply_updates(trainable_params_cpu, updates)

                trainable_params = jax.tree_util.tree_map(lambda x: jax.device_put(x, current_device), new_params_cpu)
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
                    log_to_wandb(
                        {
                            "avg_10_loss": avg_10_loss,
                        },
                        step=global_step,
                    )
                    print(
                        f"Epoch {epoch+1}, Batch {batch_idx+1:2d}: Loss = {current_loss:.4f} | Avg 10 = {avg_10_loss:.4f}"
                    )
                    last_10_losses = []
                else:
                    print(
                        f"Epoch {epoch+1}, Batch {batch_idx+1:2d}: Loss = {current_loss:.4f} ({len(last_10_losses)}/10)"
                    )

            avg_epoch_loss = np.mean(epoch_losses)

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

    print("Testing sentiment classification generation....")


if __name__ == "__main__":
    main()
