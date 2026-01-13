# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import lorax
import numpy as np
import optax
import wandb
from lorax import DORA_FREEZE
from transformers import AutoConfig, AutoTokenizer, FlaxAutoModelForCausalLM

from blacksmith.datasets.torch.sst2.sst2_dataset import SSTDataset
from blacksmith.experiments.torch.llama.configs import TrainingConfig
from blacksmith.tools.cli import generate_config, parse_cli_options


def create_batches(data: jnp.ndarray, batch_size: int = 8) -> jnp.ndarray:
    """Create training batches from input data."""
    data = jnp.array(data)  # Ensure input is a JAX array
    num_batches = len(data) // batch_size
    batched_data = data[: num_batches * batch_size].reshape(num_batches, batch_size, -1)
    return batched_data


def load_model(model_name: str) -> FlaxAutoModelForCausalLM:
    """Load and configure the Llama model for training."""
    config = AutoConfig.from_pretrained(model_name)
    # We are finetuning, so disable caching
    config.use_cache = False
    return FlaxAutoModelForCausalLM.from_pretrained(model_name, config=config, from_pt=False, dtype=jnp.bfloat16)


def load_data(training_config: TrainingConfig) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Load and preprocess the SST dataset for training.

    Args:
        training_config: Configuration object containing dataset and training parameters.

    Returns:
        Tuple of (input_ids, attention_masks, labels) as batched JAX arrays.
    """
    dataset_loader = SSTDataset(training_config, split="train")
    train_dataloader = dataset_loader.get_dataloader()

    train_input_ids = []
    train_attention_mask = []
    train_labels = []

    for batch in train_dataloader:
        for item in batch["input_ids"]:
            train_input_ids.append(np.array(item))
        for item in batch["attention_mask"]:
            train_attention_mask.append(np.array(item))
        for item in batch["labels"]:
            train_labels.append(np.array(item))

    train_input_ids = create_batches(train_input_ids, training_config.batch_size)
    train_attention_mask = create_batches(train_attention_mask, training_config.batch_size)
    train_labels = create_batches(train_labels, training_config.batch_size)

    return train_input_ids, train_attention_mask, train_labels


def create_dora_decision_fn(dora_rank: int, target_modules: list) -> Any:
    """Create DoRA decision function for parameter selection.

    Args:
        dora_rank: Rank value to apply to DoRA-adapted parameters.
        target_modules: List of module names to apply DoRA adaptation to.

    Returns:
        Decision function that returns dora_rank for target modules, DORA_FREEZE otherwise.
    """

    def decision_fn(path: Any, param: Any) -> int:
        path_str = ".".join(str(k.key) if hasattr(k, "key") else str(k) for k in path)

        # Check if any of the target modules are in the path
        for target_module in target_modules:
            if target_module in path_str:
                print(f"Applying DoRA rank {dora_rank} to: {path_str}")
                return dora_rank

        return DORA_FREEZE

    return decision_fn


def create_forward_fn(dora_model):
    """Create forward pass function that returns logits.

    Args:
        dora_model: DoRA-wrapped model for forward computation.

    Returns:
        Function that performs forward pass and returns logits given parameters and inputs.
    """

    def forward_fn(
        trainable_params: Any,
        frozen_params: Any,
        input_ids_batch: jnp.ndarray,
        attention_mask_batch: jnp.ndarray,
    ) -> jnp.ndarray:
        merged_params = lorax.merge_trainable_frozen(trainable_params, frozen_params)
        logits = dora_model(input_ids_batch, attention_mask=attention_mask_batch, params=merged_params).logits
        return logits

    return forward_fn


def create_loss_fn(forward_fn):
    """Create training loss function that uses the forward function.

    Args:
        forward_fn: Forward pass function that returns logits.

    Returns:
        Function that computes cross-entropy loss with proper masking for causal LM.
    """

    def loss_fn(
        trainable_params: Any,
        frozen_params: Any,
        input_ids_batch: jnp.ndarray,
        attention_mask_batch: jnp.ndarray,
        labels_batch: jnp.ndarray,
    ) -> jnp.ndarray:
        logits = forward_fn(trainable_params, frozen_params, input_ids_batch, attention_mask_batch)

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


def create_train_step_fn(loss_fn):
    """Create JIT-compiled training step function.

    Args:
        loss_fn: Loss function that computes training loss.

    Returns:
        JIT-compiled function that computes loss and gradients for training step.
    """

    @jax.jit
    def train_step(
        trainable_params: Any,
        frozen_params: Any,
        input_ids_batch: jnp.ndarray,
        attention_mask_batch: jnp.ndarray,
        labels_batch: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Any]:
        # Compute gradients only with respect to argnums=0 (trainable_params).
        # Frozen parameters participate in the forward pass but are treated as
        # constants for differentiation and receive no gradients.
        loss, grads = jax.value_and_grad(loss_fn, argnums=0)(
            trainable_params, frozen_params, input_ids_batch, attention_mask_batch, labels_batch
        )
        return loss, grads

    return train_step


def train(config):
    """Main training function that orchestrates DoRA fine-tuning.

    Args:
        config: Training configuration object containing all hyperparameters and settings.
    """
    # Load dataset
    input_id_batches, attention_mask_batches, label_batches = load_data(config)

    # Initialize model parameters on CPU, since jax.random.normal
    # is currently not supported on device (https://github.com/tenstorrent/tt-xla/issues/1105).
    with jax.default_device(jax.devices("cpu")[0]):
        model = load_model(config.model_name)
    # Put parameters on device
    model.params = jax.tree_util.tree_map(lambda x: jax.device_put(x, jax.devices("tt")[0]), model.params)

    # Setup wandb
    os.environ["WANDB_MODE"] = "online" if config.use_wandb else "disabled"
    # Initialize wandb if enabled
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
    )

    # Set up DoRA
    decision_fn = create_dora_decision_fn(config.lora_r, config.lora_target_modules)
    dora_spec = lorax.simple_spec(model.params, decision_fn=decision_fn, tune_vectors=False)
    # Initializing DoRA parameters on CPU, since jax.random.normal
    # is currently not supported on device (https://github.com/tenstorrent/tt-xla/issues/1105).
    with jax.default_device(jax.devices("cpu")[0]):
        dora_params = lorax.init_dora(model.params, dora_spec, jax.random.PRNGKey(seed=config.seed))
    dora_params = jax.tree_util.tree_map(lambda x: jax.device_put(x, jax.devices("tt")[0]), dora_params)

    # Split parameters into trainable and frozen sets: only DoRA‑adapted weights
    # are optimized during training, while the base model weights remain fixed.
    trainable_params, frozen_params = lorax.split_trainable_frozen(dora_params, dora_spec)

    # Optimizer is initialized on CPU as it's execution will be on CPU
    # (https://github.com/tenstorrent/tt-metal/issues/27072).
    with jax.default_device(jax.devices("cpu")[0]):
        optimizer = optax.adamw(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
        trainable_params_cpu = jax.tree_util.tree_map(
            lambda x: jax.device_put(x, jax.devices("cpu")[0]), trainable_params
        )
        opt_state = optimizer.init(trainable_params_cpu)

    # DoRA model and training functions
    dora_model = lorax.dora(model)
    forward_fn = create_forward_fn(dora_model)
    loss_fn = create_loss_fn(forward_fn)
    train_step = create_train_step_fn(loss_fn)

    print("Starting DoRA fine-tuning on SST dataset...")

    try:
        global_step = 0
        loss_buffer = {"loss": []}
        for epoch in range(config.num_epochs):
            num_batches = len(input_id_batches)

            for batch_idx in range(num_batches):
                input_ids = input_id_batches[batch_idx]
                attention_mask = attention_mask_batches[batch_idx]
                labels = label_batches[batch_idx]

                loss, grads = train_step(trainable_params, frozen_params, input_ids, attention_mask, labels)

                # Perform optimizer step on CPU because of tt-metal #27072 (pow/exp accuracy).
                # Move grads/params to CPU, compute Adam update, then move updated params back to TT.
                # See: https://github.com/tenstorrent/tt-metal/issues/27072
                with jax.default_device(jax.devices("cpu")[0]):
                    grads_cpu = jax.tree_util.tree_map(lambda x: jax.device_put(x, jax.devices("cpu")[0]), grads)
                    trainable_params_cpu = jax.tree_util.tree_map(
                        lambda x: jax.device_put(x, jax.devices("cpu")[0]), trainable_params
                    )
                    updates, new_opt_state = optimizer.update(grads_cpu, opt_state, trainable_params_cpu)
                    new_params_cpu = optax.apply_updates(trainable_params_cpu, updates)
                    opt_state = new_opt_state

                trainable_params = jax.tree_util.tree_map(
                    lambda x: jax.device_put(x, jax.devices("tt")[0]), new_params_cpu
                )

                current_loss = float(loss)
                loss_buffer["loss"].append(current_loss)
                global_step += 1

                # Log training metrics at configured frequency.
                if global_step % config.logging_steps == 0:
                    avg_metrics = {k: np.mean(loss_buffer[k]) for k in loss_buffer}
                    print(f"[step {global_step}] " f"loss={avg_metrics['loss']:.4f}")

                    wandb.log(
                        {
                            "train/loss": avg_metrics["loss"],
                            "train/epoch": epoch + 1,
                            "step": global_step,
                        },
                        step=global_step,
                    )

                    loss_buffer = {k: [] for k in loss_buffer}

        wandb.log(
            {
                "training_completed": True,
                "total_steps": global_step,
            },
            step=global_step,
        )

        print("TRAINING COMPLETED - All metrics logged to wandb!")

    except Exception as e:
        print(f"Error during training: {e}")
        wandb.log({"error": str(e), "training_failed": True})
        raise

    finally:
        if config.use_wandb:
            wandb.finish()
            print("Finished wandb run.")


if __name__ == "__main__":
    default_config = Path(__file__).parent / "test_llama_fine_tuning_jax.yaml"
    args = parse_cli_options(default_config=default_config)
    config: TrainingConfig = generate_config(TrainingConfig, args.config)
    train(config)
