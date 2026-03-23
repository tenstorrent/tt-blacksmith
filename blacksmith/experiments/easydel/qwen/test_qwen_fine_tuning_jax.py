# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

os.environ.setdefault("PJRT_DEVICE", "TT")
os.environ.setdefault("XLA_STABLEHLO_COMPILE", "1")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from datasets import load_dataset
from flax import nnx
from transformers import AutoTokenizer

from blacksmith.experiments.easydel.qwen.configs import TrainingConfig
from blacksmith.tools.cli import generate_config, parse_cli_options

DEFAULT_EXPERIMENT_NAME = "Qwen-TT-EasyDel-LoRA-Training"
DEFAULT_RUN_NAME = "qwen3-0.6b-wikitext-tt-easydel"

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
            "lora_rank": training_config.lora_rank,
            "lora_pattern": training_config.lora_pattern,
            "device": device,
            "framework": "jax_easydel",
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


def create_batches(data: np.ndarray, batch_size: int = 4) -> np.ndarray:
    """Create training batches from numpy data. Stays numpy to avoid TT slice issues."""
    num_batches = len(data) // batch_size
    batched_data = data[: num_batches * batch_size].reshape(num_batches, batch_size, -1)
    return batched_data


def load_model(model_name: str, *, dtype=jnp.bfloat16, num_hidden_layers=None, max_position_embeddings=None):
    """Load a model via EasyDel with optional config overrides.

    max_position_embeddings: override the default 40960 to avoid allocating
        a huge causal attention mask.  Set to your actual max_length to save
        hundreds of MB of DRAM.
    """
    from easydel import AutoEasyDeLModelForCausalLM

    config_overrides = {}
    if num_hidden_layers is not None:
        config_overrides["num_hidden_layers"] = num_hidden_layers
    if max_position_embeddings is not None:
        config_overrides["max_position_embeddings"] = max_position_embeddings

    kwargs = {"dtype": dtype}
    if config_overrides:
        kwargs["config_kwargs"] = config_overrides

    return AutoEasyDeLModelForCausalLM.from_pretrained(model_name, **kwargs)


def _tokenize_split(training_config: TrainingConfig, split: str) -> np.ndarray:
    """Tokenize a dataset split and return batched numpy arrays of input_ids."""
    tokenizer = AutoTokenizer.from_pretrained(training_config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading dataset {training_config.dataset_id}/{training_config.dataset_configuration} ({split})...")
    ds = load_dataset(training_config.dataset_id, training_config.dataset_configuration, split=split)
    all_text = "\n".join(line for line in ds["text"] if line.strip())

    print(f"Tokenizing {split} split...")
    all_ids = tokenizer.encode(all_text, add_special_tokens=False)
    print(f"  {split} tokens: {len(all_ids):,}")

    seq_length = training_config.max_length
    batch_size = training_config.batch_size
    num_examples = len(all_ids) // seq_length
    ids_array = np.array(all_ids[: num_examples * seq_length], dtype=np.uint32).reshape(num_examples, seq_length)

    batches = create_batches(ids_array, batch_size)
    print(f"  prepared {len(batches)} {split} batches of shape ({batch_size}, {seq_length})")
    return batches


def load_data(training_config: TrainingConfig) -> np.ndarray:
    """Load and preprocess the wikitext training set."""
    return _tokenize_split(training_config, "train")


def load_val_data(training_config: TrainingConfig) -> np.ndarray:
    """Load and preprocess the wikitext validation set."""
    return _tokenize_split(training_config, "validation")


def _set_nnx_model_mesh(module, mesh):
    """EasyDel/eformer need a mesh on config and `with mesh:` during forward."""
    cfg = module.config
    cfg.set_model_mesh(mesh)
    if getattr(cfg, "text_config", None):
        cfg.text_config.set_model_mesh(mesh)
    if getattr(cfg, "vision_config", None):
        cfg.vision_config.set_model_mesh(mesh)


def _count_params(state):
    """Count total number of scalar parameters in an NNX state pytree."""
    leaves = jax.tree.leaves(state)
    return sum(x.size for x in leaves if hasattr(x, "size"))


def create_train_step_fn(graphdef, call_signature, tx) -> Any:
    """Create a JIT-compiled training step (forward + backward + optimizer).

    Compiles the entire forward pass, gradient computation, and optimizer
    update into a single StableHLO graph.  Uses optax.softmax_cross_entropy
    with one-hot labels to avoid stablehlo.scatter (TT-MLIR limitation).
    """

    def loss_fn(lora_params, frozen_state, input_ids, *, train):
        m = nnx.merge(graphdef, lora_params, frozen_state)
        kwargs = {"input_ids": input_ids}
        if "train" in call_signature.parameters:
            kwargs["train"] = train
        if "deterministic" in call_signature.parameters:
            kwargs["deterministic"] = not train
        out = m(**kwargs)

        shift_logits = out.logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        one_hot = jax.nn.one_hot(shift_labels, shift_logits.shape[-1])
        per_token_loss = optax.softmax_cross_entropy(shift_logits, one_hot)
        return jnp.mean(per_token_loss)

    def train_step(lora_params, frozen_state, opt_state, input_ids, *, train):
        loss, grads = jax.value_and_grad(loss_fn, argnums=0)(
            lora_params, frozen_state, input_ids, train=train,
        )
        updates, new_opt_state = tx.update(grads, opt_state, lora_params)
        new_lora_params = optax.apply_updates(lora_params, updates)
        return loss, new_lora_params, new_opt_state

    return jax.jit(train_step, static_argnames=("train",))


def create_eval_step_fn(graphdef, call_signature) -> Any:
    """Create a JIT-compiled evaluation step (forward pass only, no gradients)."""

    def eval_loss_fn(lora_params, frozen_state, input_ids):
        m = nnx.merge(graphdef, lora_params, frozen_state)
        kwargs = {"input_ids": input_ids}
        if "train" in call_signature.parameters:
            kwargs["train"] = False
        if "deterministic" in call_signature.parameters:
            kwargs["deterministic"] = True
        out = m(**kwargs)

        shift_logits = out.logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        one_hot = jax.nn.one_hot(shift_labels, shift_logits.shape[-1])
        per_token_loss = optax.softmax_cross_entropy(shift_logits, one_hot)
        return jnp.mean(per_token_loss)

    return jax.jit(eval_loss_fn)


def evaluate(jit_eval_step, lora_params, frozen_state, val_batches) -> float:
    """Run evaluation on validation batches and return average loss."""
    total_loss = 0.0
    for batch in val_batches:
        loss = jit_eval_step(lora_params, frozen_state, batch)
        total_loss += float(loss)
    return total_loss / len(val_batches) if val_batches else 0.0


def main(training_config: TrainingConfig) -> None:
    """Main training function with configurable parameters."""

    cpu_device = jax.devices("cpu")[0]
    current_device, device_kind = _select_preferred_device()
    jax.config.update("jax_default_device", current_device)

    print(f"Loading {training_config.model_name} model... Using device: {device_kind} -> {current_device}")

    model = load_model(
        training_config.model_name,
        num_hidden_layers=training_config.num_hidden_layers,
        max_position_embeddings=training_config.max_length,
    )

    if device_kind == "tt":
        devices_for_mesh = tuple(jax.devices("tt")[:1])
    else:
        devices_for_mesh = tuple(jax.devices("cpu")[:1])
    mesh = jax.make_mesh((1,), ("X",), devices=devices_for_mesh)
    _set_nnx_model_mesh(model, mesh)

    print(f"  num_hidden_layers:       {model.config.num_hidden_layers}")
    print(f"  hidden_size:             {model.config.hidden_size}")
    print(f"  intermediate_size:       {model.config.intermediate_size}")
    print(f"  vocab_size:              {model.config.vocab_size}")
    print(f"  max_position_embeddings: {model.config.max_position_embeddings}")

    wandb_run = setup_wandb(training_config, enable=training_config.model_to_wandb, device=device_kind)

    batches = load_data(training_config)
    val_batches_np = load_val_data(training_config)

    # LoRA init uses he_uniform (jax.random.uniform) to initialize lora_a.
    # That RNG op must run on CPU -- TT-MLIR cannot compile the StableHLO
    # produced by the monkeypatched jax.random path on the TT device.
    print(f"\nApplying LoRA (rank={training_config.lora_rank}, pattern={training_config.lora_pattern!r})...")
    with jax.default_device(cpu_device):
        model = model.apply_lora_to_layers(
            lora_rank=training_config.lora_rank,
            lora_pattern=training_config.lora_pattern,
            verbose=True,
        )

    # NNX split: separate LoRA params (trainable) from frozen state.
    # Only lora_params is passed to jax.value_and_grad.
    graphdef, lora_params, frozen_state = nnx.split(model, nnx.LoRAParam, ...)
    call_signature = inspect.signature(model.__call__)

    n_lora = _count_params(lora_params)
    n_frozen = _count_params(frozen_state)
    print(f"\n  trainable (LoRA) params: {n_lora:,}")
    print(f"  frozen params:           {n_frozen:,}")
    print(f"  trainable fraction:      {n_lora / (n_lora + n_frozen):.4%}")

    base_tx = optax.adamw(learning_rate=training_config.learning_rate)
    accum_steps = training_config.gradient_accumulation_steps
    if accum_steps > 1:
        tx = optax.MultiSteps(base_tx, every_k_schedule=accum_steps)
        effective_batch = training_config.batch_size * accum_steps
        print(f"\n  gradient accumulation: {accum_steps} steps -> effective batch size {effective_batch}")
    else:
        tx = base_tx
    opt_state = tx.init(lora_params)

    jit_train_step = create_train_step_fn(graphdef, call_signature, tx)
    jit_eval_step = create_eval_step_fn(graphdef, call_signature)

    jnp_batches = [jnp.array(batches[i], dtype=jnp.uint32) for i in range(len(batches))]
    jnp_val_batches = [jnp.array(val_batches_np[i], dtype=jnp.uint32) for i in range(len(val_batches_np))]

    print(f"\nStarting training on {training_config.dataset_id} dataset...")
    global_step = 0
    last_10_losses = []
    step_times = []
    step_losses = []

    val_freq = training_config.val_steps_freq

    try:
        with mesh:
            if jnp_val_batches:
                val_loss = evaluate(jit_eval_step, lora_params, frozen_state, jnp_val_batches)
                print(f"  Initial val loss: {val_loss:.4f}")
                log_to_wandb({"val_loss": val_loss}, step=0)

            for epoch in range(training_config.num_epochs):
                epoch_losses = []
                num_batches = len(jnp_batches)

                for batch_idx in range(num_batches):
                    input_ids = jnp_batches[batch_idx]

                    t0 = time.perf_counter()
                    loss, lora_params, opt_state = jit_train_step(
                        lora_params, frozen_state, opt_state, input_ids, train=True,
                    )
                    elapsed = time.perf_counter() - t0

                    current_loss = float(loss)
                    epoch_losses.append(current_loss)
                    last_10_losses.append(current_loss)
                    step_times.append(elapsed)
                    step_losses.append(current_loss)
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
                            f"Epoch {epoch+1}, Batch {batch_idx+1:3d}: "
                            f"Loss = {current_loss:.4f} | Avg 10 = {avg_10_loss:.4f} | "
                            f"Time = {elapsed:.3f}s"
                        )
                        last_10_losses = []
                    else:
                        print(
                            f"Epoch {epoch+1}, Batch {batch_idx+1:3d}: "
                            f"Loss = {current_loss:.4f} ({len(last_10_losses)}/10) | "
                            f"Time = {elapsed:.3f}s"
                        )

                    if val_freq > 0 and jnp_val_batches and global_step % val_freq == 0:
                        val_loss = evaluate(jit_eval_step, lora_params, frozen_state, jnp_val_batches)
                        print(f"  [Step {global_step}] Val loss: {val_loss:.4f}")
                        log_to_wandb({"val_loss": val_loss}, step=global_step)

                avg_epoch_loss = np.mean(epoch_losses)
                print(f"\n  Epoch {epoch+1} complete — avg loss: {avg_epoch_loss:.4f}")

                if jnp_val_batches:
                    val_loss = evaluate(jit_eval_step, lora_params, frozen_state, jnp_val_batches)
                    print(f"  Epoch {epoch+1} val loss: {val_loss:.4f}")
                    log_to_wandb({"val_loss": val_loss}, step=global_step)

        log_to_wandb(
            {"training_completed": True, "total_steps": global_step},
            step=global_step,
        )

        total_time = sum(step_times)
        avg_time = total_time / len(step_times) if step_times else 0
        tokens_processed = global_step * training_config.batch_size * training_config.max_length
        print(f"\nTRAINING COMPLETED")
        print(f"  steps:            {global_step}")
        print(f"  total time:       {total_time:.2f}s")
        print(f"  avg step time:    {avg_time:.3f}s")
        print(f"  final loss:       {step_losses[-1]:.4f}")
        print(f"  tokens processed: {tokens_processed:,}")
        if total_time > 0:
            print(f"  throughput:       {tokens_processed / total_time:.0f} tok/s")

        metrics = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config": {
                "model_name": training_config.model_name,
                "dataset": f"{training_config.dataset_id}/{training_config.dataset_configuration}",
                "max_length": training_config.max_length,
                "batch_size": training_config.batch_size,
                "num_epochs": training_config.num_epochs,
                "learning_rate": training_config.learning_rate,
                "lora_rank": training_config.lora_rank,
                "lora_pattern": training_config.lora_pattern,
                "optimizer": "adamw",
                "trainable_params": n_lora,
                "frozen_params": n_frozen,
                "device": device_kind,
            },
            "results": {
                "final_loss": step_losses[-1],
                "total_time_s": round(total_time, 3),
                "avg_step_time_s": round(avg_time, 3),
                "tokens_processed": tokens_processed,
                "throughput_tok_per_s": round(tokens_processed / total_time, 1) if total_time > 0 else 0,
                "step_losses": step_losses,
                "step_times_s": [round(t, 3) for t in step_times],
            },
        }
        metrics_path = Path(__file__).parent / "lora_training_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {metrics_path}")

    except Exception as e:
        print(f"Error during training: {e}")
        log_to_wandb({"error": str(e), "training_failed": True})
        raise

    finally:
        if WANDB_ENABLED and wandb is not None:
            wandb.finish()
            print("Finished wandb run")


if __name__ == "__main__":
    default_config = Path(__file__).parent / "test_qwen_fine_tuning_jax.yaml"
    args = parse_cli_options(default_config=default_config)
    training_config: TrainingConfig = generate_config(TrainingConfig, args.config, args.test_config)
    main(training_config)
