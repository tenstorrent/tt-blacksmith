# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import logging
import os
from pathlib import Path
from typing import Any, Optional

from easydel import AutoEasyDeLModelForCausalLM

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

os.environ.setdefault("PJRT_DEVICE", "TT")
os.environ.setdefault("XLA_STABLEHLO_COMPILE", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import wandb  # noqa: E402
from flax import nnx  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from blacksmith.experiments.easydel.qwen.configs import TrainingConfig  # noqa: E402
from blacksmith.tools.cli import generate_config, parse_cli_options  # noqa: E402
from datasets import load_dataset  # noqa: E402

WANDB_ENABLED = True


def setup_wandb(training_config: TrainingConfig, enable: bool = False, device: str = "tt") -> Optional[Any]:
    """Set up wandb for experiment tracking.

    Args:
        training_config: Training configuration with wandb project/run settings.
        enable: Whether to enable wandb logging.
        device: Device kind, one of {"tt", "cpu"}.

    Returns:
        The wandb run object if enabled, None otherwise.

    """
    global WANDB_ENABLED
    WANDB_ENABLED = bool(enable and (wandb is not None))
    if not WANDB_ENABLED:
        return None
    wandb_run = wandb.init(
        project=training_config.wandb_project,
        name=training_config.wandb_run_name,
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
    logger.info(f"Started wandb run: {wandb_run.name}")
    return wandb_run


def log_to_wandb(data_dict: dict[str, Any], step: Optional[int] = None) -> None:
    """Log metrics to wandb if enabled, otherwise no-op.

    Args:
        data_dict: Dictionary of metric names and values.
        step: Training step number.

    """
    if WANDB_ENABLED and wandb is not None:
        wandb.log(data_dict, step=step)


def _select_preferred_device() -> tuple[jax.Device, str]:
    """Select TT device if available, otherwise fall back to CPU.

    Returns:
        The selected device and its kind ("tt" or "cpu").

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
    """Reshape flat numpy data into batches.

    Stays as numpy arrays to avoid TT device slice issues.

    Args:
        data: Array of shape (num_examples, seq_length).
        batch_size: Number of samples per batch.

    Returns:
        Array of shape (num_batches, batch_size, seq_length).

    """
    num_batches = len(data) // batch_size
    batched_data = data[: num_batches * batch_size].reshape(num_batches, batch_size, -1)
    return batched_data


def load_model(
    model_name: str,
    *,
    dtype: Any = jnp.bfloat16,
    max_position_embeddings: Optional[int] = None,
) -> Any:
    """Load a causal LM via EasyDel with optional config overrides.

    Args:
        model_name: HuggingFace model identifier.
        dtype: Data type for model parameters.
        max_position_embeddings: Override the default (40960) to avoid
            allocating a huge causal attention mask. Set to your actual
            max_length to save hundreds of MB of DRAM.

    Returns:
        The loaded EasyDel model.

    """

    config_overrides = {}
    if max_position_embeddings is not None:
        config_overrides["max_position_embeddings"] = max_position_embeddings

    kwargs = {"dtype": dtype}
    if config_overrides:
        kwargs["config_kwargs"] = config_overrides

    return AutoEasyDeLModelForCausalLM.from_pretrained(model_name, **kwargs)


def load_tokenizer(model_name: str) -> Any:
    """Load a HuggingFace tokenizer and ensure it has a pad token.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        The configured tokenizer.

    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_data(training_config: TrainingConfig, tokenizer: Any, split: str = "train") -> np.ndarray:
    """Load, tokenize, and batch a dataset split.

    Args:
        training_config: Training configuration with dataset and tokenization settings.
        tokenizer: Pre-loaded HuggingFace tokenizer.
        split: Dataset split name (e.g. "train", "validation").

    Returns:
        Batched numpy array of shape (num_batches, batch_size, seq_length).

    """
    logger.info(f"Loading dataset {training_config.dataset_id}/{training_config.dataset_configuration} ({split})...")
    ds = load_dataset(training_config.dataset_id, training_config.dataset_configuration, split=split)
    all_text = "\n".join(line for line in ds["text"] if line.strip())

    logger.info(f"Tokenizing {split} split...")
    all_ids = tokenizer.encode(all_text, add_special_tokens=False)
    logger.info(f"  {split} tokens: {len(all_ids):,}")

    seq_length = training_config.max_length
    batch_size = training_config.batch_size
    num_examples = len(all_ids) // seq_length
    ids_array = np.array(all_ids[: num_examples * seq_length], dtype=np.uint32).reshape(num_examples, seq_length)

    batches = create_batches(ids_array, batch_size)
    logger.info(f"  prepared {len(batches)} {split} batches of shape ({batch_size}, {seq_length})")
    return batches


def _set_nnx_model_mesh(module: Any, mesh: jax.sharding.Mesh) -> None:
    """Attach a JAX mesh to an EasyDel model config.

    EasyDel/eformer requires a mesh on the config object and an active
    ``with mesh:`` context during the forward pass.

    Args:
        module: EasyDel NNX model.
        mesh: JAX mesh to attach.

    """
    cfg = module.config
    cfg.set_model_mesh(mesh)
    if getattr(cfg, "text_config", None):
        cfg.text_config.set_model_mesh(mesh)
    if getattr(cfg, "vision_config", None):
        cfg.vision_config.set_model_mesh(mesh)


def _count_params(state: Any) -> int:
    """Count total number of scalar parameters in an NNX state pytree.

    Args:
        state: NNX state pytree (e.g. lora_params or frozen_state).

    Returns:
        Total number of scalar parameters.

    """
    leaves = jax.tree.leaves(state)
    return sum(x.size for x in leaves if hasattr(x, "size"))


def create_train_step_fn(graphdef: Any, call_signature: inspect.Signature, tx: Any) -> Any:
    """Create a JIT-compiled training step (forward + backward + optimizer).

    Compiles the entire forward pass, gradient computation, and optimizer
    update into a single StableHLO graph. Uses optax.softmax_cross_entropy
    with one-hot labels to avoid stablehlo.scatter (TT-MLIR limitation).

    Args:
        graphdef: NNX graph definition from nnx.split.
        call_signature: Model __call__ signature for keyword detection.
        tx: Optax optimizer transform.

    Returns:
        JIT-compiled train_step function.

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
            lora_params,
            frozen_state,
            input_ids,
            train=train,
        )
        updates, new_opt_state = tx.update(grads, opt_state, lora_params)
        new_lora_params = optax.apply_updates(lora_params, updates)
        return loss, new_lora_params, new_opt_state

    return jax.jit(train_step, static_argnames=("train",))


def create_eval_step_fn(graphdef: Any, call_signature: inspect.Signature) -> Any:
    """Create a JIT-compiled evaluation step (forward pass only, no gradients).

    Args:
        graphdef: NNX graph definition from nnx.split.
        call_signature: Model __call__ signature for keyword detection.

    Returns:
        JIT-compiled eval_loss_fn function.

    """

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


def evaluate(jit_eval_step: Any, lora_params: Any, frozen_state: Any, val_batches: list[jnp.ndarray]) -> float:
    """Run evaluation on validation batches and return average loss.

    Args:
        jit_eval_step: JIT-compiled evaluation function.
        lora_params: Trainable LoRA parameters.
        frozen_state: Frozen (non-trainable) model state.
        val_batches: List of validation batch arrays.

    Returns:
        Average validation loss across all batches.

    """
    total_loss = 0.0
    for batch in val_batches:
        loss = jit_eval_step(lora_params, frozen_state, batch)
        total_loss += float(loss)
    return total_loss / len(val_batches) if val_batches else 0.0


def _training_loop(
    training_config: TrainingConfig,
    jit_train_step: Any,
    jit_eval_step: Any,
    lora_params: Any,
    frozen_state: Any,
    opt_state: Any,
    jnp_train_batches: list[jnp.ndarray],
    jnp_val_batches: list[jnp.ndarray],
) -> tuple[int, list[float]]:
    """Execute the training and validation loop.

    Must be called inside a ``with mesh:`` context.

    Args:
        training_config: Training configuration with all hyperparameters.
        jit_train_step: JIT-compiled training step function.
        jit_eval_step: JIT-compiled evaluation step function.
        lora_params: Trainable LoRA parameters.
        frozen_state: Frozen (non-trainable) model state.
        opt_state: Optimizer state.
        jnp_train_batches: Pre-converted training batches.
        jnp_val_batches: Pre-converted validation batches.

    Returns:
        A tuple of (global_step, step_losses).

    """
    global_step = 0
    steps_freq = training_config.steps_freq
    running_losses: list[float] = []
    step_losses: list[float] = []

    if jnp_val_batches:
        val_loss = evaluate(jit_eval_step, lora_params, frozen_state, jnp_val_batches)
        logger.info(f"  Initial validation loss: {val_loss:.4f}")
        log_to_wandb({"val_loss": val_loss}, step=0)

    for epoch in range(training_config.num_epochs):
        epoch_losses: list[float] = []
        num_batches = len(jnp_train_batches)

        for batch_idx in range(num_batches):
            input_ids = jnp_train_batches[batch_idx]

            loss, lora_params, opt_state = jit_train_step(
                lora_params,
                frozen_state,
                opt_state,
                input_ids,
                train=True,
            )

            current_loss = float(loss)
            epoch_losses.append(current_loss)
            running_losses.append(current_loss)
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

            if len(running_losses) == steps_freq:
                avg_window_loss = np.mean(running_losses)
                log_to_wandb({"avg_window_loss": avg_window_loss}, step=global_step)
                logger.info(
                    f"Epoch {epoch+1}, Batch {batch_idx+1:3d}: "
                    f"Loss = {current_loss:.4f} | Avg {steps_freq} = {avg_window_loss:.4f}"
                )
                running_losses = []
            else:
                logger.info(
                    f"Epoch {epoch+1}, Batch {batch_idx+1:3d}: "
                    f"Loss = {current_loss:.4f} ({len(running_losses)}/{steps_freq})"
                )

            if (
                training_config.val_steps_freq is not None
                and jnp_val_batches
                and global_step % training_config.val_steps_freq == 0
            ):
                val_loss = evaluate(jit_eval_step, lora_params, frozen_state, jnp_val_batches)
                logger.info(f"  [Step {global_step}] Validation loss: {val_loss:.4f}")
                log_to_wandb({"val_loss": val_loss}, step=global_step)

        avg_epoch_loss = np.mean(epoch_losses)
        logger.info(f"Epoch {epoch+1} complete — avg loss: {avg_epoch_loss:.4f}")

        if jnp_val_batches:
            val_loss = evaluate(jit_eval_step, lora_params, frozen_state, jnp_val_batches)
            logger.info(f"  Epoch {epoch+1} validation loss: {val_loss:.4f}")
            log_to_wandb({"val_loss": val_loss}, step=global_step)

    return global_step, step_losses


def main(training_config: TrainingConfig) -> None:
    """Run full LoRA fine-tuning pipeline.

    Args:
        training_config: Training configuration with all hyperparameters.

    """

    cpu_device = jax.devices("cpu")[0]
    current_device, device_kind = _select_preferred_device()
    jax.config.update("jax_default_device", current_device)

    logger.info(f"Loading {training_config.model_name} model... Using device: {device_kind} -> {current_device}")

    model = load_model(
        training_config.model_name,
        max_position_embeddings=training_config.max_length,
    )

    num_devices = training_config.num_devices
    devices_for_mesh = tuple(jax.devices(device_kind)[:num_devices])
    mesh = jax.make_mesh((num_devices,), ("X",), devices=devices_for_mesh)
    _set_nnx_model_mesh(model, mesh)

    logger.info(f"  num_hidden_layers:       {model.config.num_hidden_layers}")
    logger.info(f"  hidden_size:             {model.config.hidden_size}")
    logger.info(f"  intermediate_size:       {model.config.intermediate_size}")
    logger.info(f"  vocab_size:              {model.config.vocab_size}")
    logger.info(f"  max_position_embeddings: {model.config.max_position_embeddings}")

    setup_wandb(training_config, enable=training_config.model_to_wandb, device=device_kind)

    tokenizer = load_tokenizer(training_config.model_name)
    train_batches = load_data(training_config, tokenizer, split="train")
    val_batches_np = load_data(training_config, tokenizer, split="validation")

    # LoRA init uses `he_uniform` (`jax.random.uniform`) to initialize `lora_a`.
    # That RNG op must run on CPU as TT-MLIR cannot compile the StableHLO
    # produced by the monkeypatched `jax.random` path on the TT device.
    logger.info(f"Applying LoRA (rank={training_config.lora_rank}, pattern={training_config.lora_pattern!r})...")
    with jax.default_device(cpu_device):
        model = model.apply_lora_to_layers(
            lora_rank=training_config.lora_rank,
            lora_pattern=training_config.lora_pattern,
            verbose=True,
        )

    # NNX split: separate LoRA params (trainable) from frozen state.
    # Only `lora_params` is passed to `jax.value_and_grad`.
    graphdef, lora_params, frozen_state = nnx.split(model, nnx.LoRAParam, ...)
    call_signature = inspect.signature(model.__call__)

    n_lora = _count_params(lora_params)
    n_frozen = _count_params(frozen_state)
    logger.info(f"  Trainable (LoRA) params: {n_lora:,}")
    logger.info(f"  Frozen params:           {n_frozen:,}")
    logger.info(f"  Trainable fraction:      {n_lora / (n_lora + n_frozen):.4%}")

    base_tx = optax.adamw(learning_rate=training_config.learning_rate)
    accum_steps = training_config.gradient_accumulation_steps
    if accum_steps > 1:
        tx = optax.MultiSteps(base_tx, every_k_schedule=accum_steps)
        effective_batch = training_config.batch_size * accum_steps
        logger.info(f"  Gradient accumulation: {accum_steps} steps -> Effective batch size {effective_batch}")
    else:
        tx = base_tx
    opt_state = tx.init(lora_params)

    jit_train_step = create_train_step_fn(graphdef, call_signature, tx)
    jit_eval_step = create_eval_step_fn(graphdef, call_signature)

    jnp_train_batches = [jnp.array(train_batches[i], dtype=jnp.uint32) for i in range(len(train_batches))]
    jnp_val_batches = [jnp.array(val_batches_np[i], dtype=jnp.uint32) for i in range(len(val_batches_np))]
    if training_config.max_val_batches is not None:
        jnp_val_batches = jnp_val_batches[: training_config.max_val_batches]
        logger.info(f"  Using {len(jnp_val_batches)} of {len(val_batches_np)} validation batches")

    logger.info(f"Starting training on {training_config.dataset_id} dataset...")

    with mesh:
        global_step, step_losses = _training_loop(
            training_config,
            jit_train_step,
            jit_eval_step,
            lora_params,
            frozen_state,
            opt_state,
            jnp_train_batches,
            jnp_val_batches,
        )

    log_to_wandb(
        {"training_completed": True, "total_steps": global_step},
        step=global_step,
    )

    logger.info("TRAINING COMPLETED")
    logger.info(f"  Steps:      {global_step}")
    logger.info(f"  Final loss: {step_losses[-1]:.4f}")

    if WANDB_ENABLED and wandb is not None:
        wandb.finish()
        logger.info("Finished wandb run")


if __name__ == "__main__":
    default_config = Path(__file__).parent / "single_chip" / "test_qwen3_0.6b_lora.yaml"
    args = parse_cli_options(default_config=default_config)
    training_config: TrainingConfig = generate_config(TrainingConfig, args.config, args.test_config)
    main(training_config)
