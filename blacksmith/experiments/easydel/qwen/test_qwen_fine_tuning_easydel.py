# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import inspect
import logging
import os
import random
from pathlib import Path
from typing import Any, Optional

from easydel import AutoEasyDeLModelForCausalLM

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import wandb  # noqa: E402
from flax import nnx  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from blacksmith.experiments.easydel.qwen.configs import TrainingConfig  # noqa: E402
from blacksmith.experiments.easydel.qwen.data_loading import (  # noqa: E402
    load_sst2_batches,
)
from blacksmith.experiments.easydel.qwen.train_steps import (  # noqa: E402
    create_eval_inspect_step_fn,
    create_eval_step_fn,
    create_train_step_fn,
    evaluate,
)
from blacksmith.tools.cli import generate_config, parse_cli_options  # noqa: E402

WANDB_ENABLED = False


def setup_wandb(
    training_config: TrainingConfig,
    enable: bool = False,
    device: str = "tt",
) -> Optional[Any]:
    """Set up wandb for experiment tracking."""
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


def log_to_wandb(
    data_dict: dict[str, Any],
    step: Optional[int] = None,
) -> None:
    """Log metrics to wandb if enabled, otherwise no-op."""
    if WANDB_ENABLED and wandb is not None:
        wandb.log(data_dict, step=step)


def _select_preferred_device(
    use_tt: bool = True,
) -> tuple[jax.Device, str]:
    """Select compute device: TT > GPU > CPU."""
    cpu = jax.devices("cpu")[0]
    if not use_tt:
        try:
            gpu_devs = jax.devices("gpu")
            if gpu_devs:
                return gpu_devs[0], "gpu"
        except Exception:
            pass
        return cpu, "cpu"
    try:
        tt_devs = jax.devices("tt")
    except Exception:
        tt_devs = []
    if tt_devs:
        return tt_devs[0], "tt"
    return cpu, "cpu"


def load_model(
    model_name: str,
    *,
    dtype: Any = jnp.bfloat16,
    mask_max_position_embeddings: Optional[int] = None,
) -> Any:
    """Load a causal LM via EasyDel with optional config overrides."""
    config_overrides = {}
    if mask_max_position_embeddings is not None:
        config_overrides["mask_max_position_embeddings"] = mask_max_position_embeddings
    kwargs = {"dtype": dtype}
    if config_overrides:
        kwargs["config_kwargs"] = config_overrides
    return AutoEasyDeLModelForCausalLM.from_pretrained(
        model_name,
        **kwargs,
    )


def load_tokenizer(model_name: str) -> Any:
    """Load a HuggingFace tokenizer, ensuring a pad token exists."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _set_nnx_model_mesh(
    module: Any,
    mesh: jax.sharding.Mesh,
) -> None:
    """Attach a JAX mesh to an EasyDel model config."""
    module.config.set_model_mesh(mesh)


def _count_params(state: Any) -> int:
    """Count total scalar parameters in an NNX state pytree."""
    leaves = jax.tree.leaves(state)
    return sum(x.size for x in leaves if hasattr(x, "size"))


def _load_and_prepare_batches(
    training_config: TrainingConfig,
) -> tuple[list[dict], list[dict]]:
    """Load SST-2 dataset and return train/val lists of batch dicts.

    Each batch dict has keys ``input_ids``, ``labels``, and
    ``attention_mask`` (all ``jnp.ndarray``).  Labels contain
    ``-100`` at prompt positions so only response tokens contribute
    to the loss.
    """
    t_ids, t_lbl, t_msk = load_sst2_batches(
        training_config,
        split="train",
    )
    v_ids, v_lbl, v_msk = load_sst2_batches(
        training_config,
        split="validation",
    )

    def _to_batch(ids, lbl, msk):
        return {
            "input_ids": jnp.array(ids, dtype=jnp.uint32),
            "labels": jnp.array(lbl, dtype=jnp.int32),
            "attention_mask": jnp.array(msk, dtype=jnp.int32),
        }

    train_batches = [_to_batch(t_ids[i], t_lbl[i], t_msk[i]) for i in range(len(t_ids))]
    val_batches = [_to_batch(v_ids[i], v_lbl[i], v_msk[i]) for i in range(len(v_ids))]

    return train_batches, val_batches


def _training_loop(
    training_config: TrainingConfig,
    jit_train_step: Any,
    jit_eval_step: Any,
    lora_params: Any,
    frozen_state: Any,
    opt_state: Any,
    train_batches: list[dict[str, Any]],
    val_batches: list[dict[str, Any]],
    vocab_size: int,
    *,
    device_kind: str = "tt",
    jit_inspect_step: Any = None,
    tokenizer: Any = None,
) -> tuple[int, list[float]]:
    """Execute the training and validation loop.

    Must be called inside a ``with mesh:`` context.

    *train_batches* / *val_batches* are lists of dicts with keys
    ``input_ids``, ``labels``, ``attention_mask``.

    Returns ``(global_step, step_losses)``.

    ``global_step`` counts micro-batches.  When
    ``gradient_accumulation_steps > 1`` the underlying optimizer
    (wrapped in ``optax.MultiSteps``) only updates weights every
    *k* micro-batches, but the step counter still increments per
    micro-batch.
    """
    global_step = 0
    steps_freq = training_config.steps_freq
    ignored = training_config.ignored_label_index
    running_losses: list[float] = []
    step_losses: list[float] = []

    inspect_kwargs: dict[str, Any] = {}
    if jit_inspect_step is not None and tokenizer is not None:
        inspect_kwargs = {
            "jit_inspect_step": jit_inspect_step,
            "tokenizer": tokenizer,
        }

    if val_batches:
        val_loss = evaluate(
            jit_eval_step,
            lora_params,
            frozen_state,
            val_batches,
            **inspect_kwargs,
        )
        logger.info(f"  Initial validation loss: {val_loss:.4f}")
        log_to_wandb({"val_loss": val_loss}, step=0)

    cpu = jax.devices("cpu")[0]
    rng = np.random.default_rng(training_config.seed)

    for epoch in range(training_config.num_epochs):
        epoch_losses: list[float] = []
        num_batches = len(train_batches)
        batch_order = rng.permutation(num_batches)
        logger.info(f"Epoch {epoch + 1}: shuffled {num_batches} " f"training batches (seed={training_config.seed})")

        for batch_idx in range(num_batches):
            batch = train_batches[batch_order[batch_idx]]
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            attention_mask = batch["attention_mask"]

            # All label prep on CPU for TT (ttnn.slice can't handle
            # eager device-only slices outside a JIT graph).
            ctx = jax.default_device(cpu) if device_kind == "tt" else contextlib.nullcontext()
            with ctx:
                shift = labels[:, 1:].astype(jnp.int32)
                valid = shift != ignored
                safe = jnp.where(valid, shift, 0)
                label_mask = valid.astype(jnp.float32)
                one_hot = jax.nn.one_hot(
                    safe,
                    vocab_size,
                ).astype(jnp.float32)

            loss, lora_params, opt_state, grad_stats = jit_train_step(
                lora_params,
                frozen_state,
                opt_state,
                input_ids,
                one_hot,
                label_mask,
                attention_mask,
                train=True,
            )

            current_loss = float(loss)
            g_norm = float(grad_stats["grad_norm"])
            g_max = float(grad_stats["grad_max"])
            epoch_losses.append(current_loss)
            running_losses.append(current_loss)
            step_losses.append(current_loss)
            global_step += 1

            log_to_wandb(
                {
                    "step_loss": current_loss,
                    "grad/global_norm": g_norm,
                    "grad/global_max": g_max,
                    "epoch": epoch + 1,
                    "batch": batch_idx + 1,
                },
                step=global_step,
            )

            if len(running_losses) == steps_freq:
                avg = np.mean(running_losses)
                log_to_wandb(
                    {"avg_window_loss": avg},
                    step=global_step,
                )
                logger.info(
                    f"Epoch {epoch + 1}, "
                    f"Batch {batch_idx + 1:3d}: "
                    f"Loss = {current_loss:.4f} | "
                    f"Avg {steps_freq} = {avg:.4f} | "
                    f"grad_norm = {g_norm:.4f}, "
                    f"grad_max = {g_max:.4f}"
                )
                running_losses = []
            else:
                logger.info(
                    f"Epoch {epoch + 1}, "
                    f"Batch {batch_idx + 1:3d}: "
                    f"Loss = {current_loss:.4f} "
                    f"({len(running_losses)}/{steps_freq}) | "
                    f"grad_norm = {g_norm:.4f}, "
                    f"grad_max = {g_max:.4f}"
                )

            if (
                training_config.val_steps_freq is not None
                and val_batches
                and global_step % training_config.val_steps_freq == 0
            ):
                val_loss = evaluate(
                    jit_eval_step,
                    lora_params,
                    frozen_state,
                    val_batches,
                    **inspect_kwargs,
                )
                logger.info(f"  [Step {global_step}] " f"Validation loss: {val_loss:.4f}")
                log_to_wandb({"val_loss": val_loss}, step=global_step)

        avg_epoch = np.mean(epoch_losses)
        logger.info(f"Epoch {epoch + 1} complete — " f"avg loss: {avg_epoch:.4f}")

        if val_batches:
            val_loss = evaluate(
                jit_eval_step,
                lora_params,
                frozen_state,
                val_batches,
                **inspect_kwargs,
            )
            logger.info(f"  Epoch {epoch + 1} " f"validation loss: {val_loss:.4f}")
            log_to_wandb({"val_loss": val_loss}, step=global_step)

    return global_step, step_losses


def main(training_config: TrainingConfig) -> None:
    """Run full LoRA fine-tuning pipeline."""
    random.seed(training_config.seed)
    np.random.seed(training_config.seed)

    cpu_device = jax.devices("cpu")[0]
    current_device, device_kind = _select_preferred_device(
        use_tt=training_config.use_tt,
    )
    jax.config.update("jax_default_device", current_device)

    if device_kind == "tt":
        from blacksmith.tools.workaround_utils_jax import apply_gqa_workaround

        apply_gqa_workaround()

    logger.info(f"Loading {training_config.model_name} model... " f"Using device: {device_kind} -> {current_device}")

    model = load_model(
        training_config.model_name,
        dtype=training_config.jax_dtype,
        mask_max_position_embeddings=(training_config.mask_max_position_embeddings),
    )

    num_devices = training_config.num_devices
    devices_for_mesh = tuple(
        jax.devices(device_kind)[:num_devices],
    )
    mesh = jax.make_mesh((num_devices,), ("X",), devices=devices_for_mesh)
    _set_nnx_model_mesh(model, mesh)

    logger.info(f"  num_hidden_layers:       " f"{model.config.num_hidden_layers}")
    logger.info(f"  hidden_size:             {model.config.hidden_size}")
    logger.info(f"  intermediate_size:       " f"{model.config.intermediate_size}")
    logger.info(f"  vocab_size:              {model.config.vocab_size}")
    logger.info(f"  max_position_embeddings: " f"{model.config.max_position_embeddings}")

    setup_wandb(
        training_config,
        enable=training_config.use_wandb,
        device=device_kind,
    )

    tokenizer = load_tokenizer(training_config.model_name)
    train_batches, val_batches = _load_and_prepare_batches(
        training_config,
    )

    logger.info(
        f"Applying LoRA " f"(rank={training_config.lora_rank}, " f"pattern={training_config.lora_pattern!r})..."
    )
    if device_kind == "tt":
        with jax.default_device(cpu_device):
            model = model.apply_lora_to_layers(
                lora_rank=training_config.lora_rank,
                lora_pattern=training_config.lora_pattern,
                verbose=True,
            )
    else:
        model = model.apply_lora_to_layers(
            lora_rank=training_config.lora_rank,
            lora_pattern=training_config.lora_pattern,
            verbose=True,
        )

    graphdef, lora_params, frozen_state = nnx.split(
        model,
        nnx.LoRAParam,
        ...,
    )
    call_signature = inspect.signature(model.__call__)

    n_lora = _count_params(lora_params)
    n_frozen = _count_params(frozen_state)
    logger.info(f"  Trainable (LoRA) params: {n_lora:,}")
    logger.info(f"  Frozen params:           {n_frozen:,}")
    logger.info(f"  Trainable fraction:      " f"{n_lora / (n_lora + n_frozen):.4%}")

    num_train_batches = len(train_batches)
    total_batches = num_train_batches * training_config.num_epochs
    accum = training_config.gradient_accumulation_steps
    total_opt_steps = total_batches // accum

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=training_config.learning_rate,
        warmup_steps=training_config.warmup_steps,
        decay_steps=total_opt_steps,
        end_value=training_config.end_learning_rate,
    )
    logger.info(
        f"  LR schedule: warmup {training_config.warmup_steps} "
        f"optimizer steps, cosine decay over {total_opt_steps} "
        f"optimizer steps "
        f"({training_config.learning_rate} -> "
        f"{training_config.end_learning_rate})"
    )

    base_tx = optax.adamw(learning_rate=schedule)
    if accum > 1:
        tx = optax.MultiSteps(base_tx, every_k_schedule=accum)
        eff = training_config.batch_size * accum
        logger.info(f"  Gradient accumulation: {accum} steps -> " f"Effective batch size {eff}")
    else:
        tx = base_tx
    opt_state = tx.init(lora_params)

    jit_train_step = create_train_step_fn(
        graphdef,
        call_signature,
        tx,
    )
    jit_eval_step = create_eval_step_fn(
        graphdef,
        call_signature,
        device_kind=device_kind,
    )
    jit_inspect_step = (
        create_eval_inspect_step_fn(
            graphdef,
            call_signature,
            device_kind=device_kind,
        )
        if training_config.print_examples
        else None
    )

    if training_config.max_val_batches is not None:
        orig = len(val_batches)
        val_batches = val_batches[: training_config.max_val_batches]
        logger.info(f"  Using {len(val_batches)} of {orig} " f"validation batches")

    logger.info("Starting training on SST-2 dataset...")

    try:
        with mesh:
            global_step, step_losses = _training_loop(
                training_config,
                jit_train_step,
                jit_eval_step,
                lora_params,
                frozen_state,
                opt_state,
                train_batches,
                val_batches,
                model.config.vocab_size,
                device_kind=device_kind,
                jit_inspect_step=jit_inspect_step,
                tokenizer=(tokenizer if training_config.print_examples else None),
            )

        log_to_wandb(
            {
                "training_completed": True,
                "total_steps": global_step,
            },
            step=global_step,
        )

        logger.info("TRAINING COMPLETED")
        logger.info(f"  Steps:      {global_step}")
        logger.info(f"  Final loss: {step_losses[-1]:.4f}")

    except Exception as e:
        logger.error(f"Error during training: {e}")
        log_to_wandb({"error": str(e), "training_failed": True})
        raise

    finally:
        if WANDB_ENABLED and wandb is not None:
            wandb.finish()
            logger.info("Finished wandb run")


if __name__ == "__main__":
    default_cfg = Path(__file__).parent / "single_chip" / "test_qwen3_0.6b_lora.yaml"
    args = parse_cli_options(default_config=default_cfg)
    training_config: TrainingConfig = generate_config(
        TrainingConfig,
        args.config,
        args.test_config,
    )

    if training_config.use_tt:
        os.environ.setdefault("PJRT_DEVICE", "TT")
        os.environ.setdefault("XLA_STABLEHLO_COMPILE", "1")

    main(training_config)
