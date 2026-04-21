# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import logging
import os
import random
from pathlib import Path
from typing import Any, Optional

from easydel import AutoEasyDeLModelForCausalLM

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from flax import nnx  # noqa: E402
from jax.typing import DTypeLike  # noqa: E402
from transformers import AutoTokenizer, PreTrainedTokenizerBase  # noqa: E402

from blacksmith.experiments.easydel.qwen.configs import TrainingConfig  # noqa: E402
from blacksmith.experiments.easydel.qwen.data_loading import (  # noqa: E402
    load_sst2_batches,
)
from blacksmith.experiments.easydel.qwen.multi_chip.sharding_config import (  # noqa: E402
    AXIS_NAME,
    ShardingConfig,
    make_tt_mesh,
)
from blacksmith.experiments.easydel.qwen.train_steps import (  # noqa: E402
    _place_batch_on_sharding,
    create_eval_inspect_step_fn,
    create_eval_step_fn,
    create_train_step_fn,
    evaluate,
)
from blacksmith.tools.cli import generate_config, parse_cli_options  # noqa: E402
from blacksmith.tools.logging_manager import TrainingLogger  # noqa: E402


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


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Load a HuggingFace tokenizer, ensuring a pad token exists."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_name: str,
    *,
    dtype: DTypeLike = jnp.bfloat16,
    mask_max_position_embeddings: Optional[int] = None,
    mesh_axis_size: int = 1,
) -> tuple[nnx.Module, PreTrainedTokenizerBase]:
    """Load a causal LM and its tokenizer via EasyDel + HuggingFace.

    *mesh_axis_size* is the product dimension for eformer's ``create_mesh`` and
    must match how many JAX devices that path sees for the accelerator (e.g. on
    TT, ``len(jax.devices("tt"))``).  It is independent of YAML ``num_devices``:
    training still uses :func:`make_tt_mesh` with ``training_config.num_devices``
    (single-chip yaml on a multi-chip host keeps ``num_devices=1`` and a 1-device
    mesh after load). The mesh axis is always ``AXIS_NAME`` (``"data"``) for
    consistency with :func:`make_tt_mesh` and the data-parallel ``ShardingConfig``.
    """
    config_overrides = {}
    if mask_max_position_embeddings is not None:
        config_overrides["mask_max_position_embeddings"] = mask_max_position_embeddings
    kwargs = {"dtype": dtype}
    if config_overrides:
        kwargs["config_kwargs"] = config_overrides
    model = AutoEasyDeLModelForCausalLM.from_pretrained(
        model_name,
        sharding_axis_dims=(mesh_axis_size,),
        sharding_axis_names=(AXIS_NAME,),
        auto_shard_model=False,
        param_dtype=dtype,
        **kwargs,
    )
    tokenizer = load_tokenizer(model_name)
    return model, tokenizer


def _set_nnx_model_mesh(
    module: nnx.Module,
    mesh: jax.sharding.Mesh,
) -> None:
    """Attach a JAX mesh to an EasyDel model config."""
    module.config.set_model_mesh(mesh)


def call_model(
    module: nnx.Module,
    input_ids: jax.Array,
    attention_mask: jax.Array,
):
    """Invoke the EasyDel Qwen3 causal LM with the kwargs it accepts.

    The call signature is hardcoded on purpose. ``Qwen3ForCausalLM.__call__``
    is part of EasyDel's stable API for this model family, so there is no
    need to feature-detect kwargs via ``inspect`` at runtime. If we ever
    switch to a model whose ``__call__`` takes different kwargs, this
    function is the single place to update.
    """
    return module(input_ids=input_ids, attention_mask=attention_mask)


def _load_and_prepare_batches(
    training_config: TrainingConfig,
) -> tuple[list[dict], list[dict]]:
    """Load SST-2 dataset and return train/val lists of batch dicts.

    Each batch dict has keys ``input_ids``, ``labels``, and
    ``attention_mask`` (all ``jnp.ndarray``).  Labels contain
    ``-100`` at prompt positions so only response tokens contribute
    to the loss.
    """
    if training_config.dataset_id != "sst2":
        raise ValueError(
            f"Unsupported dataset_id {training_config.dataset_id!r}; " "only 'sst2' is implemented for this experiment."
        )
    t_ids, t_lbl, t_msk = load_sst2_batches(
        training_config,
        split="train",
    )
    v_ids, v_lbl, v_msk = load_sst2_batches(
        training_config,
        split="validation",
    )

    # Keep batches as host numpy; per-step transfer to the target device happens
    # inside :func:`_place_batch_on_sharding` (``jax.device_put``) so the
    # accelerator is touched only once per step, not per batch at load time.
    def _to_batch(ids, lbl, msk):
        # Keep batches as host numpy arrays. If these were ``jnp.array`` and
        # ``jax.default_device`` is already set to TT, constructing tens of
        # thousands of tiny arrays triggers one eager host->device transfer
        # per tensor and stalls before training starts. JIT will transfer
        # them on first use.
        return {
            "input_ids": np.asarray(ids, dtype=np.uint32),
            "labels": np.asarray(lbl, dtype=np.int32),
            "attention_mask": np.asarray(msk, dtype=np.int32),
        }

    train_batches = [_to_batch(t_ids[i], t_lbl[i], t_msk[i]) for i in range(len(t_ids))]
    val_batches = [_to_batch(v_ids[i], v_lbl[i], v_msk[i]) for i in range(len(v_ids))]

    return train_batches, val_batches


def _training_loop(
    training_config: TrainingConfig,
    training_logger: TrainingLogger,
    jit_train_step: Any,
    jit_eval_step: Any,
    lora_params: Any,
    frozen_state: Any,
    opt_state: Any,
    train_batches: list[dict[str, Any]],
    val_batches: list[dict[str, Any]],
    vocab_size: int,
    *,
    sharding_config: Any = None,
    device_kind: str = "tt",
    jit_inspect_step: Any = None,
    tokenizer: Any = None,
) -> tuple[int, list[float]]:
    """Execute the training and validation loop.

    Must be called inside a mesh context for multichip.

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
            sharding_config=sharding_config,
            **inspect_kwargs,
        )
        training_logger.info(f"  Initial validation loss: {val_loss:.4f}")
        training_logger.log_metrics({"val/loss": val_loss}, step=0)

    cpu = jax.devices("cpu")[0]
    rng = np.random.default_rng(training_config.seed)

    for epoch in range(training_config.num_epochs):
        epoch_losses: list[float] = []
        num_batches = len(train_batches)
        batch_order = rng.permutation(num_batches)
        training_logger.info(
            f"Epoch {epoch + 1}: shuffled {num_batches} training batches (seed={training_config.seed})"
        )

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

            input_ids, one_hot, label_mask, attention_mask = _place_batch_on_sharding(
                sharding_config,
                input_ids,
                one_hot,
                label_mask,
                attention_mask,
            )

            # DP sanity check (one-shot, first batch only): print the concrete
            # sharding of the batch and a sample LoRA parameter so we can verify
            # the batch dim is split across devices and params are replicated.
            if global_step == 0 and epoch == 0 and batch_idx == 0:
                try:
                    training_logger.info(f"[DP] input_ids.shape={input_ids.shape} "
                                         f"input_ids.sharding={input_ids.sharding}")
                    leaf = jax.tree.leaves(lora_params)[0]
                    training_logger.info(f"[DP] sample lora leaf shape={leaf.shape} "
                                         f"sharding={leaf.sharding}")
                    training_logger.info("[DP] input_ids sharding visualization:")
                    jax.debug.visualize_array_sharding(input_ids)
                except Exception as e:
                    training_logger.warning(f"[DP] sharding introspection failed: {e}")

            loss, lora_params, opt_state, grad_stats = jit_train_step(
                lora_params,
                frozen_state,
                opt_state,
                input_ids,
                one_hot,
                label_mask,
                attention_mask,
            )

            current_loss = float(loss)
            g_norm = float(grad_stats["grad_norm"])
            g_max = float(grad_stats["grad_max"])
            epoch_losses.append(current_loss)
            running_losses.append(current_loss)
            step_losses.append(current_loss)
            global_step += 1

            # W&B step contract: once a step is committed, any later log to
            # the same step is silently dropped. To bundle ``train/*`` and
            # ``val/*`` at the same ``global_step`` we buffer every per-step
            # metric with ``commit=False`` and emit exactly one empty
            # ``commit=True`` flush at the end of the iteration. Mirrors the
            # repo-wide pattern (see e.g. ``torch/mnist``, ``torch/qwen``).
            training_logger.log_metrics(
                {
                    "train/loss": current_loss,
                    "grad/global_norm": g_norm,
                    "grad/global_max": g_max,
                    "epoch": epoch + 1,
                    "batch": batch_idx + 1,
                },
                step=global_step,
                commit=False,
            )

            if len(running_losses) == steps_freq:
                avg = float(np.mean(running_losses))
                training_logger.log_metrics(
                    {"train/avg_window_loss": avg},
                    step=global_step,
                    commit=False,
                )
                training_logger.info(
                    f"Epoch {epoch + 1}, "
                    f"Batch {batch_idx + 1:3d}: "
                    f"Loss = {current_loss:.4f} | "
                    f"Avg {steps_freq} = {avg:.4f} | "
                    f"grad_norm = {g_norm:.4f}, "
                    f"grad_max = {g_max:.4f}"
                )
                running_losses = []
            else:
                training_logger.info(
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
                    sharding_config=sharding_config,
                    **inspect_kwargs,
                )
                training_logger.info(f"  [Step {global_step}] Validation loss: {val_loss:.4f}")
                training_logger.log_metrics(
                    {"val/loss": val_loss},
                    step=global_step,
                    commit=False,
                )

            # Flush all buffered metrics for this ``global_step`` in one commit.
            training_logger.log_metrics({}, step=global_step, commit=True)

        avg_epoch = float(np.mean(epoch_losses))
        training_logger.info(f"Epoch {epoch + 1} complete — avg loss: {avg_epoch:.4f}")

        if val_batches:
            # The last batch of the epoch already committed ``global_step``;
            # bump by one so the end-of-epoch val lands on a fresh W&B step.
            global_step += 1
            val_loss = evaluate(
                jit_eval_step,
                lora_params,
                frozen_state,
                val_batches,
                sharding_config=sharding_config,
                **inspect_kwargs,
            )
            training_logger.info(f"  Epoch {epoch + 1} validation loss: {val_loss:.4f}")
            training_logger.log_metrics({"val/loss": val_loss}, step=global_step)

    return global_step, step_losses


def _log_dp_setup(
    training_logger: TrainingLogger,
    mesh: jax.sharding.Mesh,
    sharding_config: Any,
    device_kind: str,
) -> None:
    """Print mesh, device, and sharding spec info so DP setup is visible in logs.

    Logs once at startup (cheap, no device work). Together with the one-shot
    per-tensor sharding dump on the first training batch (see ``_training_loop``),
    this is enough to confirm that data-parallel training is actually splitting
    the batch across the requested TT devices.
    """
    try:
        tt_devs = jax.devices(device_kind)
    except Exception as e:
        training_logger.warning(f"[DP] jax.devices({device_kind!r}) failed: {e}")
        tt_devs = []
    training_logger.info(f"[DP] jax.devices({device_kind!r}): {tt_devs} (count={len(tt_devs)})")
    training_logger.info(f"[DP] mesh.shape: {dict(mesh.shape)}")
    training_logger.info(f"[DP] mesh.devices:\n{np.asarray(mesh.devices)}")
    if sharding_config is None:
        training_logger.info("[DP] sharding_config=None → single-device path (no DP collectives)")
    else:
        training_logger.info(f"[DP] param_partition: {sharding_config.param_partition}")
        training_logger.info(f"[DP] data_partition:  {sharding_config.data_partition}")
        training_logger.info(f"[DP] param_sharding:  {sharding_config.param_sharding}")
        training_logger.info(f"[DP] data_sharding:   {sharding_config.data_sharding}")


def _validate_multichip_config(cfg: TrainingConfig) -> None:
    """Sanity-check multi-chip (data-parallel) YAML settings."""
    if cfg.num_devices <= 1:
        return
    if not cfg.use_tt:
        raise ValueError("num_devices > 1 is only supported on TT (set use_tt: true)")
    if cfg.batch_size % cfg.num_devices != 0:
        raise ValueError(
            f"batch_size ({cfg.batch_size}) must be divisible by num_devices ({cfg.num_devices}) "
            "for data-parallel multi-chip training"
        )


def main(training_config: TrainingConfig) -> None:
    """Run full LoRA fine-tuning pipeline."""
    random.seed(training_config.seed)
    np.random.seed(training_config.seed)

    training_logger = TrainingLogger(training_config)

    cpu_device = jax.devices("cpu")[0]
    current_device, device_kind = _select_preferred_device(
        use_tt=training_config.use_tt,
    )

    if device_kind == "tt":
        from blacksmith.tools.workaround_utils_jax import apply_gqa_workaround

        apply_gqa_workaround()

    _validate_multichip_config(training_config)

    training_logger.info(
        f"Loading {training_config.model_name} model... Using device: {device_kind} -> {current_device}"
    )

    # eformer ``create_mesh`` uses the global JAX TT device count (not
    # ``jax.default_device``).  ``sharding_axis_dims`` must multiply to that
    # count.  Training mesh size still comes from YAML ``num_devices`` below.
    if device_kind == "tt":
        mesh_axis_size = len(jax.devices("tt"))
    else:
        mesh_axis_size = 1

    with jax.default_device(cpu_device):
        model, tokenizer = load_model(
            training_config.model_name,
            dtype=training_config.jax_dtype,
            mask_max_position_embeddings=(training_config.mask_max_position_embeddings),
            mesh_axis_size=mesh_axis_size,
        )

    num_devices = training_config.num_devices
    mesh = make_tt_mesh(num_devices, device_kind)
    sharding_config = (
        ShardingConfig(num_devices=num_devices, device_kind=device_kind)
        if device_kind == "tt" and num_devices > 1
        else None
    )
    _set_nnx_model_mesh(model, mesh)
    jax.config.update("jax_default_device", current_device)

    # DP sanity check: confirm JAX sees the expected device count and that the
    # mesh/shardings have the expected shape. Prints once at startup.
    _log_dp_setup(training_logger, mesh, sharding_config, device_kind)

    parallelism = "data_parallel" if sharding_config is not None else "single_device"
    training_logger.log_model_info(
        {
            "num_hidden_layers": model.config.num_hidden_layers,
            "hidden_size": model.config.hidden_size,
            "intermediate_size": model.config.intermediate_size,
            "vocab_size": model.config.vocab_size,
            "max_position_embeddings": model.config.max_position_embeddings,
            "device": device_kind,
            "framework": "jax_easydel",
            "num_devices": num_devices,
            "parallelism": parallelism,
            "mesh_axis": AXIS_NAME,
        }
    )

    train_batches, val_batches = _load_and_prepare_batches(
        training_config,
    )

    training_logger.info(
        f"Applying LoRA (rank={training_config.lora_rank}, pattern={training_config.lora_pattern!r})..."
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
    training_logger.info(
        f"  LR schedule: warmup {training_config.warmup_steps} "
        f"optimizer steps, cosine decay over {total_opt_steps} "
        f"optimizer steps "
        f"({training_config.learning_rate} -> {training_config.end_learning_rate})"
    )

    base_tx = optax.adamw(learning_rate=schedule)
    if accum > 1:
        tx = optax.MultiSteps(base_tx, every_k_schedule=accum)
        eff = training_config.batch_size * accum
        training_logger.info(f"  Gradient accumulation: {accum} steps -> Effective batch size {eff}")
    else:
        tx = base_tx
    opt_state = tx.init(lora_params)

    if sharding_config is not None:
        lora_params = jax.tree.map(lambda x: jax.device_put(x, sharding_config.param_sharding), lora_params)
        frozen_state = jax.tree.map(lambda x: jax.device_put(x, sharding_config.param_sharding), frozen_state)
        opt_state = jax.tree.map(lambda x: jax.device_put(x, sharding_config.param_sharding), opt_state)

    jit_train_step = create_train_step_fn(
        graphdef,
        call_model,
        tx,
        device_kind=device_kind,
        num_devices=training_config.num_devices,
        sharding_config=sharding_config,
        lora_params_template=lora_params,
    )
    jit_eval_step = create_eval_step_fn(
        graphdef,
        call_model,
        device_kind=device_kind,
        num_devices=training_config.num_devices,
        sharding_config=sharding_config,
    )
    use_inspect = training_config.print_examples
    jit_inspect_step = (
        create_eval_inspect_step_fn(
            graphdef,
            call_model,
            device_kind=device_kind,
        )
        if use_inspect
        else None
    )

    if training_config.max_val_batches is not None:
        orig = len(val_batches)
        val_batches = val_batches[: training_config.max_val_batches]
        training_logger.info(f"  Using {len(val_batches)} of {orig} validation batches")

    training_logger.info("Starting training on SST-2 dataset...")

    try:
        with mesh:
            global_step, step_losses = _training_loop(
                training_config,
                training_logger,
                jit_train_step,
                jit_eval_step,
                lora_params,
                frozen_state,
                opt_state,
                train_batches,
                val_batches,
                model.config.vocab_size,
                sharding_config=sharding_config,
                device_kind=device_kind,
                jit_inspect_step=jit_inspect_step,
                tokenizer=(tokenizer if use_inspect else None),
            )

        training_logger.log_summary(
            {
                "total_steps": global_step,
                "final_loss": float(step_losses[-1]) if step_losses else float("nan"),
            }
        )
        training_logger.info("TRAINING COMPLETED")

    except Exception as e:
        training_logger.error(f"Error during training: {e}")
        raise

    finally:
        training_logger.finish()


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
