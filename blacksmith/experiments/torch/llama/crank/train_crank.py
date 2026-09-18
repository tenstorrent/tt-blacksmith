# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Llama LoRA fine-tuning on Tenstorrent hardware through tt-crank.

The tt-xla counterpart is `blacksmith/experiments/torch/llama/xla/train.py`; this script
keeps its structure (config, logger, checkpoints, validation cadence) but runs the training
step the tt-crank experiments settled on: the loss is a function of an explicit parameter
dict fed through `torch.func.functional_call`, compiled once with the `tt` dynamo backend,
and the leaves are updated in place by fused AdamW.
"""
import contextlib
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from blacksmith.datasets.torch.dataset_utils import get_dataset
from blacksmith.experiments.torch.llama.crank.configs import CrankTrainingConfig
from blacksmith.experiments.torch.llama.crank.loss import negative_log_likelihood_loss
from blacksmith.tools.checkpoints_manager import CheckpointManager
from blacksmith.tools.cli import generate_config, parse_cli_options
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.performance_utils import (
    TtnnPerfMetrics,
    count_flop_params,
    peak_flops,
    training_flops_per_step,
    training_flops_per_token,
)
from blacksmith.tools.reproducibility_manager import ReproducibilityManager
from blacksmith.tools.torch_helpers import (
    collate_fn_for_causal_lm,
    collect_examples,
    show_examples,
)
from blacksmith.tools.workaround_utils import transform_labels

TT_DEVICE = "tt"
TT_BACKEND = "tt"
# The loss is a [1, 1, 1] tensor rather than a 0-d scalar (see loss.py), so backward needs
# an explicit seed of the same shape.
LOSS_SHAPE = (1, 1, 1)
MAX_PRINTED_EXAMPLES = 10
TFLOP = 1e12

# Storage dtype of every matmul weight the compiler lowers. bf16 is the default, so it maps
# to "leave the option unset".
WEIGHT_DTYPE_OPTIONS = ("bfp_bf8", "bfp_bf4", "bf16")


@dataclass
class FunctionalState:
    """The tensors `functional_call` substitutes for the module's own.

    `params` are the autograd leaves the optimizer updates in place; `frozen` and `buffers`
    never change. `aliases` maps every additional name of a tied weight onto its canonical
    entry so one tensor, one gradient and one set of moments feed all of them.
    """

    params: dict[str, torch.Tensor]
    frozen: dict[str, torch.Tensor]
    buffers: dict[str, torch.Tensor]
    aliases: dict[str, str]


def get_model(
    config: CrankTrainingConfig, device: torch.device, logger: TrainingLogger
) -> tuple[torch.nn.Module, torch.nn.Module, bool]:
    """Load the HF model, wrap it in LoRA and move it to `device`.

    Not `hf_models.get_model`: that loads in fp32, casts the whole model (LoRA adapters
    included) to `config.dtype` afterwards and applies tt-xla's per-tensor weight-dtype
    overrides. The tt-crank experiments load in `config.dtype` directly, keep the fp32
    adapters peft creates, and need the base model handle to re-tie the embeddings.

    Returns:
        The (possibly peft-wrapped) model, the base model and whether the LM head shares
        the embedding weight after the move.
    """
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        use_cache=False,
        low_cpu_mem_usage=True,
        dtype=config.torch_dtype(),
    )
    base_model = model

    if config.training_model_type == "lora":
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            task_type=config.lora_task_type,
        )
        model = get_peft_model(model, lora_config)
    else:
        logger.warning(
            f"training_model_type '{config.training_model_type}' is not 'lora'; "
            "falling back to full fine-tuning (all parameters trainable)."
        )

    # The module stays in eval mode for the whole run: train()/eval() is baked into the
    # compiled graph, and Llama has no dropout for train() to enable.
    model.eval()
    model.to(device)

    # .to(device) does not preserve weight tying: _has_compatible_shallow_copy_type is False
    # for a CPU->tt conversion, so nn.Module._apply replaces each Parameter rather than
    # swapping its .data, and lm_head.weight stops being embed_tokens.weight. Left alone the
    # run trains two drifting copies of one logical weight with two sets of AdamW moments.
    # No-op on CPU.
    base_model.tie_weights()

    # Read after the move and the re-tie, so `tied` describes the model that actually trains.
    embedding_weight = base_model.get_input_embeddings().weight
    output_weight = getattr(base_model.get_output_embeddings(), "weight", None)
    tied = output_weight is embedding_weight
    if getattr(base_model.config, "tie_word_embeddings", False) and not tied:
        raise RuntimeError("weight tying did not survive .to(device) -- the run would train two copies")

    return model, base_model, tied


def build_functional_state(model: torch.nn.Module) -> FunctionalState:
    """Split the module's tensors into the dicts `functional_call` substitutes.

    Trainability follows the `requires_grad` flags peft (or full fine-tuning) left on the
    module, so `CheckpointManager` sees the same trainable set.
    """
    # named_parameters() dedups a tied weight to one entry, but functional_call substitutes
    # by *name*: passing only the canonical name leaves the alias bound to the module's
    # original tensor, and the LM head then reads a stale weight on every step after the
    # first. The aliases are re-expanded inside the loss so one dict entry feeds both names.
    canonical: dict[int, str] = {}
    aliases: dict[str, str] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if id(parameter) in canonical:
            aliases[name] = canonical[id(parameter)]
        else:
            canonical[id(parameter)] = name

    params = {
        name: parameter.detach().requires_grad_(True)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    frozen = {name: parameter.detach() for name, parameter in model.named_parameters() if not parameter.requires_grad}
    buffers = {name: buffer.detach() for name, buffer in model.named_buffers()}

    # `.detach()` returns plain Tensors, and functional_call feeds them in as ordinary graph
    # args. Dynamo only auto-marks `nn.Parameter` instances as static inputs, so without this
    # every arg lands in the graph untagged: tt-crank then tags all args as inputs and
    # tt-mlir skips them in every pass gated on constant args -- const-eval hoisting and,
    # with an experimental weight dtype set, the weight dtype conversion, whose pattern
    # requires the matmul's weight operand to trace to a parameter/constant arg.
    #
    # Only `frozen` and `buffers` are marked: under LoRA those never change, so "constant
    # across calls" is the truth. `params` are updated in place by the optimizer every step,
    # and tagging them constant would let const-eval fold a weight-derived subgraph once and
    # then serve it stale.
    for tensor in (*frozen.values(), *buffers.values()):
        torch._dynamo.mark_static_address(tensor)

    return FunctionalState(params=params, frozen=frozen, buffers=buffers, aliases=aliases)


def make_loss_fn(model: torch.nn.Module, state: FunctionalState, dtype: torch.dtype) -> Callable:
    """Build the training loss as a function of the parameter dict.

    The module and the constant dicts are closed over rather than passed, so the compiled
    graph's inputs are exactly the parameters and the batch tensors.
    """
    frozen, buffers, aliases = state.frozen, state.buffers, state.aliases

    def loss_fn(params, input_ids, attention_mask, expected_output, loss_scale):
        resolved = {**frozen, **params}
        resolved.update({alias: resolved[name] for alias, name in aliases.items()})
        logits = torch.func.functional_call(
            model, (resolved, buffers), kwargs={"input_ids": input_ids, "attention_mask": attention_mask}
        ).logits
        shift_logits = logits[:, :-1, :].contiguous()
        return negative_log_likelihood_loss(shift_logits, expected_output, loss_scale, dtype)

    return loss_fn


def make_eval_fn(
    model: torch.nn.Module, state: FunctionalState, dtype: torch.dtype, with_predictions: bool
) -> Callable:
    """Build the validation function: the loss, plus argmax predictions when examples are printed.

    Compiled separately from the training loss so the forward-only graph never carries the
    prediction argmax when nobody reads it.
    """
    frozen, buffers, aliases = state.frozen, state.buffers, state.aliases

    def eval_fn(params, input_ids, attention_mask, expected_output, loss_scale):
        resolved = {**frozen, **params}
        resolved.update({alias: resolved[name] for alias, name in aliases.items()})
        logits = torch.func.functional_call(
            model, (resolved, buffers), kwargs={"input_ids": input_ids, "attention_mask": attention_mask}
        ).logits
        shift_logits = logits[:, :-1, :].contiguous()
        loss = negative_log_likelihood_loss(shift_logits, expected_output, loss_scale, dtype)
        if with_predictions:
            return loss, shift_logits.argmax(dim=-1)
        return loss

    return eval_fn


def prepare_batch(
    batch: dict[str, torch.Tensor],
    vocab_size: int,
    ignored_index: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int]:
    """Expand the pre-shifted labels into the one-hot targets and the loss scale, on the host.

    Returns:
        The device batch and the number of valid (non-ignored) target tokens.
    """
    # TODO: Refactor when https://github.com/tenstorrent/tt-blacksmith/issues/327 is resolved.
    expected_output, labels_mask = transform_labels(batch["labels"], ignored_index, vocab_size)
    mask = labels_mask.float()
    valid_tokens = mask.sum().clamp_min(1.0)
    # -mask / valid_tokens folds the sign and the token mean into one host-side weight, so the
    # graph is a plain weighted sum (see loss.py).
    loss_scale = (-mask / valid_tokens).unsqueeze(-1)

    device_batch = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "expected_output": expected_output.to(dtype).to(device),
        "loss_scale": loss_scale.to(device),
    }
    return device_batch, int(valid_tokens)


def read_loss(loss: torch.Tensor) -> float:
    """Bring the [1, 1, 1] device loss back as a float. Reading it is what syncs the step."""
    return float(loss.detach().float().cpu().reshape(()))


def build_compile_options(config: CrankTrainingConfig) -> dict:
    """Translate the config into `torch.compile(..., options=...)` for the tt backend."""
    from tt_crank.torch._compile import BfpDtype, CompileOption, MathFidelity

    # Highest numerical precision for stable fine-tuning convergence, as in the tt-xla script:
    # accumulate partial results in FP32 and use all four mantissa phases.
    options = {
        CompileOption.FP32_DEST_ACC_EN: True,
        CompileOption.MATH_FIDELITY: MathFidelity.HiFi4,
        CompileOption.OPT_LEVEL: config.opt_level,
        CompileOption.ENABLE_CONST_EVAL: config.enable_const_eval,
    }
    if config.enable_trace:
        options[CompileOption.ENABLE_TRACE] = True

    if config.experimental_weight_dtype is not None:
        if config.experimental_weight_dtype not in WEIGHT_DTYPE_OPTIONS:
            raise ValueError(
                f"Unsupported experimental_weight_dtype {config.experimental_weight_dtype!r}; "
                f"expected one of {WEIGHT_DTYPE_OPTIONS}"
            )
        weight_dtypes = {"bfp_bf8": BfpDtype.BfpBf8, "bfp_bf4": BfpDtype.BfpBf4}
        if config.experimental_weight_dtype in weight_dtypes:
            options[CompileOption.EXPERIMENTAL_WEIGHT_DTYPE] = weight_dtypes[config.experimental_weight_dtype]

    if config.perf_metrics_enabled:
        options[CompileOption.TTNN_PERF_METRICS_ENABLED] = True
        options[CompileOption.TTNN_PERF_METRICS_OUTPUT_FILE] = config.perf_metrics_file

    return options


@dataclass
class TrainStepContext:
    """Everything one optimizer step needs besides the batch."""

    loss_fn: Callable
    state: FunctionalState
    optimizer: torch.optim.Optimizer
    grad_seed: torch.Tensor
    step_tflops: float
    perf_metrics: Optional[TtnnPerfMetrics]
    dump_artifacts: Optional[Callable]


def train_step(
    ctx: TrainStepContext, device_batch: dict[str, torch.Tensor], is_first_step: bool, logger: TrainingLogger
) -> torch.Tensor:
    """Forward, backward and in-place AdamW update; returns the loss tensor still on device."""
    # The first step compiles the forward and backward graphs; read tt-mlir's exact FLOP
    # report after each so the HFU numerator covers both. Whatever the validation compile
    # left behind is not part of the step.
    if is_first_step and ctx.perf_metrics:
        ctx.perf_metrics.discard()

    loss_tensor = ctx.loss_fn(
        ctx.state.params,
        device_batch["input_ids"],
        device_batch["attention_mask"],
        device_batch["expected_output"],
        device_batch["loss_scale"],
    )
    if is_first_step and ctx.perf_metrics:
        ctx.perf_metrics.collect()

    loss_tensor.backward(ctx.grad_seed)
    if is_first_step and ctx.perf_metrics:
        ctx.perf_metrics.collect()
        if ctx.perf_metrics.peak_flops_per_sec:
            logger.info(
                f"Exact {ctx.perf_metrics.total_flops / TFLOP:.2f} TFLOP per step from tt-mlir vs analytical "
                f"{ctx.step_tflops:.2f}, weighted peak {ctx.perf_metrics.peak_flops_per_sec / TFLOP:.2f} TFLOP/s"
            )

    # A tied weight is one leaf on two paths, so autograd sums both contributions into its
    # single .grad; every leaf must have one.
    missing = [name for name, tensor in ctx.state.params.items() if tensor.grad is None]
    if missing:
        raise RuntimeError(f"no gradient reached {len(missing)} parameters, first: {missing[0]}")

    # The leaves are updated in place, so `state.params` stays valid.
    ctx.optimizer.step()
    ctx.optimizer.zero_grad(set_to_none=True)

    if is_first_step and ctx.dump_artifacts is not None:
        # Every later step re-runs these same compiled programs, so no further graph would be
        # registered: dump now instead of at block exit.
        artifacts_dir = ctx.dump_artifacts()
        logger.info(f"Wrote IR artifacts to {artifacts_dir}" if artifacts_dir else "No IR artifacts collected")

    return loss_tensor


def log_train_metrics(
    logger: TrainingLogger,
    config: CrankTrainingConfig,
    step: int,
    avg_loss: float,
    avg_step_time: float,
    step_flops: int,
    device_peak_flops: Optional[float],
    perf_metrics: Optional[TtnnPerfMetrics],
    valid_tokens: int,
) -> None:
    metrics = {
        "train/loss": avg_loss,
        "perf/step_time": avg_step_time,
        "perf/tokens_per_sec": config.batch_size * config.max_length / avg_step_time,
        "perf/tflops": step_flops / avg_step_time / TFLOP,
        "perf/valid_tokens": valid_tokens,
    }
    if device_peak_flops:
        metrics["perf/mfu"] = step_flops / avg_step_time / device_peak_flops * 100.0
    hfu = perf_metrics.hfu(avg_step_time) if perf_metrics else None
    if hfu is not None:
        metrics["perf/hfu"] = hfu
    logger.log_metrics(metrics, commit=False, step=step)


def validate(
    eval_fn: Callable,
    params: dict[str, torch.Tensor],
    val_dataloader,
    logger: TrainingLogger,
    config: CrankTrainingConfig,
    vocab_size: int,
    device: torch.device,
    tokenizer=None,
) -> float:
    logger.info("Starting validation...")
    dtype = config.torch_dtype()
    total_val_loss = 0.0
    num_val_batches = 0
    collected_examples = []

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation"):
            device_batch, _ = prepare_batch(batch, vocab_size, config.ignored_index, dtype, device)
            outputs = eval_fn(
                params,
                device_batch["input_ids"],
                device_batch["attention_mask"],
                device_batch["expected_output"],
                device_batch["loss_scale"],
            )
            loss, predictions = outputs if config.print_examples else (outputs, None)

            total_val_loss += read_loss(loss)
            num_val_batches += 1

            if config.print_examples:
                collected_examples = collect_examples(
                    batch_size=batch["labels"].shape[0],
                    collected_examples=collected_examples,
                    max_examples=MAX_PRINTED_EXAMPLES,
                    input_ids=batch["input_ids"],
                    expected_output=batch["labels"],
                    predictions=predictions,
                    num_val_batches=num_val_batches,
                )

    if config.print_examples and tokenizer is not None:
        logger.info("Printing validation examples...")
        show_examples(collected_examples, tokenizer, config, logger)

    avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
    logger.info(f"Average validation loss: {avg_val_loss}")
    return avg_val_loss


def train(
    config: CrankTrainingConfig,
    device: torch.device,
    logger: TrainingLogger,
    checkpoint_manager: CheckpointManager,
    compile_options: Optional[dict],
):
    logger.info("Starting training...")
    dtype = config.torch_dtype()

    # Load model.
    model, base_model, tied = get_model(config, device, logger)
    state = build_functional_state(model)
    hf_config = base_model.config
    vocab_size = hf_config.vocab_size

    parameter_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(t.numel() for t in state.params.values())
    logger.info(f"Loaded {config.model_name} model.")
    logger.info(f"Model parameters: {parameter_count}")
    logger.info(f"Trainable parameters: {trainable_count} in {len(state.params)} tensors")
    logger.info(f"Weight tying {'on' if tied else 'off'} after .to({device})")
    if state.aliases:
        aliases = ", ".join(f"{alias} -> {name}" for alias, name in state.aliases.items())
        logger.info(f"{len(state.aliases)} tied alias(es): {aliases}")

    # Analytical MFU numerator; the same formula as the tt-xla trainer so the numbers compare.
    head_dim = getattr(hf_config, "head_dim", None) or hf_config.hidden_size // hf_config.num_attention_heads
    flop_params = count_flop_params(
        model,
        embedding_weight=base_model.get_input_embeddings().weight,
        is_trainable=lambda name: name in state.params,
        tied_embeddings=tied,
    )
    flops_per_token = training_flops_per_token(
        flop_params,
        num_layers=hf_config.num_hidden_layers,
        num_heads=hf_config.num_attention_heads,
        head_dim=head_dim,
        seq_len=config.max_length,
    )
    step_flops = training_flops_per_step(
        flops_per_token,
        batch_size=config.batch_size,
        seq_len=config.max_length,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    device_peak_flops = peak_flops() if config.use_tt else None
    step_tflops = step_flops / TFLOP
    peak_info = f", peak {device_peak_flops / TFLOP:.2f} TFLOP/s" if device_peak_flops else ""
    logger.info(f"{step_tflops:.2f} TFLOP per step from {flop_params.total} charged parameters{peak_info}")

    # Stock fused AdamW. `fused=True` routes to aten::_fused_adamw_, which tt-crank lowers to
    # one fused kernel per parameter. The leaves stay put across steps so the optimizer owns
    # the moment state instead of it being threaded through the graph.
    optimizer = torch.optim.AdamW(
        list(state.params.values()),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
        fused=True,
    )

    # The leaves in `state.params` share storage with the module's parameters, so weights
    # restored into the module are what the next step trains.
    if config.resume_from_checkpoint:
        checkpoint_manager.load_checkpoint(model, optimizer)

    # Load dataset.
    train_dataset = get_dataset(config=config, split="train", collate_fn=collate_fn_for_causal_lm)
    train_dataloader = train_dataset.get_dataloader()
    logger.info(f"Loaded {config.dataset_id} dataset. Train dataset size: {len(train_dataloader) * config.batch_size}")

    eval_dataset = get_dataset(config=config, split="validation", collate_fn=collate_fn_for_causal_lm)
    eval_dataloader = eval_dataset.get_dataloader()
    logger.info(f"Loaded {config.dataset_id} dataset. Eval dataset size: {len(eval_dataloader) * config.batch_size}")

    tokenizer = train_dataset.tokenizer

    loss_fn = make_loss_fn(model, state, dtype)
    eval_fn = make_eval_fn(model, state, dtype, with_predictions=config.print_examples)
    if config.use_tt:
        loss_fn = torch.compile(loss_fn, backend=TT_BACKEND, dynamic=False, options=compile_options)
        eval_fn = torch.compile(eval_fn, backend=TT_BACKEND, dynamic=False, options=compile_options)

    grad_seed = torch.ones(LOSS_SHAPE, dtype=dtype).to(device)

    perf_metrics = None
    if config.use_tt and config.perf_metrics_enabled:
        perf_metrics = TtnnPerfMetrics(config.perf_metrics_file)
        perf_metrics.clear_stale()

    # While this collection is open the engine registers every compiled graph's TTIR and TTNN
    # IR. Registration order, so the forward comes before the backward and the per-parameter
    # AdamW graphs last -- which is why the collection has to stay open across
    # optimizer.step(). Capturing the TTIR costs a full module print the engine only pays
    # while a collection is active, hence dumping right after the first step.
    dump_artifacts = None
    artifacts_context = contextlib.nullcontext()
    if config.use_tt and config.artifacts_name:
        from tt_crank.torch._artifacts import collect_artifacts, dump_artifacts

        artifacts_context = collect_artifacts(config.artifacts_name)

    ctx = TrainStepContext(
        loss_fn=loss_fn,
        state=state,
        optimizer=optimizer,
        grad_seed=grad_seed,
        step_tflops=step_tflops,
        perf_metrics=perf_metrics,
        dump_artifacts=dump_artifacts,
    )
    global_step = 0
    epoch = 0

    try:
        # Initial validation
        val_loss = validate(eval_fn, state.params, eval_dataloader, logger, config, vocab_size, device, tokenizer)
        logger.log_metrics({"val/loss": val_loss}, commit=True, step=global_step)

        train_start = time.perf_counter()

        with artifacts_context:
            for epoch in range(config.num_epochs):
                running_loss = 0.0
                running_time = 0.0
                reached_max_steps = False

                for batch in tqdm(train_dataloader, desc="Training"):
                    if config.max_steps is not None and global_step >= config.max_steps:
                        reached_max_steps = True
                        break
                    step_start = time.perf_counter()
                    is_first_step = global_step == 0

                    device_batch, valid_tokens = prepare_batch(batch, vocab_size, config.ignored_index, dtype, device)
                    loss_tensor = train_step(ctx, device_batch, is_first_step, logger)

                    loss = read_loss(loss_tensor)
                    elapsed = time.perf_counter() - step_start
                    global_step += 1
                    running_loss += loss
                    running_time += elapsed

                    if global_step % config.steps_freq == 0:
                        log_train_metrics(
                            logger,
                            config,
                            global_step,
                            avg_loss=running_loss / config.steps_freq,
                            avg_step_time=running_time / config.steps_freq,
                            step_flops=step_flops,
                            device_peak_flops=device_peak_flops,
                            perf_metrics=perf_metrics,
                            valid_tokens=valid_tokens,
                        )
                        running_loss = 0.0
                        running_time = 0.0

                    # Validation
                    if global_step % config.val_steps_freq == 0:
                        val_loss = validate(
                            eval_fn, state.params, eval_dataloader, logger, config, vocab_size, device, tokenizer
                        )
                        logger.log_metrics({"val/loss": val_loss}, commit=False, step=global_step)

                    # Commit metrics to W&B.
                    logger.log_metrics({}, commit=True, step=global_step)

                    # Save step checkpoint.
                    if checkpoint_manager.should_save_checkpoint(global_step):
                        checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)

                # Save epoch checkpoint.
                if checkpoint_manager.should_save_checkpoint(global_step, epoch):
                    checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)

                if reached_max_steps:
                    break

        if config.measure_e2e_time:
            train_elapsed = time.perf_counter() - train_start
            logger.info(f"Training e2e time: {train_elapsed:.3f}s ({global_step} steps)")

        # Save final model.
        final_model_path = checkpoint_manager.save_checkpoint(
            model, global_step, epoch, optimizer, checkpoint_name="final_model.pth"
        )
        logger.log_artifact(final_model_path, artifact_type="model", name="final_model.pth")

    except Exception as e:
        traceback_str = traceback.format_exc()
        logger.error(f"Training failed with error: {str(e)}", traceback_str)
        raise
    finally:
        logger.finish()


if __name__ == "__main__":
    # Config setup
    default_config = Path(__file__).parent / "lora" / "single_chip" / "llama_3_1_8b_sst2.yaml"
    args = parse_cli_options(default_config=default_config)
    config: CrankTrainingConfig = generate_config(
        CrankTrainingConfig, args.config, args.test_config, args.test_checkpoint_path
    )

    # Reproducibility setup
    repro_manager = ReproducibilityManager(config)
    repro_manager.setup()

    # Logger setup.
    logger = TrainingLogger(config, args.test_log_filename_prefix)

    # Device setup. tt-crank reads TT_CRANK_ARTIFACTS_DIR into a static at library load, so
    # it has to be set before the extension module is imported -- setting it later is
    # silently ignored.
    compile_options = None
    if config.use_tt:
        if config.artifacts_name:
            os.environ.setdefault("TT_CRANK_ARTIFACTS_DIR", config.artifacts_dir)

        import tt_crank.torch  # noqa: F401  (registers the tt device and the "tt" dynamo backend)

        device = torch.device(TT_DEVICE)
        compile_options = build_compile_options(config)
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Checkpoint manager setup
    checkpoint_manager = CheckpointManager(config, logger, device)

    # Start training.
    train(config, device, logger, checkpoint_manager, compile_options)
