# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Building blocks of the tt-crank Llama fine-tuning step.

Model loading, the functional parameter state `torch.func.functional_call` substitutes, the
loss and eval functions compiled with the `tt` dynamo backend, batch preparation, compile
options and the single optimizer step. `train.py` wires these into the training loop.
"""
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from blacksmith.experiments.torch.llama.crank.configs import CrankTrainerConfig
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.performance_utils import TtnnPerfMetrics
from blacksmith.tools.workaround_utils import transform_labels

TT_DEVICE = "tt"
TT_BACKEND = "tt"
# The loss is a [1, 1, 1] tensor rather than a 0-d scalar (see `negative_log_likelihood_loss`), so backward needs
# an explicit seed of the same shape.
LOSS_SHAPE = (1, 1, 1)
MAX_PRINTED_EXAMPLES = 10
TFLOP = 1e12


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
    config: CrankTrainerConfig, device: torch.device, logger: TrainingLogger
) -> tuple[torch.nn.Module, torch.nn.Module, bool]:
    """Load the HF model, wrap it in LoRA and move it to `device`.

    The tt-crank experiments load in `config.dtype` directly, keep the fp32
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


def negative_log_likelihood_loss(
    shift_logits: torch.Tensor,
    expected_output: torch.Tensor,
    loss_scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Token-averaged causal-LM loss written for the ops tt-crank lowers.

    `loss_scale` is `-mask / valid_tokens`, pre-divided on the host so the graph needs no
    `aten.div`. The final sum takes an explicit dim list because only `aten.sum.dim_IntList`
    is lowered on tt, not the full-reduction overload, which is also why the result is a
    `[1, 1, 1]` tensor rather than a 0-d scalar.

    Args:
        shift_logits: `[batch, seq_len - 1, vocab]` logits for the next-token positions.
        expected_output: One-hot targets of the same shape (zeros on masked positions).
        loss_scale: `[batch, seq_len - 1, 1]` per-token weights, `-1 / valid_tokens` or 0.
        dtype: Compute dtype the log-probabilities are cast back to.

    Returns:
        The `[1, 1, 1]` loss.
    """
    probabilities = torch.softmax(shift_logits, dim=-1)
    target_probabilities = (probabilities * expected_output).sum(dim=-1, keepdim=True)
    log_probabilities = torch.log(target_probabilities.float().clamp_min(1e-12)).to(dtype)
    return (log_probabilities * loss_scale).sum(dim=[0, 1, 2], keepdim=True)


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
    # graph is a plain weighted sum (see `negative_log_likelihood_loss`).
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


def build_compile_options(config: CrankTrainerConfig) -> dict:
    """Translate the config into `torch.compile(..., options=...)` for the tt backend."""
    from tt_crank.torch._compile import BfpDtype, CompileOption, MathFidelity

    options = {
        CompileOption.FP32_DEST_ACC_EN: True,
        CompileOption.MATH_FIDELITY: MathFidelity.HiFi4,
        CompileOption.OPT_LEVEL: config.optimization_level,
        CompileOption.ENABLE_CONST_EVAL: config.enable_const_eval,
    }
    if config.enable_trace:
        options[CompileOption.ENABLE_TRACE] = True

    # The config already validated the value; bf16 leaves the option unset.
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
    config: CrankTrainerConfig,
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
