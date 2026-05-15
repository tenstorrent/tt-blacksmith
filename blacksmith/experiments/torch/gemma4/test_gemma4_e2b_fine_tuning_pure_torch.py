# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""LoRA fine-tuning for Gemma 4 E2B (text-only) on a 2x4 TT mesh."""
import copy
import traceback
from pathlib import Path

import torch
import torch_xla
import torch_xla.runtime as xr
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import Gemma4ForConditionalGeneration
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer

from blacksmith.datasets.torch.dataset_utils import get_dataset
from blacksmith.experiments.torch.llama.configs import TrainingConfig
from blacksmith.tools.checkpoints_manager import CheckpointManager
from blacksmith.tools.cli import generate_config, parse_cli_options
from blacksmith.tools.device_manager import DeviceManager
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.reproducibility_manager import ReproducibilityManager
from blacksmith.tools.torch_helpers import (
    collate_fn_for_causal_lm,
    collect_examples,
    show_examples,
)
from blacksmith.tools.workaround_utils import cross_entropy_loss, transform_labels


def _patch_embed_tokens_per_layer_split(model: torch.nn.Module) -> int:
    """Workaround ttnn.embedding silent-corruption bug at HIDDEN > 256 col-tiles.

    Gemma-4 ``embed_tokens_per_layer`` is (V, 8960) = 280 col-tiles, which falls
    in the broken kernel path. Split the lookup along the hidden dim into 2
    halves, concat, then apply ``embed_scale``.
    """
    n = 0
    for mod_name, mod in model.named_modules():
        if not mod_name.endswith("embed_tokens_per_layer"):
            continue
        if not isinstance(mod, torch.nn.Embedding):
            continue

        embed_scale = getattr(mod, "embed_scale", None)

        def _make_split_forward(embed_mod, scale):
            def _forward(input_ids):
                chunks = embed_mod.weight.chunk(2, dim=-1)
                outs = [
                    torch.nn.functional.embedding(input_ids, w.contiguous())
                    for w in chunks
                ]
                out = torch.cat(outs, dim=-1)
                if scale is not None:
                    out = out * scale
                return out

            return _forward

        mod.forward = _make_split_forward(mod, embed_scale)
        n += 1
    return n


def _strip_multimodal_towers(model: Gemma4ForConditionalGeneration) -> None:
    """Drop vision/audio towers + embedders so PEFT only attaches to text attn."""
    inner = model.model
    for attr in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
        if hasattr(inner, attr):
            delattr(inner, attr)


def get_vocab_size(model: torch.nn.Module) -> int:
    m = model
    while hasattr(m, "model") and not hasattr(m, "config"):
        m = m.model
    cfg = m.config
    return getattr(cfg, "vocab_size", None) or cfg.text_config.vocab_size


def get_model(config: TrainingConfig, device: torch.device, return_cpu_twin: bool = False):
    """Build the LoRA-wrapped model. If ``return_cpu_twin`` is True, also return
    a CPU-resident deepcopy of the model captured *before* it's moved to TT and
    ``torch.compile``-d. The twin shares numerical state with the TT model and
    is used as the PCC reference for diagnostics."""
    dtype = eval(config.dtype)

    base = Gemma4ForConditionalGeneration.from_pretrained(config.model_name, torch_dtype=dtype)
    _strip_multimodal_towers(base)
    _patch_embed_tokens_per_layer_split(base)

    if config.training_type == "lora":
        lora_cfg = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            task_type=config.lora_task_type,
        )
        model = get_peft_model(base, lora_cfg)
    else:
        raise ValueError(
            f"Only training_type='lora' is supported for Gemma 4 E2B, got '{config.training_type}'."
        )

    model.to(dtype)

    cpu_twin = None
    if return_cpu_twin:
        cpu_twin = copy.deepcopy(model)
        # Re-apply the embedding patch so its closure binds to the twin's own
        # ``embed_tokens_per_layer`` weight, not the original model's.
        _patch_embed_tokens_per_layer_split(cpu_twin)

    model.to(device)

    if config.use_tt:
        compile_options = {"tt_enable_torch_fx_fusion_pass": False, 
        "tt_legacy_compile": True, 
        #"tt_use_aot_autograd": True}
        }
        model = torch.compile(model, backend="tt", options=compile_options)

    if return_cpu_twin:
        return model, cpu_twin
    return model


def validate(model, val_data_loader, loss_fn, logger, device, config, vocab_size, tokenizer=None):
    logger.info("Starting validation...")
    total_val_loss = 0.0
    num_val_batches = 0
    collected_examples = []

    with torch.no_grad():
        for batch in tqdm(val_data_loader, desc="Validation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            # See https://github.com/tenstorrent/tt-blacksmith/issues/455.
            expected_output = batch["labels"]

            device_manager.shard_model(model)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()

            expected_output_one_hot, labels_mask = transform_labels(
                expected_output, config.ignored_index, vocab_size
            )

            if config.use_tt:
                loss = loss_fn(shift_logits, expected_output_one_hot, labels_mask)
            else:
                loss = loss_fn(shift_logits, expected_output_one_hot.to(device), labels_mask.to(device))

            predictions = shift_logits.argmax(dim=-1)
            if config.use_tt:
                torch_xla.sync(wait=True)

            total_val_loss += loss.item()
            num_val_batches += 1

            if config.print_examples:
                collected_examples = collect_examples(
                    batch_size=expected_output.shape[0],
                    collected_examples=collected_examples,
                    max_examples=10,
                    input_ids=input_ids,
                    expected_output=expected_output,
                    predictions=predictions,
                    num_val_batches=num_val_batches,
                )

            if num_val_batches > 20:
                logger.info(f"Stopping validation early after {num_val_batches} batches.")
                break

    if config.print_examples and tokenizer is not None:
        logger.info("Printing validation examples...")
        show_examples(collected_examples, tokenizer, config, logger)

    avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
    logger.info(f"Average validation loss: {avg_val_loss}")
    return avg_val_loss


# Keep large vocab-sized tensors scoped locally so they don't propagate beyond
# the step and trigger expensive CCLs in multi-chip setups.
def training_step_inner(batch, model, loss_fn, gradient_accumulation_steps):
    output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = output.logits
    shift_logits = logits[:, :-1, :].contiguous()
    loss = loss_fn(shift_logits, batch["expected_output"], batch["labels_mask"])
    scaled_loss = loss / gradient_accumulation_steps
    scaled_loss.backward()
    return loss.detach(), logits


# ---------------------------------------------------------------------------
# One-step PCC diagnostic helpers (TT vs CPU twin).
# ---------------------------------------------------------------------------
def _to_host(t: torch.Tensor) -> torch.Tensor:
    """Materialize a tensor on host. Always ``.cpu()`` BEFORE any cast so the
    device-to-host transfer stays in the source dtype (bf16) — casting on
    device first triggers a device-side reshape/typecast that pads to tile
    boundaries and blows large tensors (e.g. logits [B=1, S=1024, V=262144])
    up by 32x, causing DRAM OOM."""
    return t.detach().cpu()


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation, computed in fp64 on host. Inputs MUST already be
    on host (use ``_to_host`` first); this function does no device IO."""
    af = a.to(torch.float64).flatten()
    bf = b.to(torch.float64).flatten()
    da = af - af.mean()
    db = bf - bf.mean()
    denom = (da.norm() * db.norm()).item()
    if denom == 0.0:
        return float("nan")
    return float((da @ db).item() / denom)


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    """max|a-b| in fp64 on host. Inputs MUST already be on host."""
    return float((a.to(torch.float64) - b.to(torch.float64)).abs().max().item())


# Submodule paths inside a Gemma4TextDecoderLayer that we hook for the
# intra-layer PCC zoom-in. Order matches roughly the forward execution order.
INTRA_LAYER_SUFFIXES = (
    "input_layernorm",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.q_norm",
    "self_attn.k_norm",
    "self_attn.o_proj",
    "self_attn",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.act_fn",
    "mlp.down_proj",
    "mlp",
    "post_feedforward_layernorm",
)


def _run_step_capturing_layers(
    model, batch, gradient_accumulation_steps, intra_layer_indices=()
):
    """Wrap ``training_step_inner`` with forward hooks on every
    ``Gemma4TextDecoderLayer`` so we capture each layer's output hidden_state
    and (via ``retain_grad``) its gradient after backward.

    If ``intra_layer_indices`` is non-empty, also hook every submodule in
    ``INTRA_LAYER_SUFFIXES`` for those layer indices to capture intra-layer
    intermediates (RMSNorms, q/k/v/o projections, MLP, etc.).

    Returns ``(captured_layers, captured_intra, loss, logits)`` where
    ``captured_layers[i]`` is the layer-``i`` hidden_state output tensor and
    ``captured_intra[(layer_idx, suffix)]`` is the corresponding submodule
    output tensor. Gradients are available via ``.grad`` after backward."""
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    captured = {}
    captured_intra = {}
    handles = []

    def _make_layer_hook(idx):
        def _hook(_mod, _inputs, output):
            t = output if isinstance(output, torch.Tensor) else output[0]
            if t.requires_grad:
                t.retain_grad()
            captured[idx] = t
        return _hook

    def _make_intra_hook(layer_idx, suffix):
        def _hook(_mod, _inputs, output):
            if isinstance(output, torch.Tensor):
                t = output
            elif isinstance(output, (tuple, list)) and len(output) > 0:
                t = output[0]
            else:
                return
            if t is None:
                return
            if t.requires_grad:
                t.retain_grad()
            captured_intra[(layer_idx, suffix)] = t
        return _hook

    layer_paths = []
    for name, mod in inner.named_modules():
        if isinstance(mod, Gemma4TextDecoderLayer):
            layer_paths.append((name, mod))

    intra_set = set(intra_layer_indices)

    for idx, (layer_name, layer) in enumerate(layer_paths):
        handles.append(layer.register_forward_hook(_make_layer_hook(idx)))
        if idx not in intra_set:
            continue
        for suffix in INTRA_LAYER_SUFFIXES:
            try:
                sub = inner.get_submodule(f"{layer_name}.{suffix}")
            except AttributeError:
                continue
            handles.append(sub.register_forward_hook(_make_intra_hook(idx, suffix)))

    try:
        loss, logits = training_step_inner(
            batch, model, cross_entropy_loss, gradient_accumulation_steps
        )
    finally:
        for h in handles:
            h.remove()

    return captured, captured_intra, loss, logits


def _run_pcc_diagnostic(
    tt_model,
    cpu_twin,
    batch_cpu,
    batch_tt,
    config,
    logger,
    gradient_accumulation_steps,
    intra_layer_indices=(),
):
    """Run one fwd+bwd through both models (via ``training_step_inner``) and
    print per-layer PCC for the hidden_state output of each decoder layer
    (forward) and its gradient (backward), plus the forward PCC of logits.

    If ``intra_layer_indices`` is non-empty, also print an intra-layer PCC
    table for those layer indices, covering each submodule in
    ``INTRA_LAYER_SUFFIXES`` (RMSNorms, q/k/v/o projections, MLP pieces).

    All device->host transfers are done up-front in a single batch so the
    final PCC table is printed contiguously instead of being interleaved with
    TTNN runtime debug spam."""
    for p in tt_model.parameters():
        if p.grad is not None:
            p.grad = None
    for p in cpu_twin.parameters():
        if p.grad is not None:
            p.grad = None

    n_supervised = int(batch_cpu["labels_mask"].sum().item())
    seq_len = int(batch_cpu["input_ids"].shape[1])
    logger.info(
        f"[PCC] batch: B={batch_cpu['input_ids'].shape[0]} S={seq_len} "
        f"#supervised_tokens={n_supervised}"
    )
    if intra_layer_indices:
        logger.info(f"[PCC] intra-layer zoom-in on layers: {sorted(intra_layer_indices)}")

    logger.info("[PCC] Running CPU twin forward+backward (reference)...")
    cpu_capt, cpu_intra, cpu_loss, cpu_logits = _run_step_capturing_layers(
        cpu_twin, batch_cpu, gradient_accumulation_steps, intra_layer_indices
    )

    logger.info("[PCC] Running TT forward+backward...")
    tt_capt, tt_intra, tt_loss, tt_logits = _run_step_capturing_layers(
        tt_model, batch_tt, gradient_accumulation_steps, intra_layer_indices
    )
    if config.use_tt:
        torch_xla.sync(wait=True)

    # ---- Stage 1: materialize EVERYTHING to host (one big device->host burst)
    logger.info("[PCC] materializing tensors to host...")
    cpu_loss_h = _to_host(cpu_loss)
    tt_loss_h = _to_host(tt_loss)
    cpu_logits_h = _to_host(cpu_logits)
    tt_logits_h = _to_host(tt_logits)

    n_layers = (max(cpu_capt.keys()) + 1) if cpu_capt else 0
    layer_rows = []
    for i in range(n_layers):
        ct = cpu_capt.get(i)
        tt = tt_capt.get(i)
        if ct is None or tt is None:
            layer_rows.append((i, None, None, None, None))
            continue
        layer_rows.append((
            i,
            _to_host(ct),
            _to_host(tt),
            _to_host(ct.grad) if ct.grad is not None else None,
            _to_host(tt.grad) if tt.grad is not None else None,
        ))

    intra_rows = {}  # layer_idx -> list of (suffix, ct_h, tt_h, cg_h, tg_h)
    for layer_idx in sorted(intra_layer_indices):
        rows = []
        for suffix in INTRA_LAYER_SUFFIXES:
            ct = cpu_intra.get((layer_idx, suffix))
            tt = tt_intra.get((layer_idx, suffix))
            if ct is None or tt is None:
                rows.append((suffix, None, None, None, None))
                continue
            rows.append((
                suffix,
                _to_host(ct),
                _to_host(tt),
                _to_host(ct.grad) if ct.grad is not None else None,
                _to_host(tt.grad) if tt.grad is not None else None,
            ))
        intra_rows[layer_idx] = rows

    # ---- Stage 2: print PCC tables (host-only, no device IO from here on) --
    logger.info("=" * 88)
    logger.info(
        f"[PCC] loss   TT={tt_loss_h.item():.6f}   CPU={cpu_loss_h.item():.6f}   "
        f"d={tt_loss_h.item() - cpu_loss_h.item():+.4e}"
    )
    logger.info(
        f"[PCC] logits fwd PCC={_pcc(cpu_logits_h, tt_logits_h):.6f}  "
        f"max|d|={_max_abs(cpu_logits_h, tt_logits_h):.4e}"
    )

    logger.info("")
    logger.info(
        f"{'layer':>5} | {'fwd PCC':>10} | {'fwd max|d|':>12} | "
        f"{'grad PCC':>10} | {'grad max|d|':>13}"
    )
    logger.info("-" * 72)
    for i, ct_h, tt_h, cg_h, tg_h in layer_rows:
        if ct_h is None or tt_h is None:
            logger.info(f"{i:>5} | (missing)")
            continue
        f_pcc = _pcc(ct_h, tt_h)
        f_mae = _max_abs(ct_h, tt_h)
        if cg_h is not None and tg_h is not None:
            g_pcc_s = f"{_pcc(cg_h, tg_h):>10.6f}"
            g_mae_s = f"{_max_abs(cg_h, tg_h):>13.4e}"
        else:
            g_pcc_s = f"{'(no grad)':>10}"
            g_mae_s = f"{'-':>13}"
        logger.info(
            f"{i:>5} | {f_pcc:>10.6f} | {f_mae:>12.4e} | {g_pcc_s} | {g_mae_s}"
        )
    logger.info("=" * 72)

    for layer_idx in sorted(intra_rows.keys()):
        logger.info("")
        logger.info(f"---- Intra-layer PCC, layer {layer_idx} {'-' * 40}")
        logger.info(
            f"{'submodule':<32} | {'fwd PCC':>10} | {'fwd max|d|':>12} | "
            f"{'grad PCC':>10} | {'grad max|d|':>13}"
        )
        logger.info("-" * 88)
        for suffix, ct_h, tt_h, cg_h, tg_h in intra_rows[layer_idx]:
            if ct_h is None or tt_h is None:
                logger.info(f"{suffix:<32} | (missing)")
                continue
            f_pcc = _pcc(ct_h, tt_h)
            f_mae = _max_abs(ct_h, tt_h)
            if cg_h is not None and tg_h is not None:
                g_pcc_s = f"{_pcc(cg_h, tg_h):>10.6f}"
                g_mae_s = f"{_max_abs(cg_h, tg_h):>13.4e}"
            else:
                g_pcc_s = f"{'(no grad)':>10}"
                g_mae_s = f"{'-':>13}"
            logger.info(
                f"{suffix:<32} | {f_pcc:>10.6f} | {f_mae:>12.4e} | {g_pcc_s} | {g_mae_s}"
            )
        logger.info("=" * 88)


def train(
    config: TrainingConfig,
    device_manager: DeviceManager,
    logger: TrainingLogger,
    checkpoint_manager: CheckpointManager,
):
    logger.info("Starting training...")

    model, cpu_twin = get_model(config, device_manager.device, return_cpu_twin=True)
    vocab_size = get_vocab_size(model)
    logger.info(f"Loaded {config.model_name} (text-only view). vocab_size={vocab_size}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)

    if config.resume_from_checkpoint:
        checkpoint_manager.load_checkpoint(model, optimizer)

    train_dataset = get_dataset(config=config, split="train", collate_fn=collate_fn_for_causal_lm)
    train_dataloader = train_dataset.get_dataloader()
    logger.info(f"Loaded {config.dataset_id} dataset. Train size: {len(train_dataloader)*config.batch_size}")

    eval_dataset = get_dataset(config=config, split="validation", collate_fn=collate_fn_for_causal_lm)
    eval_dataloader = eval_dataset.get_dataloader()
    logger.info(f"Loaded {config.dataset_id} dataset. Eval size: {len(eval_dataloader)*config.batch_size}")

    tokenizer = train_dataset.tokenizer

    global_step = 0
    running_loss = 0.0

    try:
        #model.eval()
        #val_loss = validate(
        #    model,
        #    eval_dataloader,
        #    cross_entropy_loss,
        #    logger,
        #    device_manager.device,
        #    config,
        #    vocab_size,
        #    tokenizer,
        #)
        #logger.log_metrics({"val/loss": val_loss}, commit=True, step=global_step)
        model.train()

        for epoch in range(config.num_epochs):
            accumulation_step = 0

            for batch in tqdm(train_dataloader, desc="Training"):
                if accumulation_step == 0:
                    optimizer.zero_grad()

                expected_output, labels_mask = transform_labels(
                    batch["labels"], config.ignored_index, vocab_size
                )
                # Keep CPU-side tensors for the CPU twin's reference fwd+bwd.
                batch_cpu = {
                    "input_ids": batch["input_ids"].clone(),
                    "attention_mask": batch["attention_mask"].clone(),
                    "expected_output": expected_output.clone(),
                    "labels_mask": labels_mask.clone(),
                }
                batch = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "expected_output": expected_output,
                    "labels_mask": labels_mask,
                }

                batch = device_manager.prepare_batch(batch)
                device_manager.shard_model(model)

                # ONE-STEP PCC DIAGNOSTIC: per-layer hidden_state fwd + grad + logits
                # vs. an identical-weight CPU twin. Zooms into layer 9 at
                # submodule granularity to localize the 0.985 -> 0.870 backward
                # grad PCC drop happening *inside* layer-9's bwd. Don't add
                # layer 8 here: with hooks on two early layers, AOTAutograd
                # extracts a degenerate const-zero subgraph that triggers a
                # 1x8 -> 1x1 mesh-device reshape on TT-XLA and crashes the next
                # executable with "Device count mismatch: 8 vs 1".
                _run_pcc_diagnostic(
                    tt_model=model,
                    cpu_twin=cpu_twin,
                    batch_cpu=batch_cpu,
                    batch_tt=batch,
                    config=config,
                    logger=logger,
                    gradient_accumulation_steps=config.gradient_accumulation_steps,
                    intra_layer_indices=(9,),
                )

                if config.use_tt:
                    torch_xla.sync(wait=True)
                exit(0)

                loss_, _ = training_step_inner(
                    batch, model, cross_entropy_loss, config.gradient_accumulation_steps
                )

                running_loss += loss_.item()
                accumulation_step += 1

                print(f"Loss at step {global_step} is {loss_.item()}", flush=True)

                if accumulation_step == config.gradient_accumulation_steps:
                    device_manager.optimizer_step(optimizer)

                    accumulation_step = 0
                    global_step += 1

                    if global_step % config.steps_freq == 0:
                        avg_loss = running_loss / (config.steps_freq * config.gradient_accumulation_steps)
                        logger.log_metrics({"train/loss": avg_loss}, commit=False, step=global_step)
                        running_loss = 0.0

                    if global_step % config.val_steps_freq == 0:
                        model.eval()
                        val_loss = validate(
                            model,
                            eval_dataloader,
                            cross_entropy_loss,
                            logger,
                            device_manager.device,
                            config,
                            vocab_size,
                            tokenizer,
                        )
                        logger.log_metrics({"val/loss": val_loss}, commit=False, step=global_step)
                        model.train()

                    logger.log_metrics({}, commit=True, step=global_step)

                    #if config.use_tt:
                    #    xr.clear_computation_cache()

                    if checkpoint_manager.should_save_checkpoint(global_step):
                        checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)

            if checkpoint_manager.should_save_checkpoint(global_step, epoch):
                checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)

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
    default_config = Path(__file__).parent / "test_gemma4_e2b_wizardlm.yaml"
    args = parse_cli_options(default_config=default_config)
    config: TrainingConfig = generate_config(TrainingConfig, args.config, args.test_config, args.test_checkpoint_path)

    repro_manager = ReproducibilityManager(config)
    repro_manager.setup()

    logger = TrainingLogger(config, args.test_log_filename_prefix)

    device_manager = DeviceManager(config)
    logger.info(f"Using device: {device_manager.device}")

    if config.use_tt:
        torch_xla.set_custom_compile_options({"fp32_dest_acc_en": True, "math_fidelity": "hifi4"})

    checkpoint_manager = CheckpointManager(config, logger, device_manager.device)

    train(config, device_manager, logger, checkpoint_manager)
