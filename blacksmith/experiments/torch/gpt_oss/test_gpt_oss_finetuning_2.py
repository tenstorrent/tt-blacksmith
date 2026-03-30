# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import traceback
from pathlib import Path

import torch
import torch_xla
import torch_xla.runtime as xr
from tqdm import tqdm

from blacksmith.datasets.torch.dataset_utils import get_dataset
from blacksmith.experiments.torch.gpt_oss.configs import TrainingConfig
from blacksmith.models.torch.gpt_oss_overrides import get_gpt_oss_model, print_debug_intermediates
from blacksmith.tools.checkpoints_manager import CheckpointManager
from blacksmith.tools.cli import generate_config, parse_cli_options
from blacksmith.tools.device_manager import DeviceManager
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.reproducibility_manager import ReproducibilityManager
from blacksmith.tools.torch_helpers import collate_fn_for_causal_lm
from blacksmith.tools.workaround_utils import cross_entropy_loss, transform_labels

FINETUNE_LAYERS = range(12, 19)  # layers 12 to 18 inclusive



def print_all_gradients(model, global_step, accumulation_step):
    print(f"\n{'='*80}", flush=True)
    print(f"GRADIENTS at global_step={global_step}, accumulation_step={accumulation_step}", flush=True)
    print(f"{'='*80}", flush=True)
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                grad = param.grad.float()
                g_norm = grad.norm().item()
                total_norm += g_norm**2
                num_zeros = (grad == 0).sum().item()
                numel = grad.numel()
                print(
                    f"  {name}: shape={list(grad.shape)}, "
                    f"min={grad.min().item():.6e}, max={grad.max().item():.6e}, "
                    f"mean={grad.mean().item():.6e}, std={grad.std().item():.6e}, "
                    f"norm={g_norm:.6e}, zeros={num_zeros}/{numel}",
                    flush=True,
                )
            else:
                print(f"  {name}: grad=None", flush=True)
    total_norm = total_norm**0.5
    print(f"  TOTAL grad norm: {total_norm:.6e}", flush=True)
    print(f"{'='*80}\n", flush=True)


def print_batch(batch, step):
    print(f"\n{'='*80}", flush=True)
    print(f"BATCH at step={step}", flush=True)
    print(f"{'='*80}", flush=True)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={list(v.shape)}, dtype={v.dtype}", flush=True)
            if k in ("input_ids", "attention_mask", "labels_mask"):
                print(f"    values={v.tolist()}", flush=True)
            else:
                vf = v.float()
                print(
                    f"    norm={vf.norm().item():.6e}, min={vf.min().item():.6e}, "
                    f"max={vf.max().item():.6e}, mean={vf.mean().item():.6e}",
                    flush=True,
                )
        else:
            print(f"  {k}: {v}", flush=True)
    print(f"{'='*80}\n", flush=True)


# Training step extracted into a separate function to keep large vocab-sized
# tensors (e.g. logits) scoped locally. This ensures they do not propagate beyond
# the step via the computation graph, avoiding unnecessary and expensive
# CCLs in multi-chip setups.
# Issue itself should be investigated further.
def training_step_inner(batch, model, loss_fn, gradient_accumulation_steps):
    output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = output.logits
    shift_logits = logits[:, :-1, :].contiguous()
    loss = loss_fn(shift_logits, batch["expected_output"], batch["labels_mask"])
    # Scale loss by number of accumulation steps to get correct effective batch size.
    scaled_loss = loss / gradient_accumulation_steps
    scaled_loss.backward()
    return loss.detach()


def train(
    config: TrainingConfig,
    device_manager: DeviceManager,
    logger: TrainingLogger,
    checkpoint_manager: CheckpointManager,
):
    logger.info("Starting training (no validation, partial freeze layers 12-18)...")

    # Load model without LoRA.
    config.training_type = "partial_freeze"
    model = get_gpt_oss_model(config, device_manager.device, debug_router_grads=True)

    # Freeze all parameters, then unfreeze layers 12-18.
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        for layer_idx in FINETUNE_LAYERS:
            if f".layers.{layer_idx}." in name:
                param.requires_grad = True
                break

    logger.info(f"Loaded {config.model_name} model.")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"  Trainable: {name} {list(param.shape)}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)

    # Load checkpoint if needed.
    if config.resume_from_checkpoint:
        checkpoint_manager.load_checkpoint(model, optimizer)

    # Load dataset.
    train_dataset = get_dataset(config=config, split="train", collate_fn=collate_fn_for_causal_lm)
    train_dataloader = train_dataset.get_dataloader()
    logger.info(f"Loaded {config.dataset_id} dataset. Train dataset size: {len(train_dataloader)*config.batch_size}")

    global_step = 0
    running_loss = 0.0
    batch_idx = 0

    try:
        model.train()
        device_manager.shard_model(model)

        for epoch in range(config.num_epochs):
            accumulation_step = 0

            for batch in tqdm(train_dataloader, desc="Training"):
                batch_idx += 1

                # Zero out gradients at the start of accumulation cycle
                if accumulation_step == 0:
                    optimizer.zero_grad()

                # TODO: Refactor when https://github.com/tenstorrent/tt-blacksmith/issues/327 is resolved.
                expected_output, labels_mask = transform_labels(
                    batch["labels"], config.ignored_index, model.model.config.vocab_size
                )
                batch = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "expected_output": expected_output,
                    "labels_mask": labels_mask,
                }
                # Shard batch if data parallelism is used.
                batch = device_manager.prepare_batch(batch)

                
                if accumulation_step > 1:
                    exit(0)


                # Training step.
                loss_ = training_step_inner(batch, model, cross_entropy_loss, config.gradient_accumulation_steps)


                # Clamp gradient values.
                #for p in trainable_params:
                #    if p.grad is not None:
                #        p.grad = p.grad.clamp(-10_000.0, 10_000.0)

                # Clip gradient norms.
                #torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

                if config.use_tt:
                    torch_xla.sync(wait=True)


                running_loss += loss_.item()
                accumulation_step += 1

                print(f"Current loss and step: {loss_.item()} {global_step}", flush=True)
                print_all_gradients(model, global_step, accumulation_step)
                for li in FINETUNE_LAYERS:
                    print_debug_intermediates(model, li)

                
                # Only step the optimizer after accumulating gradients.
                if accumulation_step == config.gradient_accumulation_steps:
                    device_manager.optimizer_step(optimizer)

                    accumulation_step = 0
                    global_step += 1

                    if global_step % config.steps_freq == 0:
                        avg_loss = running_loss / (config.steps_freq * config.gradient_accumulation_steps)
                        logger.log_metrics({"train/loss": avg_loss}, commit=False, step=global_step)
                        running_loss = 0.0
                        # Clear XLA computation cache to avoid memory issues.
                        if config.use_tt:
                            xr.clear_computation_cache()

                    # Commit metrics to W&B.
                    logger.log_metrics({}, commit=True, step=global_step)

                    # Save step checkpoint.
                    if checkpoint_manager.should_save_checkpoint(global_step):
                        checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)

            # Save epoch checkpoint.
            if checkpoint_manager.should_save_checkpoint(global_step, epoch):
                checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)

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
    default_config = Path(__file__).parent / "test_gpt_oss_20b_finetuning.yaml"
    args = parse_cli_options(default_config=default_config)
    config: TrainingConfig = generate_config(TrainingConfig, args.config, args.test_config)

    # Reproducibility setup
    repro_manager = ReproducibilityManager(config)
    repro_manager.setup()

    # Logger setup.
    logger = TrainingLogger(config, args.test_log_filename_prefix)

    # Device setup
    device_manager = DeviceManager(config)
    logger.info(f"Using device: {device_manager.device}")

    # Use highest numerical precision for stable fine-tuning convergence.
    # fp32_dest_acc_en: accumulate partial results in FP32 to avoid precision loss.
    # math_fidelity hifi4: use all 4 mantissa phases for full precision multiplications.
    if config.use_tt:
        torch_xla.set_custom_compile_options({"fp32_dest_acc_en": True, "math_fidelity": "hifi4"})

    # Checkpoint manager setup
    checkpoint_manager = CheckpointManager(config, logger, device_manager.device)

    # Start training.
    train(config, device_manager, logger, checkpoint_manager)
