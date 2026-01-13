# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import math
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax import linen as nn
from jax.experimental import shard_map
from jax.sharding import NamedSharding, PartitionSpec
from transformers import AutoTokenizer

from blacksmith.datasets.jax.distil_bert.sst2_dataset import *
from blacksmith.experiments.jax.distil_bert.checkpoint_utils import *
from blacksmith.experiments.jax.distil_bert.configs import ExperimentConfig
from blacksmith.experiments.jax.distil_bert.multi_chip.data_parallel.sharding_config import (
    ShardingConfig,
)
from blacksmith.models.jax.distil_bert.model import init_model
from blacksmith.models.jax.distil_bert.model_utils import combine_params, split_params
from blacksmith.tools.cli import generate_config, parse_cli_options
from blacksmith.tools.jax_helpers import (
    build_schedule,
    ce_with_labels,
    cosine_embedding_loss,
    kl_divergence,
)


def create_sharded_teacher_forward(teacher, sharding_config: ShardingConfig):
    """
    Creates data-parallel teacher forward pass.
    Params are replicated and input batch is sharded.
    """

    def compute_teacher_outputs(params, input_ids, attention_mask):
        outputs = teacher(
            input_ids=input_ids,
            attention_mask=attention_mask,
            params=params,
            train=False,
            output_hidden_states=True,
        )
        return outputs.logits, outputs.hidden_states[-1]

    def teacher_forward(params, input_ids, attention_mask):
        return shard_map.shard_map(
            compute_teacher_outputs,
            mesh=sharding_config.mesh,
            in_specs=(
                sharding_config.param_partition,
                sharding_config.data_partition,
                sharding_config.data_partition,
            ),
            out_specs=(
                sharding_config.data_partition,
                sharding_config.data_partition,
            ),
            check_rep=False,
        )(params, input_ids, attention_mask)

    teacher_forward_jit = jax.jit(
        teacher_forward,
        out_shardings=(
            sharding_config.data_sharding,
            sharding_config.data_sharding,
        ),
    )

    return teacher_forward_jit


def create_sharded_training_step(student, loss_fn, config: ExperimentConfig, sharding_config: ShardingConfig):
    """
    Creates data-parallel training step.
    Params are replicated and input batch is sharded.
    Gradients are averaged across devices.
    """

    def compute_loss_and_grads(
        trainable_params, frozen_params, input_ids, attention_mask, labels, t_logits, t_hidden, rng
    ):

        # Per-shard forward + backward.
        def local_loss(p):
            return loss_fn(
                student,
                p,
                frozen_params,
                t_logits,
                t_hidden,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                },
                rng,
                config,
            )

        (loss, (loss_ce, loss_kl, loss_cos, _)), grads = jax.value_and_grad(local_loss, argnums=0, has_aux=True)(
            trainable_params
        )

        # All-gather and mean gradients.
        def gather_grads(x):
            g = jax.lax.all_gather(x, axis_name="data")
            return jnp.mean(g, axis=0)

        grads = jax.tree_util.tree_map(gather_grads, grads)

        metrics_local = {
            "loss_total": loss,
            "loss_ce": loss_ce,
            "loss_kl": loss_kl,
            "loss_cos": loss_cos,
        }

        # Average all losses across devices.
        metrics_all = jax.lax.all_gather(metrics_local, axis_name="data")

        metrics_mean = {k: jnp.mean(metrics_all[k]) for k in metrics_all}

        return metrics_mean, grads

    def train_step(trainable_params, frozen_params, input_ids, attention_mask, labels, t_logits, t_hidden, rng):

        return shard_map.shard_map(
            compute_loss_and_grads,
            mesh=sharding_config.mesh,
            in_specs=(
                sharding_config.param_partition,
                sharding_config.param_partition,
                sharding_config.data_partition,
                sharding_config.data_partition,
                sharding_config.data_partition,
                sharding_config.data_partition,
                sharding_config.data_partition,
                sharding_config.param_partition,
            ),
            out_specs=(
                sharding_config.param_partition,
                sharding_config.param_partition,
            ),
            check_rep=False,
        )(trainable_params, frozen_params, input_ids, attention_mask, labels, t_logits, t_hidden, rng)

    train_step_jit = jax.jit(
        train_step,
        out_shardings=(
            sharding_config.param_sharding,  # Loss scalar replicated.
            sharding_config.param_sharding,  # Grads replicated.
        ),
    )

    return train_step_jit


def create_sharded_eval_step(student, sharding_config: ShardingConfig):
    """
    Returns eval_step_fn(params, frozen_params, batch)
    that returns sharded logits across devices.
    """

    # Per-device logits.
    def compute_logits(trainable_params, frozen_params, input_ids, attention_mask, labels):

        params = combine_params(trainable_params, frozen_params)

        out = student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            params=params,
            train=False,
        )

        return out.logits

    def eval_step_sharded(trainable_params, frozen_params, input_ids, attention_mask, labels):

        return shard_map.shard_map(
            compute_logits,
            mesh=sharding_config.mesh,
            in_specs=(
                sharding_config.param_partition,
                sharding_config.param_partition,
                sharding_config.data_partition,
                sharding_config.data_partition,
                sharding_config.data_partition,
            ),
            out_specs=sharding_config.data_partition,
            check_rep=False,
        )(trainable_params, frozen_params, input_ids, attention_mask, labels)

    eval_step_sharded_jit = jax.jit(
        eval_step_sharded,
        out_shardings=sharding_config.data_sharding,
    )
    return eval_step_sharded_jit


def loss_fn(student, trainable_params, frozen_params, t_logits, t_hidden, batch, rng, config: ExperimentConfig):
    # Combine trainable and frozen parameters to get full model params for inference.
    params = combine_params(trainable_params, frozen_params)
    s_outputs = student(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        params=params,
        dropout_rng=rng,
        train=True,
        output_hidden_states=True,
    )
    s_logits, s_hidden = s_outputs.logits, s_outputs.hidden_states[-1]

    loss_ce = ce_with_labels(s_logits, batch["labels"])
    loss_kl = kl_divergence(t_logits, s_logits, config.temperature)
    loss_cos = cosine_embedding_loss(s_hidden, t_hidden)

    total = config.alpha_ce * loss_ce + config.alpha_kl * loss_kl + config.alpha_cos * loss_cos
    return total, (loss_ce, loss_kl, loss_cos, s_logits)


def evaluate(dataset, eval_step_fn, trainable_params, frozen_params, columns, batch_size=32):
    # Use numpy_batch_iter without shuffling for deterministic evaluation.
    eval_iter = numpy_batch_iter(dataset, batch_size, columns, shuffle=False)

    steps_per_eval = math.ceil(len(dataset) / batch_size)

    total_correct = 0
    total_samples = 0

    for _ in range(steps_per_eval):
        batch = next(eval_iter)

        # Shard data across input dimension.
        input_ids_ = jax.device_put(batch["input_ids"], sharding_config.data_sharding)
        attention_mask_ = jax.device_put(batch["attention_mask"], sharding_config.data_sharding)
        labels_ = jax.device_put(batch["labels"], sharding_config.data_sharding)

        # Get sharded logits from eval_step_fn.
        # We don't directly return accuracy from eval_step_fn to avoid
        # frequently occuring issue (https://github.com/tenstorrent/tt-xla/issues/1993).
        logits_sharded = eval_step_fn(trainable_params, frozen_params, input_ids_, attention_mask_, labels_)

        # Transfer sharded logits to CPU.
        # The logits are sharded across devices, so when we transfer them to CPU,
        # they will be gathered into the full batch shape.
        logits_cpu = jax.device_put(logits_sharded, jax.devices("cpu")[0])

        # Calculate accuracy on CPU.
        with jax.default_device(jax.devices("cpu")[0]):
            preds = jnp.argmax(logits_cpu, axis=-1)
            # Compare with original labels.
            correct = jnp.sum((preds == batch["labels"]).astype(jnp.int32))

        total_correct += int(correct)
        total_samples += len(batch["labels"])

    return total_correct / total_samples


def train(config: ExperimentConfig, sharding_config: ShardingConfig):
    # Load dataset and create batch iterator.
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    train_data, val_data, columns = load_sst2(tokenizer, max_length=config.max_length)
    train_iter = numpy_batch_iter(train_data, config.batch_size, columns, shuffle=True, seed=config.seed)

    # Initialize models and split student params into trainable and frozen,
    # where frozen params are the embedding layers. This is done to keep the
    # embeddings fixed during training as they are already well-trained on large corpora.
    teacher, teacher_params = init_model(config.teacher_model, num_labels=2)
    student, student_params = init_model(config.student_model, num_labels=2, seed=config.seed)
    trainable_params, frozen_params = split_params(student_params)

    # For data parallelism, parameters are replicated across devices.
    teacher_params = jax.device_put(teacher_params, sharding_config.param_sharding)
    trainable_params = jax.device_put(trainable_params, sharding_config.param_sharding)
    frozen_params = jax.device_put(frozen_params, sharding_config.param_sharding)

    # Create sharded JIT-compiled teacher forward function.
    forward_teacher_jit = create_sharded_teacher_forward(teacher, sharding_config)
    # Create sharded JIT-compiled training step function for student.
    train_step_jit = create_sharded_training_step(student, loss_fn, config, sharding_config)
    # Create sharded JIT-compiled evaluation step function for student.
    eval_step_jit = create_sharded_eval_step(student, sharding_config)

    os.environ["WANDB_MODE"] = "online" if config.use_wandb else "disabled"
    # Initialize wandb.
    wandb.init(
        project=config.project_name,
        name=config.experiment_name,
        job_type=config.job_name,
    )

    steps_per_epoch = math.ceil(len(train_data) / config.batch_size)
    num_train_steps = steps_per_epoch * config.num_epochs
    global_step = 0
    rng = jax.random.PRNGKey(config.seed)

    # Setup checkpointing.
    checkpoint_dir = Path(config.output_dir) / "checkpoints"
    start_step = 0

    # Load from checkpoint if resuming.
    if config.resume_from_checkpoint:
        latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            checkpoint = load_checkpoint(latest_checkpoint)
            trainable_params = checkpoint["trainable_params"]
            # Replicate checkpointed trainable params.
            trainable_params = jax.device_put(trainable_params, sharding_config.param_sharding)
            opt_state = checkpoint["opt_state"]
            rng = checkpoint["rng"]
            start_step = checkpoint["step"]
            global_step = start_step
            print(f"Resuming training from step {start_step}")
        else:
            print("No checkpoint found, starting from scratch")
    else:
        # Delete all existing checkpoints when not resuming.
        if checkpoint_dir.exists():
            cleanup_old_checkpoints(checkpoint_dir, keep_top_k=0)
            print("Cleaned up all existing checkpoints for fresh start")

    # Optimizer is initialized on CPU as it's execution will be on CPU
    # (https://github.com/tenstorrent/tt-metal/issues/27072).
    trainable_params_cpu = jax.tree_util.tree_map(lambda x: jax.device_put(x, jax.devices("cpu")[0]), trainable_params)
    with jax.default_device(jax.devices("cpu")[0]):
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                learning_rate=build_schedule(config.learning_rate, config.warmup_ratio, num_train_steps),
                weight_decay=config.weight_decay,
            ),
        )
        opt_state = optimizer.init(trainable_params_cpu)

    # Loss buffer to accumulate losses for logging.
    loss_buffer = {"loss_total": [], "loss_ce": [], "loss_kl": [], "loss_cos": []}

    for epoch in range(1, config.num_epochs + 1):
        for step in range(steps_per_epoch):
            batch = next(train_iter)

            # Shard data across input dimension.
            input_ids_ = jax.device_put(batch["input_ids"], sharding_config.data_sharding)
            attention_mask_ = jax.device_put(batch["attention_mask"], sharding_config.data_sharding)
            labels_ = jax.device_put(batch["labels"], sharding_config.data_sharding)

            # Replicate RNG.
            rng = jax.device_put(rng, sharding_config.param_sharding)

            t_logits, t_hidden = forward_teacher_jit(teacher_params, input_ids_, attention_mask_)
            metrics, grads = train_step_jit(
                trainable_params, frozen_params, input_ids_, attention_mask_, labels_, t_logits, t_hidden, rng
            )

            # Move grads and params to CPU for optimizer step.
            grads_cpu = jax.tree_util.tree_map(lambda x: jax.device_put(x, jax.devices("cpu")[0]), grads)
            trainable_params_cpu = jax.tree_util.tree_map(
                lambda x: jax.device_put(x, jax.devices("cpu")[0]), trainable_params
            )

            with jax.default_device(jax.devices("cpu")[0]):
                updates, new_opt_state = optimizer.update(grads_cpu, opt_state, trainable_params_cpu)
                new_trainable_params_cpu = optax.apply_updates(trainable_params_cpu, updates)
                opt_state = new_opt_state

            trainable_params = jax.device_put(new_trainable_params_cpu, sharding_config.param_sharding)

            # 'loss_type' can be 'loss_total', 'loss_ce', 'loss_kl', 'loss_cos'.
            for loss_type in loss_buffer:
                loss_buffer[loss_type].append(float(metrics[loss_type]))

            # Log training metrics at configured frequency.
            if global_step % config.log_every == 0:
                avg_metrics = {k: np.mean(loss_buffer[k]) for k in loss_buffer}
                print(
                    f"[step {global_step}] "
                    f"loss_total={avg_metrics['loss_total']:.4f} "
                    f"ce={avg_metrics['loss_ce']:.4f} "
                    f"kl={avg_metrics['loss_kl']:.4f} "
                    f"cos={avg_metrics['loss_cos']:.4f} "
                )

                wandb.log(
                    {
                        "train/loss_total": avg_metrics["loss_total"],
                        "train/loss_ce": avg_metrics["loss_ce"],
                        "train/loss_kl": avg_metrics["loss_kl"],
                        "train/loss_cos": avg_metrics["loss_cos"],
                        "train/epoch": epoch,
                        "step": global_step,
                    }
                )

                loss_buffer = {k: [] for k in loss_buffer}

            # Log validation metrics at configured frequency.
            if global_step % config.log_val_every == 0:
                val_acc = evaluate(
                    val_data, eval_step_jit, trainable_params, frozen_params, columns, batch_size=config.batch_size
                )
                print(f"→ step {global_step}: validation accuracy={val_acc*100:.2f}%")

                # Log validation to wandb.
                wandb.log(
                    {
                        "val/accuracy": val_acc,
                        "step": global_step,
                    }
                )

            # Save checkpoint at configured frequency.
            if config.do_checkpoint and global_step % config.checkpoint_every == 0 and global_step > 0:
                save_checkpoint(checkpoint_dir, global_step, trainable_params, opt_state, rng)
                cleanup_old_checkpoints(checkpoint_dir, config.keep_top_k_checkpoints)

            global_step += 1

    if config.use_wandb:
        wandb.finish()

    # Save model.
    output_dir = Path(config.output_dir) / "distilled_student_sst2"
    output_dir.mkdir(parents=True, exist_ok=True)
    final_params = combine_params(trainable_params, frozen_params)
    student.save_pretrained(output_dir, params=final_params)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved student to: {output_dir}")


if __name__ == "__main__":
    jax.config.update("jax_use_shardy_partitioner", True)

    default_config = Path(__file__).parents[2] / "test_distil_bert_flax.yaml"
    args = parse_cli_options(default_config=default_config)
    config: ExperimentConfig = generate_config(ExperimentConfig, args.config)

    sharding_config = ShardingConfig()
    train(config, sharding_config)
