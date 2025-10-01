# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import math
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
import wandb

from blacksmith.experiments.jax.distil_bert.configs import ExperimentConfig
from blacksmith.tools.cli import generate_config

from blacksmith.models.jax.distil_bert.model import init_teacher, init_student
from blacksmith.models.jax.distil_bert.model_utils import split_params, combine_params
from blacksmith.datasets.jax.distil_bert.sst2_dataset import get_tokenizer, load_sst2, numpy_batch_iter

# Optimizer schedule with linear warmup and linear decay.
def build_schedule(config: ExperimentConfig, num_train_steps: int):
    warmup_steps = int(config.warmup_ratio * num_train_steps)
    schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, config.learning_rate, warmup_steps),
            optax.linear_schedule(config.learning_rate, 0.0, num_train_steps - warmup_steps),
        ],
        boundaries=[warmup_steps],
    )
    return schedule


def softmax_with_temperature(logits, T):
    return nn.softmax(logits / T, axis=-1)


def kl_divergence(p_logits, q_logits, T):
    p = softmax_with_temperature(p_logits, T)
    log_p = jax.nn.log_softmax(p_logits / T, axis=-1)
    log_q = jax.nn.log_softmax(q_logits / T, axis=-1)
    kl = jnp.sum(p * (log_p - log_q), axis=-1)
    return (T**2) * jnp.mean(kl)


def ce_with_labels(logits, labels):
    num_classes = logits.shape[-1]
    one_hot_labels = jax.nn.one_hot(labels, num_classes)
    return optax.softmax_cross_entropy(logits, one_hot_labels).mean()


def cosine_embedding_loss(x, y, eps=1e-8):
    x_norm = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)
    y_norm = y / (jnp.linalg.norm(y, axis=-1, keepdims=True) + eps)
    cos_sim = jnp.sum(x_norm * y_norm, axis=-1)
    return 1.0 - jnp.mean(cos_sim)


def make_teacher_forward(teacher):
    @jax.jit
    def forward_teacher(params, batch):
        outputs = teacher(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            params=params,
            train=False,
            output_hidden_states=True,
        )
        return outputs.logits, outputs.hidden_states[-1]

    return forward_teacher


def make_student_train_step(student, config: ExperimentConfig):
    @jax.jit
    def train_step(trainable_params, frozen_params, batch, t_logits, t_hidden, rng):
        rng, dropout_rng = jax.random.split(rng)
        # Calculate gradients only with respect to trainable parameters, to keep
        # frozen ones (embeddings) unchanged.
        (loss, (loss_ce, loss_kl, loss_cos, _)), grads = jax.value_and_grad(
            lambda p: loss_fn(student, p, frozen_params, t_logits, t_hidden, batch, dropout_rng, config),
            argnums=0,
            has_aux=True,
        )(trainable_params)
        metrics = {
            "loss_total": loss,
            "loss_ce": loss_ce,
            "loss_kl": loss_kl,
            "loss_cos": loss_cos,
        }
        return grads, metrics

    return train_step


def make_student_eval_step(student):
    @jax.jit
    def eval_step(trainable_params, frozen_params, batch):
        # Combine trainable and frozen parameters to get full model params for inference.
        params = combine_params(trainable_params, frozen_params)
        s_outputs = student(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            params=params,
            train=False,
        )
        logits = s_outputs.logits
        preds = jnp.argmax(logits, axis=-1)
        acc = jnp.mean((preds == batch["labels"]).astype(jnp.bfloat16))
        return acc

    return eval_step


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
    n = len(dataset)
    total, count = 0.0, 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = {k: dataset[k][start:end] for k in columns}
        batch["input_ids"] = batch["input_ids"].astype(np.int32)
        batch["attention_mask"] = batch["attention_mask"].astype(np.int32)
        batch["labels"] = batch["labels"].astype(np.int32)
        acc = eval_step_fn(trainable_params, frozen_params, batch)
        total += float(acc) * (end - start)
        count += end - start
    return total / count


def train(config: ExperimentConfig):
    # Load dataset and create batch iterator.
    tokenizer = get_tokenizer(config.tokenizer_name)
    train_data, val_data, columns = load_sst2(tokenizer, max_length=config.max_length)
    train_iter = numpy_batch_iter(train_data, config.batch_size, columns, shuffle=True, seed=config.seed)

    # Initialize models and split student params into trainable and frozen,
    # where frozen params are the embedding layers. This is done to keep the
    # embeddings fixed during training as they are already well-trained on large corpora.
    teacher, teacher_params = init_teacher()
    student, student_params = init_student()
    trainable_params, frozen_params = split_params(student_params)

    # Create JIT-compiled teacher forward function.
    forward_teacher = make_teacher_forward(teacher)

    # Create JIT-compiled student functions.
    train_step = make_student_train_step(student, config)
    eval_step = make_student_eval_step(student)

    os.environ["WANDB_MODE"] = "online" if config.use_wandb else "disabled"

    # Initialize wandb
    wandb.init(
        project=config.project_name,
        name=config.experiment_name,
        job_type=config.job_name,
    )

    num_train_steps = math.ceil(len(train_data) / config.batch_size) * config.num_epochs
    steps_per_epoch = math.ceil(len(train_data) / config.batch_size)
    global_step = 0
    rng = jax.random.PRNGKey(config.seed)

    # Optimizer is initialized on CPU as it's execution will be on CPU
    # (https://github.com/tenstorrent/tt-metal/issues/27072).
    trainable_params_cpu = jax.tree_util.tree_map(lambda x: jax.device_put(x, jax.devices("cpu")[0]), trainable_params)
    with jax.default_device(jax.devices("cpu")[0]):
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=build_schedule(config, num_train_steps), weight_decay=config.weight_decay),
        )
        opt_state = optimizer.init(trainable_params_cpu)

    # Loss buffer to accumulate losses for logging.
    loss_buffer = {"loss_total": [], "loss_ce": [], "loss_kl": [], "loss_cos": []}

    for epoch in range(1, config.num_epochs + 1):
        for step in range(steps_per_epoch):
            batch = next(train_iter)

            t_logits, t_hidden = forward_teacher(teacher_params, batch)
            grads, metrics = train_step(trainable_params, frozen_params, batch, t_logits, t_hidden, rng)

            # Move grads and params to CPU for optimizer step.
            grads_cpu = jax.tree_util.tree_map(lambda x: jax.device_put(x, jax.devices("cpu")[0]), grads)
            trainable_params_cpu = jax.tree_util.tree_map(
                lambda x: jax.device_put(x, jax.devices("cpu")[0]), trainable_params
            )

            # Optimizer step is done on CPU (https://github.com/tenstorrent/tt-metal/issues/27072).
            with jax.default_device(jax.devices("cpu")[0]):
                updates, new_opt_state = optimizer.update(grads_cpu, opt_state, trainable_params_cpu)
                new_trainable_params_cpu = optax.apply_updates(trainable_params_cpu, updates)
                opt_state = new_opt_state

            trainable_params = jax.tree_util.tree_map(
                lambda x: jax.device_put(x, jax.devices("tt")[0]), new_trainable_params_cpu
            )

            for k in loss_buffer:
                loss_buffer[k].append(float(metrics[k]))

            # Log training metrics at configured frequency.
            if global_step % config.log_every == 0:
                avg_metrics = {k: np.mean(loss_buffer[k]) for k in loss_buffer}
                print(
                    f"[epoch {epoch} step {global_step}] "
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
                    val_data, eval_step, trainable_params, frozen_params, columns, batch_size=config.batch_size
                )
                print(f"→ step {global_step}: validation accuracy={val_acc*100:.2f}%")

                # Log validation to wandb
                wandb.log(
                    {
                        "val/accuracy": val_acc,
                        "step": global_step,
                    }
                )

            global_step += 1

    if config.use_wandb:
        wandb.finish()

    # Save model.
    output_dir = os.path.join(config.output_dir, "distilled_student_sst2")
    os.makedirs(output_dir, exist_ok=True)
    final_params = combine_params(trainable_params, frozen_params)
    student.save_pretrained(output_dir, params=final_params)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved student to: {output_dir}")


if __name__ == "__main__":
    config_file_path = os.path.join(os.path.dirname(__file__), "test_distil_bert_flax.yaml")
    config = generate_config(ExperimentConfig, config_file_path)
    train(config)
