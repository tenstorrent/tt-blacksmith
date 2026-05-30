# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import csv
import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from dataset import ShakespeareDataset
from flax.core import freeze, unfreeze
from model_jax import GPT, GPTConfig

# Defining devices.
try:
    device = jax.devices("tt")[0]
    print(f"Using Tenstorrent Device: {device}")
except Exception as e:
    print("Tenstorrent device not found, using CPU.")
    print(f"Error: {e}")
    device = jax.devices("cpu")[0]
cpu_device = jax.devices("cpu")[0]

dataset = ShakespeareDataset()
train_data = dataset.get_data("train")
val_data = dataset.get_data("val")


def get_batch(split, train, val, block_size, batch_size, device):

    data = train if split == "train" else val
    ix = np.random.randint(0, len(data) - block_size, (batch_size,))

    x_stack = np.stack([data[i : i + block_size] for i in ix])
    y_stack = np.stack([data[i + 1 : i + 1 + block_size] for i in ix])

    x_dev = jax.device_put(jnp.array(x_stack, dtype=jnp.uint32), device)
    y_dev = jax.device_put(jnp.array(y_stack, dtype=jnp.uint32), device)
    return x_dev, y_dev


# Setting up configuration and hyper parameters.
config = GPTConfig(
    block_size=256,
    vocab_size=65,
    num_layers=6,
    num_heads=6,
    num_embeds=384,
    dropout_rate=0.2,
    use_matmul_embed=True,
    dtype=jnp.float32,
)
batch_size = 64
max_iters = 200
learning_rate = 3e-4


# --- Init ---
print(f"Initializing model (V={config.vocab_size})...")
model = GPT(config)
key = jax.random.PRNGKey(1337)
key, init_key = jax.random.split(key)

# Initialization on CPU is suffitient.
with jax.default_device(cpu_device):
    variables = model.init(init_key)

variables = unfreeze(variables)
cache = freeze({"cache": variables.pop("cache")})
params = freeze(variables)

optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=learning_rate, weight_decay=1e-1))
opt_state = optimizer.init(params)

print("Moving weights to tt hardware...")
params = jax.device_put(params, device)
cache = jax.device_put(cache, device)


@partial(jax.jit, backend="tt")
def compute_grads_tt(params, cache, x, y):
    """
    We are shifting logits by their maximum value to prevent exp() overflow.
    After that we can compute log probabilities safely which results in standard cross entropy.
    Doing `jax.nn.one_hot` with manual softmax because scatter/gather are still not supported on tt-hardware.
    """

    def loss_fn(p):
        vars = {"params": p["params"], **cache}
        logits = model.apply(vars, x, deterministic=True)

        logits_max = jnp.max(logits, axis=-1, keepdims=True)
        shifted_logits = logits - jax.lax.stop_gradient(logits_max)

        log_normalizers = jnp.log(jnp.sum(jnp.exp(shifted_logits), axis=-1, keepdims=True))
        log_probs = shifted_logits - log_normalizers

        vocab_size = logits.shape[-1]
        one_hot = jax.nn.one_hot(y, vocab_size)
        loss = -jnp.sum(one_hot * log_probs, axis=-1)
        return jnp.mean(loss)

    loss_val, grads = jax.value_and_grad(loss_fn)(params)
    return loss_val, grads


@partial(jax.jit, backend="tt")
def eval_step(params, cache, x, y):
    vars = {"params": params["params"], **cache}
    logits = model.apply(vars, x, deterministic=True)
    logits_max = jnp.max(logits, axis=-1, keepdims=True)
    shifted_logits = logits - jax.lax.stop_gradient(logits_max)

    log_normalizers = jnp.log(jnp.sum(jnp.exp(shifted_logits), axis=-1, keepdims=True))
    log_probs = shifted_logits - log_normalizers

    vocab_size = logits.shape[-1]
    one_hot = jax.nn.one_hot(y, vocab_size)
    loss = -jnp.sum(one_hot * log_probs, axis=-1)
    return jnp.mean(loss)


out_dir = "output"
os.makedirs(out_dir, exist_ok=True)
log_file_path = os.path.join(out_dir, "test-logs.csv")
plot_file_path = os.path.join(out_dir, "test-plot.png")

# Logging containers.
iter_nums = []
train_losses = []
val_losses = []
val_iters = []  # Store iterations where validation happened.

print(f"Training for {max_iters} iterations...")
start_time = time.time()
eval_interval = 20
eval_iters = 5

# Open CSV for writing.
with open(log_file_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "train_loss", "val_loss", "time_sec"])

    for iter in range(max_iters):

        # Fetch Batch.
        xb, yb = get_batch("train", train_data, val_data, config.block_size, batch_size, device)

        # Compute Gradients.
        loss, grads = compute_grads_tt(params, cache, xb, yb)

        # Perform optimizer step on CPU because of tt-metal #27072 (pow/exp accuracy).
        # Move grads/params to CPU, compute Adam update, then move updated params back to TT.
        # See: https://github.com/tenstorrent/tt-metal/issues/27072
        with jax.default_device(cpu_device):
            grads_cpu = jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu_device), grads)
            params_cpu = jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu_device), params)

            # Debugging check if instability in tt comipler (e. g. pow and mul) caused
            # overflow or underflow resulting NaNs.
            first_grad_leaf = jax.tree_util.tree_leaves(grads_cpu)[0]
            if np.isnan(np.array(first_grad_leaf)).any() or np.isinf(np.array(first_grad_leaf)).any():
                raise ValueError(
                    f"CRITICAL FAILURE: Gradients corrupted (NaN/Inf) at Iteration {iter}. Hardware saved from hanging."
                )

            updates, opt_state = optimizer.update(grads_cpu, opt_state, params_cpu)
            new_params_cpu = optax.apply_updates(params_cpu, updates)

        # Strict FP32 Push to TT Device.
        params = jax.tree_util.tree_map(lambda x: jax.device_put(x.astype(jnp.float32), device), new_params_cpu)

        iter_nums.append(iter)
        train_loss_sync = float(loss)
        train_losses.append(train_loss_sync)

        if iter % eval_interval == 0 or iter == max_iters - 1:
            v_losses = []
            for _ in range(eval_iters):
                xb_val, yb_val = get_batch("val", train_data, val_data, config.block_size, batch_size, device)
                val_loss_array = eval_step(params, cache, xb_val, yb_val)
                v_losses.append(float(val_loss_array))

            val_loss_sync = sum(v_losses) / len(v_losses)
            val_iters.append(iter)
            val_losses.append(val_loss_sync)

            dt = time.time() - start_time
            print(f"Step {iter:4d}: Train Loss {train_loss_sync:.4f} | Val Loss {val_loss_sync:.4f} | Time {dt:.2f}s")

            writer.writerow([iter, train_loss_sync, val_loss_sync, dt])
            f.flush()

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
print("Saving loss plot...")
plt.figure(figsize=(10, 6))
plt.plot(iter_nums, train_losses, label="Train Loss", alpha=0.5)
plt.plot(val_iters, val_losses, label="Validation Loss", color="red", linewidth=2)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig(plot_file_path)
print(f"Plot saved to {plot_file_path}")
print(f"Log saved to {log_file_path}")
