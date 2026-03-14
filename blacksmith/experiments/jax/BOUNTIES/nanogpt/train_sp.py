import os
import time
import numpy as np
import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze
import optax
from functools import partial
from model_jax import GPT, GPTConfig
import csv
import matplotlib.pyplot as plt
from dataset import ShakespeareDataset

# Defining devices.
try:
    tt_device = jax.devices('tt')[0]
    print(f"Using Tenstorrent Device: {tt_device}")
except:
    print("Tenstorrent device not found, using CPU.")
    tt_device = jax.devices('cpu')[0]
cpu_device = jax.devices('cpu')[0]

dataset = ShakespeareDataset(tt_device)
train_data = dataset.get_data('train')
val_data = dataset.get_data('val')

def get_batch(split):
    # Slice using pure NumPy on the CPU. It is instantaneous.
    data = train_data if split == 'train' else val_data
    ix = np.random.randint(0, len(data) - config.block_size, (batch_size,))
    
    x_stack = np.stack([data[i:i+config.block_size] for i in ix])
    y_stack = np.stack([data[i+1:i+1+config.block_size] for i in ix])
    
    x_dev = jax.device_put(jnp.array(x_stack, dtype=jnp.uint32), tt_device)
    y_dev = jax.device_put(jnp.array(y_stack, dtype=jnp.uint32), tt_device)
    return x_dev, y_dev
    # return jnp.array(x_stack, dtype=jnp.uint32), jnp.array(y_stack, dtype=jnp.uint32)


# Setting up configuration and hyper parameters.
config = GPTConfig(
    block_size=256,
    vocab_size=65,
    num_layers=6,
    num_heads=6,
    num_embeds=384,
    dropout_rate=0.2,
    use_matmul_embed=True,
    dtype=jnp.float32
)
batch_size = 64
max_iters = 200
learning_rate = 3e-4


# --- Init ---
print(f"Initializing model (V={config.vocab_size})...")
model = GPT(config)
key = jax.random.PRNGKey(1337)
key, init_key = jax.random.split(key)

# Init on CPU to be safe.
with jax.default_device(cpu_device):
    variables = model.init(init_key)

variables = unfreeze(variables)
cache = freeze({'cache': variables.pop('cache')})
params = freeze(variables)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=learning_rate, weight_decay=1e-1)
)
opt_state = optimizer.init(params)

print(f"Moving weights to tt hardware...")
params = jax.device_put(params, tt_device)
cache = jax.device_put(cache, tt_device)
# opt_state = jax.device_put(opt_state, tt_device)

@partial(jax.jit, backend='tt')
def compute_grads_tt(params, cache, x, y):
    def loss_fn(p):
        vars = {'params': p['params'], **cache}
        logits = model.apply(vars, x, deterministic=True)
        
        # Shifting logits by their maximum value to prevent exp() overflow.
        logits_max = jnp.max(logits, axis=-1, keepdims=True)
        shifted_logits = logits - jax.lax.stop_gradient(logits_max)
        
        # Computing log probabilities safely.
        log_normalizers = jnp.log(jnp.sum(jnp.exp(shifted_logits), axis=-1, keepdims=True))
        log_probs = shifted_logits - log_normalizers
        
        # This is now standard cross entropy.
        vocab_size = logits.shape[-1]
        one_hot = jax.nn.one_hot(y, vocab_size)
        loss = -jnp.sum(one_hot * log_probs, axis=-1)
        return jnp.mean(loss)

    loss_val, grads = jax.value_and_grad(loss_fn)(params)
    return loss_val, grads

@partial(jax.jit, backend='tt')
def eval_step(params, cache, x, y):
    vars = {'params': params['params'], **cache}
    logits = model.apply(vars, x, deterministic=True)
    logits_max = jnp.max(logits, axis=-1, keepdims=True)
    shifted_logits = logits - jax.lax.stop_gradient(logits_max)
    
    log_normalizers = jnp.log(jnp.sum(jnp.exp(shifted_logits), axis=-1, keepdims=True))
    log_probs = shifted_logits - log_normalizers
    
    vocab_size = logits.shape[-1]
    one_hot = jax.nn.one_hot(y, vocab_size)
    loss = -jnp.sum(one_hot * log_probs, axis=-1)
    return jnp.mean(loss)


out_dir = 'output'
os.makedirs(out_dir, exist_ok=True)
log_file_path = os.path.join(out_dir, 'log.csv')
plot_file_path = os.path.join(out_dir, 'loss_plot.png')

# Logging containers.
iter_nums = []
train_losses = []
val_losses = []
val_iters = [] # Store iterations where validation happened.

print(f"Training for {max_iters} iterations...")
start_time = time.time()
eval_interval = 20
eval_iters = 5

# Open CSV for writing.
with open(log_file_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['step', 'train_loss', 'val_loss', 'time_sec'])

    for iter in range(max_iters):
        print(f"\n--- DEBUG: Starting Iteration {iter} ---")
        
        # 1. Fetch Batch
        xb, yb = get_batch('train') 
        print("DEBUG: 1. Batch pushed.")
    
        # 2. Compute Gradients (Iter 0 will pause here for ~18s to compile)
        loss, grads = compute_grads_tt(params, cache, xb, yb)
        # _ = loss.block_until_ready()
        print("DEBUG: 2. compute_grads_tt completed.")
    
        # 3. CPU Optimizer Step
        with jax.default_device(cpu_device):
            grads_cpu = jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu_device), grads)
            params_cpu = jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu_device), params)
            
            first_grad_leaf = jax.tree_util.tree_leaves(grads_cpu)[0]
            if np.isnan(np.array(first_grad_leaf)).any() or np.isinf(np.array(first_grad_leaf)).any():
                raise ValueError(f"CRITICAL FAILURE: Gradients corrupted (NaN/Inf) at Iteration {iter}. Hardware saved from hanging.")
            
            updates, opt_state = optimizer.update(grads_cpu, opt_state, params_cpu)
            new_params_cpu = optax.apply_updates(params_cpu, updates)
            print("DEBUG: 3. CPU math completed.")
    
        # 4. Strict FP32 Push to TT Device
        # .astype(jnp.float32) is critical here to prevent silent FP64 recompilation hangs
        params = jax.tree_util.tree_map(
            lambda x: jax.device_put(x.astype(jnp.float32), tt_device), 
            new_params_cpu
        )
        # _ = jax.tree_util.tree_leaves(params)[0].block_until_ready()
        print("DEBUG: 4. Strict FP32 params pushed back to TT.")
        
        iter_nums.append(iter)
        train_loss_sync = float(loss)
        train_losses.append(train_loss_sync)
        
        if iter % eval_interval == 0 or iter == max_iters - 1:
            print("DEBUG: 5. Entering Eval Block (Iter 0 will compile here)")
            v_losses = []
            for i in range(eval_iters):
                xb_val, yb_val = get_batch('val')
                val_loss_array = eval_step(params, cache, xb_val, yb_val)
                # _ = val_loss_array.block_until_ready()
                v_losses.append(float(val_loss_array))
            
            val_loss_sync = sum(v_losses) / len(v_losses)
            val_iters.append(iter)
            val_losses.append(val_loss_sync)
            print("DEBUG: 6. Eval completed.")
        
            dt = time.time() - start_time
            print(f"Step {iter:4d}: Train Loss {train_loss_sync:.4f} | Val Loss {val_loss_sync:.4f} | Time {dt:.2f}s")
            
            writer.writerow([iter, train_loss_sync, val_loss_sync, dt])
            f.flush()

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
print("Saving loss plot...")
plt.figure(figsize=(10, 6))
plt.plot(iter_nums, train_losses, label='Train Loss', alpha=0.5)
plt.plot(val_iters, val_losses, label='Validation Loss', color='red', linewidth=2)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig(plot_file_path)
print(f"Plot saved to {plot_file_path}")
print(f"Log saved to {log_file_path}")