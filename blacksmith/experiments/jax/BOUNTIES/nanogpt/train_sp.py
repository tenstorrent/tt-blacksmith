import os
import time
import numpy as np
import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze
import optax
from functools import partial
from model_jax import GPT, GPTConfig
from utils import train_step_onehot
import csv
import matplotlib.pyplot as plt

config = GPTConfig(
    block_size=256,
    vocab_size=65,
    num_layers=6,
    num_heads=6,
    num_embeds=384,
    dropout_rate=0.2,
    dtype=jnp.float32
)

batch_size = 64
max_iters = 5000
learning_rate = 1e-3

tt_device = jax.devices('tt')[0]
cpu_device = jax.devices('cpu')[0]

# data loading
data_dir = os.path.dirname(__file__)
train_data = np.fromfile(os.path.join(data_dir, 'data/train.bin'), dtype=np.uint16)
val_data = np.fromfile(os.path.join(data_dir, 'data/val.bin'), dtype=np.uint16)

def get_batch(split, key):
    data = train_data if split == 'train' else val_data
    ix = jax.random.randint(key, (batch_size,), 0, len(data) - config.block_size)
    
    # Numpy slicing for CPU data
    ix_np = np.array(ix)
    x_stack = np.stack([data[i:i+config.block_size] for i in ix_np])
    y_stack = np.stack([data[i+1:i+1+config.block_size] for i in ix_np])
    
    return jnp.array(x_stack, dtype=jnp.uint32), jnp.array(y_stack, dtype=jnp.uint32)

# --- Init ---
print(f"Initializing model (V={config.vocab_size})...")
model = GPT(config)
key = jax.random.PRNGKey(1337)
key, init_key = jax.random.split(key)

# Init on CPU to be safe
with jax.default_device(cpu_device):
    variables = model.init(init_key)

variables = unfreeze(variables)
cache = freeze({'cache': variables.pop('cache')})
params = freeze(variables)

optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=1e-1)
opt_state = optimizer.init(params)

# Move to the device
print(f"Moving to tt hardware...")

params = jax.device_put(params, tt_device)
cache = jax.device_put(cache, tt_device)
opt_state = jax.device_put(opt_state, tt_device)


# --- Compilation ---
@partial(jax.jit, backend='tt')
def eval_step(params, cache, x, y):
    vars = {'params': params['params'], **cache}
    logits = model.apply(vars, x, deterministic=True)
    
    vocab_size = logits.shape[-1]
    one_hot = jax.nn.one_hot(y, vocab_size)

    log_probs = jax.nn.log_softmax(logits)
    loss = -jnp.sum(one_hot * log_probs, axis=-1)
    return jnp.mean(loss)


out_dir = 'output'
os.makedirs(out_dir, exist_ok=True)
log_file_path = os.path.join(out_dir, 'log.csv')
plot_file_path = os.path.join(out_dir, 'loss_plot.png')

# Logging containers
iter_nums = []
train_losses = []
val_losses = []
val_iters = [] # Store iterations where validation happened

print(f"Compiling...")
t0 = time.time()
# Warmup
key, k1, k2 = jax.random.split(key, 3)
xb, yb = get_batch('train', k1)
xb, yb = jax.device_put(xb, tt_device), jax.device_put(yb, tt_device)
params, opt_state, _ = train_step_onehot(model, optimizer, params, cache, opt_state, xb, yb, k2)
print(f"Compiled in {time.time()-t0:.2f}s")

print(f"Training for {max_iters} iterations...")
start_time = time.time()
eval_interval = 250
eval_iters = 200

# Open CSV for writing
with open(log_file_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['step', 'train_loss', 'val_loss', 'time_sec'])

    for iter in range(max_iters):
        key, batch_key, drop_key = jax.random.split(key, 3)
        xb, yb = get_batch('train', batch_key)
        xb, yb = jax.device_put(xb, tt_device), jax.device_put(yb, tt_device)

        params, opt_state, loss = train_step_onehot(model, optimizer, params, cache, opt_state, xb, yb, drop_key)
        
        # Store training loss for plotting (every step might be too noisy/heavy, but fine for 5000)
        iter_nums.append(iter)
        train_losses.append(float(loss))

        if iter % eval_interval == 0 or iter == max_iters - 1:
            # Evaluate
            v_losses = []
            for _ in range(eval_iters):
                key, val_k = jax.random.split(key)
                xb_val, yb_val = get_batch('val', val_k)
                xb_val, yb_val = jax.device_put(xb_val, tt_device), jax.device_put(yb_val, tt_device)
                v_losses.append(eval_step(params, cache, xb_val, yb_val))
            
            val_loss = float(jnp.mean(jnp.array(v_losses)))
            val_iters.append(iter)
            val_losses.append(val_loss)

            dt = time.time() - start_time
            print(f"Step {iter}: Train Loss {loss:.4f} | Val Loss {val_loss:.4f} | Time {dt:.2f}s")
            
            # Write to CSV
            writer.writerow([iter, float(loss), val_loss, dt])
            f.flush() # Ensure data is written if crash happens

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