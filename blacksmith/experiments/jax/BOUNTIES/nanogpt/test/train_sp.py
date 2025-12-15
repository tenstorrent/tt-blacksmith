import os
import time
import math
import numpy as np
import jax
import jax.numpy as jnp
from flax.core import FrozenDict, unfreeze, freeze
import optax
from functools import partial
import pandas as pd

# Import your model definition
from model_jax import GPT as GPT_JAX, GPTConfig as Config_JAX

# --- HYPERPARAMETERS (NanoGPT Shakespeare Style) ---
# We use a smaller model than GPT-2 to ensure it fits and trains fast on N150
BLOCK_SIZE = 64      # Context length (T)
BATCH_SIZE = 64      # Total Batch Size (B)
MICRO_BATCH = 4      # Chunk size for Accumulation (To prevent OOM)
NUM_LAYERS = 6
NUM_HEADS = 6
NUM_EMBED = 384
DROPOUT = 0.0 # 0.0 for speed on TT
LEARNING_RATE = 1e-3
MAX_ITERS = 2000
EVAL_INTERVAL = 100
WARMUP_ITERS = 100

# --- DATA LOADER ---
def get_batch(split):
    data_dir = os.path.dirname(__file__)
    filename = os.path.join(data_dir, 'train.bin' if split == 'train' else 'val.bin')
    data = np.memmap(filename, dtype=np.uint16, mode='r')
    
    ix = np.random.randint(0, len(data) - BLOCK_SIZE, BATCH_SIZE)
    x = np.stack([data[i:i+BLOCK_SIZE] for i in ix]).astype(np.int32)
    y = np.stack([data[i+1:i+1+BLOCK_SIZE] for i in ix]).astype(np.int32)
    return x, y

# --- SETUP ---
print("--- 1. Initializing Model ---")
key = jax.random.PRNGKey(1337)
config = Config_JAX(
    num_layers=NUM_LAYERS, 
    num_heads=NUM_HEADS, 
    num_embeds=NUM_EMBED, 
    block_size=BLOCK_SIZE, 
    dropout_rate=DROPOUT,
    vocab_size=50304 # GPT-2 Vocab
)

# Init on CPU to avoid TT Compiler Crash
cpu_device = jax.devices('cpu')[0]
with jax.default_device(cpu_device):
    model = GPT_JAX(config)
    key, init_key = jax.random.split(key)
    variables = model.init(init_key)

# Move Cache to TT Device (Performance)
print("--- 2. Moving Cache to TT Device ---")
tt_device = jax.devices('tt')[0]
variables = unfreeze(variables)
cache = freeze({'cache': variables.pop('cache')})
params = freeze(variables)
cache = jax.device_put(cache, tt_device)

# --- OPTIMIZER ---
# Cosine Decay Schedule (Same as NanoGPT)
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=LEARNING_RATE,
    warmup_steps=WARMUP_ITERS,
    decay_steps=MAX_ITERS,
    end_value=LEARNING_RATE * 0.1
)
optimizer = optax.adamw(learning_rate=schedule, weight_decay=1e-1)
opt_state = optimizer.init(params)

# --- TRAIN STEP (ACCUMULATED) ---
# This uses the Unrolled Loop method we fixed earlier.
@partial(jax.jit, backend='tt')
def train_step_accum(params, cache, opt_state, inputs, targets):
    
    NUM_MICRO = BATCH_SIZE // MICRO_BATCH
    
    # Reshape for unrolled loop
    inputs_reshaped = inputs.reshape(NUM_MICRO, MICRO_BATCH, -1)
    targets_reshaped = targets.reshape(NUM_MICRO, MICRO_BATCH, -1)

    accum_grads = jax.tree.map(jnp.zeros_like, params)
    accum_loss = 0.0

    # Unrolled Python Loop (Safe for TT Compiler)
    for i in range(NUM_MICRO):
        micro_input = inputs_reshaped[i]
        micro_target = targets_reshaped[i]

        def loss_fn(p):
            variables = {'params': p['params'], **cache}
            logits = model.apply(variables, micro_input, deterministic=True)
            
            # Use BFloat16 One-Hot to save memory
            vocab_size = logits.shape[-1]
            one_hot = jax.nn.one_hot(micro_target, vocab_size, dtype=jnp.bfloat16)
            log_probs = jax.nn.log_softmax(logits)
            
            loss = -jnp.sum(one_hot * log_probs, axis=-1)
            return jnp.mean(loss)

        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        accum_grads = jax.tree.map(lambda a, b: a + b, accum_grads, grads)
        accum_loss = accum_loss + loss_val

    # Average and Update
    avg_grads = jax.tree.map(lambda g: g / NUM_MICRO, accum_grads)
    avg_loss = accum_loss / NUM_MICRO
    
    updates, new_opt_state = optimizer.update(avg_grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, avg_loss

# --- TRAINING LOOP ---
print(f"--- 3. Starting Training ({MAX_ITERS} iters) ---")
print(f"Config: B={BATCH_SIZE}, Micro={MICRO_BATCH}, Layers={NUM_LAYERS}, Embed={NUM_EMBED}")

loss_history = []
start_time = time.time()

for step in range(MAX_ITERS):
    iter_start = time.time()
    
    # Get Data
    xb, yb = get_batch('train')
    
    # Step
    params, opt_state, loss_val = train_step_accum(params, cache, opt_state, xb, yb)
    
    # Sync for timing
    loss_val.block_until_ready()
    dt = time.time() - iter_start
    
    loss_history.append(float(loss_val))
    
    if step % 10 == 0:
         print(f"Step {step} | Loss: {loss_val:.4f} | Time: {dt*1000:.2f}ms")

    if step % EVAL_INTERVAL == 0 and step > 0:
        # Save Log
        df = pd.DataFrame({'step': range(len(loss_history)), 'loss': loss_history})
        df.to_csv('jax_loss_log.csv', index=False)
        print(f"--- Log saved to jax_loss_log.csv ---")

total_time = time.time() - start_time
print(f"Training Finished in {total_time:.2f}s")