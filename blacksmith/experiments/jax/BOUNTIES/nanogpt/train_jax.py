import numpy as np
import jax
import jax.numpy as jnp
from flax.core import FrozenDict, unfreeze, freeze
import optax
from functools import partial
from model_jax import GPT as GPT_JAX, GPTConfig as Config_JAX
import time
from utils import train_step


# --------- Setup --------- 
key = jax.random.PRNGKey(42)
config = Config_JAX() 
model = GPT_JAX(config)

key, init_key = jax.random.split(key)

print("Initializing variables on CPU...")
cpu_device = jax.devices('cpu')[0]
with jax.default_device(cpu_device):
    variables = model.init(init_key)

variables = unfreeze(variables)
cache = freeze({'cache': variables.pop('cache')}) 
params = freeze(variables) 

print("Moving Cache to TT Device...")
tt_device = jax.devices('tt')[0]
cache = jax.device_put(cache, tt_device)

# --------- Optimizer ---------
lr = 1e-4
optimizer = optax.adamw(learning_rate=lr, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.1)
opt_state = optimizer.init(params)

# --------- Train loop ---------
B = 4
T = 64
NUM_STEPS = 200
losses = []

print(f"Starting Training (B={B}, Micro-Batch=4)...")

prev_time = time.time()
for step in range(NUM_STEPS):
    input_ids = np.random.randint(0, config.vocab_size, size=(B, T), dtype=np.uint32)
    targets = np.random.randint(0, config.vocab_size, size=(B, T), dtype=np.uint32)
    
    params, opt_state, loss_val = train_step(
        model,
        optimizer,
        params, 
        cache, 
        opt_state, 
        input_ids, 
        targets
    )
    
    losses.append(loss_val)
    
    if (step + 1) % 5 == 0:
        print(f"  Step {step+1}/{NUM_STEPS} | Loss: {loss_val:.6f} | Time: {time.time() - prev_time:.3f}s")
        prev_time = time.time()

print(f"Final Loss: {losses[-1]:.6f}")