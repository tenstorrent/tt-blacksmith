import numpy as np
import jax
import jax.numpy as jnp
from flax.core import FrozenDict, unfreeze, freeze
import optax
from model_jax import GPT as GPT_JAX, GPTConfig as Config_JAX
import time

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

tt_device = jax.devices('tt')[0]
cache = jax.device_put(cache, tt_device)
print(f"Target Device: {tt_device}")

B, T = 4, 64
input_ids = np.random.randint(0, config.vocab_size, size=(B, T), dtype=np.uint32)
targets = np.random.randint(0, config.vocab_size, size=(B, T), dtype=np.uint32)

lr = 1e-4
optimizer = optax.adamw(learning_rate=lr, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.1)
opt_state = optimizer.init(params)

@jax.jit
def train_step_debug(params, cache, opt_state, inputs, targets):
    
    def loss_fn(p):
        variables = {'params': p['params'], **cache}
        logits = model.apply(variables, inputs, deterministic=True)
        
        B, T, V = logits.shape
        axis = -1
        # ------------- PROBLEM -------------
        # jnp.take_along_axis poziva gather, pa mu gradient poziva scatter (writing gradients to random memory locations)
        #label_logits = jnp.take_along_axis(logits, jnp.expand_dims(targets, axis), axis=axis).take(0, axis=axis)
        # -----------------------------------
        targets_one_hot = jax.nn.one_hot(targets, V, dtype=logits.dtype)
        label_logits = jnp.sum(logits * targets_one_hot, axis=-1)
        # -----------------------------------
        log_normalizers = jax.nn.logsumexp(logits, axis=axis)
        loss = jnp.mean(log_normalizers - label_logits)
        
        return loss

    loss_val, grads = jax.value_and_grad(loss_fn)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, loss_val

print("\n--- STARTING JIT COMPILATION & EXECUTION ---")
start = time.time()

params, opt_state, loss = train_step_debug(params, cache, opt_state, input_ids, targets)
jax.block_until_ready(loss)
print(f"Step 1 Done. Loss: {loss:.4f} | Time: {time.time() - start:.3f}s")

start = time.time()
params, opt_state, loss = train_step_debug(params, cache, opt_state, input_ids, targets)
jax.block_until_ready(loss)
print(f"Step 2 Done. Loss: {loss:.4f} | Time: {time.time() - start:.3f}s")