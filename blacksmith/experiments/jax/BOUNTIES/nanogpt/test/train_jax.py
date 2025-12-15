import numpy as np
import jax
import jax.numpy as jnp
from flax.core import FrozenDict, unfreeze, freeze
import optax
from functools import partial
from model_jax import GPT as GPT_JAX, GPTConfig as Config_JAX
import time

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


@partial(jax.jit, backend='tt')
def train_step(params, cache, opt_state, inputs, targets):
    ''' Standard training step with softmax cross-entropy loss. '''
    def loss_fn(p):
        variables = {'params': p['params'], **cache}
        logits = model.apply(variables, inputs, deterministic=True)
        
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return jnp.mean(loss)

    loss_val, grads = jax.value_and_grad(loss_fn)(params)
    
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, loss_val


@partial(jax.jit, backend='tt')
def train_step_onehot(params, cache, opt_state, inputs, targets):
    ''' One-hot encoding training step with micro-batching to reduce memory usage. '''
    MICRO_BATCH = 4
    NUM_MICRO = inputs.shape[0] // MICRO_BATCH
    
    inputs_reshaped = inputs.reshape(NUM_MICRO, MICRO_BATCH, -1)
    targets_reshaped = targets.reshape(NUM_MICRO, MICRO_BATCH, -1)

    accum_grads = jax.tree.map(jnp.zeros_like, params)
    accum_loss = 0.0

    for i in range(NUM_MICRO):
        
        micro_input = inputs_reshaped[i]
        micro_target = targets_reshaped[i]

        def loss_fn(p):
            variables = {'params': p['params'], **cache}
            logits = model.apply(variables, micro_input, deterministic=True)
            
            vocab_size = logits.shape[-1]
            one_hot = jax.nn.one_hot(micro_target, vocab_size)
            
            log_probs = jax.nn.log_softmax(logits)
            loss = -jnp.sum(one_hot * log_probs, axis=-1)
            return jnp.mean(loss)

        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        
        accum_grads = jax.tree.map(lambda a, b: a + b, accum_grads, grads)
        accum_loss = accum_loss + loss_val

    avg_grads = jax.tree.map(lambda g: g / NUM_MICRO, accum_grads)
    avg_loss = accum_loss / NUM_MICRO
    
    updates, new_opt_state = optimizer.update(avg_grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, avg_loss

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
    
    params, opt_state, loss_val = train_step_onehot(
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