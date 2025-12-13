import numpy as np
import jax
import jax.numpy as jnp
from flax.core import FrozenDict, unfreeze, freeze
import optax
from functools import partial

from model_jax import GPT as GPT_JAX, GPTConfig as Config_JAX

# setup model and params
key = jax.random.PRNGKey(42)
config = Config_JAX(num_layers=1, dropout_rate=0.0)
model = GPT_JAX(config)

cpu_device = jax.devices('cpu')[0]
tt_device = jax.devices('tt')[0]

print("Model created.")
key, init_key = jax.random.split(key)
with jax.default_device(cpu_device):
    vars = model.init(init_key)

print("Model initialized.")

vars = unfreeze(vars)
params = freeze({'params': vars.pop('params')})
cache = freeze(vars)
cache = jax.device_put(cache, tt_device)
# optimizer
lr = 1e-4
optimizer = optax.adamw(learning_rate=lr, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.1)
opt_state = optimizer.init(params)

print("Optimizer initialized.")

@partial(jax.jit, backend='tt')
def train_step(params, cache, opt_state, inputs, targets):
    
    def loss_fn(params):
        logits = model.apply(
            {'params': params['params'], **cache},
            inputs,
            deterministic=True,
        )
        # We manually compute Softmax loss to ensure it stays on device
        vocab_size = logits.shape[-1]
        one_hot_targets = jax.nn.one_hot(targets, vocab_size)
        log_probs = jax.nn.log_softmax(logits)
        loss = -jnp.sum(one_hot_targets * log_probs, axis=-1)
        return jnp.mean(loss)

    loss_val, grads = jax.value_and_grad(loss_fn)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, loss_val

B, T = 4, 64
NUM_STEPS = 200
losses = []

print("Moving cache params to TT device...")
cache = jax.device_put(cache, tt_device)

print(f"Starting Split-Optimizer Training for {NUM_STEPS} steps...")

for step in range(NUM_STEPS):
    input_ids = np.random.randint(0, config.vocab_size, size=(B, T), dtype=np.uint32)
    targets = np.random.randint(0, config.vocab_size, size=(B, T), dtype=np.uint32)
    
    params, opt_state, loss_val = train_step(
        params, 
        cache, 
        opt_state, 
        input_ids, 
        targets
    )
    
    losses.append(loss_val)
    
    if (step + 1) % 20 == 0:
        print(f"  Step {step+1}/{NUM_STEPS} | Loss: {loss_val:.6f}")

print(f"Final Loss: {losses[-1]:.6f}")