import numpy as np
import jax
import jax.numpy as jnp
from flax.core import FrozenDict, unfreeze, freeze
import optax
from functools import partial

from model_jax import GPT as GPT_JAX, GPTConfig as Config_JAX

# --- 1. Setup ---
key = jax.random.PRNGKey(42)

# Full 12-layer model
config_jax = Config_JAX(num_layers=0, dropout_rate=0.0)
model_jax = GPT_JAX(config_jax)
key, init_key = jax.random.split(key)
params_jax = model_jax.init(init_key)

# --- 2. SPLIT PARAMETERS ---
# We physically separate the parameters into two PyTrees.
p = unfreeze(params_jax)
wte = p['params'].pop('wte')
wpe = p['params'].pop('wpe')

params_cpu = freeze({'params': {'wte': wte, 'wpe': wpe}}) # Lives on CPU
params_tt = freeze(p)                                     # Lives on TT

# --- 3. SPLIT OPTIMIZERS ---
# We define two independent optimizers.
# This ensures opt_state_tt lives on the device and updates happen on the device.

lr = 1e-4
# Basic config for both
tx = optax.adamw(learning_rate=lr, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.1)

# Init states separately
opt_state_cpu = tx.init(params_cpu)
opt_state_tt = tx.init(params_tt)

# --- 4. JIT-COMPILED STEPS ---

# --- PART A: CPU Operations (Embeddings & Head) ---
@partial(jax.jit, backend='cpu') 
def step_cpu_forward(params_cpu, inputs):
    # Forward pass for embeddings
    return model_jax.apply(params_cpu, inputs, method=model_jax.embed)

@partial(jax.jit, backend='cpu')
def step_cpu_loss(params_cpu, x_final, targets):
    # Forward pass for Head + Loss
    logits = model_jax.apply(params_cpu, x_final, method=model_jax.head)
    vocab_size = logits.shape[-1]
    one_hot_targets = jax.nn.one_hot(targets, vocab_size)
    log_probs = jax.nn.log_softmax(logits)
    loss = -jnp.sum(one_hot_targets * log_probs, axis=-1)
    return jnp.mean(loss)

@partial(jax.jit, backend='cpu')
def step_cpu_update(grads, opt_state, params):
    # Optimizer Step for Embeddings (Runs on CPU)
    updates, new_opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state

# --- PART B: TT Operations (Transformer Body) ---
@partial(jax.jit, backend='tt')
def step_tt_body(params_tt, x):
    # Heavy Lifting: 12 Layers of Blocks
    return model_jax.apply(params_tt, x, deterministic=True, method=model_jax.body)

@partial(jax.jit, backend='tt')
def step_tt_update(grads, opt_state, params):
    # Optimizer Step for Body (Runs on TT)
    # CRITICAL: Weights never leave the device here!
    updates, new_opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state

# --- 5. ORCHESTRATOR ---
# This runs in Python and manages the hand-offs
def train_step(params_cpu, params_tt, opt_state_cpu, opt_state_tt, inputs, targets):
    
    # Define the gradient calculation function
    def loss_fn(params_cpu, params_tt):
        # 1. CPU: Embed
        x = step_cpu_forward(params_cpu, inputs)
        
        # 2. TT: Body (Data moves CPU->TT)
        # x is treated as a constant wrt params_tt during the split, 
        # but JAX handles the chain rule via backprop through the JIT boundary
        x_final = step_tt_body(params_tt, x)
        
        # 3. CPU: Head + Loss (Data moves TT->CPU)
        loss = step_cpu_loss(params_cpu, x_final, targets)
        return loss

    # Calculate Gradients
    # JAX is amazing: value_and_grad can differentiate through the split JIT calls
    loss_val, (grads_cpu, grads_tt) = jax.value_and_grad(loss_fn, argnums=(0, 1))(params_cpu, params_tt)
    
    # Update CPU Params (On CPU)
    new_params_cpu, new_opt_state_cpu = step_cpu_update(grads_cpu, opt_state_cpu, params_cpu)
    
    # Update TT Params (On TT)
    new_params_tt, new_opt_state_tt = step_tt_update(grads_tt, opt_state_tt, params_tt)
    
    return new_params_cpu, new_params_tt, new_opt_state_cpu, new_opt_state_tt, loss_val

# --- 6. TRAINING LOOP ---
B, T = 4, 64
NUM_STEPS = 200
losses = []

print(f"Starting Split-Optimizer Training for {NUM_STEPS} steps...")

for step in range(NUM_STEPS):
    # Data Gen
    input_ids = np.random.randint(0, config_jax.vocab_size, size=(B, T), dtype=np.uint32)
    targets = np.random.randint(0, config_jax.vocab_size, size=(B, T), dtype=np.uint32)
    
    inputs_jax = jnp.array(input_ids)
    targets_jax = jnp.array(targets)
    
    # Step
    params_cpu, params_tt, opt_state_cpu, opt_state_tt, loss_val = train_step(
        params_cpu, params_tt, opt_state_cpu, opt_state_tt, inputs_jax, targets_jax
    )
    
    losses.append(loss_val)
    
    if (step + 1) % 20 == 0:
        print(f"  Step {step+1}/{NUM_STEPS} | Loss: {loss_val:.6f}")

print(f"Final Loss: {losses[-1]:.6f}")