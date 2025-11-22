import numpy as np
import jax
import jax.numpy as jnp
from flax.core import FrozenDict, unfreeze, freeze, pop
import optax
import matplotlib.pyplot as plt
from functools import partial  # <--- Added for the fix

from model_jax import GPT as GPT_JAX, GPTConfig as Config_JAX
from model import GPT as GPT_PT, GPTConfig as Config_PT
from .utils import to_torch, copy_jax_to_pt, compare_weights

key = jax.random.PRNGKey(42)

config_jax = Config_JAX(num_layers=0, dropout_rate=0.0)
model_jax = GPT_JAX(config_jax)
key, init_key = jax.random.split(key)
params_jax = model_jax.init(init_key)

params_mutable = unfreeze(params_jax)

wte_params = params_mutable['params'].pop('wte')
wpe_params = params_mutable['params'].pop('wpe')
params_cpu = freeze({'params': {'wte': wte_params, 'wpe': wpe_params}})

params_tt = freeze(params_mutable) 

B, T = 4, 64
lr = 1e-4
weight_decay = 0.1
betas = (0.9, 0.95)
eps = 1e-8

optimizer_jax = optax.adamw(learning_rate=lr, b1=betas[0], b2=betas[1], eps=eps, weight_decay=weight_decay)
opt_state_jax = optimizer_jax.init(params_jax)

@partial(jax.jit, backend='cpu') 
def step_embed(params_cpu, inputs):
    # We reconstruct a partial model just to call 'embed'
    return model_jax.apply(params_cpu, inputs, method=model_jax.embed)

# STEP B: Transformer Body (Run on TT)
# Note: If backend='tt' fails, try backend='tpu' or remove backend arg to let JAX default to TT
@partial(jax.jit, backend='tt')
def step_body(params_tt, x):
    # Run blocks + ln_f
    return model_jax.apply(params_tt, x, deterministic=True, method=model_jax.body)

# STEP C: Head + Loss (Run on CPU)
@partial(jax.jit, backend='cpu')
def step_head_loss(params_cpu, x_final, targets):
    logits = model_jax.apply(params_cpu, x_final, method=model_jax.head)
    
    # Standard Softmax Loss
    vocab_size = logits.shape[-1]
    one_hot_targets = jax.nn.one_hot(targets, vocab_size)
    log_probs = jax.nn.log_softmax(logits)
    loss = -jnp.sum(one_hot_targets * log_probs, axis=-1)
    return jnp.mean(loss)

def split_params(full_params):
    p = unfreeze(full_params)
    # Extract CPU params
    cpu_p = {'params': {'wte': p['params']['wte'], 'wpe': p['params']['wpe']}}
    # Extract TT params (Make a deep copy or just pop if you don't need to reuse p)
    # Since we want to be clean, let's just copy the whole thing and pop keys
    tt_p = unfreeze(full_params)
    tt_p['params'].pop('wte')
    tt_p['params'].pop('wpe')
    
    return freeze(cpu_p), freeze(tt_p)

# --- 6. The Orchestrator Step ---
# We do NOT JIT this outer function. It orchestrates data movement.
def train_step_orchestrator(full_params, opt_state, inputs, targets):
    
    # Define the gradient function locally so it traces through the split calls
    def loss_composition(full_params):
        p_cpu, p_tt = split_params(full_params)
        
        # 1. CPU: Embed
        x = step_embed(p_cpu, inputs)
        
        # 2. TT: Body (Gradient flows through x, which bridges CPU->TT->CPU)
        x_final = step_body(p_tt, x)
        
        # 3. CPU: Head + Loss
        loss = step_head_loss(p_cpu, x_final, targets)
        return loss

    # Calculate Gradients for the WHOLE model
    loss_val, grads = jax.value_and_grad(loss_composition)(full_params)
    
    # Update All Params
    updates, new_opt_state = optimizer_jax.update(grads, opt_state, full_params)
    new_params = optax.apply_updates(full_params, updates)
    
    return new_params, new_opt_state, loss_val

# --- 7. Training Loop ---
NUM_STEPS = 200
jax_losses = []

print(f"Starting Hybrid Training (CPU->TT->CPU) for {NUM_STEPS} steps...")

for step in range(NUM_STEPS):
    input_ids = np.random.randint(0, config_jax.vocab_size, size=(B, T), dtype=np.uint32)
    targets = np.random.randint(0, config_jax.vocab_size, size=(B, T), dtype=np.uint32)
    input_ids_jax = jnp.array(input_ids)
    targets_jax = jnp.array(targets)
    
    params_jax, opt_state_jax, loss_jax_val = train_step_orchestrator(
        params_jax, opt_state_jax, input_ids_jax, targets_jax
    )
    
    if (step + 1) % 20 == 0:
        print(f"  Step {step+1}/{NUM_STEPS} | JAX Loss: {loss_jax_val:.6f}")

print("\n--- Final Comparison ---")
print(f"Final JAX Loss: {jax_losses[-1]:.6f}")
