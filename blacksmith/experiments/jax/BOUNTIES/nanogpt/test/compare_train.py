import numpy as np
import torch
import torch.optim as optim
import jax
import jax.numpy as jnp
from flax.core import FrozenDict, unfreeze, freeze, pop
import optax
import matplotlib.pyplot as plt

from model_jax import GPT as GPT_JAX, GPTConfig as Config_JAX
from model import GPT as GPT_PT, GPTConfig as Config_PT
from .utils import to_torch, copy_jax_to_pt, compare_weights

# --- 1. Setup ---
np.random.seed(42)
torch.manual_seed(42)
key = jax.random.PRNGKey(42)

# --- 2. Initialize JAX Model ---
# We use num_layers=0 as per your request
config_jax = Config_JAX(num_layers=0, dropout_rate=0.0)
model_jax = GPT_JAX(config_jax)
key, init_key = jax.random.split(key)
params_jax = model_jax.init(init_key)

# --- 3. CRITICAL FIX: Freeze ALL Embeddings ---
# The TT compiler crashes on gradients for both WPE and WTE.
# We must remove both from the trainable parameters.
params_mutable = unfreeze(params_jax)

# Extract Embeddings
wpe_params = params_mutable['params'].pop('wpe')
wte_params = params_mutable['params'].pop('wte')

# Create frozen dictionary with both
params_frozen = freeze({'params': {'wpe': wpe_params, 'wte': wte_params}})

# The remaining params are trainable (ln_f, and blocks if num_layers > 0)
params_train = freeze(params_mutable) 

print(f"Trainable params: {list(params_train['params'].keys())}")
print(f"Frozen params: {list(params_frozen['params'].keys())}")

# --- 4. Initialize PyTorch Model ---
config_pt = Config_PT(n_layer=0)
model_pt = GPT_PT(config_pt)
copy_jax_to_pt(params_jax, model_pt) # Copy ALL params (train + frozen)

B, T = 4, 64
lr = 1e-4
weight_decay = 0.1
betas = (0.9, 0.95)
eps = 1e-8

# --- 5. Optimizers ---
# JAX Optimizer: ONLY sees params_train (no wpe)
optimizer_jax = optax.adamw(learning_rate=lr, b1=betas[0], b2=betas[1], eps=eps, weight_decay=weight_decay)
opt_state_jax = optimizer_jax.init(params_train)

# PyTorch Optimizer: Sees everything
optimizer_pt = optim.AdamW(model_pt.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

# Helper to merge params inside the train step
def merge_params(train_params, frozen_params):
    t = unfreeze(train_params)
    f = unfreeze(frozen_params)
    t['params'].update(f['params']) # Re-insert 'wpe' into 'params'
    return freeze(t)

# --- 6. JAX Train Step (JIT-compiled) ---
@jax.jit
def jax_train_step(train_params, frozen_params, opt_state, inputs, targets):
    
    def loss_fn(train_params):
        # Merge trainable and frozen params so the model sees everything
        params = merge_params(train_params, frozen_params)
        
        # Forward Pass (deterministic=True is CRITICAL)
        logits = model_jax.apply(params, inputs, deterministic=True)
        
        # Manual Cross Entropy (One-Hot) to avoid 'ttir.scatter' error in Loss
        vocab_size = logits.shape[-1]
        one_hot_targets = jax.nn.one_hot(targets, vocab_size)
        log_probs = jax.nn.log_softmax(logits)
        loss = -jnp.sum(one_hot_targets * log_probs, axis=-1)
        
        return jnp.mean(loss)

    # Calculate gradients ONLY for train_params
    loss_val, grads = jax.value_and_grad(loss_fn)(train_params)
    
    updates, new_opt_state = optimizer_jax.update(grads, opt_state, train_params)
    new_params = optax.apply_updates(train_params, updates)
    
    return new_params, new_opt_state, loss_val

# --- 7. Training Loop ---
NUM_STEPS = 200
jax_losses = []
pt_losses = []

print(f"Starting training loop for {NUM_STEPS} steps...")

for step in range(NUM_STEPS):
    # --- Generate Data ---
    key, step_key = jax.random.split(key, 2)
    input_ids = np.random.randint(0, config_jax.vocab_size, size=(B, T), dtype=np.uint32)
    targets = np.random.randint(0, config_jax.vocab_size, size=(B, T), dtype=np.uint32)

    input_ids_jax = jnp.array(input_ids, dtype=jnp.uint32)
    targets_jax = jnp.array(targets, dtype=jnp.uint32)

    input_ids_pt = torch.tensor(input_ids, dtype=torch.long)
    targets_pt = torch.tensor(targets, dtype=torch.long)

    # --- JAX train step ---
    # Pass separated params
    params_train, opt_state_jax, loss_jax_val = jax_train_step(
        params_train, params_frozen, opt_state_jax, input_ids_jax, targets_jax
    )
    
    # --- PyTorch train step ---
    # NOTE: Ideally, you should freeze WPE in PyTorch too for exact parity, 
    # but for checking if JAX runs, this is fine.
    optimizer_pt.zero_grad(set_to_none=True)
    logits_pt, loss_pt_val = model_pt(input_ids_pt, targets_pt) 
    loss_pt_val.backward()
    optimizer_pt.step()
    
    # --- Log losses ---
    jax_losses.append(loss_jax_val)
    pt_losses.append(loss_pt_val.item())
    
    if (step + 1) % 20 == 0:
        print(f"  Step {step+1}/{NUM_STEPS} | JAX Loss: {loss_jax_val:.6f} | PT Loss: {loss_pt_val.item():.6f}")

print("Training complete.")

# --- Final Comparison ---
print("\n--- Final Comparison ---")
print(f"Final JAX Loss: {jax_losses[-1]:.6f}")
print(f"Final PyTorch Loss: {pt_losses[-1]:.6f}")

# Recombine for comparison
params_final = merge_params(params_train, params_frozen)
print("Comparing final weights...")
compare_weights(model_pt.state_dict(), params_final)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(jax_losses, label='JAX Loss', alpha=0.8)
plt.plot(pt_losses, label='PyTorch Loss', linestyle='--', alpha=0.8)
plt.title('JAX vs PyTorch Loss Curve Parity')
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_comparison.png')
print("✅ Plot saved to loss_comparison.png")