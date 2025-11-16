import numpy as np
import torch
import torch.optim as optim
import jax
import jax.numpy as jnp
from flax.core import FrozenDict, unfreeze, freeze
import optax
import matplotlib.pyplot as plt

from model_jax import GPT as GPT_JAX, GPTConfig as Config_JAX
from model import GPT as GPT_PT, GPTConfig as Config_PT
from .utils import to_torch, copy_jax_to_pt, compare_weights


np.random.seed(42)
torch.manual_seed(42)
key = jax.random.PRNGKey(42)

config_jax = Config_JAX()
model_jax = GPT_JAX(config_jax)
key, init_key = jax.random.split(key)
params_jax = model_jax.init(init_key)

config_pt = Config_PT()
model_pt = GPT_PT(config_pt)
copy_jax_to_pt(params_jax, model_pt) 

B, T = 4, 64
lr = 1e-4
weight_decay = 0.1
betas = (0.9, 0.95)
eps = 1e-8

# --- Optimizers ---
optimizer_jax = optax.adamw(learning_rate=lr, b1=betas[0], b2=betas[1], eps=eps, weight_decay=weight_decay)
opt_state_jax = optimizer_jax.init(params_jax)
optimizer_pt = optim.AdamW(model_pt.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

# --- JAX Train Step (JIT-compiled) ---
@jax.jit
def jax_train_step(params, opt_state, rng, inputs, targets):
    
    def loss_fn(params, rng):
        # Using deterministic=False because we assume dropout_rate=0.0
        # If dropout_rate > 0.0, this will use RNG, while model_pt.eval() will not.
        logits = model_jax.apply(params, inputs, deterministic=False, rngs={'dropout': rng})
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return jnp.mean(loss)

    dropout_rng, new_rng = jax.random.split(rng)
    loss_val, grads = jax.value_and_grad(loss_fn)(params, dropout_rng)
    
    updates, new_opt_state = optimizer_jax.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, loss_val, new_rng

# --- 🚀 NEW: Training Loop ---
NUM_STEPS = 200
jax_losses = []
pt_losses = []

print(f"Starting training loop for {NUM_STEPS} steps...")

for step in range(NUM_STEPS):
    # --- Generate identical data for this step ---
    key, step_key = jax.random.split(key, 2)
    input_ids = np.random.randint(0, config_jax.vocab_size, size=(B, T), dtype=np.uint32)
    targets = np.random.randint(0, config_jax.vocab_size, size=(B, T), dtype=np.uint32)

    input_ids_jax = jnp.array(input_ids, dtype=jnp.uint32)
    targets_jax = jnp.array(targets, dtype=jnp.uint32)

    input_ids_pt = torch.tensor(input_ids, dtype=torch.long)
    targets_pt = torch.tensor(targets, dtype=torch.long)

    # --- JAX train step ---
    params_jax, opt_state_jax, loss_jax_val, key = jax_train_step(
        params_jax, opt_state_jax, step_key, input_ids_jax, targets_jax
    )
    
    # --- PyTorch train step ---
    optimizer_pt.zero_grad(set_to_none=True)
    logits_pt, loss_pt_val = model_pt(input_ids_pt, targets_pt) # nanoGPT calculates loss internally
    loss_pt_val.backward()
    optimizer_pt.step()
    
    # --- Log losses ---
    jax_losses.append(loss_jax_val)
    pt_losses.append(loss_pt_val.item())
    
    if (step + 1) % 20 == 0:
        print(f"  Step {step+1}/{NUM_STEPS} | JAX Loss: {loss_jax_val:.6f} | PT Loss: {loss_pt_val.item():.6f}")

print("Training complete.")

# --- 📊 NEW: Final Comparison and Plotting ---

print("\n--- Final Comparison ---")
print(f"Final JAX Loss: {jax_losses[-1]:.6f}")
print(f"Final PyTorch Loss: {pt_losses[-1]:.6f}")
print("Comparing final weights...")
compare_weights(model_pt.state_dict(), params_jax)

# Plot the results
print("\nGenerating loss plot...")
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

# Optional: Plot the difference
loss_diff = [abs(j - p) for j, p in zip(jax_losses, pt_losses)]
plt.figure(figsize=(12, 6))
plt.plot(loss_diff, label='Absolute Loss Difference (JAX - PT)', color='red')
plt.title('Absolute Difference in Loss per Step')
plt.xlabel('Training Step')
plt.ylabel('Loss Difference')
plt.yscale('log') # Use log scale to see the small differences
plt.legend()
plt.grid(True)
plt.savefig('loss_difference.png')
print("✅ Difference plot saved to loss_difference.png")