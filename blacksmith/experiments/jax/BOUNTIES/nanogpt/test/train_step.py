import numpy as np

import torch
import torch.optim as optim

import jax
import jax.numpy as jnp
import optax

from model_jax import GPT as GPT_JAX, GPTConfig as Config_JAX
from model import GPT as GPT_PT, GPTConfig as Config_PT
from .utils import copy_jax_to_pt, compare_weights


np.random.seed(42)
torch.manual_seed(42)
key = jax.random.PRNGKey(42)

config_jax = Config_JAX()
model_jax = GPT_JAX(config_jax)
params_jax = model_jax.init(key)

config_pt = Config_PT()
model_pt = GPT_PT(config_pt)
model_pt.eval() # set to eval mode (disables dropout if rate > 0)
copy_jax_to_pt(params_jax, model_pt) 

B, T = 4, 64
key, data_key = jax.random.split(key)
input_ids_jax = jax.random.randint(data_key, (B, T), 0, 50257, dtype=jnp.uint16)
targets_jax = jax.random.randint(data_key, (B, T), 0, 50257, dtype=jnp.uint16)

input_ids_pt = torch.tensor(input_ids_jax, dtype=torch.long)
targets_pt = torch.tensor(targets_jax, dtype=torch.long)

lr = 1e-4
weight_decay = 0.1
betas = (0.9, 0.95)
eps = 1e-8

optimizer_jax = optax.adamw(learning_rate=lr, b1=betas[0], b2=betas[1], eps=eps, weight_decay=weight_decay)
opt_state_jax = optimizer_jax.init(params_jax)

optimizer_pt = optim.AdamW(model_pt.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

@jax.jit
def jax_train_step(params, opt_state, rng, inputs, targets):
    
    def loss_fn(params, rng):
        logits = model_jax.apply(params, inputs, deterministic=False, rngs={'dropout': rng})
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return jnp.mean(loss)

    dropout_rng, new_rng = jax.random.split(rng)
    loss_val, grads = jax.value_and_grad(loss_fn)(params, dropout_rng)
    
    updates, new_opt_state = optimizer_jax.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, loss_val, new_rng

key, step_key = jax.random.split(key)
params_jax_new, opt_state_jax_new, loss_jax, key = jax_train_step(
    params_jax, opt_state_jax, step_key, input_ids_jax, targets_jax
)

optimizer_pt.zero_grad(set_to_none=True)
logits_pt, loss_pt = model_pt(input_ids_pt, targets_pt) # nanoGPT calculates loss internally
loss_pt.backward()
optimizer_pt.step()

print(f"JAX Loss: {loss_jax}")
print(f"PyTorch Loss: {loss_pt.item()}")
print(f"Loss difference: {abs(loss_jax - loss_pt.item())}")

print("Comparing weights after one training step...")
compare_weights(model_pt.state_dict(), params_jax_new)

# This is the REAL test
# You need to write a helper function to compare the JAX params dict
# with the PyTorch state_dict.
# compare_weights(params_jax_new, model_pt.state_dict())
# This will be tricky due to naming and transpositions, but it's the
# ultimate proof of parity.