import time
import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze
import optax
from functools import partial
from model_jax import GPT, GPTConfig
from utils import train_step, train_step_onehot

batch_size = 8
block_size = 64
num_steps = 50

config = GPTConfig(
    block_size=block_size,
    vocab_size=50304,
    num_layers=12,
    num_heads=12,
    num_embeds=768,
    dropout_rate=0.0,
    dtype=jnp.bfloat16
)

tt_device = jax.devices('tt')[0]
cpu_device = jax.devices('cpu')[0]


print(f"Initializing model on CPU...")
model = GPT(config)
key = jax.random.PRNGKey(1337)
init_key, train_key = jax.random.split(key)

with jax.default_device(cpu_device):
    variables = model.init(init_key)

variables = unfreeze(variables)
cache = freeze({'cache': variables.pop('cache')})
params = freeze(variables)

optimizer = optax.adamw(learning_rate=3e-4)
opt_state = optimizer.init(params)


params = jax.device_put(params, tt_device)
cache = jax.device_put(cache, tt_device)
opt_state = jax.device_put(opt_state, tt_device)


print("Generating dummy data...")

X = jax.random.randint(key, (batch_size, block_size), 0, config.vocab_size).astype(jnp.uint32)
Y = jax.random.randint(key, (batch_size, block_size), 0, config.vocab_size).astype(jnp.uint32)
X = jax.device_put(X, tt_device)
Y = jax.device_put(Y, tt_device)


print("Compiling (Warmup step)...")
t0 = time.time()
params, opt_state, loss = train_step_onehot(model, optimizer, params, cache, opt_state, X, Y)
loss.block_until_ready()
t1 = time.time()
print(f"Compilation finished in {t1-t0:.3f}s")

print(f"Running benchmark for {num_steps} steps...")
torch_wait = 0 # emulate pytorch cuda sync overhead? No, purely measure step time.
start_time = time.time()

for i in range(num_steps):
    params, opt_state, loss = train_step_onehot(model, optimizer, params, cache, opt_state, X, Y)


loss.block_until_ready()
end_time = time.time()

total_time = end_time - start_time
time_per_iter = total_time / num_steps
tokens_per_sec = (batch_size * block_size * num_steps) / total_time

print(f"\n--- Results ---")
print(f"Time per iter: {time_per_iter*1000:.2f} ms")
print(f"Tokens/sec:    {tokens_per_sec:.2f}")
print(f"Final Loss:    {loss}")