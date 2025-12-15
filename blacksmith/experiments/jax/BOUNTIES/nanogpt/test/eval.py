import numpy as np
import torch
import jax
import jax.numpy as jnp

from flax.core import FrozenDict, unfreeze, freeze
from model_jax import GPT as GPT_JAX, GPTConfig as Config_JAX
from model_jax import get_pretrained_params

from utils import copy_jax_to_pt

# model is appended to $PYTHONPATH to me
# so I can import it directly, but it is supposed to be
# model from Karpathy's nanoGPT repo, for comparison.
import sys
sys.path.append('/root/nanoGPT/')
from model import GPT as GPT_PT, GPTConfig as Config_PT

key = jax.random.PRNGKey(42)
config_jax = Config_JAX() 
model_jax = GPT_JAX(config_jax)

print("Loading models...")
key, init_key = jax.random.split(key)

print("Initializing model on CPU...")
cpu_device = jax.devices('cpu')[0]
with jax.default_device(cpu_device):
    variables = model_jax.init(init_key)

variables = unfreeze(variables)
cache = freeze({'cache': variables.pop('cache')}) 
params_jax = freeze(variables) 

print("Moving Cache to TT Device...")
tt_device = jax.devices('tt')[0]
cache = jax.device_put(cache, tt_device)

config_pt = Config_PT()
model_pt = GPT_PT(config_pt)
model_pt.eval()
copy_jax_to_pt(params_jax, model_pt)

print("Running forward pass.")

input_ids_np = np.array([[101, 2054, 2064, 102]], dtype=np.int64)

input_ids_jax = jnp.array(input_ids_np, dtype=jnp.uint16)
logits_jax = model_jax.apply(params_jax, input_ids_jax, deterministic=True)
logits_jax_np = np.asarray(logits_jax)[:, [-1], :]

input_ids_pt = torch.tensor(input_ids_np, dtype=torch.long)
with torch.no_grad():
    logits_pt, _ = model_pt(input_ids_pt) # pytorch model returns (logits, loss)
logits_pt_np = logits_pt.detach().cpu().numpy()

print(f"JAX logits shape: {logits_jax_np.shape}")
print(f"PyTorch logits shape: {logits_pt_np.shape}")

if np.allclose(logits_jax_np, logits_pt_np, rtol=1e-5, atol=1e-5):
    print("✅ SUCCESS: Evaluation (Inference) parity is confirmed!")
else:
    print("❌ FAILURE: Outputs do not match.")
    print("JAX logits (last token):")
    print(logits_jax_np[:, :, :10])
    print("PyTorch logits (last token):")
    print(logits_pt_np[:, :, :10])

    diff = np.abs(logits_jax_np - logits_pt_np)
    print(f"Max absolute difference: {np.max(diff)}")
    print(f"Mean absolute difference: {np.mean(diff)}")