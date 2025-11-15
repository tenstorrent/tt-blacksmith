import numpy as np
import torch
import jax
import jax.numpy as jnp

from model_jax import GPT as GPT_JAX
from model_jax import get_pretrained_params

# model is appended to $PYTHONPATH to me
# so I can import it directly, but it is supposed to be
# model from Karpathy's nanoGPT repo, for comparison.
from model import GPT as GPT_PT

print("Loading models...")

config_jax, params_jax = get_pretrained_params('gpt2')
model_jax = GPT_JAX(config_jax)

model_pt = GPT_PT.from_pretrained('gpt2')
model_pt.eval()

print("Models loaded.")

input_ids_np = np.array([[101, 2054, 2064, 102]], dtype=np.int64)

input_ids_jax = jnp.array(input_ids_np, dtype=jnp.uint16)
input_ids_pt = torch.tensor(input_ids_np, dtype=torch.long)

print("Running forward passes...")
logits_jax = model_jax.apply(params_jax, input_ids_jax, deterministic=True)
with torch.no_grad():
    logits_pt, _ = model_pt(input_ids_pt) # pytorch model returns (logits, loss)

print("Forward passes complete.")
logits_jax_np = np.asarray(logits_jax)[:, [-1], :]
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