import numpy as np
import torch
import torch.optim as optim
from model import GPT as GPT_PT, GPTConfig as Config_PT
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

config_pt = Config_PT(n_layer=0)
model_pt = GPT_PT(config_pt)
model_pt.eval()

use_tt = True
if use_tt:
    xr.runtime.set_device_type("TT")
    device = torch_xla.device()
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_pt.to(device)

B, T = 4, 64
lr = 1e-4
weight_decay = 0.1
betas = (0.9, 0.95)
eps = 1e-8
vocab_size = 50304

optimizer_pt = optim.AdamW(model_pt.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

NUM_STEPS = 200
pt_losses = []

print(f"Starting Hybrid Training (CPU->TT->CPU) for {NUM_STEPS} steps...")

for step in range(NUM_STEPS):
    input_ids = np.random.randint(0, vocab_size, size=(B, T), dtype=np.uint32)
    targets = np.random.randint(0, vocab_size, size=(B, T), dtype=np.uint32)

    input_ids_pt = torch.tensor(input_ids, dtype=torch.long).to(device)
    targets_pt = torch.tensor(targets, dtype=torch.long).to(device)
    
    optimizer_pt.zero_grad(set_to_none=True)
    logits_pt, loss_pt_val = model_pt(input_ids_pt, targets_pt)
    loss_pt_val.backward()
    optimizer_pt.step()
    
    pt_losses.append(loss_pt_val.item())
    
    if (step + 1) % 20 == 0:
        print(f"  Step {step+1}/{NUM_STEPS} | PT Loss: {loss_pt_val.item():.6f}")

print("\n--- Final Comparison ---")
print(f"Final PyTorch Loss: {pt_losses[-1]:.6f}")