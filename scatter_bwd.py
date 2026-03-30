import os

import torch
import torch_xla
import torch_xla.runtime as xr


class ScatterModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, router_logits, router_indices, router_top_value):
        val = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
        return val


# Set up TT device environment before the XLA runtime initializes
xr.set_device_type("TT")
os.environ["PJRT_DEVICE"] = "TT"
os.environ["XLA_STABLEHLO_COMPILE"] = "1"
os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
os.environ["DISABLE_NUMERIC_CC_TOKEN"] = "1"
xr.use_spmd()

# load the tensors from file
data = torch.load("scatter_debug_layer15_3.pt")

router_logits = data["routing_weights"]
router_top_value = data["topk_post"]
router_indices = data["indices"]
incoming_gradient = data["gradient_input"]

print(f"Loaded router_logits: shape={router_logits.shape}, dtype={router_logits.dtype}, min={router_logits.min().item()}, max={router_logits.max().item()}")
print(f"Loaded router_top_value: shape={router_top_value.shape}, dtype={router_top_value.dtype}, min={router_top_value.min().item()}, max={router_top_value.max().item()}")
print(f"Loaded router_indices: shape={router_indices.shape}, dtype={router_indices.dtype}, min={router_indices.min().item()}, max={router_indices.max().item()}")

compile_options = {
    "tt_enable_torch_fx_fusion_pass": False,
    "tt_legacy_compile": True,
}

model = ScatterModel()
model = torch.compile(model, backend="tt", options=compile_options)

device = torch_xla.device()
router_logits = router_logits.to(device)
router_top_value = router_top_value.to(device).detach().requires_grad_(True)
router_indices = router_indices.to(device)

output = model(router_logits, router_indices, router_top_value)
print(f"Scatter output: shape={output.shape}, dtype={output.dtype}, min={output.min().item()}, max={output.max().item()}")
torch_xla.sync(wait=True)

incoming_gradient = incoming_gradient.to(device)
output.backward(incoming_gradient)
torch_xla.sync(wait=True)
print("Backward pass completed.")

# scatter bwd is gather: grad_topk[i,k] = grad_routing_weights[i, indices[i,k]]
actual_grad = router_top_value.grad.float().cpu()
expected_grad = torch.gather(incoming_gradient.float().cpu(), 1, router_indices.cpu())

print(f"\nBackward check:")
print(f"  expected (cpu gather): norm={expected_grad.norm():.6e}, max={expected_grad.abs().max():.6e}")
print(f"  actual (accelerator):  norm={actual_grad.norm():.6e}, max={actual_grad.abs().max():.6e}")
print(f"  match (atol=1e-3):     {torch.allclose(expected_grad, actual_grad, atol=1e-3, rtol=1e-3)}")
print(f"  max abs diff:          {(expected_grad - actual_grad).abs().max():.6e}")

