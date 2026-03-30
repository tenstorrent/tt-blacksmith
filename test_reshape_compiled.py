"""
Test the reshape → embedding pattern through torch.compile(backend="tt"),
matching the exact XLA compilation path that produces the failing TTNN IR.

The bug only appears in COMPILED mode (static memory plan), not in eager TTNN.
This test goes through the actual compilation pipeline.
"""

import torch
import torch_xla
import torch_xla.core.xla_model as xm

def test_compiled_reshape_embedding():
    """
    Reproduce the exact pattern from the router backward:
      routing_weights.grad [128, 32] → reshape [4096, 1] → embedding lookup
    """
    device = xm.xla_device()

    # Simulate routing_weights.grad = small known values
    table_data = torch.ones(128, 32, dtype=torch.bfloat16) * 0.001
    table = table_data.to(device)

    # Simulate topk indices (valid 0-31) and row indices (0-127)
    row_idx = torch.arange(128, dtype=torch.int32).unsqueeze(1).expand(128, 4)  # [128, 4]
    topk_idx = torch.randint(0, 32, (128, 4), dtype=torch.int32)
    flat_idx = (row_idx * 32 + topk_idx).reshape(1, 512)  # [1, 512], values 0-4095

    flat_idx_dev = flat_idx.to(device)

    # This is the exact pattern:
    #   table_1d = table.reshape(4096, 1)     → the reshape that might be zero-copy in compiled mode
    #   result = table_1d[flat_idx_dev]        → the embedding/gather lookup

    def fn(tbl, idx):
        tbl_1d = tbl.reshape(4096, 1)
        # Use torch.index_select or direct indexing
        result = tbl_1d[idx.long()]
        return result

    # Compile with TT backend
    compiled_fn = torch.compile(fn, backend="tt")

    # Run step 1
    print("=== Step 1 ===")
    result1 = compiled_fn(table, flat_idx_dev)
    torch_xla.sync(wait=True)
    result1_cpu = result1.cpu().float()
    print(f"  result shape: {result1_cpu.shape}")
    print(f"  min={result1_cpu.min():.6e}, max={result1_cpu.max():.6e}")
    print(f"  expected=1.000000e-03")
    bad1 = (result1_cpu - 0.001).abs() > 0.0001
    print(f"  bad values: {bad1.sum().item()} / {result1_cpu.numel()}")

    # Modify table (simulate gradient accumulation writing different values)
    table2_data = torch.ones(128, 32, dtype=torch.bfloat16) * 0.002
    table2 = table2_data.to(device)

    # Run step 2 — same compiled graph, different input
    print("\n=== Step 2 ===")
    result2 = compiled_fn(table2, flat_idx_dev)
    torch_xla.sync(wait=True)
    result2_cpu = result2.cpu().float()
    print(f"  result shape: {result2_cpu.shape}")
    print(f"  min={result2_cpu.min():.6e}, max={result2_cpu.max():.6e}")
    print(f"  expected=2.000000e-03")
    bad2 = (result2_cpu - 0.002).abs() > 0.0001
    print(f"  bad values: {bad2.sum().item()} / {result2_cpu.numel()}")

    if bad2.sum().item() > 0:
        print(f"\n  *** BUG: step 2 has wrong values ***")
        print(f"  first 20 bad: {result2_cpu[bad2][:20].tolist()}")
    else:
        print(f"\n  Step 2 correct — bug requires full model-scale memory pressure")


def test_compiled_with_grad_accum():
    """
    Closer to the real scenario: use actual backward pass with gradient accumulation.
    The key is that step 2's compiled graph has EXTRA inputs (accumulated grads).
    """
    device = xm.xla_device()

    # Small "router" model
    class MiniRouter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.randn(2880, 32, dtype=torch.bfloat16) * 0.01
            )

        def forward(self, x):
            # x: [128, 2880]
            logits = x @ self.weight           # [128, 32]
            topk_val, topk_idx = torch.topk(logits, k=4, dim=-1)  # [128, 4]
            softmax_val = torch.softmax(topk_val, dim=-1)

            # Scatter into routing_weights
            routing_weights = torch.zeros(128, 32, dtype=torch.bfloat16, device=x.device)
            routing_weights.scatter_(1, topk_idx, softmax_val)

            # The loss depends on routing_weights, so backward will compute
            # routing_weights.grad → scatter backward → the gather that triggers the bug
            return routing_weights.sum()

    model = MiniRouter().to(device)

    # Compile
    compiled_model = torch.compile(model, backend="tt")

    x = torch.randn(128, 2880, dtype=torch.bfloat16, device=device)

    # Step 1 backward
    print("=== Step 1 backward ===")
    loss1 = compiled_model(x)
    loss1.backward()
    torch_xla.sync(wait=True)
    grad1 = model.weight.grad.cpu().float()
    print(f"  weight.grad: min={grad1.min():.6e}, max={grad1.max():.6e}, norm={grad1.norm():.6e}")

    if torch.isinf(grad1).any() or torch.isnan(grad1).any():
        print(f"  *** INF/NAN in step 1 ***")
    else:
        print(f"  Step 1 clean")

    # Step 2 backward (gradient accumulation — don't zero grads)
    print("\n=== Step 2 backward ===")
    loss2 = compiled_model(x)
    loss2.backward()
    torch_xla.sync(wait=True)
    grad2 = model.weight.grad.cpu().float()
    print(f"  weight.grad: min={grad2.min():.6e}, max={grad2.max():.6e}, norm={grad2.norm():.6e}")

    if torch.isinf(grad2).any() or torch.isnan(grad2).any():
        print(f"  *** INF/NAN in step 2 — BUG REPRODUCED ***")
    else:
        print(f"  Step 2 clean — bug requires full model-scale memory pressure")


if __name__ == "__main__":
    print("=" * 60)
    print("TEST A: Direct reshape + index through torch.compile")
    print("=" * 60)
    test_compiled_reshape_embedding()

    print("\n" + "=" * 60)
    print("TEST B: Mini router with gradient accumulation")
    print("=" * 60)
    test_compiled_with_grad_accum()
