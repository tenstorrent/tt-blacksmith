# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Minimal reproducer for scatter backward explosion on TT hardware.

Observed in GPT-OSS 20B partial-freeze finetuning at layer 15, accumulation step 2:
  routing_weights.grad (norm=1.54e-3, valid values) → scatter backward → -3.689e+19

Mirrors the exact operation in _debug_router_forward:
  router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
"""

import argparse
import sys

import torch

BATCH = 128
NUM_EXPERTS = 32
TOP_K = 4
HIDDEN = 2880
DTYPE = torch.bfloat16


def _stats(t: torch.Tensor) -> str:
    t = t.float()
    return (
        f"min={t.min().item():.6e}, max={t.max().item():.6e}, "
        f"norm={t.norm().item():.6e}, "
        f"inf={t.isinf().sum().item()}, nan={t.isnan().sum().item()}"
    )


def run_scatter_backward_steps(device: torch.device, seed: int = 42):
    """Run two gradient-accumulation steps of the router scatter, mimicking the training loop."""
    torch.manual_seed(seed)

    # Router weights — trainable, gradient ACCUMULATES across steps (no zero_grad),
    # exactly like partial-freeze finetuning.
    weight = torch.nn.Parameter(
        torch.randn(NUM_EXPERTS, HIDDEN, dtype=DTYPE, device=device) * 0.01
    )
    bias = torch.nn.Parameter(
        torch.randn(NUM_EXPERTS, dtype=DTYPE, device=device) * 0.01
    )

    print(f"\n{'='*60}")
    print(f"Device: {device}")
    print(f"Shapes: router_logits=[{BATCH},{NUM_EXPERTS}], topk_values=[{BATCH},{TOP_K}]")
    print(f"{'='*60}")

    for step in range(1, 3):
        torch.manual_seed(seed + step)  # different input batch each step

        # ---- Forward (mirrors _debug_router_forward exactly) ----
        hidden = torch.randn(BATCH, HIDDEN, dtype=DTYPE, device=device)
        router_logits = torch.nn.functional.linear(hidden, weight, bias)

        router_top_value_raw, router_indices = torch.topk(router_logits, TOP_K, dim=-1)
        router_indices = router_indices.clamp(0, NUM_EXPERTS - 1)

        # retain_grad on intermediate tensors (same as _keep() in _debug_router_forward)
        router_top_value_raw.retain_grad()

        router_top_value = torch.nn.functional.softmax(
            router_top_value_raw, dim=1, dtype=DTYPE
        )
        router_top_value.retain_grad()

        router_scores = torch.zeros_like(router_logits).scatter_(
            1, router_indices, router_top_value
        )
        router_scores.retain_grad()

        # ---- Downstream: loss whose d/d(router_scores) has norm ~ 1.54e-3 ----
        fake_weight = torch.randn(BATCH, NUM_EXPERTS, dtype=DTYPE, device=device) * 8e-4
        loss = (router_scores * fake_weight).sum()

        # ---- Backward — gradient accumulates on weight/bias (no zero_grad) ----
        loss.backward()

        if device.type == "xla":
            import torch_xla
            torch_xla.sync(wait=True)

        # ---- Report ----
        rw_grad = router_scores.grad
        tv_grad = router_top_value.grad
        idx_min = router_indices.min().item()
        idx_max = router_indices.max().item()

        print(f"\n--- Step {step} ---")
        print(f"  router_indices:              min={idx_min}, max={idx_max}, "
              f"oob={(router_indices<0).sum().item()+(router_indices>=NUM_EXPERTS).sum().item()}")
        if rw_grad is not None:
            print(f"  routing_weights.grad:        {_stats(rw_grad)}")
        else:
            print(f"  routing_weights.grad:        None")
        if tv_grad is not None:
            print(f"  topk_values_post_softmax.grad: {_stats(tv_grad)}")
            ok = tv_grad.abs().max().item() < 1.0
            print(f"  PASS={ok}  (expected |values| < 1.0, got max={tv_grad.abs().max().item():.3e})")
        else:
            print(f"  topk_values_post_softmax.grad: None")

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "tt"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.device == "tt":
        import torch_xla
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
    else:
        device = torch.device("cpu")

    run_scatter_backward_steps(device, seed=args.seed)


if __name__ == "__main__":
    main()
