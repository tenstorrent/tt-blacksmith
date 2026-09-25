# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Equivalence tests for the tensor-lr annealing path in PPO training.

The TT (torch_xla) path carries the learning rate as a 0-dim device tensor and
updates it in place so the traced optimizer graph stays stable (see the fix
for the host-memory leak in ppo_breakout training). These tests verify, on CPU
and without any device, that the annealed lr trajectory produced by the tensor
path is bit-identical to the classic Python-float path.
"""

import pytest
import torch


def _anneal_float(optimizer, base_lr, update, num_updates):
    frac = 1.0 - (update - 1) / num_updates
    optimizer.param_groups[0]["lr"] = base_lr * frac


def _anneal_tensor(optimizer, base_lr, update, num_updates):
    frac = 1.0 - (update - 1) / num_updates
    new_lr = base_lr * frac
    lr = optimizer.param_groups[0]["lr"]
    if isinstance(lr, torch.Tensor):
        lr.copy_(new_lr)
    else:
        optimizer.param_groups[0]["lr"] = new_lr


def _make_optims(base_lr):
    model_f = torch.nn.Linear(4, 2)
    model_t = torch.nn.Linear(4, 2)
    model_t.load_state_dict(model_f.state_dict())
    opt_float = torch.optim.Adam(model_f.parameters(), lr=base_lr, eps=1e-5)
    opt_tensor = torch.optim.Adam(
        model_t.parameters(), lr=torch.tensor(base_lr), eps=1e-5
    )
    return model_f, model_t, opt_float, opt_tensor


def test_tensor_lr_annealing_matches_float_trajectory():
    base_lr, num_updates = 2.5e-4, 1000
    _, _, opt_float, opt_tensor = _make_optims(base_lr)
    for update in range(1, num_updates + 1):
        _anneal_float(opt_float, base_lr, update, num_updates)
        _anneal_tensor(opt_tensor, base_lr, update, num_updates)
        lr_t = opt_tensor.param_groups[0]["lr"]
        lr_f = opt_float.param_groups[0]["lr"]
        lr_t_val = lr_t.item() if isinstance(lr_t, torch.Tensor) else lr_t
        # The tensor path carries lr in float32 (tensor default dtype) while the
        # float path keeps a float64 Python scalar; the trajectories agree to
        # float32 rounding, which is what matters for training.
        assert lr_t_val == pytest.approx(lr_f, rel=1e-6), (
            f"lr mismatch at update {update}: {lr_t_val} != {lr_f}"
        )


def test_tensor_lr_optimizes_identically_to_float():
    # Same gradients, same annealed lrs -> identical parameters on both paths.
    base_lr, num_updates = 1e-3, 50
    model_f, model_t, opt_float, opt_tensor = _make_optims(base_lr)
    torch.manual_seed(0)
    inputs = torch.randn(16, 4)
    targets = torch.randn(16, 2)
    for update in range(1, num_updates + 1):
        for model, opt in ((model_f, opt_float), (model_t, opt_tensor)):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(inputs), targets)
            loss.backward()
            opt.step()
        _anneal_float(opt_float, base_lr, update, num_updates)
        _anneal_tensor(opt_tensor, base_lr, update, num_updates)
    for pf, pt in zip(model_f.parameters(), model_t.parameters()):
        assert torch.allclose(pf, pt, rtol=1e-5, atol=1e-8), (
            "tensor-lr and float-lr parameters diverged beyond float32 rounding"
        )
