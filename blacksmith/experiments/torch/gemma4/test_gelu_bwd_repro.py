# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Minimal MulGelu repro — exact mirror of the PCC-diagnostic pattern in
``test_gemma4_e2b_fine_tuning_pure_torch.py``.

Module under test: ``out = GELU(a * b)``. We hook ``mul`` and ``act_fn`` the
same way the main script hooks every ``Gemma4TextDecoderLayer`` (and the
intra-layer submodules in ``INTRA_LAYER_SUFFIXES``): forward hook captures
the submodule's output and calls ``retain_grad()`` from inside the hook.

Loss is ``(out * out).sum()`` so the gradient flowing into ``out`` is
``2 * out`` — a non-constant, output-dependent ``grad_out`` that exposes
the GELU-backward kernel to a realistic spread of input-gradient magnitudes
(rather than the trivial all-1s case from ``out.sum()``). ``mul.grad`` is
then ``grad_out * GELU'(c) = 2 * GELU(c) * GELU'(c)`` — also non-constant,
so its PCC is well-defined (no degenerate-zero-std tensors).

NOTE on autograd: under ``torch.compile(backend="tt")`` the backend only
compiles the forward by default. Backward then falls into the normal
autograd dispatcher and hits ``xla::mark_tensor`` (the op TT inserts on
every input via ``tt.mark_argument``), which has no autograd kernel
registered — PyTorch warns ``xla::mark_tensor: an autograd kernel was
not registered ... silently incorrect behavior`` and every grad becomes
zero (confirmed even for matmul-bwd — it's not op-specific). We avoid
this by passing ``tt_use_aot_autograd=True`` in the compile options:
AOTAutograd traces forward+backward into a single FX graph and compiles
both into TT, so the backward never goes through the dispatcher and
``mark_tensor`` is irrelevant.

``a`` and ``b`` are stored as ``nn.Parameter``; ``forward`` also takes a
``requires_grad=True`` ``trigger`` input (sanity-check: its gradient should
be non-zero on TT after the AOT fix)."""
import os

# TT env vars must be set BEFORE the torch_xla import. Mirrors
# ``blacksmith.tools.device_manager.DeviceManager._setup_tt_environment`` for
# the multichip path -- ``test_gemma4_e2b_wizardlm.yaml`` configures a 1x8
# mesh, which triggers the extra ``XLA_ALWAYS_ALLREDUCE`` /
# ``CONVERT_SHLO_TO_SHARDY`` / ``DISABLE_NUMERIC_CC_TOKEN`` exports plus
# ``xr.use_spmd()``. We hardcode the same 1x8 mesh here so the compiled
# program matches the runtime device count (the alternative -- a 1x1 mesh
# without SPMD -- silently produces an 8-device program on a multichip box
# and fails with ``Device count mismatch: 1 vs 8`` at flatbuffer load).
os.environ.setdefault("PJRT_DEVICE", "TT")
os.environ.setdefault("XLA_STABLEHLO_COMPILE", "1")
os.environ.setdefault("XLA_ALWAYS_ALLREDUCE", "1")
os.environ.setdefault("CONVERT_SHLO_TO_SHARDY", "1")
os.environ.setdefault("DISABLE_NUMERIC_CC_TOKEN", "1")

import copy

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from transformers.activations import ACT2FN

# Mesh: 1x8 over ("batch", "model"), mirroring
# ``test_gemma4_e2b_wizardlm.yaml`` (``mesh_shape: [1, 8]``,
# ``mesh_axis_names: ["batch", "model"]``). The "batch" axis is size-1 and
# unused for sharding -- it just keeps the mesh 2D to match the underlying
# TT topology. The MulGelu parameters stay fully replicated (no sharding
# patterns), exactly like LoRA-A / norms in the gemma4 recipe.
_MESH_SHAPE = (1, 8)
_MESH_AXIS_NAMES = ("batch", "model")


# ---------------------------------------------------------------------------
# Mirrors test_gemma4_e2b_fine_tuning_pure_torch.py::_to_host / _pcc / _max_abs
# ---------------------------------------------------------------------------
def _to_host(t):
    """Same as the main script: ``.detach().cpu()``, no on-device cast."""
    if t is None:
        return None
    return t.detach().cpu()


def _pcc(a, b):
    af = a.flatten().to(torch.float64)
    bf = b.flatten().to(torch.float64)
    da = af - af.mean()
    db = bf - bf.mean()
    denom = (da.norm() * db.norm()).item()
    if denom == 0.0:
        return float("nan")
    return float((da @ db).item() / denom)


def _cossim(a, b):
    af = a.flatten().to(torch.float64)
    bf = b.flatten().to(torch.float64)
    denom = (af.norm() * bf.norm()).item()
    if denom == 0.0:
        return float("nan")
    return float((af @ bf).item() / denom)


def _max_abs(a, b):
    return float((a.to(torch.float64) - b.to(torch.float64)).abs().max().item())


# ---------------------------------------------------------------------------
# Module under test
# ---------------------------------------------------------------------------
class _Mul(torch.nn.Module):
    """``c = a * b`` wrapped as an nn.Module so a forward hook can capture
    its output (mirrors how every projection inside ``Gemma4MLP`` is its
    own nn.Module that the main script hooks)."""

    def forward(self, a, b):
        return a * b


class MulGelu(torch.nn.Module):
    """``out = GELU(a * b)`` — Gemma-4 SwiGLU's GELU-side stand-in.

    ``a`` and ``b`` live as ``nn.Parameter`` so they accumulate ``.grad``
    via the param path. ``forward`` also takes a ``trigger`` input with
    ``requires_grad=True`` — this gives Dynamo / AOTAutograd a clearly
    differentiable input at the function boundary."""

    def __init__(self, a_init: torch.Tensor, b_init: torch.Tensor, hidden_activation: str = "gelu_pytorch_tanh"):
        super().__init__()
        self.mul = _Mul()
        self.act_fn = ACT2FN[hidden_activation]
        self.a = torch.nn.Parameter(a_init.clone())
        self.b = torch.nn.Parameter(b_init.clone())

    def forward(self, trigger):
        # ``trigger`` is a scalar with requires_grad=True initialised to 0.0
        # so it does not perturb the GELU-bwd numerics we care about, but
        # injects a real differentiable input into the compiled function.
        a = self.a + trigger
        c = self.mul(a, self.b)
        out = self.act_fn(c)
        return out


# ---------------------------------------------------------------------------
# Mirrors test_gemma4_e2b_fine_tuning_pure_torch.py::training_step_inner
# ---------------------------------------------------------------------------
def training_step_inner(model, trigger):
    out = model(trigger)
    # Funky loss: ``(out * out).sum()`` ⇒ ``grad_out = 2 * out``. Non-constant,
    # so every downstream backward tensor has real variance and PCC is
    # well-defined.
    loss = (out * out).sum()
    loss.backward()
    return loss.detach(), out


# ---------------------------------------------------------------------------
# Mirrors _run_step_capturing_layers EXACTLY: hook on each named submodule,
# retain_grad inside the hook, run forward+backward, return ``captured``.
# ---------------------------------------------------------------------------
INTRA_SUFFIXES = ("mul", "act_fn")


def _run_step_capturing_layers(model, trigger):
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    captured = {}
    handles = []

    def _make_hook(name):
        def _hook(_mod, _inputs, output):
            t = output if isinstance(output, torch.Tensor) else output[0]
            if t.requires_grad:
                t.retain_grad()
            captured[name] = t

        return _hook

    for suffix in INTRA_SUFFIXES:
        try:
            sub = inner.get_submodule(suffix)
        except AttributeError:
            continue
        handles.append(sub.register_forward_hook(_make_hook(suffix)))

    try:
        loss, out = training_step_inner(model, trigger)
    finally:
        for h in handles:
            h.remove()

    return captured, loss, out


# ---------------------------------------------------------------------------
# Mirrors get_model(..., return_cpu_twin=True): build the module ONCE on
# CPU, deepcopy as the CPU twin, then move-to-device + torch.compile the TT
# version. Identical numerical state on both sides.
# ---------------------------------------------------------------------------
def get_model(a_init, b_init, device, dtype, hidden_activation, *, use_tt: bool):
    base = MulGelu(a_init, b_init, hidden_activation)
    base.to(dtype)

    cpu_twin = copy.deepcopy(base)

    base.to(device)
    if use_tt:
        # ``tt_use_aot_autograd=True`` makes the ``tt`` backend trace
        # forward+backward via AOTAutograd into a single FX graph and compile
        # both into TT. Without it, ``torch.compile(backend="tt")`` only
        # compiles the forward; backward falls into the normal autograd
        # dispatcher and hits ``xla::mark_tensor`` (no autograd kernel
        # registered) → silent zero gradients (confirmed for matmul-bwd
        # too — not GELU-specific). See
        # python_package/tt_torch/backend/backend.py for the option.
        compile_options = {
            "tt_enable_torch_fx_fusion_pass": False,
            "tt_legacy_compile": True,
            "tt_use_aot_autograd": False,
        }
        base = torch.compile(base, backend="tt", options=compile_options)

    return base, cpu_twin


# ---------------------------------------------------------------------------
# Mirrors _run_pcc_diagnostic: run on both, sync, materialize all tensors in
# one burst, then print a single contiguous PCC table.
# ---------------------------------------------------------------------------
def _run_pcc_diagnostic(tt_model, cpu_twin, trigger_tt, trigger_cpu, *, use_tt: bool):
    print("[gelu-bwd-repro] Running CPU twin forward+backward (reference)...")
    cpu_capt, cpu_loss, cpu_out = _run_step_capturing_layers(cpu_twin, trigger_cpu)

    print("[gelu-bwd-repro] Running TT forward+backward...")
    tt_capt, tt_loss, tt_out = _run_step_capturing_layers(tt_model, trigger_tt)
    if use_tt:
        torch_xla.sync(wait=True)

    print("[gelu-bwd-repro] materializing tensors to host...")
    cpu_loss_h = _to_host(cpu_loss)
    tt_loss_h = _to_host(tt_loss)
    cpu_out_h = _to_host(cpu_out)
    tt_out_h = _to_host(tt_out)

    rows = []
    for suffix in INTRA_SUFFIXES:
        ct = cpu_capt.get(suffix)
        tt = tt_capt.get(suffix)
        if ct is None or tt is None:
            rows.append((suffix, None, None, None, None))
            continue
        rows.append((
            suffix,
            _to_host(ct),
            _to_host(tt),
            _to_host(ct.grad) if ct.grad is not None else None,
            _to_host(tt.grad) if tt.grad is not None else None,
        ))

    cpu_inner = cpu_twin._orig_mod if hasattr(cpu_twin, "_orig_mod") else cpu_twin
    tt_inner = tt_model._orig_mod if hasattr(tt_model, "_orig_mod") else tt_model
    a_cpu_grad_h = _to_host(cpu_inner.a.grad)
    b_cpu_grad_h = _to_host(cpu_inner.b.grad)
    a_tt_grad_h = _to_host(tt_inner.a.grad)
    b_tt_grad_h = _to_host(tt_inner.b.grad)
    trigger_cpu_grad_h = _to_host(trigger_cpu.grad)
    trigger_tt_grad_h = _to_host(trigger_tt.grad)
    print(
        f"[trigger.grad] cpu={None if trigger_cpu_grad_h is None else trigger_cpu_grad_h.tolist()}  "
        f"tt={None if trigger_tt_grad_h is None else trigger_tt_grad_h.tolist()}"
    )

    print("=" * 96)
    print(
        f"[PCC] loss   TT={tt_loss_h.item():.6f}   CPU={cpu_loss_h.item():.6f}   "
        f"d={tt_loss_h.item() - cpu_loss_h.item():+.4e}"
    )
    print(
        f"[PCC] out    fwd PCC={_pcc(cpu_out_h, tt_out_h):.6f}  "
        f"max|d|={_max_abs(cpu_out_h, tt_out_h):.4e}"
    )

    print("")
    print(
        f"{'submodule':<12} | {'fwd PCC':>10} | {'fwd cos':>10} | {'fwd max|d|':>12} | "
        f"{'grad PCC':>10} | {'grad cos':>10} | {'grad max|d|':>13}"
    )
    print("-" * 96)
    for suffix, ct_h, tt_h, cg_h, tg_h in rows:
        if ct_h is None or tt_h is None:
            print(f"{suffix:<12} | (missing)")
            continue
        f_pcc = _pcc(ct_h, tt_h)
        f_cos = _cossim(ct_h, tt_h)
        f_mae = _max_abs(ct_h, tt_h)
        if cg_h is not None and tg_h is not None:
            g_pcc_s = f"{_pcc(cg_h, tg_h):>10.6f}"
            g_cos_s = f"{_cossim(cg_h, tg_h):>10.6f}"
            g_mae_s = f"{_max_abs(cg_h, tg_h):>13.4e}"
        else:
            g_pcc_s = f"{'(no grad)':>10}"
            g_cos_s = f"{'-':>10}"
            g_mae_s = f"{'-':>13}"
        print(
            f"{suffix:<12} | {f_pcc:>10.6f} | {f_cos:>10.6f} | {f_mae:>12.4e} | "
            f"{g_pcc_s} | {g_cos_s} | {g_mae_s}"
        )

    # Leaf-input grads (mul-bwd outputs into a, b)
    def _row(name, ct, tt):
        if ct is None or tt is None:
            print(f"{name:<12} | (cpu={ct is not None} tt={tt is not None})")
            return
        print(
            f"{name:<12} | {'-':>10} | {'-':>10} | {'-':>12} | "
            f"{_pcc(ct, tt):>10.6f} | {_cossim(ct, tt):>10.6f} | "
            f"{_max_abs(ct, tt):>13.4e}"
        )
    _row("a.grad", a_cpu_grad_h, a_tt_grad_h)
    _row("b.grad", b_cpu_grad_h, b_tt_grad_h)
    print("=" * 96)


def main():
    # Multichip / SPMD setup -- mirrors ``DeviceManager._setup_tt_environment``
    # + ``DeviceManager._create_mesh``. ``xr.use_spmd()`` must be called
    # before any device tensor is materialised.
    xr.set_device_type("TT")
    xr.use_spmd()

    num_devices = xr.global_runtime_device_count()
    assert num_devices == _MESH_SHAPE[0] * _MESH_SHAPE[1], (
        f"Mesh shape {_MESH_SHAPE} requires {_MESH_SHAPE[0] * _MESH_SHAPE[1]} "
        f"devices, but ``global_runtime_device_count() = {num_devices}``."
    )
    mesh = xs.Mesh(
        device_ids=np.array(range(num_devices)),
        mesh_shape=_MESH_SHAPE,
        axis_names=_MESH_AXIS_NAMES,
    )
    print(f"[gelu-bwd-repro] mesh={_MESH_SHAPE} axes={_MESH_AXIS_NAMES} num_devices={num_devices}")

    torch_xla.set_custom_compile_options({"fp32_dest_acc_en": True, "math_fidelity": "hifi4"})

    B, S, I = 1, 1024, 8192
    dtype = torch.bfloat16
    hidden_activation = "gelu_pytorch_tanh"
    device = torch_xla.device()

    torch.manual_seed(0)
    a_init = torch.randn(B, S, I).to(dtype)
    b_init = torch.randn(B, S, I).to(dtype)

    print(f"[gelu-bwd-repro] shape={tuple(a_init.shape)} dtype={dtype} act={hidden_activation}")

    tt_model, cpu_twin = get_model(a_init, b_init, device, dtype, hidden_activation, use_tt=True)

    # ``trigger`` is a scalar tensor with requires_grad=True, value 0.0 → does
    # not perturb numerics but gives the compiled function an explicitly
    # differentiable input. See ``MulGelu`` docstring.
    trigger_tt = torch.zeros(1, dtype=dtype, device=device, requires_grad=True)
    trigger_cpu = torch.zeros(1, dtype=dtype, device="cpu", requires_grad=True)

    _run_pcc_diagnostic(
        tt_model=tt_model,
        cpu_twin=cpu_twin,
        trigger_tt=trigger_tt,
        trigger_cpu=trigger_cpu,
        use_tt=True,
    )


if __name__ == "__main__":
    main()
