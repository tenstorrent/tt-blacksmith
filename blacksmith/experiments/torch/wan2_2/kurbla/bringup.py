# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Single-device bring-up harness for the Wan 2.2 DiT on tt-kurbla.

Runs the smallest interesting slice of the Wan2.2 TI2V-5B experiment -- an
`N`-layer `WanTransformer3DModel` (default 1) at a tiny resolution, on one chip,
no mesh -- and reports where it breaks. Written for iteration, not for training:
weights are random by default (`--pretrained` pulls the real 5B checkpoint) so the
graph shape is faithful while the run stays seconds-long.

    python -m blacksmith.experiments.torch.wan2_2.kurbla.bringup --layers 1
    python -m blacksmith.experiments.torch.wan2_2.kurbla.bringup --layers 1 --mode compile --backward
    python -m blacksmith.experiments.torch.wan2_2.kurbla.bringup --stage embed   # sub-module only

`--stage` isolates one piece of the block so a failure points at an op instead of
"the model". Stages are cumulative in spirit, not nested: each builds only what it needs.
"""
from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from typing import Optional, Tuple

import torch
import torch.nn as nn

# Wan2.2 TI2V-5B `transformer/config.json`, verbatim except `num_layers`, which the
# CLI overrides. Kept inline so bring-up needs no network access.
WAN22_TI2V_5B_DIT_CONFIG = {
    "added_kv_proj_dim": None,
    "attention_head_dim": 128,
    "cross_attn_norm": True,
    "eps": 1e-06,
    "ffn_dim": 14336,
    "freq_dim": 256,
    "image_dim": None,
    "in_channels": 48,
    "num_attention_heads": 24,
    "num_layers": 30,
    "out_channels": 48,
    "patch_size": [1, 2, 2],
    "pos_embed_seq_len": None,
    "qk_norm": "rms_norm_across_heads",
    "rope_max_seq_len": 1024,
    "text_dim": 4096,
}

# Wan2.2 TI2V-5B `vae/config.json`, verbatim except the long latents_mean/latents_std
# lists, which only matter for latent normalisation (not for the graph shape).
WAN22_TI2V_5B_VAE_CONFIG = {
    "attn_scales": [],
    "base_dim": 160,
    "clip_output": False,
    "decoder_base_dim": 256,
    "dim_mult": [1, 2, 4, 4],
    "dropout": 0.0,
    "in_channels": 12,
    "is_residual": True,
    "num_res_blocks": 2,
    "out_channels": 12,
    "patch_size": 2,
    "scale_factor_spatial": 16,
    "scale_factor_temporal": 4,
    "temperal_downsample": [False, True, True],
    "z_dim": 48,
}

_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layers", type=int, default=1, help="DiT blocks to instantiate (30 = full model).")
    p.add_argument("--h", type=int, default=64, help="Pixel height; latent height is h // 16.")
    p.add_argument("--w", type=int, default=64, help="Pixel width; latent width is w // 16.")
    p.add_argument("--frames", type=int, default=1, help="Pixel frames (4k+1); latent frames is (frames - 1) // 4 + 1.")
    p.add_argument("--text-len", type=int, default=32, help="Text sequence length (512 in the real pipeline).")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--dtype", choices=sorted(_DTYPES), default="bfloat16")
    p.add_argument("--mode", choices=("eager", "compile"), default="eager")
    p.add_argument(
        "--stage",
        default="dit",
        choices=("embed", "attn", "block", "dit", "vae-encode", "vae-decode"),
        help="What to run. The vae-* stages build the VAE instead of the transformer.",
    )
    p.add_argument("--pretrained", action="store_true", help="Load the real checkpoint (~10 GB download).")
    p.add_argument("--lora", action="store_true", help="Wrap the DiT in the experiment's LoRA adapters.")
    p.add_argument("--backward", action="store_true", help="Also run a flow-matching loss + backward.")
    p.add_argument("--device", default="tt")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Make any CPU fallback raise, so a PASS means the graph really ran on device. "
        "Without this, unimplemented ops silently execute on CPU and still produce a good pcc.",
    )
    p.add_argument("--iters", type=int, default=1, help="Forward reps (>1 exercises the compile cache).")
    p.add_argument("--no-overrides", action="store_true", help="Skip ALL patches, shared and kurbla (A/B a failure).")
    p.add_argument(
        "--no-shared-overrides",
        action="store_true",
        help="Skip only the shared blacksmith patches, keeping the tt-kurbla ones (A/B whether the "
        "tt-xla-era patches still earn their place).",
    )
    p.add_argument("--tolerance", type=float, default=0.98, help="Minimum PCC vs the CPU reference.")
    p.add_argument("--no-check", action="store_true", help="Skip the CPU reference comparison.")
    return p.parse_args(argv)


def compare(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, str]:
    """Compare a device result `a` against a CPU reference `b`.

    Returns `(score, note)` where `score` is in [-1, 1] and 1.0 is perfect.

    Pearson correlation is the metric tt tooling uses, but on its own it is unsafe
    here: correlation is undefined when either side has zero variance, and reporting
    1.0 for that case turns "the device returned a constant" -- the exact symptom of
    a dropped write -- into a perfect pass. So a degenerate (near-constant) side is
    detected first and scored by relative error against the reference instead.
    """
    a = a.detach().flatten().to(torch.float32)
    b = b.detach().flatten().to(torch.float32)
    a_std, b_std = a.std().item(), b.std().item()

    # Constant on either side: correlation says nothing. Score by relative error.
    if a_std <= 1e-12 or b_std <= 1e-12:
        scale = max(b.abs().max().item(), 1e-12)
        rel = (a - b).abs().max().item() / scale
        note = f"DEGENERATE (a_std={a_std:.2e}, b_std={b_std:.2e}); scored by rel err={rel:.2e}"
        return (1.0 - rel if rel < 1.0 else 0.0), note

    ac, bc = a - a.mean(), b - b.mean()
    return (ac @ bc / (ac.norm() * bc.norm())).item(), ""


def build_dit(args) -> nn.Module:
    from diffusers import WanTransformer3DModel

    dtype = _DTYPES[args.dtype]
    if args.pretrained:
        # `num_layers` is honoured by from_pretrained: diffusers instantiates from the
        # (overridden) config and the missing blocks simply stay unloaded.
        model = WanTransformer3DModel.from_pretrained(
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            subfolder="transformer",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            num_layers=args.layers,
            ignore_mismatched_sizes=True,
        )
    else:
        config = dict(WAN22_TI2V_5B_DIT_CONFIG, num_layers=args.layers)
        model = WanTransformer3DModel.from_config(config).to(dtype)
    return model.eval()


def build_vae(args) -> nn.Module:
    from diffusers import AutoencoderKLWan

    dtype = _DTYPES[args.dtype]
    if args.pretrained:
        return AutoencoderKLWan.from_pretrained(
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers", subfolder="vae", torch_dtype=dtype, low_cpu_mem_usage=True
        ).eval()
    return AutoencoderKLWan.from_config(WAN22_TI2V_5B_VAE_CONFIG).to(dtype).eval()


def make_vae_inputs(args, vae) -> dict:
    """A pixel-space video for the encoder, or a latent for the decoder."""
    dtype = _DTYPES[args.dtype]
    generator = torch.Generator().manual_seed(0)
    if args.stage == "vae-encode":
        video = torch.randn(args.batch, 3, args.frames, args.h, args.w, generator=generator)
        return {"x": video.clamp(-1, 1).to(dtype)}
    lat_t = (args.frames - 1) // vae.config.scale_factor_temporal + 1
    lat_h = args.h // vae.config.scale_factor_spatial
    lat_w = args.w // vae.config.scale_factor_spatial
    latent = torch.randn(args.batch, vae.config.z_dim, lat_t, lat_h, lat_w, generator=generator)
    return {"z": latent.to(dtype)}


def make_inputs(args, model) -> dict:
    """Latent + text-embed + timestep batch matching `flow_matching_step` in train.py."""
    dtype = _DTYPES[args.dtype]
    vae_spatial, vae_temporal = 16, 4
    lat_h, lat_w = args.h // vae_spatial, args.w // vae_spatial
    lat_t = (args.frames - 1) // vae_temporal + 1
    in_ch = model.config.in_channels
    text_dim = model.config.text_dim

    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(args.batch, in_ch, lat_t, lat_h, lat_w, generator=generator).to(dtype)
    text = torch.randn(args.batch, args.text_len, text_dim, generator=generator).to(dtype)
    timestep = torch.full((args.batch,), 500, dtype=torch.int64)
    return {"hidden_states": hidden, "timestep": timestep, "encoder_hidden_states": text}


class _EmbedOnly(nn.Module):
    """patch_embedding + condition embedder: the pre-block prologue of the DiT."""

    def __init__(self, dit):
        super().__init__()
        self.patch_embedding = dit.patch_embedding
        self.condition_embedder = dit.condition_embedder

    def forward(self, hidden_states, timestep, encoder_hidden_states):
        hidden_states = self.patch_embedding(hidden_states).flatten(2).transpose(1, 2)
        temb, timestep_proj, encoder_hidden_states, _ = self.condition_embedder(
            timestep, encoder_hidden_states, None
        )
        return hidden_states + temb.unsqueeze(1) + encoder_hidden_states.sum(1, keepdim=True)


class _AttnOnly(nn.Module):
    """Block 0's self-attention on an already-patchified token sequence."""

    def __init__(self, dit):
        super().__init__()
        self.patch_embedding = dit.patch_embedding
        self.attn = dit.blocks[0].attn1
        self.norm = dit.blocks[0].norm1

    def forward(self, hidden_states, timestep, encoder_hidden_states):
        tokens = self.patch_embedding(hidden_states).flatten(2).transpose(1, 2)
        return self.attn(self.norm(tokens))


class _BlockOnly(nn.Module):
    """Block 0 end to end, fed the real prologue outputs."""

    def __init__(self, dit):
        super().__init__()
        self.dit = dit

    def forward(self, hidden_states, timestep, encoder_hidden_states):
        dit = self.dit
        tokens = dit.patch_embedding(hidden_states).flatten(2).transpose(1, 2)
        temb, timestep_proj, encoder_hidden_states, _ = dit.condition_embedder(
            timestep, encoder_hidden_states, None
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))
        rotary_emb = dit.rope(hidden_states)
        return dit.blocks[0](tokens, encoder_hidden_states, timestep_proj, rotary_emb)


class _DitWrapper(nn.Module):
    """Strip the diffusers output object so dynamo sees a plain tensor."""

    def __init__(self, dit):
        super().__init__()
        self.dit = dit

    def forward(self, hidden_states, timestep, encoder_hidden_states):
        return self.dit(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]


def _vae_stages():
    # Reuse the experiment's own wrappers rather than redefining them: they are what
    # precompute.py / generate.py compile, so a bring-up failure here is a real failure.
    from blacksmith.models.torch.wan2_2.model_overrides import VAEDecoderWrapper, VAEEncoderWrapper

    return {"vae-encode": VAEEncoderWrapper, "vae-decode": VAEDecoderWrapper}


_STAGES = {"embed": _EmbedOnly, "attn": _AttnOnly, "block": _BlockOnly, "dit": _DitWrapper}


def main(argv=None) -> int:
    args = parse_args(argv)

    if not args.no_overrides and not args.no_shared_overrides:
        from blacksmith.models.torch.wan2_2.model_overrides import (
            apply_generality_overrides,
            apply_perf_overrides,
        )

        apply_generality_overrides()
        apply_perf_overrides()

    is_vae = args.stage.startswith("vae")
    base = build_vae(args) if is_vae else build_dit(args)
    if not args.no_overrides and args.device != "cpu":
        from blacksmith.experiments.torch.wan2_2.kurbla.model_overrides import apply_kurbla_overrides

        print(f"[bringup] kurbla rewrites: {apply_kurbla_overrides(base)}")
    inputs = make_vae_inputs(args, base) if is_vae else make_inputs(args, base)
    module = (_vae_stages() if is_vae else _STAGES)[args.stage](base)

    if args.lora:
        from peft import LoraConfig

        assert not is_vae, "--lora targets the DiT attention/FFN projections; it does not apply to the VAE"
        for p in base.parameters():
            p.requires_grad_(False)
        base.add_adapter(
            LoraConfig(
                r=32,
                lora_alpha=32,
                target_modules=["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
                lora_dropout=0.0,
                init_lora_weights="gaussian",
            )
        )

    params = sum(p.numel() for p in module.parameters())
    print(
        f"[bringup] stage={args.stage} layers={args.layers} params={params / 1e6:.1f}M "
        f"dtype={args.dtype} mode={args.mode} device={args.device}"
    )
    for name, tensor in inputs.items():
        print(f"[bringup]   {name}: {tuple(tensor.shape)} {tensor.dtype}")

    reference: Optional[torch.Tensor] = None
    if not args.no_check:
        with torch.no_grad():
            reference = module(**inputs)
        print(f"[bringup] cpu reference: {tuple(reference.shape)} {reference.dtype}")

    if args.device != "cpu":
        import tt_kurbla.torch  # noqa: F401  — registers the "tt" device and dynamo backend

        print(f"[bringup] chips={torch.tt.num_chips()} mesh={torch.tt.mesh_shape()}")

    module = module.to(args.device)
    inputs = {k: v.to(args.device) for k, v in inputs.items()}
    if args.mode == "compile":
        module = torch.compile(module, backend="tt")

    # `strict_no_fallback` makes the catch-all CPU fallback raise instead of running, so
    # "PASS" cannot be satisfied by ops that quietly ran on the host.
    if args.strict and args.device != "cpu":
        from tt_kurbla.torch.testing import strict_no_fallback

        fallback_guard = strict_no_fallback()
    else:
        fallback_guard = nullcontext()

    out = None
    with fallback_guard:
        for i in range(args.iters):
            start = time.perf_counter()
            with torch.no_grad() if not args.backward else torch.enable_grad():
                out = module(**inputs)
            elapsed = time.perf_counter() - start
            print(f"[bringup] forward {i}: {tuple(out.shape)} {out.dtype} in {elapsed:.2f}s")

    if args.backward:
        target = torch.zeros_like(out)
        loss = torch.nn.functional.mse_loss(out.float(), target.float())
        loss.backward()
        grads = [p for p in module.parameters() if p.grad is not None]
        print(f"[bringup] backward ok: loss={loss.item():.6f}, {len(grads)} params with grad")

    if reference is not None:
        score, note = compare(out.cpu().float(), reference.float())
        status = "PASS" if score >= args.tolerance else "FAIL"
        print(f"[bringup] pcc={score:.5f} vs cpu (tolerance {args.tolerance}) -> {status}"
              + (f"  [{note}]" if note else ""))
        return 0 if score >= args.tolerance else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
