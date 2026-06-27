# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import gc
import json
from pathlib import Path

import torch

from blacksmith.datasets.torch.diffusiondb_pixelart.diffusiondb_pixelart_dataset import (
    download_and_subset_dataset,
    pil_to_video_tensor,
    wan_latents_normalize,
)
from blacksmith.experiments.torch.wan2_2.configs import TrainingConfig
from blacksmith.models.torch.wan2_2.device import WanDeviceManager, wan_xla_compile_options
from blacksmith.models.torch.wan2_2.model_overrides import (
    UMT5Wrapper,
    VAEEncoderWrapper,
    apply_generality_overrides,
    apply_perf_overrides,
)


@torch.no_grad()
def precompute_latents_and_embeds(config: TrainingConfig, device_manager: WanDeviceManager):
    from diffusers import AutoencoderKLWan
    from transformers import AutoTokenizer, UMT5EncoderModel

    cache = Path(config.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    samples_dir = cache / "samples"
    samples_dir.mkdir(exist_ok=True)

    samples = download_and_subset_dataset(config)

    vae_dtype = config.vae_precompute_torch_dtype()
    vae = AutoencoderKLWan.from_pretrained(
        config.model_id, subfolder="vae", torch_dtype=vae_dtype, low_cpu_mem_usage=True
    ).eval()
    vae = device_manager.to_device(vae)
    device_manager.shard_model(vae)
    vae_enc_compiled = device_manager.compile(VAEEncoderWrapper(vae))

    metadata: list[dict] = []
    print(f"[precompute] VAE-encoding {len(samples)} images at {config.train_h}x{config.train_w} ...")
    for i, (img, caption) in enumerate(samples):
        video_cpu = pil_to_video_tensor(img, config.train_h, config.train_w).to(vae_dtype)
        latent = vae_enc_compiled(device_manager.to_device(video_cpu))
        latent = wan_latents_normalize(latent, vae)
        latent = latent.squeeze(0).contiguous().to("cpu")
        triggered = config.trigger + caption
        torch.save({"latent": latent, "caption": triggered}, samples_dir / f"sample_{i:04d}.pt")
        metadata.append({"idx": i, "caption": triggered})

    with open(cache / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    del vae_enc_compiled, vae
    gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        config.model_id, subfolder="text_encoder", torch_dtype=config.torch_dtype(), low_cpu_mem_usage=True
    ).eval()
    # The checkpoint stores only `shared.weight` and relies on weight-tying for
    # `encoder.embed_tokens.weight`; the pinned transformers version does not auto-tie.
    text_encoder.encoder.embed_tokens.weight = text_encoder.shared.weight
    text_encoder = device_manager.to_device(text_encoder)
    device_manager.shard_model(text_encoder)
    umt5_compiled = device_manager.compile(UMT5Wrapper(text_encoder))

    unique_captions = sorted({m["caption"] for m in metadata})
    if "" not in unique_captions:
        unique_captions.append("")
    embeds: dict[str, torch.Tensor] = {}
    print(f"[precompute] T5-encoding {len(unique_captions)} unique captions ...")
    max_seq = 512
    for cap in unique_captions:
        tok = tokenizer(cap, padding="max_length", truncation=True, max_length=max_seq, return_tensors="pt")
        input_ids = device_manager.to_device(tok.input_ids)
        attn_mask = device_manager.to_device(tok.attention_mask)
        out = umt5_compiled(input_ids, attn_mask)
        # Match WanPipeline.encode_prompt: zero out padding then keep full length.
        out = out * attn_mask.unsqueeze(-1).to(out.dtype)
        embeds[cap] = out.squeeze(0).to("cpu")

    torch.save(embeds, cache / "embeds.pt")
    del umt5_compiled, text_encoder, tokenizer
    gc.collect()
    print(f"[precompute] done. cache at {cache.resolve()}")


if __name__ == "__main__":
    import torch_xla

    from blacksmith.tools.cli import generate_config, parse_cli_options
    from blacksmith.tools.reproducibility_manager import ReproducibilityManager

    default_config = Path(__file__).parent / "lora" / "quietbox" / "wan2_2_ti2v_5b_diffusiondb.yaml"
    args = parse_cli_options(default_config=default_config)
    config: TrainingConfig = generate_config(TrainingConfig, args.config, args.test_config, args.test_checkpoint_path)

    ReproducibilityManager(config).setup()

    # Overrides must run before any diffusers/transformers model load.
    apply_generality_overrides()
    apply_perf_overrides()

    device_manager = WanDeviceManager(config)
    if config.use_tt:
        torch_xla.set_custom_compile_options(wan_xla_compile_options())

    precompute_latents_and_embeds(config, device_manager)
