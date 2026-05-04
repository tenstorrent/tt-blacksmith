# Gemma 1.1 2B Trace Enablement

This document captures the modifications made to apply the
`enable-trace-training` skill to the Gemma 1.1 2B SST-2 LoRA finetuning workload.

## Summary of changes

| File | Change |
| --- | --- |
| `blacksmith/experiments/torch/gemma11/configs.py` | Added `enable_trace: bool = False` and `trace_region_size_mb: int = 200` fields to `TrainingConfig`. |
| `blacksmith/models/torch/huggingface/hf_models.py` | Added `make_static_causal_mask_4d()` helper. In `get_model()`, conditionally call `torch_xla.set_custom_compile_options({"enable_trace": "true"})` *before* `torch.compile` when `config.enable_trace` is true. |
| `blacksmith/tools/device_manager.py` | In `_setup_tt_environment()`, set `TT_RUNTIME_TRACE_REGION_SIZE = trace_region_size_mb * 1_000_000` when trace is enabled. |
| `blacksmith/experiments/torch/gemma11/train.py` | Pre-build `mask_4d` and `position_ids` on device when trace is enabled. Pass them to the HF model in both training and validation forward calls. Use `torch_xla.sync(wait=False)` between backward and optimizer when trace is enabled. |
| `blacksmith/experiments/torch/gemma11/single_chip/gemma11_sst2_trace.yaml` | New override config that turns on trace, picks a 300 MB trace region, disables checkpointing/W&B, and limits validation. |

The optimizer change (`AdamW(..., capturable=True)`) was already present in
`train.py` as the default, so no edit was required there.

## How to run

```bash
source env/activate --xla
python3 blacksmith/experiments/torch/gemma11/train.py \
    --config blacksmith/experiments/torch/gemma11/single_chip/gemma11_sst2.yaml \
    --test-config blacksmith/experiments/torch/gemma11/single_chip/gemma11_sst2_trace.yaml
```

## Verification checklist (per skill)

1. No `TT_FATAL: Host tensor has different shape` errors during training.
2. No `TT_FATAL: Creating trace buffers ... but only ... is allocated` errors.
3. Compiled-graph count stabilizes after the first 1-2 steps (use
   `torch_xla.debug.metrics.metrics_report()` to confirm).
4. Per-step wall-clock drops vs. baseline once trace replay kicks in.

## Environment status at time of writing

The verification run could not complete during this session because the
device at PCI BDF `0000:b1:00.0` (logical chip 0) entered an unresponsive
state. `dmesg` shows:

```
tenstorrent 0000:b1:00.0: Device is unresponsive, cannot reset.
```

`tt-smi -r 0,1`, `tt-smi -r 0000:b1:00.0`, and `tt-smi --use_luwen -r ...`
all reported "Secondary bus reset not completed". This requires a host
power cycle to recover; once the chip is back, re-running the command
above should exercise all of the changes and produce the expected
trace-enabled training behavior.

The trigger appears to be a host-side `lto1: internal compiler error:
Segmentation fault` from `riscv-tt-elf-g++` while building the BRISC /
NCRISC / TRISC firmware kernels. After clearing
`~/.cache/tt-metal-cache`, the LTO crash repeated and left the chip in
the wedged state above. This compiler crash is in tt-metal's JIT firmware
build path, not in the trace enablement code, so the modifications above
should be unaffected once the device is recovered.
