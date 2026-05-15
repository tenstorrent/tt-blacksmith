# `ttnn.embedding` chunked-tilize bug — hypothesis writeup

> **Status:** unverified hypothesis based on reading the tt-metal source.
> Findings below are derived from inspecting `embedding.cpp`, `embedding_device_operation.cpp`,
> `embeddings_fused_program_factory.cpp`, `embeddings_tilize.cpp`, and `tilize_chunked.cpp`
> at the version of tt-metal vendored inside `tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/`.
> The hypothesis explains every empirically observed pass/fail combination we tested
> so far, but it has **not been confirmed with a watcher trace or a one-line code patch**.

## TL;DR

`ttnn.embedding` produces incorrect results for `(VOCAB, HIDDEN) = (262144, 8960)` bf16
weights gathered for `SEQ = 32` token ids, on Blackhole P150. The op runs the
`EmbeddingsFusedProgramFactory` reader (`embeddings_tilize.cpp`) + the
`tilize_chunked.cpp` compute kernel. We believe the bug is that those two kernels
disagree on the per-chunk byte width of the shared L1 intermediate when
`num_tiles_per_block` (i.e. `HIDDEN / TILE_WIDTH`) is not a multiple of
`tiles_per_chunk = 64`.

- Reader uses `weight_chunk_size = weight_block_size / num_chunks` per token per chunk
  (equal-split). For `HIDDEN = 8960` that's `17920 / 5 = 3584` bytes.
- Compute reads the same L1 buffer with implicit row stride
  `tiles_per_chunk * TILE_WIDTH * sizeof(bf16) = 64 * 32 * 2 = 4096` bytes.
- Same bytes, two different strides → token rows in L1 don't line up with the rows
  compute reads, so every output tile that goes back to DRAM `output` is scrambled.

The exact same op works for:

- `HIDDEN = 8192, SEQ = 32` — `num_tiles_per_block = 256`, the program factory's
  L1 budget check `required_memory_bytes > max_l1_budget_bytes` is **false**
  (`1 MiB > 1 MiB` is false because of the strict `>`), so `use_chunked_processing = false`
  and the buggy chunked path is skipped.
- `HIDDEN = 8960, SEQ = 16` — `SEQ % TILE_HEIGHT != 0` so `fused_tilized = false` in
  `ttnn::operations::embedding::EmbeddingOperation::invoke`, and the op falls back
  to `EmbeddingsRMProgramFactory` / `embeddings.cpp` which is non-chunked.

## Reproducer

```python
import ttnn, torch

VOCAB = 262144
HIDDEN = 8960       # also fails for 8224
SEQ    = 32         # SEQ = 16 works (different kernel)

with ttnn.manage_device(device_id=0) as device:
    weight_torch = 0.01 * torch.randn((VOCAB, HIDDEN), dtype=torch.float32)
    ids_torch = torch.full((1, SEQ), 1, dtype=torch.int32)

    weight = ttnn.from_torch(
        weight_torch.to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
    )
    ids = ttnn.from_torch(
        ids_torch,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        dtype=ttnn.uint32,
    )

    out_tt = ttnn.to_torch(ttnn.embedding(ids, weight)).to(torch.float32)

    weight_back = ttnn.to_torch(weight).to(torch.float32)
    out_torch = torch.nn.functional.embedding(ids_torch.long(), weight_back)

    diff = (out_tt - out_torch).abs()
    print("max abs diff:", diff.max().item())
```

`test_embed_minrepro_ttnn.py` in this directory is a more verbose form of the
above. Note the file ships with `SEQ = 16` (which happens to pass on this op
version because it picks a different program factory); the original failure that
motivated the file is `SEQ = 32` with the same weight shape.

Observed (Blackhole P150):

| `HIDDEN` | `SEQ` | Program factory chosen | `num_tiles_per_block` | `tiles_per_chunk` | `num_chunks` | Result |
|---------:|------:|-----------------------|----------------------:|-----------------:|-------------:|--------|
| 8192     | 16    | RM (`embeddings.cpp`)             | n/a | n/a | n/a | OK     |
| 8224     | 16    | RM (`embeddings.cpp`)             | n/a | n/a | n/a | (failed in earlier testing — re-verify; if it now passes, RM path is fine) |
| 8960     | 16    | RM (`embeddings.cpp`)             | n/a | n/a | n/a | OK     |
| 8192     | 32    | Fused (no chunking)               | 256 | 256 | 1   | OK     |
| 8224     | 32    | Fused + chunked                   | 257 | 64  | 5   | FAIL   |
| 8960     | 32    | Fused + chunked                   | 280 | 64  | 5   | FAIL   |

## Dispatch trail

1. **Python**: `ttnn.embedding(ids, weight)` with `weight.layout == TILE_LAYOUT`.
2. **`EmbeddingOperation::invoke`** (`ttnn/cpp/ttnn/operations/embedding/embedding.cpp`):
   - converts `weight` to `ROW_MAJOR_LAYOUT` in DRAM (line `if (mutable_weight.layout() == ttnn::TILE_LAYOUT) { mutable_weight = ttnn::to_layout(...); }`).
   - sets `fused_tilized = true` iff
     `input_tensor.padded_shape()[-1] % TILE_HEIGHT == 0`
     **and** `weight.padded_shape()[-1] % TILE_WIDTH == 0`
     **and** the weight was originally `TILE_LAYOUT` (or a `TILE_LAYOUT`
     output is requested).
     For `SEQ = 32, HIDDEN = 8960` both are multiples of 32, weight came in tiled
     → `fused_tilized = true`.
3. **`EmbeddingsDeviceOperation::select_program_factory`**:
   - `input.layout() == ROW_MAJOR` and `operation_attributes.tilized == true`
     → returns `EmbeddingsFusedProgramFactory`.
4. **`EmbeddingsFusedProgramFactory::create`** decides chunking:
   - `num_tiles_per_block = weights.padded_shape()[-1] / TILE_WIDTH = 8960 / 32 = 280`
   - `weights_single_tile_size = 32 * 32 * 2 = 2048 B`
   - `required_memory_bytes = 2 * 280 * 2048 = 1,146,880 B`
   - `max_l1_budget_bytes = 1 MiB = 1,048,576 B`
   - `use_chunked_processing = required_memory_bytes > max_l1_budget_bytes = true`
   - therefore: `tiles_per_chunk = min(280, max_double_buffer_tiles = 64) = 64`,
     `num_chunks = ceil(280 / 64) = 5`,
     `buffering = 2`.
   - reader = `embeddings_tilize.cpp`,
     compute = `tilize_chunked.cpp` (because `use_chunked_processing` is true).

For `HIDDEN = 8192` step 4 instead gives `required_memory_bytes = 1,048,576 B`
which is **not strictly greater** than `1 MiB`, so `use_chunked_processing = false`,
`num_chunks = 1`, and the per-token bf16 row is read with one NOC read of
`16384 B`. That branch works.

## The actual bug

### Reader kernel (`embeddings_tilize.cpp`)

```cpp
for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
    cb_reserve_back(cb_id_in0, tiles_per_chunk);
    uint32_t l1_write_addr = get_write_ptr(cb_id_in0);

    // ↓↓↓ THIS IS THE BUG ↓↓↓
    uint32_t weight_chunk_size   = weight_block_size / num_chunks; // 17920 / 5 = 3584
    uint32_t weight_chunk_offset = chunk * weight_chunk_size;       // chunk * 3584

    for (uint32_t k = 0; k < tile_height; ++k) {       // 32 tokens
        input_token_t token = input_l1_ptr[k];
        uint64_t src_noc_addr = get_token_noc_addr(token, weights);
        noc_async_read(src_noc_addr + weight_chunk_offset, l1_write_addr, weight_chunk_size);
        l1_write_addr += weight_chunk_size;            // advances by 3584
    }
    noc_async_read_barrier();
    cb_push_back(cb_id_in0, tiles_per_chunk);          // publishes 64 pages
}
```

This **uniformly splits** the per-token bf16 row (`weight_block_size = HIDDEN * 2 = 17,920` B)
into `num_chunks = 5` equal slices of `3584` B and uses that single number both
as the per-NOC-read size **and** as the L1 row stride.

### Compute kernel (`tilize_chunked.cpp`)

```cpp
for (uint32_t b = 0; b < per_core_block_cnt; ++b) {
    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        cb_wait_front(cb_id_in0, tiles_per_chunk);
        cb_reserve_back(cb_id_out0, tiles_per_chunk);
        tilize_block(cb_id_in0, tiles_per_chunk, cb_id_out0);
        cb_push_back(cb_id_out0, tiles_per_chunk);
        cb_pop_front(cb_id_in0, tiles_per_chunk);
    }
}
```

`tilize_block(in, tiles_per_chunk, out)` internally reads from `cb_in0` as a
`TILE_HEIGHT × (tiles_per_chunk * TILE_WIDTH)` bf16 row-major rectangle, i.e.:

- **row count** = `TILE_HEIGHT = 32` (hardcoded)
- **row stride** = `tiles_per_chunk * TILE_WIDTH * sizeof(bf16) = 64 * 32 * 2 = 4096 B`

It then produces `tiles_per_chunk = 64` tiles into `cb_out0`.

### The mismatch

The two kernels share the L1 region pointed to by `cb_in0`. For `HIDDEN = 8960`:

```
   In one chunk page (131,072 B in L1):

   READER writes 32 rows × 3584 B per row =  114,688 B used,  16,384 B left stale.
   COMPUTE reads 32 rows × 4096 B per row =  131,072 B (the whole chunk page).

   Reader's row-k starts at L1 offset k * 3584.
   Compute's row-k starts at L1 offset k * 4096.

   They only coincide for k = 0; from k = 1 onward, every "row k" that compute
   reads is a mix of the tail of the reader's row k and the head of row k+1.
```

Additionally, the reader's `weight_chunk_offset = chunk * 3584` picks the wrong
starting byte within each token's DRAM row for every chunk ≥ 1:

| Chunk | Correct DRAM byte range (cols `[c*2048, (c+1)*2048)` of each weight row) | Reader's actual DRAM byte range |
|-------|------------------------------------------------------------------------|---------------------------------|
| 0 | `[0, 4096)` | `[0, 3584)` |
| 1 | `[4096, 8192)` | `[3584, 7168)` |
| 2 | `[8192, 12288)` | `[7168, 10752)` |
| 3 | `[12288, 16384)` | `[10752, 14336)` |
| 4 | `[16384, 17920)` (only **1536 B**, last chunk is short) | `[14336, 17920)` |

So even ignoring the L1 stride mismatch, every chunk past chunk 0 also reads a
shifted slice of columns from `weight`.

### Output corruption

The writer (`writer_unary_interleaved_start_id.cpp`) is told to write exactly
`num_tiles_per_block * local_num_blocks = 280` tiles. It dutifully pops the first
280 tiles from `cb_out0` and writes them to DRAM `output` at tile indices `0..279`.
All 280 tiles contain `tilize_block`-rearranged garbage produced from the
misaligned L1 intermediate, so the entire `output` tensor is wrong.

Compute also pushes 40 extra tiles (`5 * 64 - 280 = 40`) into `cb_out0` that the
writer never consumes. They're produced from the trailing 16384 B of stale L1 +
the wrong final-chunk byte range — but they don't show up in `output`. That is a
**second** bug (no special-case handling for the trailing short chunk) but it
doesn't actually corrupt result bytes today; it just wastes L1 and TRISC cycles.

## Why each combination passes or fails

- `HIDDEN = 8192, SEQ = 32` → strict `>` in `use_chunked_processing` means the
  block fits in the L1 budget exactly, no chunking, no kernel-mismatch.
- `HIDDEN = 8960, SEQ = 16` → `fused_tilized = false`, falls into
  `EmbeddingsRMProgramFactory` / `embeddings.cpp`, which reads the whole bf16 row
  per token in one go (`noc_async_read<weight_stick_size>`). No chunking, no
  mismatch.
- `HIDDEN ∈ {8224, 8960}, SEQ = 32` → enters chunked branch with
  `num_tiles_per_block ≠ k * tiles_per_chunk`. Reader's per-chunk size
  (`weight_block_size / num_chunks`) is neither `4096` (for the full chunks) nor
  `1536` (for the trailing chunk); compute reads at the fixed `4096` stride; rows
  misalign; result is wrong.

## Expected fix shape

The reader needs to compute its per-chunk byte width from `tiles_per_chunk`
(and pass the actual trailing-chunk tile count down to the compute kernel),
not from `weight_block_size / num_chunks`. Sketch:

```cpp
constexpr uint32_t chunk_row_bytes_full = tiles_per_chunk * 32 * sizeof(uint16_t);
                                                  // bf16 hardcoded; 4096 B for tpc=64
for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
    uint32_t tiles_this_chunk =
        (chunk == num_chunks - 1)
            ? (num_tiles_per_block - chunk * tiles_per_chunk)   // 24 for HIDDEN=8960
            : tiles_per_chunk;
    uint32_t weight_chunk_size   = tiles_this_chunk * 32 * sizeof(uint16_t);
    uint32_t weight_chunk_offset = chunk * chunk_row_bytes_full;

    cb_reserve_back(cb_id_in0, tiles_this_chunk);
    uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
    for (uint32_t k = 0; k < tile_height; ++k) {
        ...
        noc_async_read(src_noc_addr + weight_chunk_offset, l1_write_addr, weight_chunk_size);
        l1_write_addr += weight_chunk_size;
    }
    noc_async_read_barrier();
    cb_push_back(cb_id_in0, tiles_this_chunk);
}
```

with a corresponding change in `tilize_chunked.cpp` to take the per-chunk tile
count as a runtime argument (or a compile-time array) so that `tilize_block`
processes 24 tiles for the trailing chunk instead of 64, and a matching tweak in
`embeddings_fused_program_factory.cpp` to push that runtime arg.

## Workarounds for callers (without patching tt-metal)

1. **Force the RM path** by passing `layout=ttnn.ROW_MAJOR_LAYOUT` to
   `ttnn.embedding`, then `ttnn.to_layout(..., TILE_LAYOUT)` afterwards. This
   side-steps the chunked tilize reader entirely.
2. **Slice the gather along `HIDDEN`** in Python: do five separate
   `ttnn.embedding` calls on column-slices of `weight` that are each ≤ `8192`
   wide, then concatenate. Each call avoids the chunked branch. This is what
   we're using as a stopgap.
3. **Slice the gather along `SEQ`** to keep each call at `SEQ < TILE_HEIGHT`
   so that `fused_tilized` is false; the RM kernel handles arbitrary `HIDDEN`.
   Less ergonomic.

## Suggested upstream filing

If this hypothesis is confirmed, a tt-metal issue should mention:

- Affected op: `ttnn.embedding` with `weight_arg.layout == TILE_LAYOUT` (or
  `layout=ttnn.TILE_LAYOUT` requested), `SEQ` a multiple of `TILE_HEIGHT`, and
  `HIDDEN / TILE_WIDTH` greater than `max_double_buffer_tiles = 64` and not a
  multiple of `64`.
- Files to patch:
  - `ttnn/cpp/ttnn/operations/embedding/device/kernels/dataflow/embeddings_tilize.cpp`
  - `ttnn/cpp/ttnn/operations/embedding/device/kernels/compute/tilize_chunked.cpp`
  - `ttnn/cpp/ttnn/operations/embedding/device/embeddings_fused_program_factory.cpp`
- Sibling bug for reference: `#37171` (sharded layernorm gather reads) — same
  family of "reader/compute disagree about chunking boundary inside an L1 CB"
  pattern.

## Caveats / things to verify

- We have not run a tt-metal watcher (`TT_METAL_WATCHER=1`) trace on the failing
  case. Doing so would confirm or rule out an alternative (NOC packet) hypothesis
  and would give a concrete data point on what `tilize_block` sees.
- We have not yet bisected on `HIDDEN` to confirm the boundary is exactly the
  L1-budget threshold (`HIDDEN > 8192` for bf16 / 1 MiB budget). Trying
  e.g. `HIDDEN ∈ {8193, 8200, 8224, 8256, 8320, ..., 8960}` should show all of
  them failing with the same symptom (every chunk past chunk 0 corrupted).
- We have not tried `dtype=ttnn.bfloat8_b` for the weight; the tile size and
  thus the boundary will shift.
- We have not exercised the sharded-output variant
  (`output_sharded == true` in the program factory). The math there uses
  `weight_block_size = shard_spec.shape[1] * 2` instead of the full row width,
  so the bug may or may not manifest depending on shard shape.
