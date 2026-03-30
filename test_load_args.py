"""
Load the exported step 2 input tensors and replay the router backward
embedding sequence eagerly with ttnn ops. Compare each intermediate
with a CPU golden reference to find which op is faulty.

Tensor mapping (from exported tensorbin shapes):
  arg340.tensorbin = 128x32 bf16 = accumulated routing_weights.grad
  arg403.tensorbin = 128x4 si32 = topk indices for layer 15
"""
import torch
import ttnn

def load_arg(path):
    t = ttnn.load_tensor(path)
    try:
        return ttnn.to_torch(t)
    except RuntimeError:
        # Multi-device tensor — try different concat dims
        for dim in range(len(t.shape)):
            try:
                composer = ttnn.ConcatMeshToTensor(dim=dim)
                return ttnn.to_torch(t, mesh_composer=composer)
            except Exception:
                continue
        # Last resort: get individual device tensors
        tensors = ttnn.get_device_tensors(t)
        return ttnn.to_torch(tensors[0])  # Just take first shard

def main():
    tensors_dir = "/home/tt-admin/pglusac/tt-blacksmith/irs_debug/tensors"

    print("Loading exported step 2 tensors...")
    # arg340 = accumulated routing_weights.grad (128x32 bf16)
    rw_grad = load_arg(f"{tensors_dir}/arg340.tensorbin")
    # Try multiple candidates for topk indices
    for argn in [349, 403, 350, 404]:
        t = load_arg(f"{tensors_dir}/arg{argn}.tensorbin")
        if t.shape == torch.Size([128, 4]) and t.dtype == torch.int32:
            vals = t.unique()
            if vals.min() >= 0 and vals.max() <= 31:
                print(f"  arg{argn}: {t.shape} {t.dtype} min={t.min()} max={t.max()} → topk indices")
                topk_idx = t
                break
    else:
        print("Could not find topk indices tensor, using arg403")
        topk_idx = load_arg(f"{tensors_dir}/arg403.tensorbin")

    print(f"\nrouting_weights.grad: shape={rw_grad.shape}, dtype={rw_grad.dtype}")
    rw_f = rw_grad.float()
    print(f"  min={rw_f.min():.6e}, max={rw_f.max():.6e}, norm={rw_f.norm():.6e}")
    print(f"topk_indices: shape={topk_idx.shape}, dtype={topk_idx.dtype}")
    print(f"  min={topk_idx.min()}, max={topk_idx.max()}, unique={topk_idx.unique().tolist()}")

    # Compute flat indices: row*32 + topk_col
    row_idx = torch.arange(128).unsqueeze(1).expand(128, 4)
    flat_idx = (row_idx * 32 + topk_idx).reshape(1, 512).to(torch.int32)
    print(f"\nFlat indices: min={flat_idx.min()}, max={flat_idx.max()}")

    # CPU golden
    table_cpu = rw_grad.reshape(4096, 1).to(torch.bfloat16).flatten().float()
    cpu_result = table_cpu[flat_idx.flatten().long()]
    print(f"CPU embedding result: min={cpu_result.min():.6e}, max={cpu_result.max():.6e}")

    device = ttnn.open_device(device_id=0)

    # ============================================================
    # Test 1: Direct embedding with pre-built row-major table
    # ============================================================
    print("\n" + "=" * 60)
    print("Test 1: Direct embedding (table already row-major)")
    print("=" * 60)
    table_rm = ttnn.from_torch(rw_grad.reshape(4096,1).to(torch.bfloat16),
                                device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    idx_tt = ttnn.from_torch(flat_idx, device=device, dtype=ttnn.uint32)
    emb1 = ttnn.embedding(idx_tt, table_rm)
    r1 = ttnn.to_torch(emb1).flatten().float()
    d1 = (r1 - cpu_result).abs()
    print(f"  TT: min={r1.min():.6e}, max={r1.max():.6e}")
    print(f"  max diff={d1.max():.6e}, bad={( d1>0.01).sum().item()}/{len(d1)}")

    # ============================================================
    # Test 2: Full on-device sequence matching TTNN IR
    #   tiled [128,32] → reshape [4096,1] → to_layout(row_major) → embedding
    # ============================================================
    print("\n" + "=" * 60)
    print("Test 2: Full on-device reshape→to_layout→embedding")
    print("=" * 60)
    src_tt = ttnn.from_torch(rw_grad.to(torch.bfloat16), device=device,
                              layout=ttnn.TILE_LAYOUT)
    reshaped = ttnn.reshape(src_tt, [4096, 1])
    table_rm2 = ttnn.to_layout(reshaped, ttnn.ROW_MAJOR_LAYOUT)
    # Check table before embedding
    table_check = ttnn.to_torch(table_rm2).flatten().float()
    td = (table_check - table_cpu).abs()
    print(f"  Table check: max diff={td.max():.6e}, bad={(td>0.01).sum().item()}/{len(td)}")
    if (td > 0.01).sum() > 0:
        bad_pos = torch.where(td > 0.01)[0][:20]
        print(f"  *** TABLE CORRUPTED at {bad_pos.tolist()}")
        print(f"  expected: {table_cpu[bad_pos].tolist()}")
        print(f"  got:      {table_check[bad_pos].tolist()}")

    emb2 = ttnn.embedding(idx_tt, table_rm2)
    r2 = ttnn.to_torch(emb2).flatten().float()
    d2 = (r2 - cpu_result).abs()
    print(f"  TT embedding: min={r2.min():.6e}, max={r2.max():.6e}")
    print(f"  max diff={d2.max():.6e}, bad={(d2>0.01).sum().item()}/{len(d2)}")

    # ============================================================
    # Test 3: Full index computation on device (matching TTNN IR)
    #   topk_idx[128,4] si32 → typecast u32 → reshape [128,4,1]
    #   → concat with row_indices [128,4,1] → typecast f32
    #   → matmul with [[32],[1]] → reshape [1,512]
    #   → typecast u32 → to_layout row_major
    # ============================================================
    print("\n" + "=" * 60)
    print("Test 3: Full index computation on device")
    print("=" * 60)

    # Row indices: arange(0,128) repeated 4 times → [128,4,1] u32
    row_const = torch.arange(128, dtype=torch.int32).unsqueeze(1).expand(128,4).unsqueeze(2)
    row_tt = ttnn.from_torch(row_const, device=device, layout=ttnn.ROW_MAJOR_LAYOUT,
                              dtype=ttnn.uint32)

    topk_tt = ttnn.from_torch(topk_idx, device=device, layout=ttnn.TILE_LAYOUT)
    topk_u32 = ttnn.typecast(topk_tt, ttnn.uint32)
    topk_reshaped = ttnn.reshape(topk_u32, [128, 4, 1])

    concat_tt = ttnn.concat([row_tt, topk_reshaped], 2)
    concat_f32 = ttnn.typecast(concat_tt, ttnn.float32)

    stride = torch.tensor([[32.0], [1.0]], dtype=torch.float32)
    stride_tt = ttnn.from_torch(stride, device=device, layout=ttnn.TILE_LAYOUT)
    flat_tt = ttnn.matmul(concat_f32, stride_tt)

    flat_reshaped = ttnn.reshape(flat_tt, [1, 512])
    flat_u32 = ttnn.typecast(flat_reshaped, ttnn.uint32)
    flat_rm = ttnn.to_layout(flat_u32, ttnn.ROW_MAJOR_LAYOUT)

    # Check computed indices
    idx_check = ttnn.to_torch(flat_rm).flatten()
    idx_expected = flat_idx.flatten().to(torch.int32)
    idx_diff = (idx_check.int() - idx_expected).abs()
    print(f"  Computed indices: min={idx_check.min()}, max={idx_check.max()}")
    print(f"  idx max diff={idx_diff.max()}, bad={(idx_diff>0).sum().item()}/{len(idx_diff)}")
    if (idx_diff > 0).sum() > 0:
        bad_idx = torch.where(idx_diff > 0)[0][:20]
        print(f"  *** INDEX MISMATCH at {bad_idx.tolist()}")
        print(f"  expected: {idx_expected[bad_idx].tolist()}")
        print(f"  got:      {idx_check[bad_idx].int().tolist()}")

    # Full embedding with computed indices
    emb3 = ttnn.embedding(flat_rm, table_rm2)
    r3 = ttnn.to_torch(emb3).flatten().float()
    d3 = (r3 - cpu_result).abs()
    print(f"  TT embedding: min={r3.min():.6e}, max={r3.max():.6e}")
    print(f"  max diff={d3.max():.6e}, bad={(d3>0.01).sum().item()}/{len(d3)}")

    if (d3 > 0.01).sum() > 0:
        print("\n  *** BUG FOUND IN EAGER EXECUTION ***")
    else:
        print("\n  All tests pass in eager mode.")
        print("  Bug is in COMPILED graph execution, not individual ops.")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
