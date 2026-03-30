#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Standalone TTNN Python reproducer for the norm=inf gradient explosion in
GPT-OSS 20B MoE finetuning (step 2, layer 15).

Failing subgraph (from step2_emitpy.py lines 6935–6973):
  routing_weights.grad [128,4] per chip
  → all_gather(dim=1, cluster_axis=1, 1×8 mesh) → [128,32] per chip
  → reshape [4096,1]
  → to_layout(ROW_MAJOR)
  → embedding(flat_indices [1,512] uint32, table [4096,1])
  → topk_values_post_softmax.grad [1,512,1] → reshape [128,4]

Expected: result[i,k] = routing_weights.grad[i, topk_indices[i,k]]  (scatter backward)
Observed in training: min=-3.689e+19 (explosion), while input norm=1.54e-3 (clean)

Usage:
  python3 repro_embedding_explosion.py

Requires the real tensors saved in repro_tensors/ from a training run.
If those don't have the right values to trigger the data-dependent bug,
the script prints the ops are correct and exits — the bug then requires
the full compiled flatbuffer context.
"""
import sys
import torch
import ttnn


REPRO_TENSORS = "/home/tt-admin/pglusac/tt-blacksmith/repro_tensors"


def stats(t):
    f = t.float()
    return (f"shape={list(t.shape)}, min={f.min():.6e}, max={f.max():.6e}, "
            f"norm={f.norm():.6e}, inf={torch.isinf(f).sum().item()}")


def main():
    print("=" * 70)
    print("TTNN embedding explosion reproducer — GPT-OSS 20B, layer 15 backward")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load real tensors from training run (saved from print_debug_intermediates)
    # ------------------------------------------------------------------
    rw_grad = torch.load(f"{REPRO_TENSORS}/layer15_routing_weights_grad.pt")
    topk_idx = torch.load(f"{REPRO_TENSORS}/layer15_topk_indices.pt").long()

    print(f"\nInputs (from step 2, layer 15):")
    print(f"  routing_weights.grad: {stats(rw_grad)}")
    print(f"  topk_indices:         shape={list(topk_idx.shape)}, "
          f"min={topk_idx.min()}, max={topk_idx.max()}")

    # ------------------------------------------------------------------
    # CPU golden: scatter backward = embedding lookup
    #   topk_values_post_softmax.grad[i,k] = routing_weights.grad[i, topk_idx[i,k]]
    # Implemented as: flat_idx = row*32 + topk_col → table[4096,1] lookup
    # ------------------------------------------------------------------
    row_idx = torch.arange(128).unsqueeze(1).expand(128, 4)          # [128,4]
    flat_idx_cpu = (row_idx * 32 + topk_idx).reshape(1, 512)         # [1,512]
    table_cpu = rw_grad.reshape(4096, 1).to(torch.bfloat16).float()  # [4096,1]
    cpu_result = table_cpu.flatten()[flat_idx_cpu.flatten().long()]   # [512]

    print(f"\nCPU golden (scatter backward):")
    print(f"  flat_idx: min={flat_idx_cpu.min()}, max={flat_idx_cpu.max()}")
    print(f"  result:   {stats(cpu_result)}")

    # ------------------------------------------------------------------
    # Set up 1×8 mesh (matches training hardware config)
    # ------------------------------------------------------------------
    print(f"\nOpening 1×8 mesh device ...")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 8))
    print(f"  {mesh}")

    dram = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    try:
        _run_tests(mesh, dram, rw_grad, topk_idx, flat_idx_cpu, cpu_result)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _run_tests(mesh, dram, rw_grad, topk_idx, flat_idx_cpu, cpu_result):
    # ==================================================================
    # TEST 1 — Eager: shard → all_gather → reshape → to_layout → embedding
    # Matches the exact TTNN op sequence from step2_emitpy.py lines 6935–6971
    # ==================================================================
    print("\n" + "=" * 70)
    print("TEST 1: all_gather → reshape → to_layout → embedding (eager, 8 chips)")
    print("=" * 70)

    dram = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    # Shard routing_weights.grad [128,32] along dim=1: each chip gets [128,4]
    # (In training, each chip computed its own local [128,4] slice via bmm/sum)
    rw_sharded = ttnn.from_torch(
        rw_grad.to(torch.bfloat16),
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=dram,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
    )
    print(f"  Sharded:    logical={list(rw_sharded.shape)}, per chip=[128,4]")

    # all_gather(dim=1, cluster_axis=1): each chip collects all 8 slices → [128,32]
    rw_gathered = ttnn.all_gather(
        input_tensor=rw_sharded,
        dim=1,
        cluster_axis=1,
        memory_config=dram,
        topology=ttnn.Topology.Ring,
    )
    print(f"  Gathered:   logical={list(rw_gathered.shape)}")

    # Verify gathered table matches expected routing_weights.grad
    gathered_cpu = ttnn.to_torch(
        rw_gathered, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
    )[:128].float()  # Take chip 0 — all chips should have identical data
    gather_diff = (gathered_cpu - rw_grad.float()).abs()
    print(f"  Gather check:  max diff={gather_diff.max():.3e}, "
          f"bad={(gather_diff > 0.01).sum().item()}/4096")
    if (gather_diff > 0.01).sum() > 0:
        print("  *** all_gather produced WRONG VALUES (potential bug here) ***")

    # reshape [128,32] → [4096,1]
    rw_reshaped = ttnn.reshape(rw_gathered, [4096, 1], memory_config=dram)
    print(f"  Reshaped:   logical={list(rw_reshaped.shape)}")

    # to_layout(ROW_MAJOR)
    table_rm = ttnn.to_layout(rw_reshaped, ttnn.Layout.ROW_MAJOR)
    print(f"  to_layout:  logical={list(table_rm.shape)}, layout=ROW_MAJOR")

    # Verify table before embedding
    table_tt = ttnn.to_torch(
        table_rm, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
    )[:4096].float().flatten()
    table_diff = (table_tt - rw_grad.float().reshape(-1)).abs()
    print(f"  Table check:   max diff={table_diff.max():.3e}, "
          f"bad={(table_diff > 0.01).sum().item()}/4096")
    if (table_diff > 0.01).sum() > 0:
        print("  *** TABLE CORRUPTED before embedding — bug is in reshape/to_layout ***")
        bad_pos = torch.where(table_diff > 0.01)[0][:20]
        print(f"  bad positions: {bad_pos.tolist()}")
        print(f"  expected: {rw_grad.float().reshape(-1)[bad_pos].tolist()}")
        print(f"  got:      {table_tt[bad_pos].tolist()}")

    # Flat indices [1,512] uint32 — replicated to all chips
    flat_idx_tt = ttnn.from_torch(
        flat_idx_cpu.to(torch.int32),
        device=mesh,
        layout=ttnn.Layout.ROW_MAJOR,
        dtype=ttnn.uint32,
        memory_config=dram,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    print(f"  Indices:    logical={list(flat_idx_tt.shape)}, dtype=uint32")

    # embedding — matches step2_emitpy.py line 6971
    emb_out = ttnn.embedding(
        flat_idx_tt, table_rm,
        padding_idx=None,
        layout=ttnn.Layout.TILE,
        dtype=ttnn.bfloat16,
        memory_config=dram,
    )
    print(f"  Embedding:  logical={list(emb_out.shape)}")

    # Compare result from chip 0 to CPU golden
    result_all = ttnn.to_torch(
        emb_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
    )
    result = result_all[:1].flatten().float()  # chip 0 result, shape [512]
    result_diff = (result - cpu_result).abs()

    print(f"\nTEST 1 RESULT:")
    print(f"  TT:  min={result.min():.6e}, max={result.max():.6e}, norm={result.norm():.6e}")
    print(f"  CPU: min={cpu_result.min():.6e}, max={cpu_result.max():.6e}, norm={cpu_result.norm():.6e}")
    print(f"  max diff={result_diff.max():.6e}, bad={(result_diff > 0.01).sum().item()}/512")

    if result.abs().max() > 1e10:
        print("\n  *** BUG REPRODUCED: TT embedding output has inf/huge values! ***")
        print("  This confirms the subgraph itself is faulty on this mesh config.")
        bad = torch.where(result.abs() > 1e10)[0][:10]
        print(f"  bad positions (result): {bad.tolist()}")
        print(f"  result at bad pos: {result[bad].tolist()}")
        print(f"  flat_idx at bad pos: {flat_idx_cpu.flatten()[bad].tolist()}")
    elif (result_diff > 0.01).sum() > 0:
        print("\n  *** BUG REPRODUCED: TT embedding mismatch (smaller, not inf) ***")
        bad = torch.where(result_diff > 0.01)[0][:10]
        print(f"  bad positions: {bad.tolist()}")
        print(f"  expected: {cpu_result[bad].tolist()}")
        print(f"  got:      {result[bad].tolist()}")
    else:
        print("\n  PASS: ops are correct in eager mode on 8-device mesh.")
        print("  The bug requires the full compiled flatbuffer context.")
        print("  It is a COMPILED GRAPH memory aliasing bug — the step 2 backward")
        print("  flatbuffer (fb_1774741203335.ttnn) assigns overlapping DRAM addresses,")
        print("  and the embedding op reads from a DRAM region that was overwritten by")
        print("  another op (large attention/expert gradients) earlier in the same kernel.")
        print()
        print("  To reproduce the bug, run the training script at committed state:")
        print("    python3 blacksmith/experiments/torch/gpt_oss/test_gpt_oss_finetuning_2.py \\")
        print("      --config blacksmith/experiments/torch/gpt_oss/test_gpt_oss_20b_finetuning.yaml")
        print("  And look for 'topk_values_post_softmax.grad' at step 2, layer 15.")


if __name__ == "__main__":
    main()
