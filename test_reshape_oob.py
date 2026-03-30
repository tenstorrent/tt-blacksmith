"""
Reproducer for ttnn.reshape zero-copy OOB bug.

Strategy: pollute DRAM with large values, free them, then do the reshape.
If reshape is zero-copy, the OOB reads will pick up the polluted values.
"""

import torch
import ttnn

def run_test():
    device = ttnn.open_device(device_id=0)

    print("=" * 60)
    print("TEST 1: Clean DRAM (baseline)")
    print("=" * 60)
    test_reshape(device, pollute=False)

    print("\n" + "=" * 60)
    print("TEST 2: Polluted DRAM (simulates step 2)")
    print("=" * 60)
    test_reshape(device, pollute=True)

    print("\n" + "=" * 60)
    print("TEST 3: Polluted DRAM + deallocate source before to_layout")
    print("=" * 60)
    test_reshape_dealloc(device)

    ttnn.close_device(device)


def test_reshape(device, pollute=False):
    if pollute:
        # Allocate and free large tensors to fill DRAM with large values
        # Use value -3.69e+19 to match the observed explosion
        polluters = []
        for i in range(20):
            big = torch.full((256, 32), fill_value=-3.69e+19, dtype=torch.bfloat16)
            tt_big = ttnn.from_torch(big, device=device, layout=ttnn.TILE_LAYOUT)
            polluters.append(tt_big)
        # Free them — DRAM now contains -3.69e+19 in many regions
        for p in polluters:
            ttnn.deallocate(p)
        print("  DRAM polluted with 20 x [256x32] tensors of -3.69e+19, then freed")

    # Create the [128, 32] tensor with all 1.0
    torch_input = torch.ones(128, 32, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, device=device, layout=ttnn.TILE_LAYOUT)

    # Reshape [128, 32] -> [4096, 1]
    tt_reshaped = ttnn.reshape(tt_input, [4096, 1])

    # Convert to row-major
    tt_row_major = ttnn.to_layout(tt_reshaped, ttnn.ROW_MAJOR_LAYOUT)

    # Read back
    result = ttnn.to_torch(tt_row_major).flatten().float()
    check_result(result)

    ttnn.deallocate(tt_row_major)
    ttnn.deallocate(tt_reshaped)
    ttnn.deallocate(tt_input)


def test_reshape_dealloc(device):
    """
    Match the exact IR pattern: reshape then deallocate source BEFORE to_layout.
    This is what the TTNN IR does:
      %2139 = reshape(%2138)
      deallocate(%2138)          <-- source freed here
      ... (other ops)
      %2145 = to_layout(%2139)   <-- reads from %2139 (view of freed %2138?)
    """
    # Pollute DRAM first
    polluters = []
    for i in range(20):
        big = torch.full((256, 32), fill_value=-3.69e+19, dtype=torch.bfloat16)
        tt_big = ttnn.from_torch(big, device=device, layout=ttnn.TILE_LAYOUT)
        polluters.append(tt_big)
    for p in polluters:
        ttnn.deallocate(p)
    print("  DRAM polluted, then freed")

    # Create [128, 32] tensor
    torch_input = torch.ones(128, 32, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, device=device, layout=ttnn.TILE_LAYOUT)

    # Reshape [128, 32] -> [4096, 1]
    tt_reshaped = ttnn.reshape(tt_input, [4096, 1])

    # DEALLOCATE THE SOURCE — matching the IR pattern
    ttnn.deallocate(tt_input)
    print("  Source tensor deallocated after reshape (matches IR pattern)")

    # Allocate some small tensors to potentially reuse the freed buffer
    fillers = []
    for i in range(5):
        small = torch.full((128, 4), fill_value=9.99e+18, dtype=torch.bfloat16)
        tt_small = ttnn.from_torch(small, device=device, layout=ttnn.TILE_LAYOUT)
        fillers.append(tt_small)
    print(f"  Allocated {len(fillers)} small tensors to pressure allocator")

    # NOW do to_layout on the reshaped tensor
    tt_row_major = ttnn.to_layout(tt_reshaped, ttnn.ROW_MAJOR_LAYOUT)

    result = ttnn.to_torch(tt_row_major).flatten().float()
    check_result(result)

    for f in fillers:
        ttnn.deallocate(f)
    ttnn.deallocate(tt_row_major)
    ttnn.deallocate(tt_reshaped)


def check_result(result):
    valid = result[:128]    # positions 0-127
    oob = result[128:]      # positions 128-4095

    print(f"\n  VALID region (pos 0-127):")
    print(f"    min={valid.min():.6e}, max={valid.max():.6e}")
    valid_ok = torch.allclose(valid, torch.ones_like(valid))
    print(f"    all_ones={valid_ok}")

    print(f"  SUSPECT OOB region (pos 128-4095):")
    print(f"    min={oob.min():.6e}, max={oob.max():.6e}, norm={oob.norm():.6e}")
    nonone = (oob - 1.0).abs() > 0.01
    nonone_count = nonone.sum().item()
    print(f"    non-1.0 count: {nonone_count} / {len(oob)}")

    if nonone_count > 0:
        vals = oob[nonone]
        print(f"    first 20 bad values: {vals[:20].tolist()}")
        print(f"    *** BUG CONFIRMED: OOB reads returned garbage ***")
    else:
        print(f"    All values correct (1.0)")


if __name__ == "__main__":
    run_test()
