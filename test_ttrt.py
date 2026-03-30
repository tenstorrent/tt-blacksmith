"""
Inspect the step 2 flatbuffer to find embedding op indices,
then list all ops around them.
"""
import sys
sys.path.insert(0, "/home/tt-admin/pglusac/tt-xla/third_party/tt-mlir/src/tt-mlir/build/python_packages")

import ttrt.binary as tb

FB = "/home/tt-admin/pglusac/tt-blacksmith/irs_debug/fb_1774741203335.ttnn"

fbb = tb.load_binary_from_path(FB)
print(f"Programs: {fbb.get_num_programs()}")

# Get ops for program 0 (main backward)
ops = tb.program_ops_as_dict(fbb, 0)
print(f"Program 0 has {len(ops)} ops\n")

# Find embedding ops
for i, op in enumerate(ops):
    op_type = str(op.get("type", ""))
    loc = str(op.get("loc", ""))
    debug = str(op.get("debug_info", ""))
    if "embedding" in op_type.lower() or "embedding" in loc.lower() or "embedding" in debug.lower():
        print(f"Op {i}: type={op_type}")
        print(f"  loc={loc}")
        # Print surrounding ops
        for j in range(max(0,i-3), min(len(ops), i+4)):
            marker = " >>>" if j == i else "    "
            print(f"{marker} op {j}: {ops[j].get('type', '?')} loc={ops[j].get('loc', '?')[:60]}")
        print()
