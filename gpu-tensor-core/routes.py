"""Check how the GPU regions of a program are launched, in the LLVM IR that `vxc --emit-llvm` writes.

A region that is a matrix product (`c = a @b`) is not compiled to a kernel. Its launch record
says `kind=matmul`, and at run time Vx's CUDA library hands it to cuBLAS: `cublasSgemm` for f32,
and `cublasGemmEx` with f32 accumulation for f16 and bf16, which can use tensor cores. So the
module should hold one launch record of that kind for each product, and no PTX image at all.

    python3 -I routes.py <file.ll> <number of products>
"""

import re
import sys

ir = open(sys.argv[1]).read()
want = int(sys.argv[2])

records = re.findall(r'@vx_npu_kernel_\d+_str\("([^"]*)"', ir)
kinds = [re.search(r"kind=(\w+)", r).group(1) for r in records]
images = len(re.findall(r"image=", ir))

failed = []
if len(records) != want:
    failed.append(f"{len(records)} launch records, expected {want}")
if any(k != "matmul" for k in kinds):
    failed.append(f"launch kinds {sorted(set(kinds))}, expected only matmul")
if images:
    failed.append(f"{images} PTX images, expected none: a matmul is not a generated kernel")
if failed:
    for f in failed:
        print(f"FAIL: {f}")
    sys.exit(1)
print(f"routes: {want} launches of kind matmul (cuBLAS), no PTX image")
