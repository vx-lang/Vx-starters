"""Pull the GPU kernels out of the LLVM IR that `vxc --emit-llvm` writes.

Each GPU region becomes one launch, and each launch carries the program's GPU module as PTX, in a
string after `image=` with non-printable bytes escaped as \\XX. The module holds one kernel (an
`.entry`) per region. Checks that there is one launch per region, that the module targets sm_80
and has that many kernels, and that it calls nothing outside itself except libdevice (`__nv_*`)
functions. Writes the module to <dir>/kernels.ptx.

    python3 -I kernels.py <file.ll> <dir> <number of GPU regions>
"""

import re
import sys

ir_path, out_dir, want = sys.argv[1], sys.argv[2], int(sys.argv[3])
ir = open(ir_path).read()

images = set()
for m in re.finditer(r"image=", ir):
    end = ir.find("\\00", m.end())
    images.add(re.sub(r"\\([0-9A-F]{2})", lambda h: chr(int(h.group(1), 16)), ir[m.end():end]))
launches = len(re.findall(r"image=", ir))

failed = []
if launches != want:
    failed.append(f"{launches} launches carry a kernel, expected {want}")
if len(images) != 1:
    failed.append(f"the launches carry {len(images)} different modules, expected one")
ptx = images.pop() if images else ""
entries = re.findall(r"\.visible \.entry (\w+)", ptx)
outside = [l.strip() for l in ptx.splitlines() if ".extern .func" in l and "__nv_" not in l]
if ".target sm_80" not in ptx:
    failed.append("the module does not target sm_80")
if len(set(entries)) != want:
    failed.append(f"the module has {len(set(entries))} kernels, expected {want}")
if outside:
    failed.append(f"the module calls functions it does not contain: {outside}")
if failed:
    for f in failed:
        print(f"FAIL: {f}")
    sys.exit(1)
with open(f"{out_dir}/kernels.ptx", "w") as f:
    f.write(ptx)
print(f"kernels: {', '.join(entries)}, one sm_80 module of {len(ptx)} bytes of PTX")
