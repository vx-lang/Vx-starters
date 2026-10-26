"""Have the Vx compiler check what the planner believes.

Reads the lines `emit.vx` prints (on stdin) and, for each, writes a Vx program and compiles it
against a machine file with `vxc --machine ... -a print-ast --diagnostics-json`:

  hw    writes the machine file of a node (admit.py's `node` files): the SKU's own file from the
        toolchain, with the node's links added at the rates of hw.vx. Also checks that the SKU
        file declares the HBM size hw.vx has.
  fit   the resident set of one GPU: weights, cache, and a reserve for the part of HBM that a
        deployment does not use (the `policy` line, cost.vx's `usable`; the machine files leave
        it to the admission program). The compiler must admit it exactly when `fits` said so, and
        count the bytes it was given.
  wire  a cache sent from one node to another over the NIC: the time the compiler derives for the
        NIC hop must be the planner's.

Also the cases that are facts of the machine files: fp8 weights are refused on an A100.

    python3 -I admit.py <VX_HOME> <scratch dir> < lines
"""

import concurrent.futures
import json
import math
import os
import re
import subprocess
import sys

vx_home, scratch = sys.argv[1], sys.argv[2]
fleet = os.path.join(vx_home, "fleet")
vxc = os.path.join(vx_home, "bin", "vxc")
host = os.path.join(fleet, "host-x86-e5-2666v3.vx")
ROW = 8192  # numbers in a row of every tensor written here

FILE = {"A100-80": "a100-80.vx", "H200": "h200.vx", "B200": "b200.vx"}
failed = []
checked = {"fit": 0, "fit-admitted": 0, "fit-rejected": 0, "wire": 0, "skipped": 0}


def fail(msg):
    failed.append(msg)
    print(f"FAIL: {msg}")


def dtype(nbytes, fp8):
    """The element type of weights or cache of `nbytes` each on a SKU with or without fp8 cores."""
    if nbytes == 2:
        return "f16", 2
    if nbytes == 1:
        return ("f8e4m3" if fp8 else "i8"), 1
    raise SystemExit(f"no element type of {nbytes} bytes")


def compile_program(name, source, machine):
    path = os.path.join(scratch, name + ".vx")
    out = os.path.join(scratch, name + ".json")
    with open(path, "w") as f:
        f.write(source)
    subprocess.run([vxc, "--machine", machine, "--host", host, "-a", "print-ast", "--diagnostics-json", out, path],
                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(out) as f:
        return json.load(f)


def tensor(name, ty, rows):
    return (f"  let {name}_h : Tensor<{ty}, [{rows}, {ROW}]> = Tensor<{ty}>([{rows}, {ROW}]);\n"
            f"  let {name} = transfer({name}_h, Memory::HBM);\n")


def rows_for(nbytes, size):
    return max(1, math.ceil(nbytes / (ROW * size)))


def fit_program(wty, wrows, kty, krows, rrows):
    src = "fn admit() -> i32 {\n"
    src += tensor("w", wty, wrows) + tensor("k", kty, krows) + tensor("r", "i8", rrows)
    # The step reads all three together: that is what makes them resident at once.
    src += "  let _w = transfer(w, Memory::CPU_DRAM);\n  let _k = transfer(k, Memory::CPU_DRAM);\n"
    src += "  let _r = transfer(r, Memory::CPU_DRAM);\n  return 0;\n}\n\nfn main() -> i32 {\n  return admit();\n}\n"
    return src


# ---------------------------------------------------------------------------------------------
# The node machine files

def node_machine(name, gpus, hbm, nic, nvlink):
    """A node's machine file from the SKU's own, plus the links."""
    with open(os.path.join(fleet, FILE[name])) as f:
        text = f.read()
    cut = text.index("Topology Device {")
    head = text[:cut]
    topo = text[cut:]
    keep = [l for l in topo.splitlines() if re.match(r"\s*(arch|levels|chip|dtypes):", l)]
    transfers = [l.strip().rstrip(",") for l in topo.splitlines() if l.strip().startswith("transfer ")]
    mib = hbm // (1024 * 1024)
    peers = gpus - 1
    gbs = lambda v: f"{v / 1e9:g} GB/s"
    out = head
    out += f"""// The other GPUs of the node, reached over NVLink: one space, as in fleet/node-8gpu.vx.
Memory PEER_HBM {{
  capacity: {peers * mib} MiB, bandwidth: {gbs(nvlink)}, managed: explicit, scope: device
}}

Topology Node {{
"""
    out += "\n".join(keep) + "\n"
    out += "  memory: Memory::HBM,\n  visible: [Memory::HBM, Memory::L2, Memory::SMEM, Memory::REG, Memory::PEER_HBM],\n"
    extra = [
        f"transfer Memory::HBM -> Memory::PEER_HBM : {gbs(nvlink)}",
        f"transfer Memory::PEER_HBM -> Memory::HBM : {gbs(nvlink)}",
        "transfer Memory::HBM -> Memory::NIC_RAM : 63 GB/s",
        f"transfer Memory::NIC_RAM -> Memory::Remote_HBM : {gbs(nic)}",
        f"transfer Memory::Remote_HBM -> Memory::NIC_RAM : {gbs(nic)}",
        "transfer Memory::NIC_RAM -> Memory::HBM : 63 GB/s",
    ]
    out += ",\n".join("  " + t for t in transfers + extra) + "\n}\n"
    return out


def run_hw(parts):
    name, hbm, nic, nvlink, gpus, fp8 = parts[0], int(float(parts[1])), float(parts[2]), float(parts[3]), int(parts[4]), int(parts[5])
    HW[name] = (hbm, nic, nvlink, fp8)
    path = os.path.join(scratch, f"node-{name}-{gpus}.vx")
    with open(path, "w") as f:
        f.write(node_machine(name, gpus, hbm, nic, nvlink))
    NODE[(name, gpus)] = path
    # The SKU file declares the HBM size hw.vx has, to the byte.
    doc = compile_program(f"cap-{name}-{gpus}", fit_program("f16", 1, "f16", 1, 1), path)
    cap = doc["resident_sets"][0]["capacity_bytes"] if doc["resident_sets"] else None
    if cap != hbm:
        fail(f"{name}: hw.vx has {hbm} bytes of HBM, the machine file {cap}")
    if doc["verdict"] != "admitted":
        fail(f"{name}: the node machine file does not admit a small program: {doc['diagnostics']}")


HW = {}
NODE = {}


def run_fit(parts):
    tag, name = parts[0], parts[1]
    wbytes, wsize, kbytes, ksize, expect = float(parts[2]), int(float(parts[3])), float(parts[4]), int(float(parts[5])), parts[6]
    hbm, nic, nvlink, fp8 = HW[name]
    wty, wsz = dtype(wsize, fp8)
    kty, ksz = dtype(ksize, fp8)
    usable = POLICY * hbm
    reserve = hbm - usable
    margin = usable - (wbytes + kbytes)
    if abs(margin) < 8 * ROW * 2:
        checked["skipped"] += 1  # on the edge of one row of rounding
        return
    wrows, krows, rrows = rows_for(wbytes, wsz), rows_for(kbytes, ksz), rows_for(reserve, 1)
    doc = compile_program(f"fit-{tag}", fit_program(wty, wrows, kty, krows, rrows), os.path.join(fleet, FILE[name]))
    checked["fit"] += 1
    checked["fit-" + expect] += 1
    if doc["verdict"] != expect:
        codes = [d["code"] for d in doc["diagnostics"] if d["level"] == "error"]
        fail(f"fit {tag} on {name}: cost.vx says {expect}, the compiler {doc['verdict']} {codes} "
             f"(weights {wbytes:.0f}, cache {kbytes:.0f}, margin {margin:.0f})")
        return
    if expect == "rejected":
        codes = [d["code"] for d in doc["diagnostics"] if d["level"] == "error"]
        # E6010: the resident set is over capacity; E6009: one tensor alone is.
        if "E6010" not in codes or any(c not in ("E6009", "E6010") for c in codes):
            fail(f"fit {tag} on {name}: rejected with {codes}, expected over capacity (E6010, maybe E6009)")
        return
    total = doc["resident_sets"][0]["total_bytes"]
    want = (wrows * wsz + krows * ksz + rrows) * ROW
    if total != want:
        fail(f"fit {tag} on {name}: the compiler counts {total} resident bytes, the program has {want}")


def run_wire(parts):
    tag, a, b = parts[0], parts[1], parts[2]
    per_stream, seconds = float(parts[3]), float(parts[4])
    # The machine file of the side with the slower NIC prices the wire; the other must not price it slower.
    prices = []
    for name in (a, b):
        hbm, nic, nvlink, fp8 = HW[name]
        path = NODE[(name, 8)]
        ty, sz = "i8", 1  # the bytes are what the wire is priced by
        rows = rows_for(per_stream, sz)
        src = (f"fn send() -> i32 {{\n" + tensor("c", ty, rows) +
               "  let away = transfer(c, Memory::Remote_HBM);\n  let back = transfer(away, Memory::CPU_DRAM);\n  return 0;\n}\n\n"
               "fn main() -> i32 {\n  return send();\n}\n")
        doc = compile_program(f"wire-{tag}-{name}", src, path)
        hops = [r for r in doc["routes"] if r["path"] == ["NIC_RAM", "Remote_HBM"]]
        if doc["verdict"] != "admitted" or len(hops) != 1:
            fail(f"wire {tag}: the compiler does not price the NIC hop on {name}: {doc['verdict']} {doc['diagnostics']}")
            return
        prices.append((hops[0]["derived_cost"] * 1e-12, rows * ROW * sz))
    slow = max(p[0] for p in prices)
    nbytes = prices[0][1]
    # The rows are rounded up, so compare for the bytes the program has.
    want = seconds * nbytes / per_stream
    checked["wire"] += 1
    if abs(slow - want) > 1e-9 * want:
        fail(f"wire {tag} {a} -> {b}: the compiler prices the NIC hop at {slow}, the planner at {want}")


def refused_facts():
    """What the machine files say about number formats, which a layout must respect."""
    a100 = os.path.join(fleet, "a100-80.vx")
    doc = compile_program("fp8-on-a100", fit_program("f8e4m3", 1, "f16", 1, 1), a100)
    codes = [d["code"] for d in doc["diagnostics"] if d["level"] == "error"]
    if doc["verdict"] != "rejected" or codes != ["E6026"]:
        fail(f"fp8 weights on an A100: expected E6026, got {doc['verdict']} {codes}")
    doc = compile_program("i8-on-a100", fit_program("i8", 1, "i8", 1, 1), a100)
    if doc["verdict"] != "admitted":
        fail(f"int8 weights on an A100 are refused: {doc['diagnostics']}")
    h200 = os.path.join(fleet, "h200.vx")
    doc = compile_program("fp8-on-h200", fit_program("f8e4m3", 1, "f8e4m3", 1, 1), h200)
    if doc["verdict"] != "admitted":
        fail(f"fp8 on an H200 is refused: {doc['diagnostics']}")


lines = [l.split() for l in sys.stdin if l.split()[:1] in (["policy"], ["hw"], ["fit"], ["wire"])]
POLICY = float([l for l in lines if l[0] == "policy"][0][1])
for l in lines:
    if l[0] == "hw":
        run_hw(l[1:])
work = [l for l in lines if l[0] in ("fit", "wire")]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    futures = [pool.submit(run_fit if l[0] == "fit" else run_wire, l[1:]) for l in work]
    for fut in futures:
        fut.result()
refused_facts()

print(f"fit: {checked['fit']} cells ({checked['fit-admitted']} admitted, {checked['fit-rejected']} rejected), "
      f"{checked['skipped']} on the edge skipped; wire: {checked['wire']} cells")
if failed:
    sys.exit(1)
print("ok: the compiler agrees with the planner on memory, formats and the wire")
