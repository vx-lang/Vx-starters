"""Turn the lines of measure.py into the factors of cost.vx.

  python3 -I calibrate.py < out-2node.txt
  python3 -I calibrate.py --selftest

For each factor it prints what cost.vx assumes, what was measured, and the ratio. Nothing is
changed: read the table, then edit cost.vx (mfu, mem_eff, comm_eff, layer_overhead) and hw.vx
yourself, and put the output next to the change.

Specs are the figures of hw.vx for the GPU named in the `machine` line (a name that contains the
key below).
"""

import sys

# hbm bytes/s, bf16 FLOP/s, fp8 FLOP/s, nvlink bytes/s, nic bytes/s -- as in hw.vx
SPEC = {
    "H100": (3.35e12, 989e12, 1979e12, 450e9, 50e9),
    "H200": (4.8e12, 989e12, 1979e12, 450e9, 50e9),
    "A100": (2.039e12, 312e12, 0.0, 300e9, 25e9),
    "B200": (7.7e12, 2250e12, 4500e12, 900e9, 50e9),
}
# cost.vx
ASSUMED = {"mem_eff": 0.75, "mfu": 0.4, "comm_eff": 0.7, "layer_overhead_s": 0.00003}


def parse(text):
    rows = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2 or parts[0] != "measured":
            continue
        rows[parts[1]] = {k: v for k, v in (p.split("=", 1) for p in parts[2:])}
    return rows


def f(row, key):
    return float(row[key])


def factors(rows):
    """-> list of (name, assumed, measured, note)."""
    mach = rows["machine"]
    spec = next((v for k, v in SPEC.items() if k in mach["gpu"]), None)
    if spec is None:
        raise SystemExit(f"no spec for {mach['gpu']}: add it to SPEC")
    hbm, bf16, fp8, nvl, nic = spec
    out = []
    out.append(("mem_eff (copy)", ASSUMED["mem_eff"], f(rows["hbm_copy"], "bytes_per_s") / hbm, "achieved / spec"))
    out.append(("mem_eff (read)", ASSUMED["mem_eff"], f(rows["hbm_read"], "bytes_per_s") / hbm, "achieved / spec"))
    out.append(("mfu (bf16)", ASSUMED["mfu"], f(rows["gemm_bf16"], "flops_per_s") / bf16, "8192^3 matmul"))
    if "gemm_fp8" in rows and "flops_per_s" in rows["gemm_fp8"] and fp8 > 0:
        out.append(("mfu (fp8)", ASSUMED["mfu"], f(rows["gemm_fp8"], "flops_per_s") / fp8, "8192^3 scaled_mm"))
    out.append(("launch (us per kernel)", ASSUMED["layer_overhead_s"] * 1e6, f(rows["launch"], "us"), "cost.vx counts one overhead per LAYER, a layer has many kernels"))
    a = rows["a2a_local"]
    r = f(a, "ranks")
    # The buffer a GPU sends has 1/r of itself kept; the rest goes over NVLink.
    out.append(("comm_eff (NVLink all-to-all)", ASSUMED["comm_eff"], f(a, "bytes_per_s") * r / (r - 1) / nvl, "remote bytes / spec"))
    if "a2a_world" in rows:
        w = rows["a2a_world"]
        n = f(w, "ranks")
        gpn = f(rows["machine"], "gpus_per_node")
        # cost.vx: remote share (n - gpn) / n of the bytes over `cross` * comm_eff.
        out.append(("comm_eff (network all-to-all)", ASSUMED["comm_eff"], f(w, "bytes_per_s") * n / (n - gpn) / nic, "remote bytes / NIC spec"))
        out.append(("lat (us, network all-to-all)", 6.0, f(w, "lat_s") / 2 * 1e6, "cost.vx adds 2 * lat; 6 us is the example's guess"))
    p = rows["p2p"]
    out.append(("p2p bytes/s / NIC" if "a2a_world" in rows else "p2p bytes/s / NVLink", ASSUMED["comm_eff"], f(p, "bytes_per_s") / (nic if "a2a_world" in rows else nvl), "pipeline.vx send_time uses comm_eff"))
    out.append(("p2p lat (us)", 6.0, f(p, "lat_s") * 1e6, "pipeline.vx adds lat once per send"))
    return out


def show(table):
    print(f"{'factor':34} {'assumed':>10} {'measured':>10} {'ratio':>7}")
    for name, assumed, got, note in table:
        print(f"{name:34} {assumed:10.4g} {got:10.4g} {got / assumed:7.2f}  {note}")


def selftest():
    """Lines made from known factors must give those factors back. The numbers are invented: this
    checks the arithmetic of this file, not any GPU."""
    hbm, bf16, fp8, nvl, nic = SPEC["H100"]
    ranks, gpn, world = 8.0, 8.0, 16.0
    lines = [
        f"measured machine gpu=NVIDIA_H100_80GB_HBM3 sm=132 hbm_bytes=1 world={int(world)} gpus_per_node={int(gpn)} torch=x cuda=y",
        f"measured hbm_copy bytes_per_s={0.9 * hbm}",
        f"measured hbm_read bytes_per_s={0.8 * hbm}",
        f"measured gemm_bf16 flops_per_s={0.6 * bf16}",
        f"measured gemm_fp8 flops_per_s={0.5 * fp8}",
        "measured launch us=4.5",
        f"measured a2a_local ranks=8 lat_s=1e-05 bytes_per_s={0.8 * nvl * (ranks - 1) / ranks}",
        f"measured a2a_world ranks=16 lat_s=2e-05 bytes_per_s={0.6 * nic * (world - gpn) / world}",
        f"measured p2p peer=8 lat_s=8e-06 bytes_per_s={0.75 * nic}",
    ]
    got = {n: m for n, _, m, _ in factors(parse("\n".join(lines)))}
    want = {"mem_eff (copy)": 0.9, "mem_eff (read)": 0.8, "mfu (bf16)": 0.6, "mfu (fp8)": 0.5,
            "launch (us per kernel)": 4.5, "comm_eff (NVLink all-to-all)": 0.8,
            "comm_eff (network all-to-all)": 0.6, "lat (us, network all-to-all)": 10.0,
            "p2p bytes/s / NIC": 0.75, "p2p lat (us)": 8.0}
    bad = [n for n, v in want.items() if abs(got[n] - v) > 1e-9 * max(1.0, abs(v))]
    if set(got) != set(want) or bad:
        print("FAIL:", bad or set(got) ^ set(want))
        return 1
    print("ok: calibrate.py recovers the factors it was built from")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        sys.exit(selftest())
    show(factors(parse(sys.stdin.read())))
