"""Check the numbers gpu-inference-ops prints against the same formulas in Python (f64).

Reads the program's output on stdin. Exits non-zero if an entry differs by more than 1e-4 plus
1e-4 relative to the entry's size, or if a chosen expert differs.
"""

import math
import sys


def x(i):
    return ((i * 37) % 101) / 8.0 - 6.0


def angle(p, j):
    return ((p * 3 + j * 5) % 64) / 64.0


def rope(t, h, d):
    j = d // 2
    a, b = x((t * 3 + h) * 64 + 2 * j) / 4.0, x((t * 3 + h) * 64 + 2 * j + 1) / 4.0
    c, s = math.cos(angle(4 + t, j)), math.sin(angle(4 + t, j))
    return a * c - b * s if d % 2 == 0 else a * s + b * c


def swiglu(i):
    g = x(i) / 4.0
    return g / (1.0 + math.exp(-g)) * x(i + 3) / 8.0


def matvec(r):
    return sum(x(r * 70 + c) / 8.0 * x(c + 9) / 4.0 for c in range(70))


def matvec_block(r):
    return sum(x(r * 512 + c) / 8.0 * x(c + 9) / 4.0 for c in range(512))


def router(t):
    s = [(((t * 5 + e * 3) % 8) * 8 + (t % 7)) / 16.0 for e in range(8)]
    order = sorted(range(8), key=lambda e: -s[e])
    g = 1.0 / (1.0 + math.exp(s[order[1]] - s[order[0]]))
    return order[0], order[1], g


def decode(h, d):
    q = [x(h * 64 + k) / 16.0 for k in range(64)]
    sc = []
    for t in range(77):
        k = [x((h * 128 + t) * 64 + e + 1) / 16.0 for e in range(64)]
        sc.append(sum(a * b for a, b in zip(q, k, strict=True)) * 0.125)
    m = max(sc)
    z = sum(math.exp(v - m) for v in sc)
    return sum(math.exp(sc[t] - m) * x((h * 128 + t) * 64 + d + 2) / 4.0 for t in range(77)) / z


want = {
    "rope_forward": [rope(0, 0, 0), rope(2, 1, 33), rope(4, 2, 63)],
    "swiglu": [swiglu(0), swiglu(350), swiglu(699)],
    "matvec": [matvec(0), matvec(150), matvec(299)],
    "matvec_by_block": [matvec_block(0), matvec_block(10), matvec_block(19)],
    "router_gates": [router(0)[2], 1.0 - router(99)[2], router(199)[2]],
    "router_ids": [router(0)[0], router(99)[1], router(199)[0]],
    "attention_decode": [decode(0, 0), decode(2, 31), decode(3, 63)],
}

failed = 0
seen = set()
for line in sys.stdin:
    name, _, rest = line.partition(": ")
    if name not in want:
        continue
    seen.add(name)
    got = [float(v) for v in rest.split()]
    for g, w in zip(got, want[name], strict=True):
        tol = 0 if name == "router_ids" else 1e-4 + 1e-4 * abs(w)
        if abs(g - w) > tol:
            print(f"FAIL: {name} printed {g}, Python gives {w}")
            failed += 1
for name in want.keys() - seen:
    print(f"FAIL: no {name} line")
    failed += 1
if failed:
    sys.exit(1)
print("ok: Python computes the same entries")
