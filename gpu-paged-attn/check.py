"""Check the numbers gpu-paged-attn prints against the same attention in Python (f64).

Reads the program's output on stdin. Exits non-zero if an entry differs by more than 1e-4 plus
1e-4 relative to the entry's size, or if a length differs.
"""

import math
import sys

LENS = [37, 130, 5]


def x(i):
    return ((i * 37) % 101) / 8.0 - 6.0


def key(s, h, pos, d):
    return x(((s * 2 + h) * 256 + pos) * 64 + d) / 16.0 * (1.0 + pos / 64.0)


def val(s, h, pos, d):
    return x(((s * 2 + h) * 256 + pos) * 64 + d + 5) / 4.0


def qry(s, h, d):
    return x((s * 2 + h) * 64 + d + 9) / 16.0


def decode(s, h, d):
    n = LENS[s] + 1
    sc = [sum(qry(s, h, e) * key(s, h, pos, e) for e in range(64)) * 0.125 for pos in range(n)]
    m = max(sc)
    z = sum(math.exp(v - m) for v in sc)
    return sum(math.exp(sc[pos] - m) * val(s, h, pos, d) for pos in range(n)) / z


want = {"append": [n + 1 for n in LENS], "decode": [decode(0, 0, 0), decode(1, 1, 31), decode(2, 0, 63)]}

failed = 0
seen = set()
for line in sys.stdin:
    name, _, rest = line.partition(": ")
    if name not in want:
        continue
    seen.add(name)
    got = [float(v) for v in rest.split()]
    for g, w in zip(got, want[name], strict=True):
        tol = 0 if name == "append" else 1e-4 + 1e-4 * abs(w)
        if abs(g - w) > tol:
            print(f"FAIL: {name} printed {g}, Python gives {w}")
            failed += 1
for name in want.keys() - seen:
    print(f"FAIL: no {name} line")
    failed += 1
if failed:
    sys.exit(1)
print("ok: Python computes the same entries")
