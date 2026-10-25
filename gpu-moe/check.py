"""Check the numbers gpu-moe prints against the same layer in Python (f64).

Reads the program's output on stdin. Exits non-zero if an entry differs by more than 1e-4 plus
1e-4 relative to the entry's size, or if a count of the sort is wrong.
"""

import math
import sys

T, E, K, H, F = 20, 6, 2, 48, 64


def x(i):
    return ((i * 37) % 101) / 8.0 - 6.0


def pick(t, s):
    first = (t * 5 + 1) % 6
    return first if s == 0 else (first + 1 + t % 4) % 6


def gate(t, s):
    g0 = (4 + t % 3) / 10.0
    return g0 if s == 0 else 1.0 - g0


def layer(t, c):
    xs = [x(t * H + k) / 4.0 for k in range(H)]
    total = 0.0
    for s in range(K):
        e = pick(t, s)
        h = []
        for f in range(F):
            sg = sum(x((e * F + f) * H + k) / 64.0 * xs[k] for k in range(H))
            su = sum(x((e * F + f) * H + k + 11) / 64.0 * xs[k] for k in range(H))
            h.append(sg / (1.0 + math.exp(-sg)) * su)
        acc = sum(x((e * H + c) * F + f + 23) / 64.0 * h[f] for f in range(F))
        total += gate(t, s) * acc
    return total


counts = [sum(1 for t in range(T) for s in range(K) if pick(t, s) == e) for e in range(E)]
want = {"moe": [layer(0, 0), layer(7, 20), layer(19, 47)], "sort": [counts[0], counts[3], T * K]}

failed = 0
seen = set()
for line in sys.stdin:
    name, _, rest = line.partition(": ")
    if name not in want:
        continue
    seen.add(name)
    got = [float(v) for v in rest.split()]
    for g, w in zip(got, want[name], strict=True):
        tol = 0 if name == "sort" else 1e-4 + 1e-4 * abs(w)
        if abs(g - w) > tol:
            print(f"FAIL: {name} printed {g}, Python gives {w}")
            failed += 1
for name in want.keys() - seen:
    print(f"FAIL: no {name} line")
    failed += 1
if failed:
    sys.exit(1)
print("ok: Python computes the same entries")
