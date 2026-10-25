"""Check the numbers gpu-flash-prefill prints against the same attention in Python (f64).

Reads the program's output on stdin. Exits non-zero if an entry differs by more than 1e-4 plus
1e-4 relative to the entry's size.
"""

import math
import sys

SEQ = 50


def x(i):
    return ((i * 37) % 101) / 8.0 - 6.0


def q(h, i, d):
    return x((h * 50 + i) * 64 + d + 3) / 16.0


def k(h, i, d):
    return x((h * 50 + i) * 64 + d + 7) / 16.0 * (1.0 + i / 32.0)


def v(h, i, d):
    return x((h * 50 + i) * 64 + d + 11) / 4.0


def attend(h, i, d):
    sc = [sum(q(h, i, e) * k(h, j, e) for e in range(64)) * 0.125 for j in range(i + 1)]
    m = max(sc)
    z = sum(math.exp(s - m) for s in sc)
    return sum(math.exp(sc[j] - m) * v(h, j, d) for j in range(i + 1)) / z


want = {"prefill": [attend(0, 0, 0), attend(1, 33, 17), attend(1, 49, 63)]}

failed = 0
seen = set()
for line in sys.stdin:
    name, _, rest = line.partition(": ")
    if name not in want:
        continue
    seen.add(name)
    got = [float(t) for t in rest.split()]
    for g, w in zip(got, want[name], strict=True):
        if abs(g - w) > 1e-4 + 1e-4 * abs(w):
            print(f"FAIL: {name} printed {g}, Python gives {w}")
            failed += 1
for name in want.keys() - seen:
    print(f"FAIL: no {name} line")
    failed += 1
if failed:
    sys.exit(1)
print("ok: Python computes the same entries")
