"""Check the numbers gpu-norms prints against the same formulas in Python (f64).

Reads the program's output on stdin. Exits non-zero if an entry differs by more than 1e-5,
relative to the entry's size.
"""

import math
import sys

R, C, EPS = 32, 128, 0.00001
x = [[((i * 37 + j * 11) % 101) / 8.0 - 6.0 for j in range(C)] for i in range(R)]


def layernorm(row):
    m = sum(row) / C
    v = sum((a - m) ** 2 for a in row) / C
    return [(a - m) / math.sqrt(v + EPS) for a in row]


def rmsnorm(row):
    ms = sum(a * a for a in row) / C
    return [a / math.sqrt(ms + EPS) for a in row]


def softmax(row):
    m = max(row)
    z = sum(math.exp(a - m) for a in row)
    return [math.exp(a - m) / z for a in row]


want = {}
for name, f in [("layernorm", layernorm), ("rmsnorm", rmsnorm), ("softmax", softmax)]:
    want[name] = [f(x[0])[0], f(x[7])[50], f(x[31])[127]]

failed = 0
seen = set()
for line in sys.stdin:
    name, _, rest = line.partition(": ")
    if name not in want:
        continue
    seen.add(name)
    got = [float(v) for v in rest.split()]
    for g, w in zip(got, want[name], strict=True):
        if abs(g - w) > 1e-5 * max(abs(w), 1e-30):
            print(f"FAIL: {name} printed {g}, Python gives {w}")
            failed += 1
for name in want.keys() - seen:
    print(f"FAIL: no {name} line")
    failed += 1
if failed:
    sys.exit(1)
print("ok: Python computes the same entries")
