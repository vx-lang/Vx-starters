"""Check the numbers gpu-tensor-core prints against the same products in Python.

The data is small whole numbers, so the products are exact: entries must be equal.
Reads the program's output on stdin.
"""

import sys


def a(i, k):
    return (i * 7 + k * k + 1) % 3


def b(k, j):
    return (k * 5 + j * j + k * j) % 3


def e(j, l):
    return (j * j + l * 5 + j) % 3


def c(i, j):
    return sum(a(i, k) * b(k, j) for k in range(48))


def d(i, l):
    return sum(c(i, j) * e(j, l) for j in range(32))


product = [c(0, 0), c(7, 13), c(15, 31)]
want = {"f32": product, "f16": product, "bf16": product, "chain": [d(0, 0), d(9, 4), d(15, 7)]}

failed = 0
seen = set()
for line in sys.stdin:
    name, _, rest = line.partition(": ")
    if name not in want:
        continue
    seen.add(name)
    got = [float(v) for v in rest.split()]
    if got != [float(v) for v in want[name]]:
        print(f"FAIL: {name} printed {got}, Python gives {want[name]}")
        failed += 1
for name in want.keys() - seen:
    print(f"FAIL: no {name} line")
    failed += 1
if failed:
    sys.exit(1)
print("ok: Python computes the same entries")
