"""Check the tokens gpu-sampling prints against the same sampler in Python (f64).

Reads the program's output on stdin. Tokens must be equal.
"""

import math
import sys

ROWS, VOCAB, CAP = 64, 1000, 16


def logit(r, c):
    pat = r % 4
    if pat >= 2 and c % 50 == 7:
        return -math.inf
    if pat == 1 and c in (70, 300, 582):
        return 20.0
    v = ((c * 37 + pat * 11) % 101) / 16.0 - 3.0
    if c % 123 == pat:
        v += 8.0
    return v


def sample(r, t, k, mass, ratio):
    lg = [logit(r, c) for c in range(VOCAB)]
    order = sorted(range(VOCAB), key=lambda c: (-lg[c], c))[:CAP]
    top = lg[order[0]]
    e = [math.exp((v - top) / t) for v in lg]
    z = sum(e)
    keep = set()
    cum = 0.0
    for j, c in enumerate(order):
        p = e[c] / z
        if j < k and cum < mass and p >= ratio * (1.0 / z) and p > 0.0:
            keep.add(c)
        cum += p
    kept_sum = sum(e[c] / z for c in keep)
    u = (r + 0.5) / 64.0
    total = 0.0
    last = 0
    for c in sorted(keep):
        total += e[c] / z / kept_sum
        last = c
        if u < total:
            return c
    return last


def greedy(r):
    return max(range(VOCAB), key=lambda c: (logit(r, c), -c))


want = {"greedy": [greedy(r) for r in (0, 31, 63)]}
for name, cfg in [("top_k", (0.8, 5, 2.0, 0.0)), ("top_p", (0.5, 16, 0.9, 0.0)), ("min_p", (0.5, 16, 2.0, 0.3)), ("all_three", (0.6, 8, 0.95, 0.1))]:
    want[name] = [sample(r, *cfg) for r in (0, 31, 63)]

failed = 0
seen = set()
for line in sys.stdin:
    name, _, rest = line.partition(": ")
    if name not in want:
        continue
    seen.add(name)
    got = [int(v) for v in rest.split()]
    if got != want[name]:
        print(f"FAIL: {name} printed {got}, Python gives {want[name]}")
        failed += 1
for name in want.keys() - seen:
    print(f"FAIL: no {name} line")
    failed += 1
if failed:
    sys.exit(1)
print("ok: Python computes the same tokens")
