"""Check the picks and weights gpu-router prints against the same router in Python (f64).

Reads the program's output on stdin. Experts must match exactly, weights to 1e-4.
"""

import math
import sys

TOKENS, EXPERTS = 50, 64


def logit(t, e, ties):
    if ties == 2:
        return ((t + e // 4) % 3) / 2.0
    if ties:
        return ((t * 3 + e * 7) % 9) / 4.0 - 1.0
    return (((t * EXPERTS + e) * 37) % 101) / 8.0 - 6.0


def bias(e, ties):
    if ties == 2:
        return 0.0
    return (e % 4) / 64.0 if ties else -(e % 5) / 128.0


def sigmoid(v):
    return 1.0 / (1.0 + math.exp(-v))


def route(t, k, groups, topk_group, ties, scale):
    size = EXPERTS // groups
    score = [sigmoid(logit(t, e, ties)) + bias(e, ties) for e in range(EXPERTS)]
    gscore = []
    for g in range(groups):
        top = sorted(score[g * size : (g + 1) * size], reverse=True)
        gscore.append(top[0] + (top[1] if size > 1 else 0.0))
    chosen = sorted(range(groups), key=lambda g: (-gscore[g], g))[:topk_group]
    eligible = [e for e in range(EXPERTS) if e // size in chosen]
    picks = sorted(eligible, key=lambda e: (-score[e], e))[:k]
    w = [sigmoid(logit(t, e, ties)) for e in picks]
    total = sum(w)
    return picks, [v / (total + 1e-20) * scale for v in w]


want = {}
for name, k, groups, topk_group, ties, scale in [("grouped", 8, 8, 4, 1, 2.5), ("plain", 6, 1, 1, 0, 1.0), ("tied", 8, 16, 3, 2, 1.0)]:
    i0, _ = route(0, k, groups, topk_group, ties, scale)
    i49, _ = route(49, k, groups, topk_group, ties, scale)
    _, w17 = route(17, k, groups, topk_group, ties, scale)
    want[f"{name}_ids"] = i0 + [i49[0], i49[k - 1]]
    want[f"{name}_weights"] = [w17[0], w17[1], w17[k - 1]]

failed = 0
seen = set()
for line in sys.stdin:
    name, _, rest = line.partition(": ")
    if name not in want:
        continue
    seen.add(name)
    got = [float(v) for v in rest.split()]
    for g, w in zip(got, want[name], strict=True):
        tol = 0 if name.endswith("_ids") else 1e-4
        if abs(g - w) > tol:
            print(f"FAIL: {name} printed {g}, Python gives {w}")
            failed += 1
for name in want.keys() - seen:
    print(f"FAIL: no {name} line")
    failed += 1
if failed:
    sys.exit(1)
print("ok: Python computes the same entries")
