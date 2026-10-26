"""Check the entries gpu-logit-processors prints against the same processing in Python.

Reads the program's output on stdin. All values are dyadic, so they must be equal.
"""

import sys

ROWS, VOCAB = 6, 700
NEG = float("-inf")


def base(r, c):
    return ((c * 29 + r * 13) % 41) / 8.0 - 2.5


def lens(r):
    return 20 + r * 55


def hist(r, p):
    return ((p * 7 + r) % 90) * 7 + r % 3


def extra(r):
    return (r * 101 + 17) % VOCAB


def rep(r):
    return 1.0 + r / 4.0


def freq(r):
    return r / 8.0


def pres(r):
    return 0.5 - r / 16.0


def bias(c):
    return (c % 5) / 4.0 - 0.5


def allowed(r, c):
    return 0 if (c + r) % 6 == 0 else 1


def temp(r):
    return 0.0 if r == 3 else 0.5 + r / 2.0


def count(r, c):
    return sum(1 for p in range(lens(r)) if hist(r, p) == c) + (1 if extra(r) == c else 0)


def penalty(r, c):
    x, n = base(r, c), count(r, c)
    if n == 0:
        return x
    y = x / rep(r) if x > 0 else x * rep(r)
    return y - freq(r) * n - pres(r)


def trunc8(v):
    return int(v * 8.0)


want = {
    "counts": [count(0, extra(0)), count(1, hist(1, 3)), count(5, hist(5, 0))],
    "penalties": [trunc8(penalty(2, hist(2, 0))), trunc8(penalty(4, hist(4, 5))), trunc8(penalty(5, extra(5)))],
    "bias_mask": [trunc8(base(1, 2) + bias(2)), trunc8(base(2, 3) + bias(3)), sum(1 for r in range(ROWS) for c in range(VOCAB) if not allowed(r, c))],
    "temperature": [trunc8(base(0, 1) / temp(0)), trunc8(base(3, 1)), trunc8(base(5, 9) / temp(5))],
}


def pipe(r, c):
    if not allowed(r, c):
        return NEG
    v = penalty(r, c) + bias(c)
    return v / temp(r) if temp(r) > 0 else v


want["pipeline"] = [trunc8(pipe(1, hist(1, 2))), trunc8(pipe(3, hist(3, 4))), trunc8(pipe(4, extra(4)))]

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
print("ok: Python computes the same entries")
