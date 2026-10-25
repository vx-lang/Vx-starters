"""Check the numbers gpu-llm-ops prints against the same formulas in Python (f64).

Reads the program's output on stdin. Exits non-zero if an entry differs by more than 1e-4 plus
1e-4 relative to the entry's size.
"""

import math
import sys


def x(i):
    return ((i * 37) % 101) / 8.0 - 6.0


def gelu(v):
    return 0.5 * v * (1.0 + math.tanh(0.7978846 * (v + 0.044715 * v**3)))


def gelu_grad(v):
    th = math.tanh(0.7978846 * (v + 0.044715 * v**3))
    return 0.5 * (1.0 + th) + v * 0.5 * (1.0 - th * th) * 0.7978846 * (1.0 + 3.0 * 0.044715 * v * v)


def adamw(i):
    p, g, m, v = x(i), x(i + 1) / 16.0, x(i + 2) / 32.0, (i % 7) / 16.0
    mi = 0.9 * m + 0.1 * g
    vi = 0.999 * v + 0.001 * g * g
    return p - 0.01 * ((mi / 0.271) / (math.sqrt(vi / 0.002997001) + 1e-8) + 0.1 * p)


def inp(r, c):
    return x(r * 40 + c) / 4.0


def w(o, c):
    return x(o * 40 + c + 5) / 4.0


def matmul(r, o):
    return x(o + 3) + sum(inp(r, c) * w(o, c) for c in range(40))


def bias_grad(o):
    return sum(matmul(r, o) for r in range(6))


def ix(b, t):
    return (b * 7 + t * 3) % 5 + 10


def wte(v, c):
    return x(v * 768 + c) / 4.0


def wpe(t, c):
    return x(t * 768 + c + 9) / 4.0


def dwte(row, c):
    return sum(x((b * 5 + t) * 768 + c) / 4.0 for b in range(2) for t in range(5) if ix(b, t) == row)


def qkv(b, n, s, h, d):
    return x((((b * 4 + n) * 3 + s) * 12 + h) * 64 + d)


def preatt(h, t, c):
    return x((h * 8 + t) * 8 + c)


def causal(h, t, c):
    if c > t:
        return 0.0
    row = [preatt(h, t, k) * 0.125 for k in range(t + 1)]
    m = max(row)
    z = sum(math.exp(a - m) for a in row)
    return math.exp(row[c] - m) / z


def prob(b, t, v):
    row = [math.exp(x((b * 5 + t) * 50 + k) / 4.0) for k in range(50)]
    return row[v] / sum(row)


def target(b, t):
    return (b * 11 + t * 7) % 50


def loss(b, t):
    return -math.log(prob(b, t, target(b, t)))


def dlogits(b, t, v):
    ind = 1.0 if v == target(b, t) else 0.0
    return x(v) / 8.0 + (prob(b, t, v) - ind) * x(b * 5 + t) / 8.0


want = {
    "residual": [x(i) + x(i + 1) / 4.0 for i in (0, 500, 999)],
    "gelu_forward": [gelu(x(i)) for i in (0, 500, 999)],
    "gelu_backward": [x(i + 1) / 4.0 + gelu_grad(x(i)) * x(i) for i in (0, 500, 999)],
    "adamw": [adamw(i) for i in (0, 500, 999)],
    "matmul_forward": [matmul(0, 0), matmul(3, 7), matmul(5, 19)],
    "bias_by_column": [bias_grad(0), bias_grad(7), bias_grad(19)],
    "bias_atomic": [bias_grad(0), bias_grad(7), bias_grad(19)],
    "encoder_forward": [wte(ix(0, 0), 0) + wpe(0, 0), wte(ix(1, 3), 100) + wpe(3, 100), wte(ix(1, 4), 767) + wpe(4, 767)],
    "encoder_backward": [dwte(10, 0), dwte(11, 5), dwte(14, 767)],
    "attention_permute": [qkv(1, 2, 0, 5, 7), qkv(0, 3, 1, 11, 63), qkv(1, 1, 2, 2, 0)],
    "causal_softmax": [causal(0, 3, 1), causal(1, 7, 7), causal(1, 6, 7)],
    "crossentropy_forward": [loss(0, 0), loss(1, 2), loss(1, 4)],
    "crossentropy_backward": [dlogits(0, 0, 0), dlogits(1, 2, 13), dlogits(1, 4, 49)],
}

failed = 0
seen = set()
for line in sys.stdin:
    name, _, rest = line.partition(": ")
    if name not in want:
        continue
    seen.add(name)
    got = [float(v) for v in rest.split()]
    for g, w_ in zip(got, want[name], strict=True):
        if abs(g - w_) > 1e-4 + 1e-4 * abs(w_):
            print(f"FAIL: {name} printed {g}, Python gives {w_}")
            failed += 1
for name in want.keys() - seen:
    print(f"FAIL: no {name} line")
    failed += 1
if failed:
    sys.exit(1)
print("ok: Python computes the same entries")
