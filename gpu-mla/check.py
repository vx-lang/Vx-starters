"""Check the outputs gpu-mla prints against attention in the expanded form in Python (f64).

Reads the program's output on stdin. Outputs are printed in thousandths, cut toward zero; the
device works in f32, so a value may differ from Python's by one thousandth.
"""

import math
import sys

HEADS, DN, DV, SCALE = 2, 16, 16, 0.5


def c_val(s, t, j):
    return (((t * 13 + j * 7 + s * 5) % 17) - 8) / 16.0 * (1.0 + t / 50.0)


def pe_val(s, t, d):
    return ((t * 3 + d * 5 + s) % 11 - 5) / 16.0


def qn_val(r, h, d):
    return ((r * 7 + h * 3 + d * 5) % 9 - 4) / 8.0


def qp_val(r, h, d):
    return ((r * 5 + h * 11 + d * 3) % 7 - 3) / 8.0


def wuk_val(h, d, j):
    return ((h * 3 + d * 5 + j * 11) % 13 - 6) / 16.0


def wuv_val(h, j, d):
    return ((h * 7 + j * 3 + d * 5) % 11 - 5) / 4.0


def attend(r, h, s, length, d_out):
    scores = []
    for t in range(length):
        dot = 0.0
        for d in range(DN):
            k = sum(c_val(s, t, j) * wuk_val(h, d, j) for j in range(512))
            dot += qn_val(r, h, d) * k
        dot += sum(qp_val(r, h, d) * pe_val(s, t, d) for d in range(64))
        scores.append(dot * SCALE)
    top = max(scores)
    e = [math.exp(v - top) for v in scores]
    z = sum(e)
    v = [sum(c_val(s, t, j) * wuv_val(h, j, d_out) for j in range(512)) for t in range(length)]
    return sum(e[t] * v[t] for t in range(length)) / z


def thousandths(x):
    return int(x * 1000.0)


want = {
    "decode": [thousandths(attend(0, 0, 0, 150, 0)), thousandths(attend(0, 1, 0, 150, 5)), thousandths(attend(1, 1, 1, 61, 15))],
    "prefill": [thousandths(attend(0, 0, 0, 1, 0)), thousandths(attend(70, 1, 0, 71, 3)), thousandths(attend(139, 0, 0, 140, 9))],
}

failed = 0
seen = set()
for line in sys.stdin:
    name, _, rest = line.partition(": ")
    if name not in want:
        continue
    seen.add(name)
    got = [int(v) for v in rest.split()]
    if len(got) != 3 or any(abs(a - b) > 1 for a, b in zip(got, want[name])):
        print(f"FAIL: {name} printed {got}, Python gives {want[name]}")
        failed += 1
for name in want.keys() - seen:
    print(f"FAIL: no {name} line")
    failed += 1
if failed:
    sys.exit(1)
print("ok: Python computes the same outputs")
