"""Check the numbers gpu-quant-ops prints against the same products in Python (f64).

Decodes the weights with its own code (struct for bf16, a formula for fp8), builds the same
matrices as test.vx, and multiplies in f64. Reads the program's output on stdin. Exits non-zero
if an entry differs by more than 1e-4 plus 1e-4 relative to the entry's size.
"""

import struct
import sys

ROWS, COLS, BATCH = 40, 320, 3


def x(i):
    return ((i * 37) % 101) / 8.0 - 6.0


def bf16(bits):
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def fp8(b):
    e, m = (b >> 3) & 15, b & 7
    v = m / 512.0 if e == 0 else (8 + m) * 2.0 ** (e - 10)
    return -v if b >= 128 else v


def scale(r):
    return (1 + r % 5) / 16.0


def w_bf16(r, c):
    e, m, s = 120 + (r * 7 + c * 3) % 11, (r * 13 + c * 5) % 128, (r + c) % 2
    return bf16(s * 32768 + e * 128 + m)


def w_fp8(r, c):
    e = 0 if c % 17 == 0 else 3 + (r * 5 + c * 7) % 8
    return scale(r) * fp8(((r + c) % 2) * 128 + e * 8 + (r * 3 + c) % 8)


def w_int8(r, c):
    return scale(r) * ((r * 31 + c * 17) % 255 - 127)


def w_int4(r, c):
    n = (r * 3 + c * 5) % 16
    return (n - 16 if n >= 8 else n) * (1 + (r + c // 64) % 3) / 16.0


def product(w, m, r):
    xs = [x(c) / 4.0 for c in range(COLS)] if m is None else [x(m * COLS + c) / 4.0 for c in range(COLS)]
    return sum(w(r, c) * xs[c] for c in range(COLS))


want = {}
for name, w in [("bf16", w_bf16), ("fp8", w_fp8), ("int8", w_int8), ("int4", w_int4)]:
    want[f"{name}_matvec"] = [product(w, None, r) for r in (0, 19, 39)]
    want[f"{name}_matmul"] = [product(w, 0, 0), product(w, 1, 19), product(w, 2, 39)]

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
