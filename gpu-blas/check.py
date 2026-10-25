"""Check the numbers gpu-blas prints against the same formulas in Python (f64).

Reads the program's output on stdin. Exits non-zero if an entry differs by more than 1e-3 plus
1e-4 relative to the entry's size.
"""

import sys


def x(i):
    return ((i * 37) % 101) / 8.0 - 6.0


def matrix(rows, cols, f):
    return [[f(i, j) for j in range(cols)] for i in range(rows)]


def mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


# gemm
ga = matrix(48, 48, lambda i, j: x(i * 48 + j) / 4.0)
gb = matrix(48, 48, lambda i, j: x(i * 48 + j + 17) / 4.0)
gc = mul(ga, gb)

# 2mm
ma = matrix(10, 12, lambda i, k: x(i * 12 + k) / 4.0)
mb = matrix(12, 35, lambda k, j: x(k * 35 + j + 1) / 4.0)
mc = matrix(35, 9, lambda k, j: x(k * 9 + j + 2) / 4.0)
md = matrix(10, 9, lambda i, j: x(i * 9 + j + 3))
tmp = [[1.5 * v for v in row] for row in mul(ma, mb)]
prod = mul(tmp, mc)
d = [[md[i][j] * 0.5 + prod[i][j] for j in range(9)] for i in range(10)]

# atax
aa = matrix(300, 270, lambda i, j: x(i * 270 + j) / 8.0)
ax = [x(j + 4) / 4.0 for j in range(270)]
at = [sum(aa[i][j] * ax[j] for j in range(270)) for i in range(300)]
ay = [sum(aa[i][j] * at[i] for i in range(300)) for j in range(270)]

# gemver
v = [[x(i + 40 * s) / 4.0 for i in range(40)] for s in range(8)]
u1, v1, u2, v2, y, z, xv, w = v
a = matrix(40, 40, lambda i, j: x(i * 40 + j) / 8.0)
a = [[a[i][j] + u1[i] * v1[j] + u2[i] * v2[j] for j in range(40)] for i in range(40)]
xn = [xv[i] + sum(0.5 * a[j][i] * y[j] for j in range(40)) + z[i] for i in range(40)]
wn = [w[i] + sum(1.5 * a[i][j] * xn[j] for j in range(40)) for i in range(40)]

want = {
    "gemm": [gc[0][0], gc[17][40], gc[47][47]],
    "two_mm_first": [tmp[0][0], tmp[4][20], tmp[9][34]],
    "two_mm_second": [d[0][0], d[5][4], d[9][8]],
    "atax_first": [at[0], at[150], at[299]],
    "atax_second": [ay[0], ay[100], ay[269]],
    "gemver_outer": [a[0][0], a[13][27], a[39][39]],
    "gemver_transposed": [xn[0], xn[20], xn[39]],
    "gemver_product": [wn[0], wn[20], wn[39]],
}

failed = 0
seen = set()
for line in sys.stdin:
    name, _, rest = line.partition(": ")
    if name not in want:
        continue
    seen.add(name)
    got = [float(t) for t in rest.split()]
    for g, w_ in zip(got, want[name], strict=True):
        if abs(g - w_) > 1e-3 + 1e-4 * abs(w_):
            print(f"FAIL: {name} printed {g}, Python gives {w_}")
            failed += 1
for name in want.keys() - seen:
    print(f"FAIL: no {name} line")
    failed += 1
if failed:
    sys.exit(1)
print("ok: Python computes the same entries")
