"""Check the numbers gpu-stencil prints against the same formulas in Python (f64).

Reads the program's output on stdin. Exits non-zero if an entry differs by more than 1e-3 plus
1e-4 relative to the entry's size.
"""

import sys


def x(i):
    return ((i * 37) % 101) / 8.0 - 6.0


# jacobi_2d: 40 x 40, 3 steps, the border stays.
a = [[x(i * 40 + j) / 4.0 for j in range(40)] for i in range(40)]
for _ in range(3):
    b = [row[:] for row in a]
    for i in range(1, 39):
        for j in range(1, 39):
            b[i][j] = 0.2 * (a[i][j] + a[i][j - 1] + a[i][j + 1] + a[i + 1][j] + a[i - 1][j])
    a = b

# hotspot_step: 20 x 30, two steps.
t = [[x(i * 30 + j) / 4.0 + 60.0 for j in range(30)] for i in range(20)]
p = [[x(i * 30 + j + 5) / 64.0 for j in range(30)] for i in range(20)]
for _ in range(2):
    n = [row[:] for row in t]
    for i in range(20):
        for j in range(30):
            c = t[i][j]
            north = t[i - 1][j] if i > 0 else c
            south = t[i + 1][j] if i < 19 else c
            west = t[i][j - 1] if j > 0 else c
            east = t[i][j + 1] if j < 29 else c
            n[i][j] = c + 0.5 / 2.0 * (p[i][j] + (north + south - 2 * c) / 8.0 + (east + west - 2 * c) / 4.0 + (80.0 - c) / 16.0)
    t = n

# hotspot_pyramid_inner_write: four 16 x 16 tiles that overlap by 4.
T = [[x(i * 28 + j) / 4.0 + 60.0 for j in range(28)] for i in range(28)]
P = [[x(i * 28 + j + 5) / 16.0 for j in range(28)] for i in range(28)]
out = [[0.0] * 28 for _ in range(28)]
for bi in range(2):
    for bj in range(2):
        now = [[T[bi * 12 + y][bj * 12 + z] for z in range(16)] for y in range(16)]
        for _ in range(2):
            nxt = [row[:] for row in now]
            for y in range(1, 15):
                for z in range(1, 15):
                    c = now[y][z]
                    nxt[y][z] = c + 0.25 * (P[bi * 12 + y][bj * 12 + z] + now[y - 1][z] + now[y + 1][z] + now[y][z - 1] + now[y][z + 1] - 4 * c)
            now = nxt
        for wy in range(12):
            for wx in range(12):
                out[bi * 12 + wy + 2][bj * 12 + wx + 2] = now[wy + 2][wx + 2]

want = {
    "jacobi_2d": [a[1][1], a[20][17], a[38][38]],
    "hotspot_step": [t[0][0], t[10][15], t[19][29]],
    "hotspot_pyramid": [out[2][2], out[13][14], out[25][25]],
}

failed = 0
seen = set()
for line in sys.stdin:
    name, _, rest = line.partition(": ")
    if name not in want:
        continue
    seen.add(name)
    got = [float(v) for v in rest.split()]
    for g, w in zip(got, want[name], strict=True):
        if abs(g - w) > 1e-3 + 1e-4 * abs(w):
            print(f"FAIL: {name} printed {g}, Python gives {w}")
            failed += 1
for name in want.keys() - seen:
    print(f"FAIL: no {name} line")
    failed += 1
if failed:
    sys.exit(1)
print("ok: Python computes the same entries")
