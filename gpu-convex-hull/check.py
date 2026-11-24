"""Check what gpu-convex-hull prints against the same steps in Python.

Reads the program's output on stdin. For each input, Python makes the same points, finds the
extreme points, counts the points that the filter keeps, and finds the hull with its own
monotone chain. All numbers are integers, so they must be equal.
"""

import sys


def lcg(seed):
    """The generator of test.vx: each call gives a number in 0..32768."""
    state = seed
    while True:
        state = (state * 1103515245 + 12345) % 2**31
        yield state // 65536


def isqrt(v):
    lo, hi = 0, 32768
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if mid * mid <= v:
            lo = mid
        else:
            hi = mid
    return lo


def make_points(kind, n, seed):
    rng = lcg(seed)
    pts = []
    for i in range(n):
        if kind == "square":
            pts.append((next(rng), next(rng)))
        elif kind == "disc":
            while True:
                x, y = next(rng), next(rng)
                if (x - 16384) ** 2 + (y - 16384) ** 2 <= 16000**2:
                    pts.append((x, y))
                    break
        elif kind == "circle":
            a = i // 4 + 1
            h = isqrt(16383**2 - (a - 16384) ** 2)
            q = i % 4
            pts.append([(a, 16384 + h), (a, 16384 - h), (16384 + h, a), (16384 - h, a)][q])
        elif kind == "lines":
            line = next(rng) % 5
            pts.append((3000 + 6000 * line, next(rng)))
        elif kind == "copies":
            r = next(rng) % 6
            pts.append([(100, 100), (30000, 200), (16000, 31000), (100, 100), (15000, 12000), (16000, 31000)][r])
        elif kind == "one_line":
            x = next(rng) % 16000
            pts.append((x, 2 * x + 7))
        elif kind == "corners":
            r = next(rng)
            pts.append((r % 2 * 32767, r // 2 % 2 * 32767))
    return pts


# The 8 directions, counter-clockwise from straight down.
DIRS = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]


def cross(a, b, p):
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def extremes(pts):
    """For each direction, the lowest index of a point with the largest score."""
    out = []
    for dx, dy in DIRS:
        scores = [dx * x + dy * y for x, y in pts]
        out.append(scores.index(max(scores)))
    return out


def kept(pts, at):
    """The number of points that are not strictly inside the octagon. An edge of length 0 does
    not count, and with no edge left, every point is kept."""
    edges = [(pts[at[j]], pts[at[(j + 1) % 8]]) for j in range(8)]
    edges = [(a, b) for a, b in edges if a != b]
    if not edges:
        return len(pts)
    return sum(1 for p in pts if not all(cross(a, b, p) > 0 for a, b in edges))


def hull(pts):
    """Andrew's monotone chain: the vertices counter-clockwise from the lowest x, then y."""
    pts = sorted(set(pts))
    if len(pts) <= 1:
        return pts
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def pairs(flat):
    return [(flat[2 * i], flat[2 * i + 1]) for i in range(len(flat) // 2)]


INPUTS = {
    "no_points": [],
    "one_point": [(4, 4)],
    "two_points": pairs([7, 1, 3, 4]),
    "triangle": pairs([0, 10, 10, 0, 0, 0]),
    "square_and_middle": pairs([5, 5, 10, 10, 0, 0, 0, 10, 10, 0]),
    "point_on_an_edge": pairs([0, 0, 5, 0, 10, 0, 10, 10, 0, 10]),
    "collinear": pairs([3, 3, 10, 10, 0, 0, 5, 5]),
    "copies": pairs([2, 2, 8, 2, 2, 2, 5, 9, 8, 2, 2, 2]),
    "one_point_three_times": [(6, 6)] * 3,
}
for name, kind, n, seed in [
    ("square_1000", "square", 1000, 1),
    ("square_2000", "square", 2000, 2),
    ("square_1000000", "square", 1000000, 3),
    ("disc_100000", "disc", 100000, 4),
    ("circle_131068", "circle", 131068, 0),
    ("lines_20000", "lines", 20000, 6),
    ("copies_5000", "copies", 5000, 7),
    ("one_line_3000", "one_line", 3000, 8),
    ("corners_4000", "corners", 4000, 9),
]:
    INPUTS[name] = make_points(kind, n, seed)

got = {}
for line in sys.stdin:
    head, sep, rest = line.partition(": ")
    if sep and head.split(" ")[0] in ("input", "extremes", "hull"):
        # "input" lines have words between the numbers: keep the numbers.
        got[head] = [v for v in rest.split() if v.lstrip("-").isdigit()]

failed = 0


def expect(key, want):
    global failed
    if key not in got:
        print(f"FAIL: no line '{key}'")
        failed += 1
    elif [int(v) for v in got[key]] != want:
        print(f"FAIL: {key} printed {' '.join(got[key][:12])} ..., Python gives {' '.join(map(str, want[:12]))} ...")
        failed += 1


for name, pts in INPUTS.items():
    n = len(pts)
    at = extremes(pts) if n else []
    expect(f"input {name}", [n, kept(pts, at) if n else 0])
    if n:
        expect(f"extremes {name}", at)
    h = hull(pts)
    expect(f"hull {name}", [len(h)] + [c for p in h for c in p])

if failed:
    sys.exit(1)
print(f"ok: Python finds the same extreme points, kept points and hull for {len(INPUTS)} inputs")
