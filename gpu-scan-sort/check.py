"""Check the entries gpu-scan-sort prints against the same scan, histogram and sort in Python.

Reads the program's output on stdin. All numbers are integers, so they must be equal.
"""

import sys


def num_at(i):
    return (i * 37 + i // 7) % 11 - 3


def key_at(i):
    return (i * 73 + i // 5 * 11) % 256


def scan(n):
    out, run = [], 0
    for i in range(n):
        out.append(run)
        run += num_at(i)
    return [out[n - 1], out[n // 2], out[0]]


want = {f"scan_{n}": scan(n) for n in (1, 255, 256, 257, 5000, 65536)}
keys = [key_at(i) for i in range(5000)]
bins = [keys.count(b) for b in range(256)]
want["histogram"] = [bins[0], bins[100], bins[255]]
order = sorted(range(5000), key=lambda i: keys[i])  # Python's sort is stable
want["sort"] = [keys[order[0]], keys[order[2500]], order[4999]]

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
