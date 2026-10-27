"""Check the table printed by examples/procure.vx.

Reads it on stdin. The price of each fleet is worked out again from the counts (72 GPUs at 4 for a rack,
8 at 2.5 for an H200 node, 8 at 2 for an H100 node, 8 at 1 for an A100 node); the cheapest fleet that carries the load must be
a fleet of the table with no unmet load, and no fleet of the table with no unmet load may cost less.
"""

import re
import sys

text = sys.stdin.read()
failed = []


def fail(msg):
    failed.append(msg)
    print(f"FAIL: {msg}")


rows = []
for m in re.finditer(r"^  (\d+) racks, (\d+) H200 nodes, (\d+) H100 nodes, (\d+) A100 nodes: (\d+), unmet ([\d.]+)$", text, re.M):
    racks, h, g, a, price, unmet = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)), float(m.group(6))
    want = racks * 72 * 4.0 + h * 8 * 2.5 + g * 8 * 2.0 + a * 8 * 1.0
    if abs(price - want) > 0.5:
        fail(f"{racks} racks, {h} H200, {g} H100, {a} A100 is priced {price}, not {want}")
    rows.append((racks, h, g, a, price, unmet))
if len(rows) != 80:
    fail(f"expected 80 fleets in the table, found {len(rows)}")

m = re.search(r"^cheapest that carries it: (\d+) racks, (\d+) H200 nodes, (\d+) H100 nodes, (\d+) A100 nodes, (\d+) an hour$", text, re.M)
if not m:
    fail("no cheapest line")
else:
    key = tuple(int(m.group(i)) for i in range(1, 6))
    feasible = [r for r in rows if r[5] < 0.005]
    if not feasible:
        fail("the table has no fleet with nothing unmet")
    else:
        cheapest = min(feasible, key=lambda r: r[4])
        if key != cheapest[:5]:
            fail(f"the program says {key} is the cheapest, the table says {cheapest[:5]}")
# The report that follows is of that fleet: its services are all met.
for m in re.finditer(r"\((\d+)% unmet\)", text):
    if int(m.group(1)) != 0:
        fail("the report of the cheapest fleet has a service unmet")
if failed:
    sys.exit(1)
print(f"ok: the table of {len(rows)} fleets adds up, and the cheapest is the cheapest")
