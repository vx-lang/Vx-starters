"""Check the table printed by examples/pipeline.vx.

Reads it on stdin. The pipeline and cost formulas themselves are priced again in Python by check.py
(the `pipeline` line of test.vx); here the table must be consistent with itself: every token time
within the target, tokens a second equal to batch over token time, a batch that is a multiple of 16
and below the limit of the search, a microbatch count that is a power of two no bigger than the
batch. And the claims the README makes of it, read off the table, must hold.
"""

import re
import sys

text = sys.stdin.read()
SLO = 0.05
rows = []
failed = []


def fail(msg):
    failed.append(msg)
    print(f"FAIL: {msg}")


pat = re.compile(r"^pp (\S+) ([\d.]+) (\d+) (\d+) ep (\d+) ([\d.]+) ([\d.]+) pipe (\d+) (\d+) ([\d.]+) ([\d.]+)$", re.M)
for m in pat.finditer(text):
    sku, nic, nodes, ctx = m.group(1), float(m.group(2)), int(m.group(3)), int(m.group(4))
    eb, et, etps = int(m.group(5)), float(m.group(6)), float(m.group(7))
    pb, pm, pt, ptps = int(m.group(8)), int(m.group(9)), float(m.group(10)), float(m.group(11))
    line = m.group(0)
    for b, t, tps, what in ((eb, et, etps, "ep"), (pb, pt, ptps, "pipe")):
        if b % 16 or not 16 <= b <= 8192:
            fail(f"{what} batch {b}: {line}")
        if t > SLO + 5e-6:
            fail(f"{what} token time {t} over the target: {line}")
        if abs(b / t - tps) > 0.001 * tps + 0.1:
            fail(f"{what} tokens/s {tps} is not {b}/{t}: {line}")
    if pm & (pm - 1) or not 1 <= pm <= pb:
        fail(f"microbatches {pm}: {line}")
    rows.append((sku, nic, nodes, ctx, etps, ptps))
if len(rows) != 22:
    fail(f"expected 22 lines, found {len(rows)}")

# Read off the table, as the README does: the sweep is the last ten lines.
sweep = {(r[0], r[1]): r for r in rows[-10:] if r[2] == 2 and r[3] == 8192}
for nic in (12.5, 25.0, 50.0, 100.0, 200.0):
    for sku in ("H100", "H200"):
        if (sku, nic) not in sweep:
            fail(f"no {sku} line at {nic} GB/s in the sweep")
if not failed:
    # Both ways only gain from a faster network, and the pipeline does not use it for its
    # all-to-all, so its rate is the same (to the search's step) at every NIC speed.
    for sku in ("H100", "H200"):
        ep = [sweep[(sku, n)][4] for n in (12.5, 25.0, 50.0, 100.0, 200.0)]
        pp = [sweep[(sku, n)][5] for n in (12.5, 25.0, 50.0, 100.0, 200.0)]
        if ep != sorted(ep):
            fail(f"{sku}: the all-to-all rate falls with a faster network: {ep}")
        if max(pp) > 1.02 * min(pp):
            fail(f"{sku}: the pipeline's rate moves with the network by more than 2%: {pp}")
if failed:
    sys.exit(1)
print(f"ok: the pipeline table of {len(rows)} lines is consistent")
