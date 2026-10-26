"""Check that the report of examples/datacenter.vx adds up.

Reads the report on stdin. The services' GPUs and costs, the placements' domains and the fleet's
counts are printed separately and must agree: every domain is in one placement at most, the GPUs of a
service are those of its placements, the domains in use are the ones listed, the spend is the sum of
the services' costs, and what was served is what the placements served.
"""

import re
import sys

text = sys.stdin.read()
failed = []


def fail(msg):
    failed.append(msg)
    print(f"FAIL: {msg}")


services = {}
for m in re.finditer(r"^  (\S+): (\S+), (\d+) \+ (\d+) tokens, ([\d.]+) asked, ([\d.]+) served \((\d+)% unmet\), (\d+) GPUs, (\d+)$", text, re.M):
    name = m.group(1)
    services[name] = dict(asked=float(m.group(5)), served=float(m.group(6)), unmet=int(m.group(7)), gpus=int(m.group(8)), cost=int(m.group(9)))
if len(services) != 5:
    fail(f"expected 5 services in the report, found {len(services)}")

placed = {name: dict(gpus=0, served=0.0, n=0, prefill=0.0, decode=0.0) for name in services}
domains = []
for m in re.finditer(r"^  (\S+) (both|prefill|decode)\s+(\S+) x(\d+) of (\d+) \(domains ([\d ]+)\),.* HBM (\d+)%, ([\d.]+) of ([\d.]+) a second$", text, re.M):
    name, role = m.group(1), m.group(2)
    count, per = int(m.group(4)), int(m.group(5))
    members = [int(v) for v in m.group(6).split()]
    if len(members) != count:
        fail(f"{name} {role}: {count} domains but {len(members)} listed")
    domains += members
    p = placed[name]
    p["gpus"] += count * per
    p["n"] += 1
    if float(m.group(8)) > float(m.group(9)) + 0.005:
        fail(f"{name} {role}: serves {m.group(8)} of {m.group(9)} a second")
    if int(m.group(7)) > 90:
        fail(f"{name} {role}: HBM {m.group(7)}% is over the 90% policy")
    p[{"both": "served", "prefill": "prefill", "decode": "decode"}[role]] += float(m.group(8))
if len(domains) == 0:
    fail("no placements in the report")
if len(set(domains)) != len(domains):
    fail("a domain is in two placements")

for name, s in services.items():
    p = placed[name]
    if p["gpus"] != s["gpus"]:
        fail(f"{name}: the service line says {s['gpus']} GPUs, the placements {p['gpus']}")
    if p["n"] == 0 and s["served"] > 0:
        fail(f"{name}: served with no placement")
    # What a service is served is what its replicas serve: both jobs in one, or the prefill pool
    # (prefill is only counted up to what the decode pool takes).
    got = p["served"] if p["prefill"] == 0 and p["decode"] == 0 else p["prefill"]
    if abs(got - s["served"]) > 0.011 * max(1, p["n"]):
        fail(f"{name}: the service line says {s['served']} served, the placements {got}")
    if p["decode"] > 0 and p["decode"] + 0.011 * p["n"] < s["served"]:
        fail(f"{name}: served {s['served']} but the decode pool serves {p['decode']}")

for name, s in services.items():
    want = 100.0 * (s["asked"] - s["served"]) / s["asked"]
    if abs(want - s["unmet"]) > 0.5 + 1.0 / s["asked"]:
        fail(f"{name}: {s['unmet']}% unmet, but {s['asked']} asked and {s['served']} served is {want:.1f}%")

total = sum(s["cost"] for s in services.values())
m = re.search(r"^  spend (\d+) an hour$", text, re.M)
if not m or abs(int(m.group(1)) - total) > len(services):
    fail(f"spend {m.group(1) if m else None} is not the sum of the services' costs {total} (each is rounded)")

in_use = 0
for m in re.finditer(r"^  (\S+) x(\d+): (\d+) domains \((\d+) GPUs\), (\d+) in use, (\d+) free$", text, re.M):
    n, gpus, used, free = int(m.group(3)), int(m.group(4)), int(m.group(5)), int(m.group(6))
    if used + free != n or gpus != n * int(m.group(2)):
        fail(f"fleet line {m.group(0).strip()} does not add up")
    in_use += used
if in_use != len(domains):
    fail(f"the fleet says {in_use} domains are in use, the placements list {len(domains)}")

net = re.findall(r"^  leaf (\d+): ([\d.]+) out, ([\d.]+) in, of (\d+) \((\d+)%\)$", text, re.M)
tol = 0.011 * max(1, len(net))
out_sum = sum(float(o) for _, o, _, _, _ in net)
in_sum = sum(float(i) for _, _, i, _, _ in net)
if abs(out_sum - in_sum) > tol:
    fail(f"the leaves send {out_sum} GB a second to the spine and receive {in_sum}")
for leaf, o, i, cap, util in net:
    want = 100.0 * max(float(o), float(i)) / float(cap)
    if abs(want - int(util)) > 0.6:
        fail(f"leaf {leaf}: {util}% but {o} out, {i} in of {cap}")
if "network (GB a second" not in text:
    fail("no network section")
if not re.search(r"^headroom: [\d.]+ (times this load|of this load)", text, re.M):
    fail("no headroom line")
if "losses" not in text or "the worst leaf switch" not in text:
    fail("no losses section")
if failed:
    sys.exit(1)
print(f"ok: the report adds up ({len(services)} services, {len(domains)} domains in use)")
