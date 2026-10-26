"""Check the numbers fleet-graph prints against the same formulas in Python (f64).

Reads the program's output on stdin. The model counts, the cost of six layouts and the layout of
every placement in the schedule are worked out again here from the architectures and the
formulas of cost.vx, written out a second time, and must agree to 1e-9.
"""

import sys

GIB = 1073741824.0
TERA, GIGA = 1e12, 1e9

# name: (hbm, hbm_bw, bf16, fp8, nvlink, nic)
SKU = {
    "A100-80": (80 * GIB, 2.039 * TERA, 312 * TERA, 0.0, 300 * GIGA, 25 * GIGA),
    "H100": (80 * GIB, 3.35 * TERA, 989 * TERA, 1979 * TERA, 450 * GIGA, 50 * GIGA),
    "H200": (141 * GIB, 4.8 * TERA, 989 * TERA, 1979 * TERA, 450 * GIGA, 50 * GIGA),
    "B200": (183359 * 1048576.0, 7.7 * TERA, 2250 * TERA, 4500 * TERA, 900 * GIGA, 50 * GIGA),
    "toy": (1e30, 1e12, 1e30, 0.0, 1e30, 1e30),
    "toy-flops": (1e30, 1e30, 1e15, 0.0, 1e30, 1e30),
}

MFU, MEM_EFF, COMM_EFF, LAYER_OVERHEAD, USABLE = 0.4, 0.75, 0.7, 0.00003, 0.9


class Model:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.moe_layers = self.layers - self.dense_layers

    def attn(self):
        h = self.hidden
        if self.kind == "mla":
            return (h * self.q_lora + self.q_lora * self.heads * (self.nope + self.rope)
                    + h * (self.kv_lora + self.rope) + self.kv_lora * self.heads * (self.nope + self.v)
                    + self.heads * self.v * h)
        return h * self.heads * self.hd + 2 * h * self.kv_heads * self.hd + self.heads * self.hd * h

    def expert(self):
        return 3 * self.hidden * self.moe_inter

    def routed(self):
        return self.moe_layers * self.experts * self.expert()

    def total(self):
        dense = self.dense_layers * 3 * self.hidden * self.dense_inter
        shared = self.moe_layers * self.shared * self.expert()
        router = self.moe_layers * self.hidden * self.experts
        return self.layers * self.attn() + dense + self.routed() + shared + router + 2 * self.vocab * self.hidden

    def layer_active(self):
        dense = self.dense_layers * 3 * self.hidden * self.dense_inter
        moe = self.moe_layers * ((self.topk + self.shared) * self.expert() + self.hidden * self.experts)
        return self.layers * self.attn() + dense + moe

    def active(self):
        return self.layer_active() + self.vocab * self.hidden

    def resident(self):
        return self.total() - self.routed() - self.vocab * self.hidden

    def dense_part(self):
        return self.total() - self.routed()

    def kv_elems(self):
        if self.kind == "mla":
            return self.layers * (self.kv_lora + self.rope)
        return self.layers * 2 * self.kv_heads * self.hd

    def pair(self, decode):
        if self.kind == "mla":
            if decode:
                return 2 * self.heads * (2 * self.kv_lora + self.rope)
            return 2 * self.heads * (self.nope + self.rope + self.v)
        return 4 * self.heads * self.hd

    def out_dim(self):
        return self.v if self.kind == "mla" else self.hd


DS = Model(kind="mla", layers=61, dense_layers=3, hidden=7168, vocab=129280, heads=128, q_lora=1536,
           kv_lora=512, nope=128, rope=64, v=128, dense_inter=18432, experts=256, topk=8, shared=1, moe_inter=2048)
GLM = Model(kind="gqa", layers=92, dense_layers=3, hidden=5120, vocab=151552, heads=96, kv_heads=8, hd=128,
            dense_inter=12288, experts=160, topk=8, shared=1, moe_inter=1536)


class Dep:
    def __init__(self, sku, domains, gpd, dp, cp, wb, kb, cross, lat):
        self.hbm, self.bw, self.bf16, self.fp8, self.nvl, self.nic = SKU[sku]
        self.domains, self.gpd, self.dp, self.cp, self.wb, self.kb = domains, gpd, dp, cp, wb, kb
        self.cross, self.lat = cross, lat
        self.g = domains * gpd

    def peak(self):
        return self.fp8 if self.wb <= 1.0 and self.fp8 > 0 else self.bf16

    def group_bw(self):
        return self.nvl * COMM_EFF if self.domains == 1 else min(self.nvl, self.cross) * COMM_EFF


def seqs_per_group(d, b):
    if d.dp == 1:
        groups = d.g // d.cp
        return (b + groups - 1) // groups
    return b


def kv_shard(m, d):
    if d.dp == 1 or m.kind == "mla":
        return d.cp
    return d.cp * min(d.g // d.cp, m.kv_heads)


def weight_bytes(m, d):
    routed = m.routed() * d.wb / d.g
    return routed + m.dense_part() * d.wb if d.dp == 1 else routed + m.dense_part() * d.wb / d.g


def kv_bytes(m, d, b, s):
    return seqs_per_group(d, b) * s * m.kv_elems() * d.kb / kv_shard(m, d)


def touched(m, tokens):
    return m.experts * (1.0 - (1.0 - m.topk / m.experts) ** tokens)


def all_to_all(m, d, tokens):
    per_gpu = tokens / d.g
    dispatch = 1.0 if d.wb <= 1.0 else 2.0
    nbytes = per_gpu * m.topk * m.hidden * (dispatch + 2.0)
    local = nbytes * (d.gpd - 1) / d.g / (d.nvl * COMM_EFF)
    if d.domains == 1:
        return local
    remote = nbytes * (d.g - d.gpd) / d.g / (d.cross * COMM_EFF)
    return max(local, remote) + 2.0 * d.lat


def all_reduce(d, nbytes):
    t = 2.0 * (d.g - 1.0) / d.g * nbytes / d.group_bw()
    return t + 2.0 * d.lat if d.domains > 1 else t


def decode_step(m, d, b, s, t=1.0):
    tok = b * t
    routed_read = m.moe_layers * touched(m, tok) / d.g * m.expert() * d.wb
    resident = m.resident() * d.wb / (1 if d.dp == 1 else d.g)
    mem = (routed_read + resident + kv_bytes(m, d, b, s)) / (d.bw * MEM_EFF)
    if d.dp == 1:
        attn = seqs_per_group(d, b) * t * s / d.cp * m.layers * m.pair(True)
    else:
        attn = tok * s * m.layers * m.pair(True) / d.g
    comp = (2.0 * m.active() * tok / d.g + attn) / (d.peak() * MFU)
    comm = m.moe_layers * all_to_all(m, d, tok)
    if d.dp == 0:
        comm += m.layers * all_reduce(d, tok * m.hidden * 2.0)
    if d.cp > 1:
        partial = seqs_per_group(d, b) * t * m.heads * (m.out_dim() + 2) * 4.0
        comm += m.layers * ((d.cp - 1) / d.cp * partial / d.group_bw() + 2.0 * d.lat)
    return max(mem, comp) + comm + m.layers * LAYER_OVERHEAD


def prefill_time(m, d, n, hit=0.0):
    fresh = n * (1.0 - hit)
    old = n - fresh
    flops = 2.0 * m.layer_active() * fresh + (fresh * old + fresh * fresh / 2.0) * m.layers * m.pair(False)
    comp = flops / (d.g * d.peak() * MFU)
    resident = m.resident() * d.wb / (1 if d.dp == 1 else d.g)
    mem = (m.routed() * d.wb / d.g + resident) / (d.bw * MEM_EFF)
    comm = m.moe_layers * all_to_all(m, d, fresh)
    if d.dp == 0:
        comm += m.layers * all_reduce(d, fresh * m.hidden * 2.0)
    kv_layer = fresh * m.kv_elems() / m.layers * d.kb
    comm += m.layers * ((d.g - 1.0) / d.g * kv_layer / d.group_bw() + 2.0 * d.lat)
    return max(mem, comp) + comm + m.layers * LAYER_OVERHEAD


# The services of the test, by the tag of the placement lines: model, prompt, output, the target
# for a token and for the first token. All have a largest batch of 1024.
SERVICES = {
    "placed": {0: (DS, 8192, 1024, 0.05, 5.0, 1.0, 1.0, 0.0), 1: (GLM, 32768, 512, 0.05, 10.0, 1.0, 1.0, 0.0)},
    "pooled": {0: (DS, 32768, 512, 0.06, 0.5, 1.0, 1.0, 0.0)},
    # speculative decoding (2 tokens a step, 1.8 given) and half of each prompt cached
    "spec0": {0: (DS, 8192, 1024, 0.03, 5.0, 2.0, 1.8, 0.5)},
    "spec1": {0: (DS, 8192, 1024, 0.03, 5.0, 2.0, 1.8, 0.5)},
}

want = {
    "models": [DS.total(), DS.active(), GLM.total()],
    "models_glm": [GLM.active(), DS.kv_elems(), GLM.kv_elems()],
}
toy = Dep("toy", 1, 8, 1, 1, 1.0, 1.0, 0.0, 0.0)
toyf = Dep("toy-flops", 1, 8, 1, 1, 2.0, 2.0, 0.0, 0.0)
want["cost_toy"] = [decode_step(DS, toy, 1, 1000), prefill_time(DS, toyf, 1000), prefill_time(DS, toyf, 1000)]
n72 = Dep("B200", 1, 72, 1, 1, 1.0, 1.0, 0.0, 0.0)
want["cost_nvl72"] = [weight_bytes(DS, n72), decode_step(DS, n72, 512, 8192), prefill_time(DS, n72, 8192)]
two = Dep("H200", 2, 8, 1, 1, 1.0, 1.0, 50e9, 2e-6)
want["cost_two_nodes"] = [weight_bytes(DS, two), decode_step(DS, two, 256, 8192), prefill_time(DS, two, 8192)]
tg = Dep("H200", 1, 8, 0, 1, 2.0, 2.0, 0.0, 0.0)
want["cost_tp_glm"] = [weight_bytes(GLM, tg), decode_step(GLM, tg, 32, 8192), prefill_time(GLM, tg, 8192)]
cut = Dep("B200", 1, 72, 1, 8, 1.0, 1.0, 0.0, 0.0)
want["cost_cp8"] = [kv_bytes(DS, cut, 18, 1000000), decode_step(DS, cut, 18, 200000), prefill_time(DS, cut, 200000)]


want["cost_spec"] = [decode_step(GLM, tg, 32, 8192, 2.0), decode_step(DS, cut, 18, 200000, 2.0), prefill_time(GLM, tg, 8192, 0.25)]


def close(a, b):
    return abs(a - b) <= 1e-9 * max(abs(a), abs(b), 1e-300)


failed = 0
seen = set()
fmt_seen = False
placed = 0
for line in sys.stdin:
    name, _, rest = line.partition(": ")
    if name in SERVICES:
        f = rest.split()
        svc, role, sku = int(f[0]), int(f[1]), f[2]
        domains, gpd, dp, cp = (int(v) for v in f[3:7])
        wb, kb, cross, lat = (float(v) for v in f[7:11])
        batch, step, ttft, xfer, xnic, xs, xlat = int(f[11]), float(f[12]), float(f[13]), float(f[14]), float(f[15]), int(f[16]), float(f[17])
        m, prompt, out, slo_tpot, slo_ttft, spec_t, spec_acc, hit = SERVICES[name][svc]
        dep = Dep(sku, domains, gpd, dp, cp, wb, kb, cross, lat)
        placed += 1
        ctx = prompt + out

        def ok(b):
            return (weight_bytes(m, dep) + kv_bytes(m, dep, b, ctx) <= USABLE * dep.hbm
                    and decode_step(m, dep, b, prompt + out // 2, spec_t) / spec_acc <= slo_tpot)

        if role in (0, 2):
            # The step is priced again, and the batch is the largest that fits and meets the target.
            want_step = decode_step(m, dep, batch, prompt + out // 2, spec_t) / spec_acc
            if not close(step, want_step):
                print(f"FAIL: decode step {step} of {line.strip()} is {want_step} in Python")
                failed += 1
            if not ok(batch) or (batch < 1024 and ok(batch + 1)):
                print(f"FAIL: batch {batch} is not the largest that fits and meets the target ({line.strip()})")
                failed += 1
        if role == 0:
            want_ttft = prefill_time(m, dep, prompt, hit)
            if not close(ttft, want_ttft):
                print(f"FAIL: first token {ttft} of {line.strip()} is {want_ttft} in Python")
                failed += 1
        if role == 1:
            # Prefill plus the transfer of the prompt's cache over the slower NIC of the two sides, as
            # many GPUs in parallel as the smaller side has.
            nic = min(dep.nic, xnic)
            streams = min(dep.g, xs)
            want_xfer = prompt * m.kv_elems() * dep.kb / (nic * streams) + xlat
            want_ttft = prefill_time(m, dep, prompt, hit) + want_xfer
            if not (close(xfer, want_xfer) and close(ttft, want_ttft)):
                print(f"FAIL: prefill replica {line.strip()} is priced {want_xfer} and {want_ttft} in Python")
                failed += 1
            if weight_bytes(m, dep) + kv_bytes(m, dep, 1, prompt) > USABLE * dep.hbm or want_ttft > slo_ttft:
                print(f"FAIL: prefill replica {line.strip()} does not fit or is late")
                failed += 1
        continue
    if name == "fmt":
        got = rest.split()
        want_fmt = [f"{3.14159:.2f}", f"{2.5:.1f}", f"{0.0004:.2f}", f"{123456.789:.2f}", f"{7.0:.0f}", f"{-1.234:.1f}", f"{99.6:.0f}"]
        formatted = got
        if got != want_fmt:
            print(f"FAIL: print_f printed {got}, Python formats {want_fmt}")
            failed += 1
        fmt_seen = True
        continue
    if name not in want:
        continue
    seen.add(name)
    got = [float(v) for v in rest.split()]
    if len(got) != 3 or not all(close(a, b) for a, b in zip(got, want[name])):
        print(f"FAIL: {name} printed {got}, Python gives {want[name]}")
        failed += 1
for name in want.keys() - seen:
    print(f"FAIL: no {name} line")
    failed += 1
if not fmt_seen:
    print("FAIL: no fmt line")
    failed += 1
if placed == 0:
    print("FAIL: no placements printed")
    failed += 1
if failed:
    sys.exit(1)
print("ok: Python prices the same layouts and placements")
