# fleet-graph

A data center as a graph, a roofline cost model for serving mixture-of-experts models on it, and a
scheduler that places services on it. The question it answers: *a fleet of A100, H200 and B200
nodes and NVL72 racks must serve DeepSeek-V3 and GLM-4.5 at different batch sizes and contexts
from 8k to 1M tokens; which model goes where, how is it split, and what is left over?*

```console
$ ./build.sh
...
orders: 0.13595401956524414 0.9272457855670815 0.13595401956524414
placed: 0 H200 1 8 1 1 1 1 0 0 920 0.04198911367110642 0.14984472083737915
placed: 0 H200 1 8 1 1 1 1 0 0 920 0.04198911367110642 0.14984472083737915
...
plan: 0 1.3595401956524413 8
ok
ok: Python prices the same layouts and placements
```

Plain host code: no GPU and no CUDA. Every figure is an estimate (see *What it is not* below).

## The parts

| File | What it holds |
| --- | --- |
| `hw.vx` | `Sku` (A100-40, A100-80, H200, B200: HBM size and bandwidth, bf16 and fp8 FLOP/s, NVLink and NIC rate per GPU, a relative cost) and `Domain` (a node or rack on one NVLink fabric: `hgx8(sku)`, `nvl72(sku)`) |
| `graph.vx` | An undirected graph with a bandwidth and a latency on each edge; `widest` (the largest bottleneck over all paths) and `fastest` (least total latency), both Floyd-Warshall |
| `fleet.vx` | A fleet: a vertex for each domain, one leaf switch for each group of domains, one spine, with the leaf-to-spine links cut by an oversubscription factor. `group_bandwidth` and `group_latency` of a set of domains read off the graph, with a leaf link shared by the members under it; `pick_group` chooses domains of one class that are as close as the graph allows |
| `model.vx` | `Model`: layers, dense layers, hidden size, MLA or GQA attention, experts. Parameter counts, KV size of a token, and the attention FLOPs of a query-key pair all follow from those fields. `deepseek_v3()` and `glm_4_5()` |
| `cost.vx` | `Deploy`: a group of whole domains, attention data or tensor parallel, a context-parallel degree, number formats. `fits`, `decode_step` and `prefill_time` |
| `plan.vx` | `Service` (model, request rate, prompt, output, targets for time per token and time to first token); `evaluate`, `best_for`, `schedule` |

## The cost model

A decode step takes as long as the slower of its memory traffic and its arithmetic, plus the
communication, plus 30 microseconds a layer:

* **Memory.** The routed experts a step touches (under uniform routing, `E (1 - (1 - k/E)^b)` of
  the E experts of a layer for a batch of b), the other weights, and the KV cache, over HBM
  bandwidth at 75%.
* **Arithmetic.** Two FLOPs for each active parameter of each token, plus attention over the cache
  (MLA in the absorbed form for decode), over the peak rate at 40%; fp8 weights use the fp8 rate
  where the GPU has one.
* **Communication.** The all-to-all of every MoE layer (dispatch in fp8 or bf16, combine in bf16),
  split into the part inside a domain at the NVLink rate and the part between domains at the
  network rate the graph gives, at 70%; an all-reduce for each layer with tensor parallel
  attention; the combine of partial softmaxes with context parallelism.
* **Memory fit.** Weights per GPU (experts spread over all GPUs; the rest replicated with data
  parallel attention, split with tensor parallel), plus the cache of the sequences the GPU holds
  (an MLA cache is not split by tensor parallelism; a GQA cache is split over the KV heads),
  within 90% of HBM.

A prefill of n tokens is taken to be sequence parallel over the group: the FLOPs of the layers and
the causal attention over `g` GPUs, the weights read once, the all-to-alls, and a gather of each
layer's cache.

## The scheduler

One algorithm, with no rule about a SKU or a model:

1. A *candidate* is a group of free domains of one class (found by `pick_group`, so its bandwidth
   is what the wiring gives) with a way to split the model: data or tensor parallel attention, a
   context-parallel degree up to 64, weights in bf16 or fp8, a cache in bf16 or fp8 (as far as the
   service allows).
2. `evaluate` finds the largest batch that fits in memory and meets the time-per-token target
   (memory and step time only grow with the batch, so a bisection finds it), the time to first
   token, and from them the requests a second one replica serves, `1 / (ttft + out * step / batch)`,
   and what a request costs.
3. `best_for` tries every candidate the free part of the fleet offers and takes the cheapest
   request that meets both targets.
4. `schedule` places a replica at a time until a service's rate is met or nothing feasible is left,
   for each service in turn. A greedy depends on the order, since the first service takes the best
   of what is free, so it tries the order with the most demanding service first and, for up to 5
   services, every other order, and keeps the plan with the least rate unmet.

In the test, a mixed fleet (one NVL72 rack, four H200 nodes, eight A100-80 nodes) serves 20
requests a second of DeepSeek-V3 at 8k and 10 of GLM-4.5 at 32k. Taking GLM first gives it the
NVL72 rack and leaves 93% of the demand unmet; the best order puts DeepSeek on four H200 nodes (fp8
weights and cache, batch 920) and GLM on the rack and all eight A100 nodes (in groups of 3, 3 and 2), leaving 14% of the GLM rate
unmet. A plan that is best over all orders is still a greedy's: it is not the best split of the fleet.

## What the tests check

`test.vx` returns the number of failed checks and prints `ok` when there are none.

1. **Models.** The derived totals against the published ones (DeepSeek-V3 671B and 37B active,
   GLM-4.5 355B and 32B), and hand-worked sizes: an expert of 44,040,192 numbers, the routed
   experts of 653,908,770,816, the cache of a token.
2. **Graph.** A case worked out by hand, then the widest paths of 12 random graphs against a search
   by thresholds, and the least latencies against Bellman-Ford.
3. **Fleet.** Rates and latencies of groups worked out by hand for a small fleet, including a leaf
   link shared by two members and oversubscription 4, and the choice of a group.
4. **Cost.** A decode step on a machine with only memory bandwidth, and a prefill on one with only
   arithmetic, each equal to the hand sum; weights and cache per GPU for each parallel layout; what
   fits where (DeepSeek-V3 in bf16 does not fit 8 H200; in fp8 it does; a million tokens of bf16
   cache need the cache cut in two); a step never gets shorter with a bigger batch, a longer
   context, a slower network or a slower SKU.
5. **Planner.** No candidate tried is cheaper than the chosen one; the chosen batch is the largest
   that is ok; in the schedule no domain is used twice, every placement meets its targets, a
   replica serves no more than its rate, served plus unmet is the demand, and the same inputs give
   the same plan; less fleet leaves at least as much unmet in the tested cases; an impossible
   service is left whole.
6. **Python.** `check.py` prices the same layouts again from the architectures and the formulas,
   written out a second time, and every placement of the schedule too; they agree to 1e-9.

## What it is not

* **The numbers are not measured.** Hardware figures are typed from memory of the vendor sheets and
  not checked (see the note in `hw.vx`); the cost of a GPU is a placeholder, not a price. The 40%,
  75% and 70% are the usual size, not a measurement. Routing is uniform, and a real router is not,
  so a real step is slower. Put your own figures in and measure before using a plan.
* **A replica is co-located.** Prefill and decode share the GPUs and take turns. Splitting them onto
  different pools (prefill on compute-rich GPUs, decode on bandwidth-rich ones, with the cache moved
  between them) is the next thing to add; the pieces are here.
* **No queueing.** A service is a steady rate, with no bursts and no waiting.
* **Missing:** pipeline parallelism, speculative
  decoding and prefix caching.

## Toolchain

It needs Vx `main` newer than v0.0.3 (39a12e0a or later): `std::vec` and structs with `Copy`.
No CUDA is needed. Run `./build.sh`; set `VX_HOME` for a toolchain other than `~/.vx/current`.
