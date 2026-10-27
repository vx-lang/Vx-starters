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
| `network.vx` | The traffic a plan puts on the links between the leaf switches and the spine, which all services share: the all-to-all, the cache gather and the cache handoff, against each leaf link's capacity |
| `report.vx`, `examples/datacenter.vx`, `examples/procure.vx` | A plan as text (services, placements, fleet use) and what losing a domain or a leaf switch does to it, and the headroom; a whole datacenter in one program, and the cheapest fleet that carries a load (below) |
| `emit.vx`, `admit.py` | The bridge to the compiler (below): what the planner believes, printed as lines, and a Vx program for each, compiled against the machine files of the toolchain |
| `plan.vx` | `Service` (model, request rate, prompt, output, targets for time per token and time to first token, and whether it is served co-located or on separate prefill and decode pools); `evaluate`, `evaluate_prefill`, `evaluate_decode`, `best_role`, `schedule` |

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
3. `best_role` tries every candidate the free part of the fleet offers and takes the one that
   covers the rate still needed for the least an hour in *whole replicas* (`price`: a rack that
   could serve a hundred times the rate still costs a whole rack), then the one with the cheaper
   request.
4. `schedule` places a replica at a time until a service's rate is met or nothing feasible is left,
   for each service in turn. A greedy depends on the order, since the first service takes the best
   of what is free, so it tries the order with the most demanding service first and, for up to 5
   services, every other order, and keeps the plan with the least rate unmet.

### Prefill and decode on separate pools

A service's `mode` says how it is served: 0 co-located, 1 disaggregated, 2 whichever leaves less of
it unmet and then costs less.

* **Co-located.** A replica takes its prefills and its decode steps in turns.
* **Disaggregated.** The prompt is read on a pool of *prefill* replicas, its KV cache is sent over
  the network, and the tokens are written on a pool of *decode* replicas. Each pool is chosen for
  its own work and can sit on other GPUs and be split another way: prefill is arithmetic and wants
  FLOP/s, decode is memory traffic and wants HBM and bandwidth, and no prefill steals a decode
  step. A prefill replica serves `1 / prefill_time` requests a second and a decode replica
  `batch / (out * step)`. The first token waits for the prefill and for the transfer of the
  prompt's cache (prompt x cache bytes per token), which crosses at the slower NIC of the two
  sides over as many GPUs in parallel as the smaller side has, plus the latency the graph gives.
  The two pools are grown together, the one with less capacity getting the next replica, so that
  neither takes every domain the other needs. The first decode replica fixes the cache format and
  what the prefill replicas are priced against; a later decode replica must be no worse for the
  transfer. If no prefill replica fits, the decode replicas are given back.

On the test fleet of one B200 node and four H200 nodes, DeepSeek-V3 with a 32k prompt, 512 output
tokens, 2.4 requests a second and a first token in half a second:

| | placement | cost an hour |
| --- | --- | --- |
| co-located | the B200 node, and all four H200 nodes as one group of 32 (the only H200 layout that is fast enough) | 112 |
| disaggregated | prefill on the B200 node, decode on one H200 node (tensor parallel attention, the cache cut in 8) | 52 |

On a fleet with a big NVL72 rack the co-located replicas win, since one rack does both jobs: the
pools pay for two sets of weights. The scheduler takes whichever is better in each case.

In the test, the mixed fleet (one NVL72 rack, four H200 nodes, eight A100-80 nodes) serves 20
requests a second of DeepSeek-V3 at 8k and 10 of GLM-4.5 at 32k. Taking GLM first gives it the
NVL72 rack and leaves 93% of the demand unmet; the best order puts DeepSeek on four H200 nodes (fp8
weights and cache, batch 920) and GLM on the rack and all eight A100 nodes (in groups of 3, 3 and 2),
leaving 14% of the GLM rate unmet. A plan that is best over all orders is still a greedy's: it is
not the best split of the fleet.

## Speculative decoding and the prefix cache

Two things that change what a service costs, as parameters of a `Service`:

* `with_spec(s, t, acc)`: speculative decoding (DeepSeek's and GLM's multi-token prediction). A step
  works on `t` tokens of each sequence (the token it has and `t - 1` drafts to check) and gives it
  `acc` tokens on average, so the time between tokens is `step / acc`. The weights, the cache and the
  layers are read once for all `t` tokens; the arithmetic, the attention and the traffic between GPUs
  grow with `t`; and `t` tokens touch more experts than one (for one sequence, 8 of 256 become 15.75).
* `with_hit(s, hit)`: a share `hit` of each prompt is in the prefix cache. Only the rest is computed,
  but it attends to all of it, and the weights are read once either way.

At 30 ms a token on an H200 node (DeepSeek-V3, 8k prompt), speculation with 2 tokens a step and 1.8
given lets the batch grow from 112 to 678 and the rate go from 2.4 to 5.1 requests a second; half of the
prompt cached takes the first token from 0.15 to 0.08 s. The tests check both against hand sums on
machines with one limit, and `check.py` against Python.

## A datacenter in one program

`examples/datacenter.vx` is the whole of the library in 30 lines of Vx: the fleet (two NVL72 racks,
twelve H200 nodes, twenty-four A100-80 nodes, seven leaf switches with the spine links cut to half),
five services (the chat services with speculation and half of each prompt cached, the agents with
speculation and 60% cached), `schedule`, and the report. `vxc -a run-jit examples/datacenter.vx`
prints:

```
services (requests a second; cost in units an hour)
  chat-ds: DeepSeek-V3, 8192 + 512 tokens, 30.00 asked, 30.00 served (0% unmet), 40 GPUs, 76
  chat-glm: GLM-4.5, 4096 + 512 tokens, 20.00 asked, 20.00 served (0% unmet), 56 GPUs, 68
  docs-ds: DeepSeek-V3, 131072 + 1024 tokens, 2.00 asked, 2.00 served (0% unmet), 88 GPUs, 304
  repo-glm: GLM-4.5, 1000000 + 256 tokens, 0.0200 asked, 0.0200 served (0% unmet), 72 GPUs, 288
  agents-ds: DeepSeek-V3, 32768 + 256 tokens, 15.00 asked, 15.00 served (0% unmet), 64 GPUs, 160
placements
  chat-ds both    H200 x1 of 8 (domains 2), attention data parallel, context split 1, weights 1B, cache 1B, batch 976, 41.9 ms a token, first token 0.08 s, HBM 90%, 9.58 of 9.58 a second
  ...
  chat-glm decode  A100-80 x4 of 8 (domains 16 17 18 19), attention data parallel, context split 1, weights 2B, cache 2B, batch 416, 37.8 ms a token, HBM 90%, 20.00 of 21.50 a second
  chat-glm prefill H200 x1 of 8 (domains 13), attention data parallel, context split 1, weights 2B, cache 2B, first token 0.07 s, HBM 79%, 15.20 of 15.20 a second
  chat-glm prefill A100-80 x2 of 8 (domains 20 21), attention data parallel, context split 1, weights 2B, cache 2B, first token 0.18 s, HBM 90%, 4.80 of 5.61 a second
  docs-ds both    B200 x1 of 72 (domains 1), attention data parallel, context split 8, weights 1B, cache 1B, batch 128, 7.8 ms a token, first token 0.44 s, HBM 18%, 1.99 of 1.99 a second
  repo-glm both    B200 x1 of 72 (domains 0), attention data parallel, context split 4, weights 2B, cache 2B, batch 18, 25.6 ms a token, first token 36.79 s, HBM 71%, 0.0200 of 0.0269 a second
  ...
fleet
  B200 x72: 2 domains (144 GPUs), 2 in use, 0 free
  H200 x8: 12 domains (96 GPUs), 12 in use, 0 free
  A100-80 x8: 24 domains (192 GPUs), 10 in use, 14 free
  spend 896 an hour
network (GB a second between a leaf switch and the spine)
  leaf 3: 23.46 out, 0.00 in, of 1200 (2%)
  leaf 4: 0.00 out, 23.46 in, of 800 (3%)
headroom: 1.05 times this load before something goes unmet
losses (the services placed again, in the same order)
  a domain in use (24 tried): 14 leave something unmet
  the worst, domain 0 (B200 x72):
    1.49 of a service more unmet, 176 an hour less
  the worst leaf switch, 0 (1 domains):
    1.49 of a service more unmet, 176 an hour less
```

Read it as a design review. The GLM chat service is split across SKUs: it decodes on four A100 nodes
and prefills on an H200 node and two pairs of A100 nodes. The million-token GLM service takes a whole
NVL72 rack, with the context cut in 4, and the long documents take the other. The 14 free A100 nodes
are the slack, and they are of little use to the rest: the fleet carries only 1.05 times this load, and
losing a rack leaves a service and a half unmet. A planner that says that before the hardware is bought,
or before a rack fails, is the point of writing it as a program.

The build checks that the report adds up (`check_example.py`): every domain is in at most one
placement, the GPUs of a service are those of its placements, the domains in use are the ones listed,
the spend is the sum of the services' costs, what a service is served is what its replicas serve, and
the unmet share follows from the asked and served.

### The network

The planner prices a group's own traffic with the links the group has to itself. The spine is shared:
every placement that spans two leaves, and every cache handed from a prefill pool to a decode pool on
another leaf, puts bytes on the same leaf links. `network.vx` adds them up. Per request a decode or
co-located replica moves the all-to-all of every MoE layer for each output token, a co-located or
prefill replica the all-to-all of the prompt's new tokens and the gather of their cache between the GPUs
of the group, and a prefill replica the cache of the whole prompt to the decode pool (split by what each
decode replica serves). Bytes in a group are taken to be spread evenly over the pairs of GPUs, so the
flow from one domain to another is the total times the share of GPUs each has; a flow between leaves
goes up the sender's link and down the receiver's; and the flows are scaled by the requests a second
each placement serves.

In the example the only traffic over the spine is the cache of the GLM chat prompts, 15.2 requests a
second from the H200 prefill node on leaf 3 to the A100 decode nodes on leaf 4, 23.5 GB a second, 2% of
the 1200 GB a second of leaf 3's link. The tests work the numbers out by hand: two H200 nodes on two
leaves serving one request a second move 91.2 GB between their GPUs per request, a quarter of it each
way across; a cache handoff of 32768-token prompts at 2.4 a second is 2.76 GB a second; three nodes,
two on a leaf, send 2/9 of the bytes out of each leaf. It is a report, not a constraint: the planner
does not yet move a placement because a link is full.

### Headroom

`headroom` answers "how much more load does the fleet carry?": the largest factor, to about 1%, by
which every rate can be multiplied with nothing left unmet (below 1 when the load as given is not
met), found by doubling and bisection. On the example fleet it is 1.05. On two H200 nodes carrying half
a replica's rate it is 4.00, as it must be (two nodes, one replica each, and half a replica's rate
asked), and with three replicas' rate asked it is 0.66, which the test checks. The greedy is not
monotone, so this is the boundary of one search, not a promise for every placement.

### What to buy

`examples/procure.vx` runs the planner over a grid of fleets (0 to 2 NVL72 racks, 0 to 12 H200 nodes,
0 to 12 H100 nodes, 0 to 24 A100 nodes) for four services (the chat services, the long documents, the
agents) and prints a line for each: the price (the hourly cost of all its GPUs, used or not) and the
unmet share. The cheapest fleet that carries the load is one rack, twelve H200 nodes and six H100
nodes, 624 an hour. Without H100 in the grid it was one rack, twelve H200 nodes and twenty-four A100
nodes, 720; before speculation and the prefix cache were in the model, two racks and twelve H200
nodes, 816. The H100 cost (2.0, against 2.5 for an H200 and 1.0 for an A100) is a placeholder, so the
choice of H100 over A100 here is the answer to that number as much as to the planner: put your own
prices in before reading it as advice.
`check_procure.py` works the prices out again and
checks that no cheaper fleet of the table carries the load. The grid is coarse and the services are
placed in one order; it is a way to see where the money goes, not an optimizer.

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
6. **Pools.** The cache transfer worked out by hand; the price of whole replicas; a decode replica
   serves `batch / (out * step)`, a prefill replica `1 / prefill_time`, and the first token waits
   for the transfer; on the B200 + H200 fleet the pools cost less than the co-located replicas and
   mode 2 takes them; mode 2 leaves no more unmet than either way, and costs no more, in four
   settings; decode replicas with no prefill to feed them are given back; the pools of a service
   share one cache format.
7. **Losses and the report.** A domain that is down is not used and not free, the search over orders
   keeps to the same fleet, and a service whose every host is down is left whole; the number formatter
   against Python's.
8. **Headroom.** On two nodes: 4.0 times the load for half a replica's rate, 0.66 for three
   replicas' rate, and at the boundary the load is met and 5% more is not.
9. **Speculation and the cache.** A step on one token is the plain step; a compute-bound step on two
   tokens takes twice as long, a memory-bound one reads the extra experts only; a prefill with half
   cached is the work of the new tokens over the old; the batch grows and the rate doubles at 30 ms a
   token; plans with both, in both modes, priced again in Python.
10. **The network.** Bytes a request moves between the GPUs of a group, by hand; the shares of the
    flows between leaves for two and for three nodes; the cache handoff split over two decode
    replicas; no flow for nodes on one leaf; what leaves the leaves enters them.
11. **Python.** `check.py` prices the same layouts again from the architectures and the formulas,
   written out a second time, and every placement of the schedule too; they agree to 1e-9.

## The compiler checks the planner

A planner written in plain code is only as good as its arithmetic. The Vx compiler already knows the
hardware: the machine files in the toolchain's `fleet/` declare each SKU's memory, its number formats
and the rate of every link, and `vxc --machine` admits or refuses a program against them (E6010: the
resident set is over capacity) and prices its transfers. `./build.sh` makes the compiler the second
opinion:

* **Memory.** For 238 layouts and placements `emit.vx` prints the resident set of one GPU (weights,
  cache) and whether `fits` says it fits. `admit.py` writes a Vx program that places tensors of those
  sizes in `Memory::HBM` of the SKU, plus a reserve for the part of HBM a deployment does not use (the
  planner's 10%; the machine files leave that policy to the admission program), compiles it with
  `vxc --machine fleet/<sku>.vx`, and requires the same verdict: admitted exactly when `fits`
  says so, with E6010 when not, and the compiler's count of resident bytes equal to the program's.
  The layouts are a sweep (SKU, nodes, attention parallelism, context parallelism, formats, context
  of 8k and 256k) at the largest batch that fits and the next one, plus every placement of the
  mixed-fleet and pool scenarios.
* **Hardware table.** `hw.vx` and the machine files must agree on HBM to the byte (A100-80 85899345920,
  H200 151397597184, B200 192265846784).
* **Number formats.** The compiler refuses fp8 tensors on an A100 (E6026: the file declares no
  `f8e4m3`) and admits int8. So the 1-byte format of a layout is fp8 where the GPU has fp8 cores
  and int8 weight-only where it has not, and the programs are written that way; compute on an A100
  runs at the bf16 rate either way, as `cost.vx` has it.
* **The wire.** For 19 cache transfers `admit.py` generates a machine file for each node (the SKU's
  own file plus NVLink, PCIe and NIC links at the rates of `hw.vx`), sends a tensor of the bytes one
  GPU sends from HBM to `Memory::Remote_HBM`, and reads the time the compiler derives for the NIC hop
  (`derived_cost`, in picoseconds). It must be the planner's wire time, `bytes / (NIC x streams)`
  with the slower NIC of the two sides, to 1e-9.

This checks the planner's *thresholds* and *prices* against the compiler's; it does not check how
the planner arrives at the weight bytes of a layout (the tests do, by hand sums). The compiler also
prices the whole path with each hop in turn (GPU to NIC at PCIe rate, then the NIC): the planner
counts only the wire, as the transfer is pipelined, so the compiler's sum is an upper bound.

I broke the bridge five ways (a wrong HBM size, the cache split over one NIC instead of all, `fits`
without the weights, with the wrong comparison, with the whole of HBM as the limit); each fails.

## How far to trust the numbers

A rough check, from memory of the figures DeepSeek published for its own deployment (about 14.8 thousand
output tokens a second per H800 node at about 50 ms a token and an average context of 4989 tokens, and
about 73.7 thousand input tokens a second per node for prefill, 56% of them cache hits; I have not
re-checked these). The model gives one H200 node 33.9 thousand output tokens a second at 50 ms a token
and a context of 4989, and 58 thousand prompt tokens a second of prefill. An H200 has 1.43 times the
memory bandwidth of an H800 (4.8 against 3.35 TB/s), which would put the decode figure near 21 thousand,
so the model is about 1.6 times too fast there; on prefill, with the same arithmetic rate and 44% of the
prompt computed, the published figure is about 32 thousand computed tokens a second, and the model is
about 1.8 times too fast. That is the direction the README says to expect (uniform routing, efficiency
factors that are guesses, no queueing), and under a factor of two. Before trusting an absolute number,
set the factors in `cost.vx` from a measured run; the comparison between two fleets or two layouts is
less sensitive to them than the throughput of either.

## What it is not

* **The numbers are not measured.** Hardware figures are typed from memory of the vendor sheets and
  not checked (see the note in `hw.vx`); the cost of a GPU is a placeholder, not a price. The 40%,
  75% and 70% are the usual size, not a measurement. Routing is uniform, and a real router is not,
  so a real step is slower. Put your own figures in and measure before using a plan.
* **The pools are a heuristic.** The decode pool comes first and fixes the cache format; the two
  grow together. It does not try the prefill pool first, a different cache format for each pool
  (with a conversion), or a rebalance after the fact. The transfer is priced as the cache over the
  NICs of one replica; the spine, shared by all the services at once, is not.
* **No queueing.** A service is a steady rate, with no bursts and no waiting.
* **Missing:** pipeline parallelism, speculative
  decoding and prefix caching.

## Toolchain

It needs Vx `main` newer than v0.0.3 (39a12e0a or later): `std::vec` and structs with `Copy`.
No CUDA is needed. Run `./build.sh`; set `VX_HOME` for a toolchain other than `~/.vx/current`.
