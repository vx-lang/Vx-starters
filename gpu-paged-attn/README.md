# gpu-paged-attn

Decode attention for a batch of sequences over a paged key-value cache on the GPU, of any length,
and the kernel that appends a token to the cache.

```console
$ ./build.sh
...
ok
ok: Python computes the same entries
kernels: vx_npu_kernel_0, vx_npu_kernel_1, vx_npu_kernel_2, one sm_80 module of 36447 bytes of PTX
ok: ptxas assembles them for sm_80 (A100)
ok: the executable prints the same
```

The kernels are in `ops.vx`. Each is a `spawn on(Topology::Device)` region with `across` loops.
They are the GPU form of the [`paged-kv`](../paged-kv) starter.

The cache is a pool of pages of 16 positions, and a block table maps a sequence's page number to a
page of the pool, so sequences of different lengths share the pool and a page can be anywhere.
The layout is flat, so the writes that go to a page the table names can be atomic stores (the
race proof refuses a plain write at an index that only the data knows):

    element (page, head, slot, d) is at ((page * heads + head) * 16 + slot) * 64 + d

| Kernel | What it does |
| --- | --- |
| `kv_append` | Writes each sequence's new keys and values (`[seqs][heads][64]`) at position `lens[s]`; the page must already be in the table |
| `kv_advance` | `lens[s] += 1` |
| `paged_attention_decode` | One new token of each sequence, all heads: `softmax(q . K * scale) . V` over `lens[s]` positions, any length |

`paged_attention_decode` has no length limit. A block (sequence, head) walks the cache in chunks of
128 positions with an online softmax: a running maximum, a running sum and the running weighted
sum of V, rescaled when a later chunk raises the maximum. The loop over chunks is bounded by
`max_len`, a host value, and not by `lens[s]`: the race proof wants the control around a lane
loop to be the same in every lane, so the lanes mask positions at or past their sequence's
length. Every sequence needs at least one position.

The test has three sequences of 37, 130 and 5 positions (so the second needs two chunks) in a pool
of 64 pages, with the pages scattered by a permutation. It appends one token to each, advances
the lengths, and decodes. Keys grow with position, so a later chunk raises the maximum and the
rescaling is used. The host reference computes the keys and values from their formula and does a
plain softmax over all positions, so it never reads the cache. The append is also checked
directly: the new token's keys are where the table says, and the number of nonzero cells in the
cache is exactly the number of positions that should hold a number.

An fp8 cache would be a variant of these kernels that decodes on read, as
[`gpu-quant-ops`](../gpu-quant-ops) does for weights; it is not here.

`test.vx` imports the library, moves inputs to GPU memory with `transfer(t, Memory::HBM)`, runs
each kernel, brings the result back, and compares it with the same formula written as host
code. The inputs are whole numbers scaled by 1/8, so Python reads the same matrices. `build.sh`
checks four things:

1. **The results.** The program prints `ok` when the append put the keys where the table says and the decode is within 1e-4 of the host attention.
2. **Python.** `check.py` computes three entries of each result in f64.
3. **The kernels.** `vxc --emit-llvm` carries the GPU module as PTX in a string. `kernels.py`
   pulls it out, checks it targets `sm_80`, has one kernel for each region and calls nothing
   outside itself, and `ptxas` must assemble it.
4. **An executable.** `vxc -a emit-obj`, a link by hand (vx-lang/Vx#1499), and the executable
   must print the same.

No GPU is needed. Without one, the dispatch library runs the same regions on the host, so
this project has only been run that way. Where a GPU is present, the same program uses it.

## Toolchain

It needs Vx `main` from 2026-10-25 or later (d25deea9): the machine file `fleet/a100-40.vx`
uses `levels:`, which v0.0.3 and the 2026-10-13 build do not read. `across` loops need it too.
Assembling the kernels needs `ptxas` (CUDA toolkit), and a CUDA toolchain on a machine with no
NVIDIA driver links against the toolkit's stub, as in [`gpu-norms`](../gpu-norms).
