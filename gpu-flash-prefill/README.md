# gpu-flash-prefill

Causal multi-head attention for a whole prompt on the GPU, tiled with an online softmax, so the
matrix of scores is never written.

```console
$ ./build.sh
...
ok
ok: Python computes the same entries
kernels: vx_npu_kernel_0, one sm_80 module of 36640 bytes of PTX
ok: ptxas assembles them for sm_80 (A100)
ok: the executable prints the same
```

The kernel is in `ops.vx`: `flash_prefill`, a `spawn on(Topology::Device)` region with `across`
loops. It is the GPU form of the [`flash-attn`](../flash-attn) starter.

Prefill attends over a whole prompt. Writing the T x T matrix of scores costs memory that grows
with the square of the prompt, so the kernel never writes it: block (head, query tile) handles 16
queries and walks the keys and values of the same head in tiles of 16, only up to its own tile
(the causal mask). For each query it keeps a running maximum, a running sum and the running
weighted sum of V, and a tile that raises a query's maximum rescales them. The lanes are 16 x 16:
lane (ty, tx) scores query ty against key tx, and later owns columns tx, tx + 16, .. of the 64
outputs of query ty. A tile of Q, K, V and the output being built are in shared memory, 4 KiB each.

`q`, `k`, `v` and `out` are `[heads][T][64]`. T can be any length: a key at or past T is past
every valid query, so the causal mask hides it, and queries at or past T are not written.

The row reductions over 16 scores are serial loops in the code of the block, which runs on one
lane. `lanes::max` and `lanes::sum` over a `[16][16]` tensor would spread them; this version
shows the structure, not the speed.

Not here, from the ticket: the MLA form (attention over a compressed latent), which glm-vx needs,
and a non-causal option.

The test runs 2 heads over 50 positions, three full tiles and one of 2, against naive attention
on the host that builds every row of scores. Keys grow with position, so later tiles raise the
maximum and the rescale is used. Position 0 must equal its own value exactly.

`test.vx` imports the library, moves inputs to GPU memory with `transfer(t, Memory::HBM)`, runs
each kernel, brings the result back, and compares it with the same formula written as host
code. The inputs are whole numbers scaled by 1/8, so Python reads the same matrices. `build.sh`
checks four things:

1. **The results.** The program prints `ok` when the attention is within 1e-4 of the naive host attention.
2. **Python.** `check.py` computes three entries of each result in f64.
3. **The kernels.** `vxc --emit-llvm` carries the GPU module as PTX in a string. `kernels.py`
   pulls it out, checks it targets `sm_80`, has one kernel for each kernel function and calls nothing
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
