# gpu-moe

A mixture-of-experts layer on the GPU with the work grouped by expert: a device-side sort of
(token, expert) pairs, then one block for each pair.

```console
$ ./build.sh
...
ok
ok: Python computes the same entries
kernels: vx_npu_kernel_0, ..., vx_npu_kernel_5, one sm_80 module of 44015 bytes of PTX
ok: ptxas assembles them for sm_80 (A100)
ok: the executable prints the same
```

The kernels are in `ops.vx`, as functions that take tensors in GPU memory, and `moe_forward`
runs them in order. Each is a `spawn on(Topology::Device)` region with `across` loops.

A mixture-of-experts layer sends each token to k experts, and each expert is a SwiGLU network,
`down(silu(gate x) * (up x))`. A kernel that handles one token at a time reads an expert's
weights again for each token that picks it. Here the (token, slot) pairs are sorted by expert
first, so one block works on one pair and the pairs of an expert sit together. This is what
prefill and decoding of a batch need; glm-vx's fused kernels handle a batch of one.

| Kernel | What it does |
| --- | --- |
| `moe_reset` | Zeroes the per-expert counters |
| `moe_count` | `counts[e]` = pairs that pick expert e (atomic add) |
| `moe_offsets` | Prefix sum of the counts: the first position of each expert |
| `moe_scatter` | Each pair takes the next free position in its expert's range (atomic add), and records `pos_of[p]` and `pair_at[pos]` |
| `moe_expert_ffn` | Block (e, j) runs the network of the j-th pair of expert e: `h` in shared memory, then the down projection |
| `moe_combine` | A token's output is the gate-weighted sum of its k results |

The race proof refuses a plain write at a position that only the data knows, so those writes are
atomic stores (`pair_at`, and the rows of `ysorted`); the one plain write, `pos_of[p]`, is at the
pair's own index. The order in which pairs take positions varies from run to run, but each
pair's result is written once and `moe_combine` adds a token's results in slot order, so the
output does not depend on it.

The test uses 20 tokens, 6 experts, k = 2, hidden 48 and ffn 64. It checks the layer against a
host loop over tokens, and the sort itself: counts, offsets, that every pair has its own
position inside its expert's range, and that `pair_at` is the inverse of `pos_of`.

`test.vx` imports the library, moves inputs to GPU memory with `transfer(t, Memory::HBM)`, runs
each kernel, brings the result back, and compares it with the same formula written as host
code. The inputs are whole numbers scaled by 1/8, so Python reads the same matrices. `build.sh`
checks four things:

1. **The results.** The program prints `ok` when the layer is within 1e-5 of the host code and the sort is right.
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
