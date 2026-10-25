# gpu-inference-ops

The GPU kernels of one decode step of a transformer: rotary embeddings, SwiGLU, matrix-vector
products, a top-2 expert router, and attention of one new token over a key-value cache.

```console
$ ./build.sh
...
ok
ok: Python computes the same entries
kernels: vx_npu_kernel_0, ..., vx_npu_kernel_5, one sm_80 module of 46465 bytes of PTX
ok: ptxas assembles them for sm_80 (A100)
ok: the executable prints the same
```

The kernels are in `ops.vx`, as functions that take tensors in GPU memory. Each is a
`spawn on(Topology::Device)` region with `across` loops: `across blocks` becomes CUDA blocks and
`across lanes` becomes threads. They are the GPU forms of what [`rope`](../rope),
[`moe-router`](../moe-router) and [`paged-kv`](../paged-kv) do on the CPU, and they are new
code for this project: the corpus of [vx-lang/Vx#2127](https://github.com/vx-lang/Vx/pull/2127)
has no decode-step kernels. The compiler's race checker accepts all six; two needed a rewrite
first ([vx-lang/Vx#2220](https://github.com/vx-lang/Vx/issues/2220),
[#2221](https://github.com/vx-lang/Vx/issues/2221)).

| Kernel | What it does |
| --- | --- |
| `rope_forward` | Rotary embeddings for tokens at positions `pos0 + t`, from cos and sin tables. One write for each lane, because the checker refuses two (#2220) |
| `swiglu` | `silu(gate) * up`, one element for each lane |
| `matvec` | `y = W x`, one output for each lane |
| `matvec_by_block` | `y = W x` for 512 columns, one block for each output, the lanes' parts added with `lanes::sum` |
| `router_top2` | The two best experts of each token and their gates (a softmax of the two scores) |
| `attention_decode` | One new token's attention over a key-value cache of up to 128 entries, one block for each head |

`test.vx` imports the library, moves inputs to GPU memory with `transfer(t, Memory::HBM)`, runs
each kernel, brings the result back, and compares it with the same formula written as host
code. The inputs are whole numbers scaled by 1/8, so Python reads the same matrices. `build.sh`
checks four things:

1. **The results.** The program prints `ok` when every kernel is within its tolerance of the host
   code.
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
