# gpu-llm-ops

The GPU kernels of a GPT-2 training step, after [llm.c](https://github.com/karpathy/llm.c)'s CUDA code, as a library: token and position embedding with its gradient, GELU forward and backward, residual add, `matmul` with bias and its bias gradient, the attention layout changes, causal softmax, cross-entropy with its fused backward, and an AdamW step.

```console
$ ./build.sh
...
ok
ok: Python computes the same entries
kernels: vx_npu_kernel_0, ..., vx_npu_kernel_13, one sm_80 module of 84345 bytes of PTX
ok: ptxas assembles them for sm_80 (A100)
ok: the executable prints the same
```

The kernels are in ``ops.vx``, as functions that take tensors in GPU memory. Each is a
`spawn on(Topology::Device)` region with `across` loops: `across blocks` becomes CUDA blocks and
`across lanes` becomes threads. The kernels come from the `across` test corpus in
[vx-lang/Vx#2127](https://github.com/vx-lang/Vx/pull/2127), where the compiler's checker accepts
each of them. Shapes are GPT-2 small where llm.c fixes them (768 channels, 12 heads of 64). The gradient of the token embedding uses an atomic add, because two tokens with one id write to one row; the bias gradient comes in two forms, one atomic and one where each lane owns a column.

llm.c's LayerNorm and softmax are not here: [`gpu-norms`](../gpu-norms) has them.

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
