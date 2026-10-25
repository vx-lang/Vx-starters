# gpu-stencil

Grid time-stepping on the GPU: Jacobi relaxation from PolyBench/GPU, and the heat step of Rodinia's HotSpot, once with a launch for each step and once with two steps inside a tile in shared memory.

```console
$ ./build.sh
...
ok
ok: Python computes the same entries
kernels: vx_npu_kernel_0, ..., vx_npu_kernel_3, one sm_80 module of 18749 bytes of PTX
ok: ptxas assembles them for sm_80 (A100)
ok: the executable prints the same
```

The kernels are in ``stencil.vx``, as functions that take tensors in GPU memory. Each is a
`spawn on(Topology::Device)` region with `across` loops: `across blocks` becomes CUDA blocks and
`across lanes` becomes threads. The kernels come from the `across` test corpus in
[vx-lang/Vx#2127](https://github.com/vx-lang/Vx/pull/2127), where the compiler's checker accepts
each of them. The time loop is on the host, with one launch for each step: a launch ends before the next one starts, which is how a step sees the cells of the one before. `hotspot_pyramid_inner_write` changes Rodinia's kernel so that the tiles, which overlap by 4 cells, write only their inner 12 x 12 cells, which is what lets the checker show that two blocks never write one cell.

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
