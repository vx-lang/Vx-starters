# gpu-norms

LayerNorm, RMSNorm and softmax over the rows of a matrix, each written as a GPU kernel in Vx,
compiled to NVIDIA PTX and built into a standalone executable. No GPU is needed to build or check
it.

```console
$ ./build.sh
layernorm: -1.7142419815063477 -1.678718090057373 -1.0746632814407349
rmsnorm: -1.6458110809326172 -1.603589415550232 -0.9897774457931519
softmax: 0.0000003452998669217777 0.00000038410462366300635 0.0000035375935567572014
ok
ok: Python computes the same entries
kernels: vx_npu_kernel_0, vx_npu_kernel_1, vx_npu_kernel_2, one sm_80 module of 30633 bytes of PTX
ok: ptxas assembles them for sm_80 (A100)
ok: the executable prints the same
```

`src/main.vx` puts a 32 x 128 matrix in GPU memory with `transfer(x, Memory::HBM)`, runs three
kernels, and brings the results back with `transfer(y, Memory::CPU_DRAM)`. Each kernel is a
`spawn on(Topology::Device)` region with `across` loops: one block for each row and one lane
for each column, a row in shared memory, and `lanes::sum` for the sums (the maximum of softmax is
one pass over the row). After the transfer the tensor's type says it lives in GPU memory, and
reading it on the host without bringing it back is a compile error (`E6003: 'ln' lives in HBM
but CPU sees only [CPU_DRAM, ..]`).

`build.sh` checks four things:

1. **The results.** The program compares each kernel with the same formula written as plain host
   code (within 1e-5, and 1e-7 for softmax), and checks that every softmax row sums to 1.
2. **Python.** `check.py` computes three entries of each result in f64, and they must agree to a
   relative 1e-5.
3. **The kernels.** `vxc --emit-llvm` shows the GPU module that the program carries, as PTX in a
   string. `kernels.py` pulls it out and checks that it targets `sm_80` (A100), has one kernel
   per region, and calls nothing outside itself except libdevice's `__nv_*` math. Then NVIDIA's
   assembler, `ptxas`, must accept it. `sqrt` and `exp` come from `core::num`, and the compiler
   links libdevice (from `$CUDA_HOME`) so they become device code.
4. **An executable.** `vxc -a emit-obj`, then a link by hand (vx-lang/Vx#1499), and the
   executable must print the same as the run.

Where a GPU and its driver are present, the kernels run on the GPU. Without one, the dispatch
library runs the same regions on the host, so this project has only been run that way: building
and checking the kernels needs the CUDA toolkit, not a GPU.

## Toolchain

It needs Vx `main` from 2026-10-25 or later (d25deea9): `across` loops and the machine file
`fleet/a100-40.vx` (which `build.sh` passes with `--machine`) use `levels:`, which v0.0.3 does
not read. An earlier version of this project, with one plain loop for each row in a
`spawn on(Topology::GPU)` region, passed on v0.0.3 as well; it is in the git history.

Without `ptxas` (from the CUDA toolkit, at `$CUDA_HOME/bin` or on `PATH`), step 3 still checks the
PTX but does not assemble it. Installing just these Ubuntu packages from NVIDIA's repository is
enough: `cuda-cudart-dev-13-1 libcublas-dev-13-1 cuda-driver-dev-13-1 cuda-nvcc-13-1`.

A CUDA toolchain on a machine without an NVIDIA driver cannot link any program, because its
dispatch library needs `libcuda.so.1` (vx-lang/Vx#1737). `build.sh` then puts the toolkit's stub
on `LD_LIBRARY_PATH` under that name; the stub reports no device at run time.

## Not done yet

- Running on a real GPU, and timing it.
