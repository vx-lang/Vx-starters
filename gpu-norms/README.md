# gpu-norms

LayerNorm, RMSNorm and softmax over the rows of a matrix, each written as a GPU kernel in Vx,
compiled to NVIDIA PTX and built into a standalone executable. No GPU is needed to build or check
it.

```console
$ ./build.sh
layernorm: -1.7142406702041626 -1.6787168979644775 -1.0746625661849976
rmsnorm: -1.6458110809326172 -1.603589415550232 -0.9897774457931519
softmax: 0.0000003452998669217777 0.0000003841046520847158 0.0000035375931020098506
ok
ok: Python computes the same entries
kernels: vx_npu_kernel_0, vx_npu_kernel_1, vx_npu_kernel_2, one sm_80 module of 40346 bytes of PTX
ok: ptxas assembles them for sm_80 (A100)
ok: the executable prints the same
```

`src/main.vx` puts a 32 x 128 matrix in GPU memory with `transfer(x, Memory::GPU_HBM)`, runs three
`spawn on(Topology::GPU) { .. }` regions, one per operation, and brings the results back with
`transfer(y, Memory::CPU_DRAM)`. After the transfer the tensor's type says it lives in GPU
memory, and reading it on the host without bringing it back is a compile error
(`E6003: 'ln' lives in GPU_HBM but CPU sees only [CPU_DRAM, NPU_HBM]`).

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

## Toolchains

It passes with the v0.0.3 release, with Vx built from `main`, and with Vx built from `main` with
CUDA (a toolchain packaged with `VX_TARGET_SUFFIX=-cuda`, whose dispatch library calls CUDA).

Without `ptxas` (from the CUDA toolkit, at `$CUDA_HOME/bin` or on `PATH`), step 3 still checks the
PTX but does not assemble it. Installing just these Ubuntu packages from NVIDIA's repository is
enough: `cuda-cudart-dev-13-1 libcublas-dev-13-1 cuda-driver-dev-13-1 cuda-nvcc-13-1`.

A CUDA toolchain on a machine without an NVIDIA driver cannot link any program, because its
dispatch library needs `libcuda.so.1` (vx-lang/Vx#1737). `build.sh` then puts the toolkit's stub
on `LD_LIBRARY_PATH` under that name; the stub reports no device at run time.

## Not done yet

- Running on a real GPU, and timing it.
- One GPU thread per row. Each region is written as a plain loop; mapping work to GPU blocks and
  threads is still being built (vx-lang/Vx#379).
