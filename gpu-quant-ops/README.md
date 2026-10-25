# gpu-quant-ops

Matrix-vector and matrix-matrix products for the GPU with weights stored as bf16, fp8, int8 or
int4, decoded in the lanes.

```console
$ ./build.sh
...
ok
ok: Python computes the same entries
kernels: vx_npu_kernel_0, ..., vx_npu_kernel_7, one sm_80 module of 51515 bytes of PTX
ok: ptxas assembles them for sm_80 (A100)
ok: the executable prints the same
```

The kernels are in `ops.vx`, as functions that take tensors in GPU memory. Each is a
`spawn on(Topology::Device)` region with `across` loops. They are new code for this project
(glm-vx has kernels of the same kind, but no license file), and they pair with the
[`quantize`](../quantize) and [`bf16`](../bf16) starters, which do the same formats on the CPU.

| Format | Weights | Scales |
| --- | --- | --- |
| `bf16` | `u16` bits | none |
| `fp8` | `u8`, e4m3fn (bias 7, `0x7f` and `0xff` are NaN) | one for each row |
| `int8` | `i8` | one for each row |
| `int4` | `u8` holding two signed 4-bit values, low nibble first | one for each group of 64 columns |

Each format has `<format>_matvec` (`y = W x`) and `<format>_matmul` (`out[m][r]` for a batch of
`x` rows), so 8 kernels. One block computes one output: 256 lanes each sum a strided slice of
the columns (lane `l` takes `l`, `l + 256`, ..), `lanes::sum` adds the lanes' parts, and the
row's scale is applied once. The weights stay in their small type in memory and are decoded in the
lane; the decoders are macros, because the race proof refuses a call to a function in an
`across` loop, and `f32::from_bits` ([vx-lang/Vx#2222](https://github.com/vx-lang/Vx/issues/2222)).

The tests use 40 x 320 weights (320 is not a multiple of 256, so the strided loop runs a second
pass in some lanes) and a batch of 3, and the host reference follows the same sum order.
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
