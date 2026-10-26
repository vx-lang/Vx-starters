# gpu-logit-processors

What a serving loop does to the logits before it picks a token, on the GPU: repetition,
frequency and presence penalties, a bias for each token, a mask of allowed tokens, and
temperature. Run them before [`gpu-sampling`](../gpu-sampling).

```console
$ ./build.sh
...
ok
ok: Python computes the same entries
kernels: vx_npu_kernel_0, vx_npu_kernel_1, vx_npu_kernel_2, vx_npu_kernel_3, vx_npu_kernel_4, one sm_80 module of 15496 bytes of PTX
ok: ptxas assembles them for sm_80 (A100)
ok: the executable prints the same
```

The kernels are in `ops.vx`, each a `spawn on(Topology::Device)` region with `across` loops: a
block for each row, a lane for each token, 256 at a time. They run in this order:

| Kernel | What it does |
| --- | --- |
| `count_history` | `counts[r * vocab + t]` += 1 for each token `t` of the row's prompt and output so far (atomic adds) |
| `count_token` | The same for the token just picked, one for each row |
| `apply_penalties` | For each token that has appeared: a positive logit is divided by `rep[r]`, a negative one multiplied; then minus `freq[r]` for each time and minus `pres[r]` once |
| `apply_bias_mask` | Adds `bias[c]` (one vector for all rows) and turns the logit of a token with `allowed[r][c] == 0` to -infinity |
| `scale_temperature` | Divides row `r` by `temp[r]`; a temperature of 0 or less leaves the row alone (greedy) |

Every row has its own penalty and temperature, so one batch can mix requests. After
`scale_temperature`, call `sample_rows` with a temperature of 1.0, or the logits are scaled twice.

The test runs 6 rows of a vocabulary of 700 (not a multiple of the lanes), with histories of 20
to 295 tokens that repeat many tokens, a different penalty setting for each row, and a row with
temperature 0. Each kernel is checked alone against host code for every entry, and then the
four in a row. All values are dyadic, so the device, the host and Python agree exactly.

`test.vx` imports the library, moves inputs to GPU memory with `transfer(t, Memory::HBM)`, runs
each kernel, brings the result back, and compares it with the same formula written as host
code. `build.sh` checks four things:

1. **The results.** The program prints `ok` when every entry equals the host's.
2. **Python.** `check.py` computes some entries of each step again.
3. **The kernels.** `vxc --emit-llvm` carries the GPU module as PTX in a string. `kernels.py`
   pulls it out, checks it targets `sm_80`, has one kernel for each region and calls nothing
   outside itself, and `ptxas` must assemble it.
4. **An executable.** `vxc -a emit-obj`, a link by hand (vx-lang/Vx#1499), and the executable
   must print the same.

No GPU is needed. Without one, the dispatch library runs the same regions on the host, so
this project has only been run that way. Where a GPU is present, the same program uses it.

## Toolchain

It needs Vx `main` from 2026-10-25 or later (d25deea9): the machine file `fleet/a100-40.vx`
uses `levels:`, which v0.0.3 and the 2026-10-13 build do not read. Assembling the kernels needs
`ptxas` (CUDA toolkit), and a CUDA toolchain on a machine with no NVIDIA driver links against
the toolkit's stub, as in [`gpu-norms`](../gpu-norms).
