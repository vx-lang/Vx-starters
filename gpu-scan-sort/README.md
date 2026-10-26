# gpu-scan-sort

Prefix scan, histogram and a stable radix sort on the GPU. A scan gives the same order on every
run; an atomic counter, as in [`gpu-moe`](../gpu-moe), does not (the lane that gets the counter
first gets the lower slot).

```console
$ ./build.sh
...
ok
ok: Python computes the same entries
kernels: vx_npu_kernel_0, vx_npu_kernel_1, vx_npu_kernel_2, vx_npu_kernel_3, vx_npu_kernel_4, one sm_80 module of ... bytes of PTX
ok: ptxas assembles them for sm_80 (A100)
ok: the executable prints the same
```

The kernels are in `ops.vx`, each a `spawn on(Topology::Device)` region with `across` loops, a
block for each tile of 256 numbers.

| Function | What it does |
| --- | --- |
| `scan_tiles` | Exclusive scan inside each tile of 256, in shared memory: 8 steps of Hillis-Steele between two buffers, so no lane reads a number another lane is writing. Also gives each tile's sum |
| `scan_add` | Adds the scanned tile sums back |
| `scan_exclusive` | `out[i]` = the sum of `inp[0..i)` for up to 65536 numbers: `scan_tiles`, a scan of the tile sums, `scan_add` (three launches) |
| `histogram` | `bins[keys[i]] += 1` with atomic adds |
| `bit_flags`, `split_pass` | One stable pass of a radix sort on one bit: the entries with a 0 go first, in order, then the ones with a 1; positions from `scan_exclusive` |
| `radix_sort` | Stable sort of non-negative i32 keys by their low `bits` bits (`bits` even), with a value for each key; sort `0..n` as the values to get the permutation |

The test scans 1, 255, 256, 257, 5000 and 65536 numbers (around the tile edges and the limit)
against a running sum, counts 5000 keys, and sorts 5000 keys of 8 bits with many equal ones.
Each entry must be where a stable sort puts it, with its own position as the value.

The sort moves one bit at a time (the writes of a pass go to positions only the data knows, so
they are atomic stores), which costs 5 launches for each bit (the flags, a scan of 3 launches, the move). A wider digit
per pass, with a histogram for each tile, would use fewer passes; that is not here.

## Could `gpu-moe` use it?

Yes, and it would make `moe_scatter` deterministic: sort the `tokens * k` expert ids with their
pair numbers as values, and the pairs of each expert come out in pair order on every run.
`gpu-moe` is left as it is, with its atomic cursor: its results do not depend on the order within
an expert, and the sort costs 40 launches for 8 bits of expert id against 1 for the cursor.
Use the sort when a run must be repeatable bit for bit.

`test.vx` imports the library, moves inputs to GPU memory with `transfer(t, Memory::HBM)`, runs
each function, brings the result back, and compares it with host code. `build.sh` checks four
things:

1. **The results.** The program prints `ok` when every entry equals the host's.
2. **Python.** `check.py` computes some entries again.
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
