# gpu-sampling

Picking the next token from the logits on the GPU, with the rules of the `sampler` starter:
greedy, temperature, top-k, top-p and min-p.

```console
$ ./build.sh
...
ok
ok: Python computes the same entries
kernels: vx_npu_kernel_0, vx_npu_kernel_1, vx_npu_kernel_2, one sm_80 module of 53065 bytes of PTX
ok: ptxas assembles them for sm_80 (A100)
ok: the executable prints the same
```

The kernels are in `ops.vx`. Each is a `spawn on(Topology::Device)` region with `across` loops,
one block of 256 lanes for each row. They are the GPU form of the [`sampler`](../sampler)
starter and follow its rules: probabilities are `softmax(logits / T)` over the whole vocabulary;
top-k keeps the k most likely tokens (ties to the lower id); top-p keeps the most likely tokens,
in order, until their total reaches `mass`; min-p keeps tokens at least `ratio` times as likely
as the most likely one; kept probabilities are scaled to add up to 1; a logit of -infinity is a
masked token; a draw takes a uniform `u` in [0, 1) and returns the first kept token, in id
order, whose running total passes it.

| Kernel | What it does |
| --- | --- |
| `argmax_rows` | Greedy: the largest logit of each row, the lower id on a tie (`lanes::argmax`) |
| `topk_candidates` | The `cap` largest logits of each row, in order: `cap` rounds of a block argmax, each blotting out its pick in a scratch copy of the row |
| `sample_rows` | One token for each row: `lanes::sum` adds up the whole row's softmax, the three cuts are intersected on the candidates, and the draw is made in id order |

The uniform `u[r]` is the caller's, so the host can use any seeded stream, as `sampler` does.
Every kept token must be among the `cap` candidates (`cap` is at most 64): top-p and min-p are
exact when their set has at most `cap` tokens, and otherwise the draw is from the best `cap`
of them. To switch a cut off pass `k = cap`, `mass = 2.0` or `ratio = 0.0`. A plain draw from the
whole vocabulary is not here: it would need a different kernel, with an inverse-CDF search over
the row.

The test runs 64 rows of a vocabulary of 1000 (not a multiple of the lanes), with `u` spread over
(0, 1), for greedy, top-k, top-p, min-p and the three together. The logits are on a grid that
ties many tokens, a few tokens hold most of the mass, rows 2 and 3 mask tokens, and rows 1, 5, ..
have a three-way tie for the top (two of them in the same lane). The host sampler ranks the
candidates by repeated scans and draws over the whole vocabulary in id order, as `sampler` does,
and every token must be equal.

`test.vx` imports the library, moves inputs to GPU memory with `transfer(t, Memory::HBM)`, runs
each kernel, brings the result back, and compares it with the same formula written as host
code. The inputs are whole numbers scaled by 1/8, so Python reads the same matrices. `build.sh`
checks four things:

1. **The results.** The program prints `ok` when every token equals the host sampler's.
2. **Python.** `check.py` computes the tokens of three rows of each configuration.
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
