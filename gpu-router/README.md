# gpu-router

The router of a mixture-of-experts layer for a batch of tokens on the GPU: sigmoid scores with a
selection bias, optionally top-k over groups, as in DeepSeek-V3 and GLM.

```console
$ ./build.sh
...
ok
ok: Python computes the same entries
kernels: vx_npu_kernel_0, one sm_80 module of 37913 bytes of PTX
ok: ptxas assembles them for sm_80 (A100)
ok: the executable prints the same
```

The kernel is in `ops.vx`: `router_topk`, one `spawn on(Topology::Device)` region with `across`
loops, one token for each lane.

Scoring is the one DeepSeek-V3 and GLM use. An expert's score is `sigmoid(logit) + bias[e]`. The
bias steers which experts are picked but not their weights: a picked expert's weight is
`sigmoid(logit)`, the k weights are divided by their sum (plus 1e-20) and multiplied by `scale`.
Ties go to the lower expert id.

With `groups` > 1 the experts form equal groups of consecutive ids. A group scores the sum of
its two best experts, only the `topk_group` best groups may send experts, and the k picks are
the best experts among them. One group and `topk_group` = 1 is a plain top-k.

The scan over the experts is serial in each lane, repeated for each pick, and the picks are read
back from `ids` to exclude them. That is exact and fine for a few hundred experts; a version
with a warp for each token would be faster. glm-vx's own `router` (one token, a warp, up to 8 of
256 experts, `lanes::argmax`) is the single-token form of this, without groups.

The test runs three configurations over 50 tokens and 64 experts, and compares every pick with a
host router that is written differently (by ranks, not repeated scans):

| Configuration | What it covers |
| --- | --- |
| `grouped`: 8 groups of 8, 4 groups, k = 8, scale 2.5 | Many tied logits, told apart by the bias or the id |
| `plain`: one group, k = 6 | Distinct scores, a plain top-k |
| `tied`: 16 groups of 4, 3 groups, k = 8 | Whole groups with the same score, and a cut after 3 groups that falls inside a tie: the lower group ids must win |

`test.vx` imports the library, moves inputs to GPU memory with `transfer(t, Memory::HBM)`, runs
each kernel, brings the result back, and compares it with the same formula written as host
code. The inputs are whole numbers scaled by 1/8, so Python reads the same matrices. `build.sh`
checks four things:

1. **The results.** The program prints `ok` when every token's picks equal the host router's and the weights are within 1e-5.
2. **Python.** `check.py` computes three entries of each result in f64.
3. **The kernels.** `vxc --emit-llvm` carries the GPU module as PTX in a string. `kernels.py`
   pulls it out, checks it targets `sm_80`, has one kernel for each kernel function and calls nothing
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
