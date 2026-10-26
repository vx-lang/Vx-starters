# gpu-mla

Multi-head latent attention (MLA), the attention of DeepSeek and GLM models, on the GPU. The
cache holds, for each position, one latent vector `c` of 512 numbers and one rope key `pe` of 64
numbers, shared by all heads, instead of a key and a value for each head. Head `h` makes its keys
and values from the latent with two matrices, `wuk[h]` and `wuv[h]`.

```console
$ ./build.sh
...
ok
ok: Python computes the same outputs
kernels: vx_npu_kernel_0, vx_npu_kernel_1, vx_npu_kernel_2, vx_npu_kernel_3, vx_npu_kernel_4, one sm_80 module of ... bytes of PTX
ok: ptxas assembles them for sm_80 (A100)
ok: the executable prints the same
```

The kernels are in `ops.vx`, each a `spawn on(Topology::Device)` region with `across` loops. They
use the absorbed form, which never makes the keys and values: `wuk` is folded into the query
once, and attention runs over the latents themselves.

| Kernel | What it does |
| --- | --- |
| `mla_absorb_q` | `qlat[r][h] = q_nope[r][h] . wuk[h]`: the query moved into the latent space, 512 numbers |
| `mla_attention` | `score[t] = (qlat . c[t] + q_pe . pe[t]) * scale`, then the softmax-weighted sum of the latents, `olat`. Chunks of 128 positions with an online softmax, so the length is not limited |
| `mla_expand_out` | `out[r][h] = olat[r][h] . wuv[h]`: the head's output, `dv` numbers |
| `mla_append` | Writes a new token's latent and rope key at position `lens[s]` of each sequence (atomic stores) |
| `mla_advance` | `lens[s] += 1` |

`mla_attention` serves decode and prefill. A row is one query: it reads sequence `qseq[row]` of the
cache and attends to its first `qlen[row]` positions. Decode is one row for each sequence, with
`qlen` its length. Prefill is one row for each position `t` of a prompt, with `qlen = t + 1`,
which is the causal mask. The cache is flat: position `t` of sequence `s` is row `s * stride + t`,
at `row * 512` in the latents and `row * 64` in the rope keys. The loop over chunks is bounded by
the host value `max_len`; each lane masks positions at or past its row's length.

The test checks the absorbed form against the expanded one, which is a different way to the same
numbers: it makes every key and value from the latents with `wuk` and `wuv` on the host, then
runs plain softmax attention. Decode has two sequences of 150 and 61 positions (the longer crosses
the chunk of 128), the last position of each put in the cache by `mla_append`. Prefill is 140
rows of the first sequence, each attending to its own prefix. The latents grow with the position,
so a later chunk raises the maximum and the rescale matters. Outputs agree to 1e-4 (they are
about 0.1 to 1). `check.py` computes some outputs again in Python (f64).

`test.vx` imports the library, moves inputs to GPU memory with `transfer(t, Memory::HBM)`, runs the
kernels, brings the result back, and compares it with host code. `build.sh` checks four things:

1. **The results.** The program prints `ok` when every output is within 1e-4 of the host's.
2. **Python.** `check.py` computes some outputs again.
3. **The kernels.** `vxc --emit-llvm` carries the GPU module as PTX in a string. `kernels.py`
   pulls it out, checks it targets `sm_80`, has one kernel for each region and calls nothing
   outside itself, and `ptxas` must assemble it.
4. **An executable.** `vxc -a emit-obj`, a link by hand (vx-lang/Vx#1499), and the executable
   must print the same.

## What is not here

* The latent is 512 numbers and the rope key 64 (the sizes of DeepSeek-V3 and GLM). Other sizes
  need the constants in `ops.vx` changed.
* The cache is not paged (compare [`gpu-paged-attn`](../gpu-paged-attn)), and not fp8.
* The query and key projections, and the rope on `q_pe` and `pe`, are outside: use
  [`gpu-inference-ops`](../gpu-inference-ops) for rope and the matmul kernels for the projections.
* A lane scores one position at a time over 576 numbers, so reads of the cache are not coalesced
  across lanes. That is the first thing to look at when it runs on a GPU.

No GPU is needed. Without one, the dispatch library runs the same regions on the host, so
this project has only been run that way. Where a GPU is present, the same program uses it.

## Toolchain

It needs Vx `main` from 2026-10-25 or later (d25deea9): the machine file `fleet/a100-40.vx`
uses `levels:`, which v0.0.3 and the 2026-10-13 build do not read. Assembling the kernels needs
`ptxas` (CUDA toolkit), and a CUDA toolchain on a machine with no NVIDIA driver links against
the toolkit's stub, as in [`gpu-norms`](../gpu-norms).
