# Vx starter projects

Small, complete programs written in [Vx](https://github.com/vx-lang/Vx), most of them about the
software around machine learning. They have two jobs:

1. Show what Vx does, one idea per project, on an ordinary CPU.
2. Find out what Vx's build system and package manager have to do
   ([vx-lang/Vx#1275](https://github.com/vx-lang/Vx/issues/1275)). Each project is built with
   plain scripts first, and everything that gets in the way goes in [`FRICTION.md`](FRICTION.md).

## Projects

### Language basics

| Project | Shows | Build-system question |
| --- | --- | --- |
| [`wordcount`](wordcount) | Files, structs, loops, raw buffers | One package, one binary: how to build, run and test it |
| [`textkit`](textkit) + [`textstat`](textstat) | Modules, traits, generics | A local library dependency; two packages with the same function name |
| [`bigint`](bigint), [`rational`](rational), [`interval`](interval) | Operators on user types, `Vec`, exact arithmetic | A chain of libraries: `interval` needs `rational`, which needs `bigint` |

### Re-checking published proofs

| Project | Shows | Build-system question |
| --- | --- | --- |
| [`cycle-clique`](cycle-clique) | A published proof's computer check, ported and run natively | Optimization level as a build setting; expected output that comes from another project |
| [`hellinger`](hellinger) | Traits over three number types, `i128`, an interval proof by branch and bound, `grad` to check its derivatives | A build that goes through LLVM IR and the bundled `clang`, because `vxc -O1` is too slow |

The three number libraries are for checking the computer-assisted parts of proofs, such as those
in [openai/math](https://github.com/openai/math), where the checkers need integers of any size,
exact fractions and interval bounds, and Vx's `std` has none of them. `cycle-clique` and `hellinger` are the
first such checks ([vx-lang/Vx#1489](https://github.com/vx-lang/Vx/issues/1489)).

### Training

| Project | Shows | Build-system question |
| --- | --- | --- |
| [`tiny-mlp`](tiny-mlp) | `grad` on training math: textbook derivatives, backprop checked three ways, Adam on XOR | Optional toolchain parts (Enzyme); a program that only runs under `run-jit` |
| [`lr-schedule`](lr-schedule) | Warmup and cosine decay, clipping by global norm, gradient accumulation checked with `grad`; one bad label that clipping survives | The same optional toolchain part (Enzyme) in a second project |
| [`layernorm`](layernorm) | Variance in f32 (one pass fails, two-pass and Welford do not), merging statistics across shards, LayerNorm's backward against `grad` | Numerical tolerances as part of a test's contract |
| [`data-loader`](data-loader) | Per-epoch shuffling, sharding across ranks, exact resume, the same global batches on any number of ranks | A program whose output must not change between runs or machines |
| [`bf16`](bf16) | bf16 and fp16 rounding as bit operations, checked against Vx's own types; a stalled bf16 sum, loss scaling, dynamic loss scaling | Checking against the toolchain's own number types, whose conversions depend on the CPU |

### Inference and serving

| Project | Shows | Build-system question |
| --- | --- | --- |
| [`flash-attn`](flash-attn) | FlashAttention's tiled online softmax, checked against naive attention | A project that needs a compiler newer than the last release |
| [`rope`](rope) | Rotary position embeddings: rotation keeps length, attention sees only relative positions; constants computed by `comptime` | Values worked out by the compiler, and checked against the run-time library |
| [`sampler`](sampler) | Greedy, temperature, top-k, top-p and min-p sampling, with masked tokens; 200,000 draws per setting checked by chi-square | Statistical tests in the test step: a seeded RNG from `std::rand` and a pass threshold |
| [`spec-decode`](spec-decode) | Speculative decoding: accept/reject keeps the target model's distribution exactly; tokens per pass match the formula | Randomized tests with fixed seeds and statistical limits |
| [`bpe-tokenizer`](bpe-tokenizer) | A GPT-style byte-pair tokenizer: training, encoding, decoding | Data files from outside the project as test inputs |
| [`paged-kv`](paged-kv) | A vLLM-style paged KV cache with copy-on-write; contracts prove its index arithmetic | Proofs as part of compiling: a broken contract is a build error |
| [`batch-sched`](batch-sched) | Continuous vs static batching in a serving simulator, checked against a case worked by hand | Tests with exact expected numbers next to tests of properties |
| [`quantize`](quantize) | int8/int4 weight quantization per tensor and per group, int4 packing, an int8 matmul within its rounding bound; contracts prove the clamp and packing ranges | A test step that is partly a proof: some breaks fail the compile, others the run |
| [`moe-router`](moe-router) | Mixture-of-experts top-k routing with expert capacity, dropped tokens, the balance loss, and dispatch/combine | One more project of the same shape: what a template or `vx new` would generate |

### Distributed

| Project | Shows | Build-system question |
| --- | --- | --- |
| [`ring-allreduce`](ring-allreduce) | Data-parallel all-reduce as reduce-scatter plus all-gather on simulated ranks; each rank sends 2(N-1)/N of the vector; contracts prove the ring's chunk numbers | A program about parallel hardware with no threads to run on: simulation in place of a runtime |
| [`reshard`](reshard) | Row- and column-parallel splits with remainders, and direct resharding from k1 to k2 shards; contracts prove the overlap arithmetic | Checkpoint layouts as data the build and tests share |

### Files, C libraries and hardware

| Project | Shows | Build-system question |
| --- | --- | --- |
| [`safetensors`](safetensors) | The checkpoint format: a JSON header parser, byte layouts, broken files refused; Python reads the same file | A test that needs another language's tools (`python3`) |
| [`ffi-zlib`](ffi-zlib) | Calling a C library: zlib's crc32, adler32, compress2 and uncompress through an `extern` block, checked against known values and a CRC-32 written in Vx | Linking native libraries: `vxc` cannot add `-lz`, so the build compiles to an object and links it by hand |
| [`will-it-fit`](will-it-fit) | `--machine`: one program admitted or refused per GPU (A100 40 GB, H100); capacity (E6010) and dtype (E6026) errors predicted by a run-time calculation and matched exactly | Machine models as build inputs, per target: the same source, one compile per machine file |
| [`prefill-decode`](prefill-decode) | `spawn on` GPU and NPU regions, `transfer` / `to_device` of a KV cache, `--verify-seams` proving every hop and refusing a relaxed one | Optional tools (`z3`) in the test step: skipped with a message when missing |
| [`gpu-norms`](gpu-norms) | LayerNorm, RMSNorm and softmax as `spawn on(Topology::GPU)` kernels: PTX for sm_80 that `ptxas` accepts, a standalone executable, results checked against host code and Python, no GPU needed | Device code as a build output: the PTX is dug out of LLVM IR, the CUDA toolkit is an optional tool, and a CUDA toolchain needs the driver's stub to link |
| [`gpu-llm-ops`](gpu-llm-ops) | The kernels of a GPT-2 training step (embedding, GELU, matmul, attention layout, causal softmax, cross-entropy, AdamW) as a library of `across` kernels, from llm.c; same checks as `gpu-norms` | A library whose code is device code: the tests import it, and the PTX still has to be dug out of the test program |
| [`gpu-moe`](gpu-moe) | A mixture-of-experts layer with work grouped by expert: a sort of (token, expert) pairs on the device from atomic adds, then one block for each pair; checked against a host loop and for the sort's own invariants | Atomics as the way to write at a position only the data knows (the race proof refuses a plain write) |
| [`gpu-blas`](gpu-blas) | A tiled matrix product, 2mm, atax and gemver as `across` kernels, from PolyBench/GPU; edge sizes that are not a multiple of the block | A toolchain newer than the last release, because the machine file needs `levels:` |
| [`gpu-stencil`](gpu-stencil) | Jacobi and HotSpot grid steps as `across` kernels: a launch for each step, and two steps inside a shared-memory tile | A host loop that launches kernels: the build must count launches, not functions |

## Getting the toolchain

```sh
curl -fsSL https://vxlang.org/install.sh | sh
```

Every project builds with `./build.sh` against `~/.vx/current`, or the toolchain `VX_HOME` names.

The projects up to `tiny-mlp` were written against v0.0.3. From `flash-attn` on, they use Vx
built from `main` (39a12e0a or later; d25deea9 for the `gpu-llm-ops`, `gpu-blas` and `gpu-stencil` kernels), which has fixes the release lacks; there is no nightly
installer yet, so that means building Vx and packaging it with `scripts/release/package.sh`.
Every project passes with that build too (rechecked on b31ef3ca, built with and without CUDA),
except `rational` and `interval`, which no longer
compile on `main` ([vx-lang/Vx#1661](https://github.com/vx-lang/Vx/issues/1661)); they still
pass on v0.0.3.

## One rule for the package manager

Vx checks function bodies in parallel because every global fact is known before any body is
checked ([language rules](https://github.com/vx-lang/Vx/blob/main/docs/language_rules_for_a_parallel_compiler.md)).
The package manager must keep that true: a build step that runs before the compiler, such as one
that generates source files, is fine; anything that adds declarations while bodies are being
checked, as Rust's procedural macros can, is not.

## License

Apache License v2.0 with LLVM Exceptions, the same as Vx. See [`LICENSE`](LICENSE).
