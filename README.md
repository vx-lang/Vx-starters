# Vx starter projects

Small, complete programs written in [Vx](https://github.com/vx-lang/Vx), built with a released
toolchain. They have two jobs:

1. Show what Vx does, one idea per project, on an ordinary CPU.
2. Find out what Vx's build system and package manager have to do
   ([vx-lang/Vx#1275](https://github.com/vx-lang/Vx/issues/1275)). Each project is built with
   plain scripts first, and everything that gets in the way goes in [`FRICTION.md`](FRICTION.md).

## Projects

| Project | Shows | Build-system question |
| --- | --- | --- |
| [`wordcount`](wordcount) | Files, structs, loops, raw buffers | One package, one binary: how to build, run and test it |
| [`textkit`](textkit) + [`textstat`](textstat) | Modules, traits, generics | A local library dependency; two packages with the same function name |
| [`bigint`](bigint), [`rational`](rational), [`interval`](interval) | Operators on user types, `Vec`, exact arithmetic | A chain of libraries: `interval` needs `rational`, which needs `bigint` |
| [`cycle-clique`](cycle-clique) | A published proof's computer check, ported and run natively | Optimization level as a build setting; expected output that comes from another project |
| [`hellinger`](hellinger) | Traits over three number types, `i128`, an interval proof by branch and bound, `grad` to check its derivatives | A build that goes through LLVM IR and the bundled `clang`, because `vxc -O1` is too slow |
| [`tiny-mlp`](tiny-mlp) | `grad` on training math: textbook derivatives, backprop checked three ways, Adam on XOR | Optional toolchain parts (Enzyme); a program that only runs under `run-jit` |
| [`flash-attn`](flash-attn) | FlashAttention's tiled online softmax, checked against naive attention | A project that needs a compiler newer than the last release |
| [`bpe-tokenizer`](bpe-tokenizer) | A GPT-style byte-pair tokenizer: training, encoding, decoding | Data files from outside the project as test inputs |

The three number libraries are for checking the computer-assisted parts of proofs, such as those
in [openai/math](https://github.com/openai/math), where the checkers need integers of any size,
exact fractions and interval bounds, and Vx's `std` has none of them. `cycle-clique` and `hellinger` are the
first such checks ([vx-lang/Vx#1489](https://github.com/vx-lang/Vx/issues/1489)).

Planned, each adding one new question:

| Project | Shows | Build-system question |
| --- | --- | --- |
| `will-it-fit` | `--machine`, capacity and dtype errors | Machine models as build inputs, per target |
| `prefill-decode` | `spawn on`, `transfer`, `--verify-seams` | Optional tools (`z3`) in the test step |
| `ffi-zlib` | Calling a C library | Linking native libraries; build scripts |

## Getting the toolchain

```
curl -fsSL https://vxlang.org/install.sh | sh
```

Every project builds with `./build.sh` against `~/.vx/current`, or the toolchain `VX_HOME` names.

The projects up to `tiny-mlp` were written against v0.0.3. From `flash-attn` on, they use Vx
built from `main` (39a12e0a or later), which has fixes the release lacks; there is no nightly
installer yet, so that means building Vx and packaging it with `scripts/release/package.sh`.
Every project passes with that build too.

## One rule for the package manager

Vx checks function bodies in parallel because every global fact is known before any body is
checked ([language rules](https://github.com/vx-lang/Vx/blob/main/docs/language_rules_for_a_parallel_compiler.md)).
The package manager must keep that true: a build step that runs before the compiler, such as one
that generates source files, is fine; anything that adds declarations while bodies are being
checked, as Rust's procedural macros can, is not.
