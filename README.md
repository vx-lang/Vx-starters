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

Planned, each adding one new question:

| Project | Shows | Build-system question |
| --- | --- | --- |
| `textkit` + a tool that uses it | Modules, traits, generics | A local library dependency; two packages with the same function name |
| `will-it-fit` | `--machine`, capacity and dtype errors | Machine models as build inputs, per target |
| `tiny-mlp` | Tensors, `matmul`, `grad` | Optional toolchain parts (Enzyme) |
| `prefill-decode` | `spawn on`, `transfer`, `--verify-seams` | Optional tools (`z3`) in the test step |
| `ffi-zlib` | Calling a C library | Linking native libraries; build scripts |

## Getting the toolchain

```
curl -fsSL https://vxlang.org/install.sh | sh
```

Every project builds with `./build.sh` against `~/.vx/current`, or the toolchain `VX_HOME` names.

## One rule for the package manager

Vx checks function bodies in parallel because every global fact is known before any body is
checked ([language rules](https://github.com/vx-lang/Vx/blob/main/docs/language_rules_for_a_parallel_compiler.md)).
The package manager must keep that true: a build step that runs before the compiler, such as one
that generates source files, is fine; anything that adds declarations while bodies are being
checked, as Rust's procedural macros can, is not.
