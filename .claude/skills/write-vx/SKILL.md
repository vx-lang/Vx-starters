---
name: write-vx
description: Write, fix or explain a Vx program (a .vx file) in Vx-starters. Use when adding a starter project, changing one, or when a .vx file does not compile.
---

# Writing Vx in Vx-starters

Vx looks like Rust, and Rust habits are the most common reason Vx does not compile. Before
writing Vx, read these pages. They are short, and their programs are compiled by Vx's test suite,
so they match the compiler:

- [Vx for Rust programmers](https://vxlang.org/docs/for-rust-programmers.html): what to change
  when writing Vx the Rust way.
- [Vx in one page](https://vxlang.org/docs/quick-reference.html): the language on one page, with
  complete programs. Until vx-lang/Vx#1728 is merged, read the
  [tour](https://vxlang.org/docs/tour.html) instead.

[llms.txt](https://vxlang.org/llms.txt) lists every other page. For a standard library type, read
its source in `$VX_HOME/stdlib/`, from the toolchain you build with, not from a Vx checkout: the
two can differ. Do not guess a method's name or signature. The starters here are working
programs too; `grep -rl` them for a feature before writing it from scratch.

## Toolchain

`VX_HOME` picks the toolchain, and `~/.vx/current` is the default. Projects up to `tiny-mlp` build
with v0.0.3; newer ones need Vx built from `main` (see the end of `README.md`). Run a program with:

```sh
"$VX_HOME/bin/vxc" -a run-jit src/main.vx
```

Lines starting `[JIT]` are the compiler's own progress, not the program's output. Run anything
whose time or memory you do not know under a limit, for example
`( ulimit -v 4000000; timeout 300 ./build.sh )`.

When it fails, fix the first error and compile again: later errors often follow from the first.
For an error with a code, such as `E3004`, read `https://vxlang.org/errors/E3004`.

## A project

Each project is one directory with `README.md` and `build.sh`. A program keeps its source in
`src/` (usually `src/main.vx`); a library keeps its modules at the top with a `test.vx`, like
`bigint`. Copy the layout of a neighbouring project.

- `build.sh` runs the program and checks its output, and exits non-zero when a check fails.
  Judge a run by that exit code, not by the last line printed.
- Every check is compared with an independent answer: a naive version, a Python script, a
  formula or a proof. Break each check on purpose, on a copy, and see it fail before trusting it.
- Add the project to a table in the top `README.md`, with what it shows and the build-system
  question it raises.

## When the compiler is wrong

An error that names a compiler source file, a `Codegen Error`, or a `Vx Compiler Internal Error`
is a compiler bug, not a mistake in the program. So is a program that compiles but prints the
wrong answer. Reduce it to a few lines, check the open issues on vx-lang/Vx, and report it. Do
not leave a workaround in a program without saying which issue it works around, and add the
obstacle to `FRICTION.md`.
