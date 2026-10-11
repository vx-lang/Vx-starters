# Friction log

Everything that got in the way while writing these projects with a released Vx toolchain.
Each entry is a requirement for the build system or package manager
([vx-lang/Vx#1275](https://github.com/vx-lang/Vx/issues/1275)), or a gap in the language or
standard library. Add to it as you go; do not tidy it into a design.

Format: the project, the toolchain version, what happened, and what we did instead.

## Building

- **wordcount, v0.0.3: no way to build an executable.** `vxc` can run a program (`run-jit`) or
  emit an object file (`emit-obj`), but not link one. `build.sh` links the object against
  `$VX_HOME/lib/libvx_std_core.so` with the bundled clang, and sets an rpath so the binary finds
  it. A user has to know the library's name and where it lives.
- **wordcount, v0.0.3: no project root.** Imports resolve from the toolchain's `stdlib/`, then the
  directory `vxc` runs in. There is no manifest, so `build.sh` `cd`s to the project first.

## Libraries

- **textkit, v0.0.3: a library is found only through `VX_STD_PATH`.** `import textkit::runs`
  looks for `textkit/runs.vx` under the stdlib paths and the current directory. To use a library
  in a sibling directory, textstat's `build.sh` sets `VX_STD_PATH` to the repository root, and
  since `VX_STD_PATH` replaces the stdlib paths, it lists them again:
  `$VX_HOME/stdlib/std:$VX_HOME/stdlib:..`. Without them, `import std::fs` fails. The library's
  name is its directory's name, and every directory under the root becomes importable.
- **textkit, v0.0.3: two libraries with the same function name cannot both be used.** Every top-level name
  goes into one namespace. When two modules declare the same function, the compiler renames all
  copies but one (the program's own copy, if it has one). So:
  - textstat can declare its own `is_space` next to a library's `is_space`: each calls its own.
  - Two imported libraries can both declare `is_space`, as long as the program does not use it.
  - If the program calls it, the error is clear (`is_space is declared by both other::scan and
    textkit::scan, and this module sees both through its imports`), but there is no way to say
    which one is meant: `textkit::scan::is_space(..)` does not parse, and `import` has no `as` and
    no list of names. Only the compiler's internal spelling works: `textkit_scan__is_space(..)`.
- **textkit, v0.0.3: no private names, and imports are transitive.** Every top-level name in a
  library is visible to its users, helpers included. textstat used `Upper` from
  `textkit::classes` while importing only `textkit::runs`, and it compiled. Adding a helper to a
  library can break a program that uses the same name through two imports.
- **textkit, v0.0.3: a precompiled `.vxlib` does not work yet.** `vxc -a emit-interface` writes
  `scan.vxlib`, and imports prefer it over `scan.vx`. But a library function that calls another
  function in the same library fails to link (`undefined reference to 'is_space'`), under
  `run-jit`, `-j` and `--link-interface` alike, and `emit-obj` writes no object
  ([vx-lang/Vx#1455](https://github.com/vx-lang/Vx/issues/1455)). Calling a `.vxlib` function
  that takes a `u8` with the literal `32` fails in codegen with `'i8' vs 'i32'`
  ([vx-lang/Vx#1456](https://github.com/vx-lang/Vx/issues/1456)). textkit is used from source.
- **textkit, v0.0.3: no test runner.** `textkit/test.vx` is a program that returns the number of
  failed checks; `run-jit` exits non-zero when it does, and `build.sh` runs it.

## Running

- **wordcount, v0.0.3: `run-jit` and a built binary see different arguments.** Under `run-jit`
  the program's first argument is the `.vx` source path, then the user's arguments; a built
  binary sees only the user's ([vx-lang/Vx#1454](https://github.com/vx-lang/Vx/issues/1454)).
  wordcount takes its file from the last argument to work both ways, and under `run-jit` with
  no file it counts its own source instead of printing its usage line.

## Tools

- **wordcount, v0.0.3: `vx-format` breaks `\n` in strings.** It writes each `\n` escape back as
  a real line break inside the string literal
  ([vx-lang/Vx#636](https://github.com/vx-lang/Vx/issues/636)). wordcount is not formatted.
- **textkit, v0.0.3: the compiler prints notes about itself on every build.** Each compile of
  textkit's tests and of textstat writes
  `[flat-codegen] emit declined: an emitter gap at src/codegen/flat/emit/aggregate.rs:219` and
  `program outside the flat subset; using the AST path` to stderr. The program is fine; the notes
  are for compiler developers. `self.count += 1` on a field adds a third; textkit writes
  `self.count = self.count + 1`.

## Language

- **textkit, v0.0.3: `unsafe { .. } as T` does not parse.** `let n = unsafe { f() } as i64;` is
  a parse error (`Unexpected token As`); the cast goes inside: `unsafe { f() as i64 }`
  ([vx-lang/Vx#1462](https://github.com/vx-lang/Vx/issues/1462)).
- **textkit, v0.0.3: no pointer arithmetic, and the error has no location.** `p + 2` on a
  `*mut u8` is `Error[E3004] at 0:0: Type mismatch in binary operation`, and `core::ptr` has no
  `add` ([vx-lang/Vx#1463](https://github.com/vx-lang/Vx/issues/1463)). `Runs::feed` takes a
  start and an end index instead.
- **textkit, v0.0.3: generic calls do not convert arguments.** Calling a generic function with
  a `*mut u8` where it takes `*const u8`, or the literal `7` where it takes `i64`, fails with
  `Failed to deduce types for generic function`. The same calls to a non-generic function work
  ([vx-lang/Vx#1464](https://github.com/vx-lang/Vx/issues/1464)).

## Standard library

- **wordcount, v0.0.3: no command-line arguments in `std`.** The runtime has `vx_env_args_count`
  and `vx_env_arg`; wordcount declares them in an `extern` block itself.
- **wordcount, v0.0.3: a missing file looks like an empty one.** `File::open` cannot report a
  failed open, so `wordcount nosuchfile` prints `0 0 0`.
- **wordcount, v0.0.3: no string splitting and no map with string keys.** The first plan was "the
  ten most common words"; it became a `wc` clone. `HashMap` takes only `i32` keys.
