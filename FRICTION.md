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

## Standard library

- **wordcount, v0.0.3: no command-line arguments in `std`.** The runtime has `vx_env_args_count`
  and `vx_env_arg`; wordcount declares them in an `extern` block itself.
- **wordcount, v0.0.3: a missing file looks like an empty one.** `File::open` cannot report a
  failed open, so `wordcount nosuchfile` prints `0 0 0`.
- **wordcount, v0.0.3: no string splitting and no map with string keys.** The first plan was "the
  ten most common words"; it became a `wc` clone. `HashMap` takes only `i32` keys.
