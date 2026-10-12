# Friction log

Everything that got in the way while writing these projects with a Vx toolchain: v0.0.3, and
from `flash-attn` on, Vx built from `main`.
Each entry is a requirement for the build system or package manager
([vx-lang/Vx#1275](https://github.com/vx-lang/Vx/issues/1275)), or a gap in the language or
standard library. Add to it as you go; do not tidy it into a design.

Format: the project, the toolchain version, what happened, and what we did instead.

## Building

- **flash-attn, main 39a12e0a: no way to get a toolchain newer than the release.** The bug fixes
  since v0.0.3 are only on `main`. Using them meant building Vx (a full `cargo build --release
  --workspace`) and packaging it with `scripts/release/package.sh`, borrowing
  the LLVM directory from the installed v0.0.3. A package needs a way to say "needs Vx newer than
  v0.0.3", and the installer a way to install a nightly.

- **wordcount, v0.0.3: no way to build an executable.** `vxc` can run a program (`run-jit`) or
  emit an object file (`emit-obj`), but not link one. `build.sh` links the object against
  `$VX_HOME/lib/libvx_std_core.so` with the bundled clang, and sets an rpath so the binary finds
  it. A user has to know the library's name and where it lives.
- **wordcount, v0.0.3: no project root.** Imports resolve from the toolchain's `stdlib/`, then the
  directory `vxc` runs in. There is no manifest, so `build.sh` `cd`s to the project first.

- **cycle-clique, v0.0.3: optimization is off unless asked for.** `vxc` defaults to `-O0`.
  `build.sh` passes `-O3`, which makes the check several times faster. A build tool needs a
  debug and a release setting, and a way to set it per project.

- **hellinger, v0.0.3: optimized builds take minutes.** `vxc -O1`, `-O2` and `-O3` did not
  finish in 250 s on a 700-line program that `-O0` compiles in 1 s. Compile time grows about
  threefold with each level of nested calls, as if every call were inlined
  ([vx-lang/Vx#1589](https://github.com/vx-lang/Vx/issues/1589)). `build.sh` runs
  `vxc -a emit-llvm`, the bundled `mlir-translate` and `clang -O2` instead, which takes 3 s and
  gives code within 20% of the C++ original. `-a emit-llvm` writes MLIR's LLVM dialect to
  standard output and ignores `-o`; a build tool has to know that `mlir-translate` comes next.

- **hellinger, v0.0.3: a program that uses `grad` runs only under `run-jit`.** `-a emit-obj`
  fails with `Symbols not found: [ __enzyme_autodiff_grad_f ]`, and `-a emit-llvm` leaves that
  call undefined ([vx-lang/Vx#1594](https://github.com/vx-lang/Vx/issues/1594)). The derivative
  cross-check is a second program that `build.sh` runs with `run-jit`, next to the built search.

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
- **bigint, v0.0.3: nothing records that one library needs another.** `interval` imports
  `rational`, which imports `bigint`. It works because all three sit under the repository root,
  which `VX_STD_PATH` names; a program that copies `interval` alone gets an import error, and
  nothing says which versions go together.
- **textkit, v0.0.3: no test runner.** `textkit/test.vx` is a program that returns the number of
  failed checks; `run-jit` exits non-zero when it does, and `build.sh` runs it.

## Wrong results and crashes

Found while writing `bigint`, `rational` and `interval`. A whole program is compiled by one of
two code generators. It uses the newer "flat" one unless any function in it is outside what that
one handles; then the whole program goes through the older "AST" one, and the compiler prints
`program outside the flat subset; using the AST path`. A program that uses `Option<BigInt>` is
on the AST path, so these libraries are, and the same function can work in one program and
fail in another.

- **bigint, v0.0.3: `&&` and `||` evaluate both sides on the AST path.** In
  `while a.len() > 0 && a.get(a.len() - 1) == 0`, the read runs when the vector is empty and
  aborts with `index -1 out of bounds`. On the flat path it is fine
  ([vx-lang/Vx#1485](https://github.com/vx-lang/Vx/issues/1485)). The libraries never guard an
  index with `&&` or `||`.
- **bigint, v0.0.3: `u32` and `u64` convert to `f64` as if they were signed, on the AST path.**
  `3000000000u32 as f64` is `-1294967296`
  ([vx-lang/Vx#1486](https://github.com/vx-lang/Vx/issues/1486)). `to_f64` converts through
  `i64`, and a test checks it.
- **bigint, v0.0.3: a `u128` literal from 2^63 to 2^64 - 1 is sign-extended.**
  `let m : u128 = 18446744073709551615;` gives `m >> 64` equal to 2^64 - 1, not 0. The same
  value through a `u64` variable and `as u128` is right
  ([vx-lang/Vx#1487](https://github.com/vx-lang/Vx/issues/1487)). The libraries use `u32` digits
  and `u64` products, and no `u128`.
- **rational, v0.0.3: a method that calls itself overflows the compiler's stack.** Even
  `return self.down(n - 1) + 1;` kills `vxc` with `thread 'main' has overflowed its stack`, at
  any action that compiles ([vx-lang/Vx#1488](https://github.com/vx-lang/Vx/issues/1488)). A
  free function that calls itself works. `Rational::pow` uses no recursion.

- **cycle-clique, v0.0.3: storing into a field or through `&mut` does not free the old value.**
  `b.v = make(1000);` and `*out = make(1000);` leak the old `Vec`'s buffer; assigning to a local
  frees it ([vx-lang/Vx#1573](https://github.com/vx-lang/Vx/issues/1573)). A hash set that grew
  by `set.keys = bigger` made the checker peak at 600 MB, and a variant that ran longer reached
  57 GB and set off the machine's out-of-memory killer, which also killed other programs.
  The checker keeps its hash set in a local vector now (4 MB). `bigint`'s division wrote its
  quotient and remainder through `&mut` and lost about 145 bytes per call; it returns a struct
  now, and `bigint/build.sh` checks that 400,000 divisions stay under 20 MB.
- **bigint, v0.0.3: moving a field out of a struct frees it twice.** `let d = f(); return d.q;`
  aborts with `free(): double free`, because `d` still drops `q`
  ([vx-lang/Vx#1507](https://github.com/vx-lang/Vx/issues/1507)). `let S { a, b } = s;` does not
  parse, and `core::mem::replace`, the usual way out, does not compile for a `Vec`
  ([vx-lang/Vx#1585](https://github.com/vx-lang/Vx/issues/1585)). So a function cannot hand back
  two vectors without a copy: `div_rem` returns a struct, and its callers `clone()` the fields.

- **hellinger, v0.0.3: the prover cannot prove anything about a returned struct.** It ignores
  struct construction, so `ensures !return.ok || return.a <= return.b` fails on every `return`
  ([vx-lang/Vx#1590](https://github.com/vx-lang/Vx/issues/1590)). The checker's intervals have no
  contracts.

- **bigint, main 39a12e0a: a loop that swaps values is now a borrow error.** In `gcd`,
  `let r = a.rem(&b); a = b; b = r;` inside a `while` is `E4007: b is moved while it is
  borrowed`; v0.0.3 accepts it, and `r` holds no borrow
  ([vx-lang/Vx#1646](https://github.com/vx-lang/Vx/issues/1646)). Binding the borrow first,
  `let by = &b; let r = a.rem(by);`, is accepted.

- **paged-kv, main 39a12e0a: the prover knows nothing about `let mut`, `/` or `%`.** A loop
  invariant about a variable the loop changes cannot be proved even on entry, and
  `let mut r = pos; r = r + 1;` loses `pos >= 0` (vx-lang/Vx#1353, #1356). Contracts went on
  small straight-line functions, and the cache keeps lengths as (blocks, fill) so that nothing
  proved needs a division.
- **paged-kv, main 39a12e0a: a parameter used only in a contract is "unused".** `W1009: Unused
  function parameter 'blocks'`, with no location, for the pool size in `slot`'s `requires`
  ([vx-lang/Vx#1647](https://github.com/vx-lang/Vx/issues/1647)).

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

- **flash-attn, main 39a12e0a: every widening `as` is a warning.** `(d as f64)` with `d : i32`
  is `W1007: this cast only widens ... f64::from(..) says so`. The older projects print many
  of these; new code uses `f64::from` and imports `core::convert`.
- **bpe-tokenizer: no way to print one byte.** `b as char` is `Unsupported cast: cannot cast
  from Scalar(U8) to Struct("char")`. The program writes the byte and a zero into a
  two-byte buffer and prints that as a C string. In v0.0.3 even that fails, since a `*mut i8`
  cannot be cast to `*const i8`.

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

- **bigint, v0.0.3: no hex literals.** Masks such as 2^32 - 1 are written `4294967295`
  ([vx-lang/Vx#783](https://github.com/vx-lang/Vx/issues/783)).
- **bigint, v0.0.3: `print!` of a `u128` does not compile** (`unsupported print arg type i128`,
  then `MLIR verification failed`). Cast to `u64` first.
- **bigint, v0.0.3: the borrow checker treats a method's result as a borrow of its receiver.**
  After `let y = x.plus(&d);`, the assignment `x = y;` is `E4007: x is moved while it is
  borrowed`, though `y` is a new value. Using `&mut r` again after a call such as `r.abs()` in
  the same function is `E4003`. `isqrt` loops on a flag, and the tests use fresh variables.
- **bigint, v0.0.3: `loop { .. return x; .. }` counts as falling off the end**
  (`E3028: body can finish without returning a value`), though the loop has no `break`.
- **bigint, v0.0.3: a range's variable is `i32`, and is not converted to `i64`.** After
  `for i in 1..31`, `BigInt::from_i64(i)` is `E3003`; it takes `i as i64`. Likewise
  `-9223372036854775807 - 1` where an `i64` is expected is `E3004: i64 vs i32`, at `0:0`.
- **cycle-clique, v0.0.3: `print!` of a range's variable can fail to compile.** In
  `for k in 5..18 { print!(k); }`, `k` has the internal type `index`: `unsupported print arg type
  index`, then `MLIR verification failed`. `print!(k as i32)` works.
- **hellinger, v0.0.3: `grad` fails on integer struct code in what it differentiates.** When the
  function calls code that builds an `i64` interval struct, even only for constants, Enzyme stops
  with `opt failed`, `Cannot deduce type of phi`, and a page of IR. The constants are made
  through the number trait (`of_int`) or computed beforehand.
- **hellinger, v0.0.3: a float literal is `f32`.** `let s = 1000000000000.0;` next to an `f64`
  is `E3004: f64 vs f32`, at `0:0`. Every literal in `gradcheck.vx` has a typed `let`
  (the book's autodiff chapter mentions it, vx-lang/Vx#1446).
- **tiny-mlp, v0.0.3: no gradient for a tensor of weights.** `vjp` returns the function's
  result type, not the differentiated argument's
  ([vx-lang/Vx#1600](https://github.com/vx-lang/Vx/issues/1600)), and a pointer passed after the
  moved parameter breaks `grad` with an Enzyme shadow error
  ([vx-lang/Vx#1601](https://github.com/vx-lang/Vx/issues/1601)). tiny-mlp takes one `grad` per
  weight, with the weights in a struct of `f64` fields passed by value.
- **bigint, v0.0.3: `panic!` does not exist.** It is the function `panic("..")`; `panic!(..)` is
  `Macro panic not found`.

- **batch-sched, main 39a12e0a: `==` on two C strings compares pointers.** `name != "static"`,
  with `name : *const i8` passed `"static"`, is true: the two literals are different pointers.
  It compiles without a warning, and it is also what pushed the program off the flat code
  generator (`emitter gap at src/codegen/flat/emit/arith.rs:302`). The function takes a `bool`
  instead.

- **safetensors, main 39a12e0a: a `void` function ending in `if`/`else` sends the program to the
  old code generator, where more things fail.** On that path `print!(k)` of a range variable is
  `print cannot print a value of type index`, while `print!(k as i32)` gets `W1013: the cast does
  nothing`; and `f32::from(i16)` fails in `mlir-translate`
  ([vx-lang/Vx#1651](https://github.com/vx-lang/Vx/issues/1651)). A `return;` at the end keeps
  the program on the flat path.
- **safetensors, main 39a12e0a: `on` is a keyword.** A parameter called `on` is `Expected
  parameter name`, with nothing saying why (as with `across`,
  [vx-lang/Vx#1650](https://github.com/vx-lang/Vx/issues/1650)).

- **reshard, main 39a12e0a: a keyword as a name gives a vague error.** `let mut across = ..` is
  `Expected identifier after let`; `across` is a keyword, and the message does not say so
  ([vx-lang/Vx#1650](https://github.com/vx-lang/Vx/issues/1650)). Renamed to `span`.
- **quantize, main 39a12e0a: a contract makes the prover warn about a float division.** In a
  function with `requires`, `let s = big / f64::from(lim);` prints `the prover ignores a
  condition here: it cannot reason about the Div operator`, with no location, though nothing in
  the contract involves `s`. The division went into a small function with no contract.
- **bf16, main 39a12e0a: `grad` is a keyword.** `let mut grad : f32 = ..` is `Expected identifier
  after let`, so the variable is `gradient`.
- **bf16, main 39a12e0a: `x as bf16` cannot be the reference.** `core::libm` says it flushes
  subnormals to zero on CPUs with AVX512-BF16, and offers `libm_bf16_bits_from_f32` instead. The
  bf16 check compares with that function; the fp16 check uses `x as f16`.
- **ring-allreduce, main 39a12e0a: `f64::from` has no `i64` version.** `f64::from(x)` with
  `x : i64` is `E3036`, since the conversion can round; it stays `x as f64`, which gives no
  warning. (`quantize` hit the same.)

## Standard library

- **ring-allreduce, main 39a12e0a: no threads, so ranks are simulated.** `core::sync::atomic`
  says "Vx has no threads yet", and `spawn on` is device dispatch. The ranks are rows of one
  array, and a step buffers every send before applying any.

- **wordcount, v0.0.3: no command-line arguments in `std`.** The runtime has `vx_env_args_count`
  and `vx_env_arg`; wordcount declares them in an `extern` block itself.
- **wordcount, v0.0.3: a missing file looks like an empty one.** `File::open` cannot report a
  failed open, so `wordcount nosuchfile` prints `0 0 0`.
- **wordcount, v0.0.3: no string splitting and no map with string keys.** The first plan was "the
  ten most common words"; it became a `wc` clone. `HashMap` takes only `i32` keys.
- **bigint, v0.0.3: no big integers, fractions, intervals or complex numbers, and no control of
  the floating-point rounding mode.** Checkers for computer-assisted proofs need all of these;
  `bigint`, `rational` and `interval` supply the first three. Exact fraction ends take the place
  of outward rounding.
