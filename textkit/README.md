# textkit

A small library for counting things in text: words, numbers, runs of capital letters.
[`textstat`](../textstat) is a tool that uses it.

```console
$ ./build.sh
ok
```

`classes.vx` defines the trait `ByteClass` and some sets of bytes that implement it: `Space`,
`Digit`, `Upper`, and `Not<C>`, every byte not in `C`. `runs.vx` defines `Runs<C>`, which counts
the runs of bytes in class `C`. Text can be fed to it in pieces, and a run that crosses two
pieces counts once:

```rust
import textkit::runs;

let mut w = words();             // Runs<Not<Space>>
w.feed(p, 0, n);                 // the bytes p[0] to p[n - 1]
print!(w.count);
```

## Using it from another project

Vx looks for `import textkit::runs` as `textkit/runs.vx` under each import path. The import
paths are the toolchain's `stdlib/` and the current directory, so the directory that holds
`textkit/` must be added. `VX_STD_PATH` does that, but it replaces the stdlib paths, so they are
listed again:

```sh
export VX_STD_PATH="$VX_HOME/stdlib/std:$VX_HOME/stdlib:.."
```

Every top-level name in textkit, such as `words` or `Space`, is visible to the program that
imports it. See [`../FRICTION.md`](../FRICTION.md) for what happens when two libraries use the
same name.
