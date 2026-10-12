# ffi-zlib

Vx calling a C library: the system's zlib, through an `extern` block.

```
$ ./build.sh
zlib 1.3.1
crc32("123456789") = 3421780262, adler32("Wikipedia") = 300286872
CRC-32 in Vx vs zlib, 201 buffers: 0 differ; chained crc32 vs whole: 0 differ
text, 7320 bytes, compressed at levels 0 1 6 9: 7331 262 262 262
random bytes, 20000 bytes, compressed at levels 0 1 6 9: 20011 20016 20016 20016
broken header: -3, broken checksum: -3, output one byte short: -5
ok
```

`src/main.vx` declares six zlib functions in an `extern` block: `zlibVersion`, `crc32`,
`adler32`, `compressBound`, `compress2` and `uncompress`. It calls them with pointers into
`Vec<u8>` buffers, through small safe wrappers that always pass a buffer's own pointer and length
together. On 64-bit Linux, zlib's `uLong` is `u64` and `uInt` is `u32`.

## What is checked

1. **Known values.** `crc32("123456789")` is `0xCBF43926` (3421780262), the standard CRC-32
   check value, and `adler32("Wikipedia")` is `0x11E60398` (300286872), the example on Wikipedia.
2. **CRC-32 in Vx.** A table-driven CRC-32 written in Vx gives the same result as zlib's on the
   check string and on 200 random buffers of up to 5000 bytes. zlib's `crc32` of two pieces,
   chained, equals its `crc32` of the whole.
3. **Round trips.** `compress2` then `uncompress` gives back exactly the same bytes, for text and
   for random bytes, at levels 0 (store only), 1, 6 and 9. The repetitive text shrinks from 7320
   bytes to 262; random bytes cannot shrink, and come out slightly larger.
4. **Broken streams are refused.** Flipping a bit in the first byte (the header) or the last byte
   (the Adler-32 of the data) gives `Z_DATA_ERROR` (-3). An output buffer one byte too small gives
   `Z_BUF_ERROR` (-5).

Each of these changes makes it fail:

- the wrong CRC polynomial in the Vx table;
- leaving out the CRC's final XOR;
- starting Adler-32 at 0 instead of 1;
- not trimming the compressed buffer to the length zlib reports;
- compressing one byte too few;
- expecting `Z_OK` from a too-small buffer;
- leaving `-lz` out of the link, which fails with `undefined reference to 'crc32'`.

## Linking a C library

This is the build-system question here, and `vxc` has no answer yet.

- `vxc -a run-jit` builds an executable with clang and links a fixed list of libraries (in Vx's
  `src/jit.rs`, ending in `-lm`). It has no flag or environment variable to add `-lz`, and an
  `extern` block cannot name its library
  ([vx-lang/Vx#1652](https://github.com/vx-lang/Vx/issues/1652)). Run that way, this program
  fails with `undefined reference to 'crc32'`.
- So `build.sh` compiles to an object with `vxc -a emit-obj` and links it with the toolchain's
  own `clang`. It must name everything `run-jit` would: `libvx_std_core` from `$VX_HOME/lib`, and
  `libmlir_c_runner_utils` and `libmlir_runner_utils` from `$VX_HOME/llvm/lib`, each with an
  rpath. Then it adds `-lz -lm`. Nothing documents that list
  ([vx-lang/Vx#1499](https://github.com/vx-lang/Vx/issues/1499)); it was read from `src/jit.rs`.
- Another way that works: point `CLANG_PATH` at a script that runs the real clang with `"$@" -lz`.
  `vxc -a run-jit` links through `CLANG_PATH`, so the program then runs under `run-jit`. This
  relies on how `vxc` links inside, which can change.

A package manager would need each package to declare the native libraries it links. The simplest
place is the `extern` block itself, as Rust's `#[link(name = "z")]` does. The build would then
pass them on to the link, and could check that they exist before compiling.

The CRC table is built at run time. The plan was to build it at compile time, but the
compile-time evaluator cannot run `>>`, `^` or `&` yet
([vx-lang/Vx#1653](https://github.com/vx-lang/Vx/issues/1653)).

Needs Vx newer than v0.0.3 (built from `main`), and zlib (`libz.so`, as in Debian and Ubuntu's
`zlib1g-dev`); see the [top-level README](../README.md).
