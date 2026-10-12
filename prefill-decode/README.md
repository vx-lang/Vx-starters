# prefill-decode

One request served the way large deployments split it: prefill on one device, decode on another,
with the KV cache moved between them by explicit transfers that the compiler proves safe.

```
$ ./build.sh
prompt:      3 8 13 2 7 12 1 6
GPU + NPU:   14 4 9 11 15 5 10 0
host alone:  14 4 9 11 15 5 10 0
ok
ok: Python decodes the same tokens
[seam] 3 obligation(s)
E6004] at 141:20: relaxed transfer of 'staged' across the CPUDRAM -> NPUHBM seam
ok: seams proved, and the relaxed copy refused
```

Prefill reads the whole prompt at once and is limited by compute. Decode writes one token at a
time and is limited by memory bandwidth. "Disaggregated serving" runs them on different machines,
which means the KV cache that prefill builds has to be shipped to the decoder. `src/main.vx`
does that with Vx's placement types:

1. `spawn on(Topology::GPU) { .. }` prefills: one key/value row per prompt token, written into a
   cache whose type says where it lives, `Tensor<f32, [16, 4], Memory::GPU_HBM>`.
2. `transfer(cache, Memory::CPU_DRAM)` brings it to the host, and `.to_device()` sends it on to
   NPU memory. Each step changes the tensor's type, so using the cache in the wrong place is a
   compile error: passing the host copy to the NPU's decode function is `E3003: Expected
   &Tensor<f32, [16, 4], Memory::NPU_HBM>, got &Tensor<f32, [16, 4], Memory::CPU_DRAM>`.
3. `spawn on(Topology::NPU[0]) { .. }` decodes 8 tokens greedily, adding each one's row to the
   cache.
4. The tokens come back to the host by `transfer` and must equal what the same model gives when it
   runs entirely on the host. `build.sh` also runs `check.py`, the model again in plain Python,
   and compares.

The model is a toy, 16 tokens with 4-number embeddings. A decode step weights each cached value
by `(q . k + 4)^2` and picks the unused token that best matches the result. All its numbers are
small whole numbers, so f32 on the devices, f32 on the host and Python agree exactly.

This machine has no GPU or NPU. Vx then runs both regions on the host, and the placements,
transfers and topology rules are checked all the same.

## Seams and z3

A seam is one hop of a transfer between memories. `vxc --verify-seams` proves, with the z3 SMT
solver, that the reader of every hop sees the written data. A plain transfer synchronizes. A
`_relaxed` one does not, and the reader could see a stale buffer. Here there are 3 seams: GPU to
host, host to NPU, and NPU to host for the tokens.

`build.sh` proves all three, then makes a copy of the program with `.to_device_relaxed()` for the
host-to-NPU hop and checks that it is refused with `E6004`, with z3's counterexample. Without
`--verify-seams`, that copy compiles and runs, and prints the right tokens, because on one CPU
nothing can be stale. That is why the check is a proof and not a test.

z3 is optional. When it is not on `PATH`, `--verify-seams` refuses every compile (`E6024`, which
names the hop and says how to install z3), so `build.sh` skips that step and says so:

```
skipped: seam verification needs z3 on PATH
```

## What did not work as planned

- **The model's code cannot be shared between devices.** A Vx function belongs to one topology:
  an ordinary function is host code, and a GPU or NPU region cannot call it (`E6001: Function
  'embed' requires topology 'CPU', but is called from 'GPU[0]'`). The formulas are therefore
  macros, which each region expands. A macro can only stand for an expression, not for a loop, so
  the decode step is a function written twice, `next_on_npu` (placed `on Topology::NPU[0]`) and
  `next_on_host`. If the two drift apart, the comparison fails.
- **A macro call cannot be another macro's argument**, `print!` included: `embed!(prompt!(p), i)`
  fails with `Macro failed to expand`, six times, with no location
  ([vx-lang/Vx#1655](https://github.com/vx-lang/Vx/issues/1655)). Results go through a `let`.
- A macro call at the end of an `if` condition takes the `{` of the `if` body, so
  `if x == embed!(t, 3) {` does not parse
  ([vx-lang/Vx#1657](https://github.com/vx-lang/Vx/issues/1657)). The decode step reads the
  embedding into local variables first.

Each of these changes makes it fail:

- prefill writing the wrong token's row (the tokens differ from the host's);
- decode not appending its rows (the tokens differ);
- the host copy of the decode step querying the wrong row (the two copies drift);
- both copies using `s^2` as the weight (only Python sees it);
- a relaxed transfer to the NPU (`E6004` under `--verify-seams`);
- the NPU decoding from the host copy of the cache (a type error).

Needs Vx newer than v0.0.3 (built from `main`) and `python3`; z3 is optional. See the
[top-level README](../README.md).
