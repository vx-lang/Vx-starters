# will-it-fit

"Will this model fit on this GPU?", answered by the compiler: the same program is compiled
against two machine files, an A100 40 GB and an H100 80 GB, and refused where it does not fit.

```console
$ ./build.sh
weights, 32 x 4096 x 4096 x 12:
  bf16 12884901888 bytes (12 GiB)
  fp8  6442450944 bytes (6 GiB)
  int4 6442450944 bytes (6 GiB)
KV cache, batch 32 x 8192 tokens, bf16: 34359738368 bytes (32 GiB)
activations, bf16: 2147483648 bytes (2 GiB)
total with bf16 weights: 49392123904 bytes (46 GiB)
total with fp8 weights:  42949672960 bytes (40 GiB)
total with int4 weights: 42949672960 bytes (40 GiB)
total with int4 weights and batch 33: 44090523648 bytes (41.0625 GiB)
bf16 a100-40 rejected E6010
bf16 h100-sxm admitted
fp8 a100-40 rejected E6026
fp8 h100-sxm admitted
int4 a100-40 admitted
int4 h100-sxm rejected E6026
int4-33 a100-40 rejected E6010
int4-33 h100-sxm rejected E6026

compiler: bf16     on a100-40   rejected E6010
compiler: bf16     on h100-sxm  admitted
compiler: fp8      on a100-40   rejected E6026
compiler: fp8      on h100-sxm  admitted
compiler: int4     on a100-40   admitted
compiler: int4     on h100-sxm  rejected E6026
compiler: int4-33  on a100-40   rejected E6010
compiler: int4-33  on h100-sxm  rejected E6026
ok
```

The model is 8B-class (32 layers, 32 query heads and 8 KV heads of 128, as in Llama-3-8B),
served at batch 32 with 8192 tokens of context each. Three things sit in GPU memory during a
decode step:

| | Size | bf16 |
| --- | --- | --- |
| Weights | `12 x layers x dim^2` (attention `4 dim^2`, MLP `8 dim^2`) | 12 GiB |
| KV cache | `2 x layers x batch x context x kv_dim` | 32 GiB |
| Activations | `batch x context x dim` | 2 GiB |

## How the compiler answers

`src/bf16.vx` declares the three tensors with these sizes, as const-generic arithmetic, and
places them in `Memory::HBM`. It is never run, only compiled:

```sh
vxc --machine $VX_HOME/fleet/a100-40.vx --host $VX_HOME/fleet/host-x86-e5-2666v3.vx -a print-ast src/bf16.vx
Error[E6010]: the working set placed in memory space 'HBM' (3 tiles) sums to 49392123904 bytes, over its 42949672960 byte capacity
```

A machine file declares a GPU's memories (capacity, bandwidth), the links between them, and the
element types its hardware has (`dtypes:`). The program does not change between machines; only
`--machine` does. Two errors matter here:

- **E6010**: what is placed in HBM at the same time adds up to more than its capacity.
- **E6026**: a tensor's element type is not one the machine has. The A100 has int4 and no fp8;
  the H100 has fp8 and no int4. So `src/fp8.vx` (fp8 weights) fits only the H100, and
  `src/int4.vx` (int4 weights) only the A100. Newer is not always more.

`src/sizes.vx` does the same arithmetic at run time and predicts every verdict; `build.sh`
compiles each program against each machine and checks that the compiler says the same thing,
error code included. With int4 weights at batch 32, the A100's 40 GiB is filled to the byte and
the program is admitted. `build.sh` then makes a copy with batch 33, and that is refused.

The verdict is read from `--diagnostics-json`, a JSON record of each compile with the bytes
required and available. `print-ast` runs every check and stops before code generation, which is
needed: code for fp8 tensors is not written yet ("sub-8-bit float element types are
capacity/declaration-only").

## Things to know

- The three tensors must be placed in one function and read after the last placement. A tensor
  nobody reads is released where it is made, so without the reads the compiler sees one tensor
  at a time, and the bf16 program is admitted on the A100. (`fleet/admit.vx` in the Vx repository
  explains this rule; this project follows it.)
- An int4 weight counts as a whole byte, the same as fp8, because that is what Vx allocates for
  one ([vx-lang/Vx#1202](https://github.com/vx-lang/Vx/issues/1202)). A real int4 checkpoint
  packs two to a byte, which would be 3 GiB of weights here, not 6
  ([vx-lang/Vx#1437](https://github.com/vx-lang/Vx/issues/1437)).
- The machine files declare the hardware's full capacity. A serving engine keeps some of it for
  itself (vLLM uses 90% by default), so "admitted" at 100% can still run out of memory.
- Not every machine file can be used yet: `h200.vx` and `mi300x.vx` have no route from GPU memory
  back to the host, which a program reading its tensors back needs
  ([vx-lang/Vx#1403](https://github.com/vx-lang/Vx/issues/1403)).

Each of these changes makes it fail:

- dropping the reads that make the three tensors co-resident (the compiler then admits bf16 on
  the A100);
- weights of `4 x layers x dim^2` (attention only);
- a KV cache sized by the query heads instead of the KV heads;
- predicting that the A100 has fp8;
- treating "exactly full" as "over".

Needs Vx newer than v0.0.3 (built from `main`); see the [top-level README](../README.md).
