# ring-allreduce

The all-reduce of data-parallel training, done the way NCCL's ring does it, on simulated ranks.

```console
$ ./build.sh
2 ranks: each sends 1 vectors (at most 1000 numbers); through rank 0 it would send 1; reals off by 0
3 ranks: each sends 1.3333333333333333 vectors (at most 1334 numbers); through rank 0 it would send 2; reals off by 4.4e-16
4 ranks: each sends 1.5 vectors (at most 1500 numbers); through rank 0 it would send 3; reals off by 4.4e-16
5 ranks: each sends 1.6 vectors (at most 1600 numbers); through rank 0 it would send 4; reals off by 8.9e-16
6 ranks: each sends 1.6666666666666667 vectors (at most 1668 numbers); through rank 0 it would send 5; reals off by 8.9e-16
7 ranks: each sends 1.7142857142857142 vectors (at most 1715 numbers); through rank 0 it would send 6; reals off by 8.9e-16
8 ranks: each sends 1.75 vectors (at most 1750 numbers); through rank 0 it would send 7; reals off by 8.9e-16
ok
```

(The real output prints the last column in full.)

In data-parallel training every rank (GPU) computes a gradient, and every rank needs the sum
before the optimizer step. Ring all-reduce puts the N ranks in a ring, each sending only to its
right neighbour, and cuts the vector into N chunks:

1. **Reduce-scatter**, N - 1 steps. At step s, rank r sends chunk `r - s` (mod N) to rank r + 1,
   which adds it into its own copy. Afterwards rank r holds the full sum of chunk r + 1.
2. **All-gather**, N - 1 steps. At step s, rank r sends chunk `r + 1 - s` (mod N), now a full
   sum, to rank r + 1, which overwrites its copy. Afterwards every rank has every sum.

Each rank sends `2 (N - 1) / N` of the vector, under 2 for any N. Sending everything to rank 0 and
back makes rank 0 send, and receive, N - 1 whole vectors. That is why the ring is used.

Vx has no threads yet, so the ranks are rows of one array. A step copies every rank's outgoing
chunk into a buffer before anyone adds, as if all sends happen at once.

## What is checked

For N = 2 to 8 ranks and a vector of 1000 numbers (so 3, 6 and 7 ranks get chunks of unequal
size; the first `1000 % N` chunks get one more):

- With whole numbers, every rank ends with exactly the plain sum.
- With random reals, every rank ends with the same bits (all-gather copies), within 10^-12 of the
  plain sum; the ring adds in a different order.
- The ranks send `2 (N - 1)` vectors in all, and no rank more than `2 (N - 1)` of the largest
  chunk.
- Contracts prove that every chunk and rank number (`r - s mod N`, `r + 1 - s mod N`, `r + 1 mod
  N`) is in `0..N`. They are proved when the program compiles. The prover cannot use `%`, so
  `wrap` takes `x mod n` for `x` within one `n` of the range, with two `if`s.

Each of these changes makes it fail: adding nothing in reduce-scatter (overwriting instead),
all-gather sending the wrong chunk, one reduce-scatter step too few, chunks that ignore the
remainder, a sent count that is off by one per step, and `wrap` leaving `x == n` alone (that one
stops the compile). One change does not: applying each send at once instead of through the
buffer. At each step a rank sends a different chunk from the one it receives, so the order
within a step does not matter.

Needs Vx newer than v0.0.3 (built from `main`); see the [top-level README](../README.md).
