# cycle-clique

Re-runs the finite check in the proof that the Ramsey number of a cycle and a clique is

    R(C_m, K_n) = (m - 1)(n - 1) + 1    for all m >= n >= 3 except m = n = 3,

the cycle-clique conjecture of Erdős, Faudree, Rousseau and Schelp (1978). The proof is
"Cycle-clique Ramsey numbers" in [openai/math](https://github.com/openai/math)
(`preprints/Cycle-clique-Ramsey-numbers-September-25-2026`). Its argument reduces the remaining
cases to 3,099 small graph patterns, and a program has to rule out each one.

```console
$ ./build.sh
ok: the same table as the paper
```

`src/main.vx` is a port of the paper's `verification/code/original_checker.py` (Apache License
2.0). It does the same steps in the same order; sets of small numbers are `u32` bit masks. For
each k = m - 1 from 5 to 17 it prints k and three counts: patterns it could not rule out,
patterns ruled out by a packing bound, and patterns ruled out after deduction. The proof needs
the first count to be 0 for every k. `build.sh` builds a native binary with `-O3` and compares
its table with the one the paper ships.

| | Time |
| --- | --- |
| The paper's Python checker | 21 s |
| This port, `vxc -a run-jit` (including compiling) | 13 s |
| This port, `-O3` binary | 3.0 s |

What this shows, and what it does not:

- A second implementation, in another language and compiler, gets the same answer from the
  same algorithm. A bug in Python, or one that only shows up in Python, would likely show here.
- It is not an independent algorithm: a mistake in the algorithm itself would be copied. The
  paper also ships `independent_checker.py`, a different method; porting it is the next step.
- The rest of the proof, the reduction to these patterns, is in the paper.

The comparison is not empty. Each of these changes alters the table, and the ones marked * leave
some pattern unexcluded, so the proof would fail:

- turning off the packing search*;
- counting a packing whose weight equals the bound;
- one fewer forbidden cycle length;
- path lengths off by one*;
- never adding a vertex when attaching*.

Raising one packing weight bound by one does not change the table; that bound has room to spare
on these patterns.

The hash set in `exact` is a local vector, not a struct: Vx v0.0.3 does not free a vector stored
into a field ([vx-lang/Vx#1573](https://github.com/vx-lang/Vx/issues/1573)), and with a struct
the program used 600 MB instead of 4 MB.
