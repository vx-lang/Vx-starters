# data-loader

The sample order of a distributed training run: shuffled every epoch, split across ranks, and
the same after a restart.

```
$ ./build.sh
1000 samples, batches of 96 on 4 ranks: each epoch reads 960 distinct samples and drops 40
restart at step 37 of 60: every rank reads the same samples as without the restart
the same global batches on 1, 2, 4, 8 and 16 ranks
first of 8 over 20000 shuffles: chi-square 8.239999999999998 (7 degrees of freedom, limit 24.32)
ok
```

The loader:

- Each epoch is a Fisher-Yates shuffle of the sample numbers, with a generator seeded from
  (seed, epoch) by splitmix64.
- Global batch `s` of an epoch is positions `s*B .. s*B + B` of that shuffle, and rank `r` takes
  the `r`-th slice of `B / R`. A partial last batch is dropped.
- The loader's whole state is the step number, which is what a checkpoint saves.

`src/main.vx` checks four properties. Real training runs have been bitten by each one:

1. **Every sample once per epoch.** With 1000 samples, global batches of 96 and 4 ranks, each of
   5 epochs reads 960 different samples across the ranks, none twice, and no two epochs share an
   order.
2. **Restarts read the same data.** Reading steps 0 to 37, then 37 to 60 as a restarted run would,
   gives every rank exactly what one run of 60 steps reads, across an epoch boundary.
3. **Elastic training.** On 1, 2, 4, 8 or 16 ranks, the rank slices of each step, put back
   together, are the same 96 samples. So a run can restart on a different number of ranks and
   see the same batches.
4. **A uniform shuffle.** Over 20,000 shuffles of 8 samples, each sample comes first about
   equally often (chi-square below its 0.999 limit). Drawing `j` from `0..i` instead of
   `0..=i`, a common slip, gives Sattolo's algorithm: it never leaves a sample in place, and
   sample 0 is never first.

Each of these changes makes it fail:

- that Sattolo slip;
- a seed that ignores the epoch;
- a restarted run counting epochs from the restart;
- rank slices that overlap;
- an epoch that keeps its partial last batch (stops on the `Vec` bounds check).

Needs Vx newer than v0.0.3 (built from `main`); see the [top-level README](../README.md).
