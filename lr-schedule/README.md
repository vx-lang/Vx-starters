# lr-schedule

The parts of a training loop around the gradient: the learning-rate schedule, gradient clipping
and gradient accumulation, with gradients from Vx's `grad`.

```console
$ ./build.sh
schedule: step 0 0.0000029999999999999997, step 99 0.0003, step 550 0.00016499999999999997, step 999 0.000030000822466198297
clipping: norm 5 to 1 gives norm 1, same direction (cosine 1)
accumulation: 4 micro-batches of 8 vs one batch of 32 differ by 0.0000000000000006661338147750939; grad vs the formula by 0.0000000000000004440892098500626
one bad label in 256, clean loss after 400 steps: clipped 0.014269961640316855 (worst on the way 5.426456398145676), not clipped 22.91800695123287 (worst 269.3169737713126)
clipped weights: 2.0868838727632233 -2.7014909294049474 0.6872923574700077 1.0886071178963768 (the data says 2 -3 0.5 1)
ok
```

(Two lines from the compiler about the code generator come first; see below.)

The model is linear regression with four weights, `y = w0 x0 + w1 x1 + w2 x2 + b`, and a mean
squared error. `src/main.vx` checks four things:

1. **The schedule most LLMs train with.** A linear warmup to the peak rate over 100 steps, then
   cosine decay to a floor at step 1000:
   - the rate rises at every warmup step and falls at every step after;
   - it reaches the peak exactly at the end of warmup;
   - halfway through the decay it is halfway between the peak and the floor;
   - it never leaves `(0, peak]`, nor goes below the floor after warmup.
2. **Clipping by global norm.** A gradient of norm 5 clipped to 1 has norm 1 and points the
   same way; clipped to 10 it is unchanged.
3. **Gradient accumulation.** The mean of the gradients of 4 micro-batches of 8 equals the
   gradient of the batch of 32, to 10^-15. The gradients come from `grad`, and they match the
   formula `mean((prediction - label) x_k)` to 10^-15.
4. **Why clipping is there.** One of the 256 training samples has a corrupted label, 1000
   instead of about 1. SGD with the schedule meets it once every 8 steps. With clipping at 1,
   the loss on the clean data ends at 0.014 (the noise alone gives 0.0025). Without clipping it
   ends at 23, after reaching 269 on the way. The check asks for at least 100 times.

Each of these changes makes it fail:

- warmup starting from 0 instead of `peak / warmup`;
- cosine decay missing its `/ 2`;
- clipping by `norm / limit` instead of `limit / norm`;
- micro-batches that overlap;
- clipping never applied during training;
- cosine decay spread over all steps instead of the steps after warmup.

## Notes

- **One `grad` per weight.** As in [`tiny-mlp`](../tiny-mlp): there is no gradient for a tensor
  of weights yet ([vx-lang/Vx#1600](https://github.com/vx-lang/Vx/issues/1600),
  [#1601](https://github.com/vx-lang/Vx/issues/1601)), and a program that uses `grad` runs only
  under `run-jit` ([#1594](https://github.com/vx-lang/Vx/issues/1594)).
- **The compiler's two lines.** `grad` with a struct argument puts the program on the older code
  generator, and the compiler says so.
- **No integer arithmetic in the data.** The data comes from sines of the sample number, because
  Enzyme stops on integer arithmetic in the code it differentiates (see
  [`hellinger`](../hellinger)).
- **No `1e-15` literals.** Vx has no exponent notation for numbers yet, so small tolerances are
  written out in full ([vx-lang/Vx#1654](https://github.com/vx-lang/Vx/issues/1654)).

Needs Vx newer than v0.0.3 (built from `main`); see the [top-level README](../README.md).
