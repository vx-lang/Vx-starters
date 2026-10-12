# layernorm

The statistics behind LayerNorm and RMSNorm, computed without losing them in f32, and
LayerNorm's backward formula checked against `grad`.

```console
$ ./build.sh
variance of 20 f32 rows of 4096 around 1000, spread 0.1, relative error:
  one pass E[x^2] - E[x]^2: 3623.723577027836 (0 rows negative)
  two passes: 0.0009340031933617565
  Welford: 0.0009284402089811698
  Welford of two parts, merged: 0.012425035484767692, its mean off by 0.0008714348077774048
RMSNorm vs LayerNorm on a row with mean 0: largest difference 0
LayerNorm backward, formula vs grad, 5 rows x 6 inputs: largest difference 0.0000000000000004440892098500626
ok
```

(Two lines from the compiler about the code generator come first, as in
[`lr-schedule`](../lr-schedule).)

LayerNorm needs each row's mean and variance. The textbook one-pass formula
`var = E[x^2] - E[x]^2` subtracts two nearly equal numbers when the mean is large next to the
spread, and in f32 the difference is mostly rounding error.

`src/main.vx` checks:

1. **Variance in f32.** Rows of 4096 values around 1000 with spread 0.1 (variance about 0.003):
   - the one-pass formula is off by a factor of thousands;
   - two passes (the mean first, then squared deviations) and Welford's running update are
     within 0.1% of the f64 answer.
2. **Merging statistics.** Chan's formula combines the Welford statistics of two parts (1500 and
   2596 values, the second shifted by 0.1). The result matches the whole row: the variance within
   5% and the mean within 0.005. This is how a row split across devices, in sequence or tensor
   parallelism, gets normalized. The 5% is f32 rounding in the parts' means; leaving out the
   term for their difference costs about 45%.
3. **RMSNorm**, `x / sqrt(mean(x^2) + eps)`, equals LayerNorm without its scale and shift on a
   row whose mean is zero.
4. **The backward formula.** For an upstream gradient `g` and output `y`,
   `dx = (g - mean(g) - y mean(g y)) / sigma`. It matches `grad` to 10^-15 on every input of 5
   random rows of 6.

Each of these changes makes it fail:

- Welford's update using the old mean twice;
- the merge leaving out the mean-difference term;
- the backward formula leaving out the `y mean(g y)` term;
- RMSNorm missing its division by the row length;
- "two passes" that are really one;
- the merged mean weighted by the wrong part.

The program calls `grad` from a small function, `by_grad`, rather than inside the loop. Passed
straight to `grad`, the loop variable reaches Enzyme as the wrong integer type
([vx-lang/Vx#1656](https://github.com/vx-lang/Vx/issues/1656)).

Needs Vx newer than v0.0.3 (built from `main`); see the [top-level README](../README.md).
