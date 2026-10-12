# flash-attn

Attention computed two ways: the textbook way, and the way FlashAttention does it. They agree.

```
$ ./build.sh
non-causal: tiled vs naive, 6 tile shapes, largest difference 2.8e-16
causal:     tiled vs naive, 6 tile shapes, largest difference 1.7e-16
large scores: tiled vs naive differ by 1.1e-16; softmax without the max subtracted gives NaN
score memory at n = 50: naive 2500 numbers, tiled (16 x 16 tiles) 256
ok
```

(The real output prints the differences in full.)

Naive attention builds the whole `n x n` score matrix `S = Q K^T / sqrt(d)`, takes a softmax of
each row, and multiplies by `V`. FlashAttention never builds `S`. It walks over `K` and `V` in
tiles, and for each query row it keeps three running values:

- `m`, the largest score seen so far;
- `l`, the sum of `exp(score - m)`;
- `acc`, the sum of `exp(score - m)` times the rows of `V`.

When a tile brings a larger score, the old `l` and `acc` are multiplied by `exp(m_old - m_new)`,
so everything stays relative to the new maximum. This is the "online softmax". At the end the
output is `acc / l`. Only one tile of scores exists at a time.

`src/main.vx` checks, on random 50 x 16 inputs:

- The tiled result equals the naive one, within 10^-12, for six tile shapes (square, tall, wide,
  sizes that do not divide 50, one tile for everything, one key per tile), with and without a
  causal mask.
- With scores up to about 2500, both stay finite, because both subtract the maximum. A softmax
  that does not subtract it overflows to NaN.

Each of these changes makes it fail: not rescaling `acc`, not rescaling `l`, dropping the causal
mask from the tiled version, and replacing the running maximum with the tile's own maximum.
That last one is still correct algebra; it fails only with the large scores, when
`exp(m_old - m_new)` overflows. That is why the running maximum is there.

Needs Vx newer than v0.0.3 (built from `main`); see the [top-level README](../README.md).
