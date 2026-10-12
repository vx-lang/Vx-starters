# rope

Rotary position embeddings (RoPE), the way LLaMA and most recent LLMs encode positions, with
their constants worked out by the compiler.

```
$ ./build.sh
compile-time cos and sin of the 4 frequencies vs run time: largest difference 1.1e-16
table vs cos and sin, 16 positions x 4 frequencies: largest difference 1.7e-15
position 0 is the identity: largest difference 0
rotation keeps the length: largest difference 1.1e-15
q(m) . k(n) = q(m - n) . k(0), all 256 pairs: largest difference 3.2e-15
table rotation vs complex multiplication: largest difference 1.7e-15
ok
```

(The real output prints the differences in full.)

RoPE encodes a token's position `m` by rotating its query and key vectors. The vector is split
into pairs `(x[2i], x[2i+1])`, and pair `i` is turned by the angle `m * theta_i`, with
`theta_i = 10000^(-2i/d)`. A rotation by `a` followed by one by `-b` is a rotation by `a - b`, so
the dot product of a query at position `m` with a key at position `n` depends only on `m - n`:
attention sees relative positions. Here the head size `d` is 8 (four pairs) and positions run
from 0 to 15.

The cos and sin of each `theta_i` are computed in `comptime` blocks, so the program carries them
as constants. The compile-time evaluator cannot call `cos` or `sin`, so they come from plain
arithmetic: `10000^(-1/4)` by Newton's method, then Taylor series. The table for every position
is built from them at run time, by turning position `m` once more to get `m + 1`.

`src/main.vx` checks:

1. The compile-time constants equal `cos` and `sin` at run time, within 10^-15, and the table
   built from them within 10^-14.
2. Position 0 changes nothing.
3. Rotation keeps a vector's length.
4. For all 256 pairs of positions, `q(m) . k(n)` equals `q(m - n) . k(0)` (or `q(0) . k(n - m)`).
5. The result equals the complex-number form: pair `i` as `z = x[2i] + i x[2i+1]`, times
   `e^(i m theta_i)`, using `cos` and `sin` at run time.

Each of these changes makes it fail:
- the wrong sign in the rotation;
- the wrong base, 1000 instead of 10000;
- a Taylor series cut to 5 terms;
- the wrong sign in the angle-addition step;
- reading the table one position late;
- a rotation that stretches one coordinate by 10^-6.

The wrong base is caught only by checks 1 and 5. Any set of angles keeps lengths and relative
positions, so checks 3 and 4 alone would not notice.

## What comptime could not do

The plan was to build the whole 16 x 4 table at compile time. Two things stopped that:

- A tensor of `f64` made at compile time comes out wrong. Whole numbers become `i32` and others
  `f32`, ignoring the declared type. Read directly, the values are silently cut (0.5 becomes 0);
  passed by reference, the program fails to compile
  ([vx-lang/Vx#1649](https://github.com/vx-lang/Vx/issues/1649)). A single `f64` comes out
  exactly, so each of the eight constants is its own `comptime` block.
- A cast inside a function that `comptime` calls, such as `i as f64`, cannot be evaluated
  ([vx-lang/Vx#1648](https://github.com/vx-lang/Vx/issues/1648)). Angles are counted in `f64`
  instead. `Tensor::new()` cannot be evaluated either, so the zero tables are written out as
  literals.

Needs Vx newer than v0.0.3 (built from `main`); see the [top-level README](../README.md).
