# quantize

int8 and int4 weight quantization, the way models are shrunk for inference, and an int8 matrix
product.

```
$ ./build.sh
[0.5, -1, 0.25, 1] in int8: [64, -127, 32, 127]
64 x 256 weights, no outlier
  int8 per tensor  : mean error 0.0001499358225039982, max error 0.00029790299212598126
  int8 groups of 64: mean error 0.00009917032128521752, max error 0.00029249244094487923
  int4 per tensor  : mean error 0.0027082695309012268, max error 0.005405278571428571
  int4 groups of 64: mean error 0.0018056037617710712, max error 0.0053646685714285715
the same with one outlier of 1.0
  int4 per tensor  : mean error 0.01601475568028044, max error 0.07107014285714285
  int4 groups of 64: mean error 0.001835628874511717, max error 0.04471810999999999
int4 packing: 256 pairs, 0 wrong
int8 matmul, 8 x 256 by 256 x 64: largest error 0.012753569608003251 (1.2407525746136856% of the largest output), at most 11.656322651388756% of its worst-case bound
ok
```

Symmetric absmax quantization gives a block of weights one scale, `s = max |w| / L`, where `L`
is 127 for int8 and 7 for int4. Each weight is stored as `q = round(w / s)`, clamped to
`[-L, L]`, and read back as `q * s`. The block is either the whole tensor or a group of 64
weights along a row. A group's scale comes from its own largest weight, so one outlier spoils
only its own group: with an outlier 50 times the typical weight, per-group int4 has about a
ninth of the per-tensor error.

`src/main.vx` checks:

1. A worked example: `[0.5, -1, 0.25, 1]` in int8 is `[64, -127, 32, 127]` (63.5 rounds away from
   zero).
2. Every weight comes back within half a step (`s / 2`) of where it was, for int8 and int4, per
   tensor and per group.
3. With one outlier, per-group int4 has at most a quarter of the per-tensor error.
4. int4 packing, two values to a byte, gives back every pair in `[-8, 7]`.
5. An int8 x int8 product of 8 inputs by a 256 x 64 layer is summed in `i32` and scaled back
   once per output: `y = (sum of qx * qw) * sx * sw`. Each factor is off by at most half its step,
   so each product is off by at most `|x| sw / 2 + |w| sx / 2 + sx sw / 4`. The error must stay
   inside the sum of those bounds; it uses about 12% of it.
6. Contracts, proved while compiling: `clamp` returns a value in `[-L, L]`, and `pack` returns a
   byte in `[0, 255]`.

Each of these changes makes it fail. Two stop the compile: a clamp that lets `L + 1` through, and
packing the high half with an offset of 7 instead of 8. Five fail the run: rounding down instead
of to nearest, a scale of `max / (L + 1)`, unpacking the high half from the wrong bit, scaling
every output by the first input's scale, and groups that take the whole tensor's maximum.

The prover warns that it "cannot reason about the `Div` operator" when a function with a
contract has `let s = big / ...`, even for floats it does not reason about anyway. So the
division sits in a small function of its own, `step`.

Needs Vx newer than v0.0.3 (built from `main`); see the [top-level README](../README.md).
