# bf16

The two 16-bit float formats of mixed-precision training, built from bit operations, and the
three problems every mixed-precision run has to handle.

```console
$ ./build.sh
known values (decimal bits):
  bf16(1.0)  0x3F80 = 16256
  bf16(0.1)  0x3DCD = 15821
  bf16(pi)   0x4049 = 16457
  fp16(1.0)  0x3C00 = 15360
  fp16(0.1)  0x2E66 = 11878
  fp16(65504) 0x7BFF = 31743
  fp16(65520) 0x7C00 = 31744
  fp16(2^-24) 0x0001 = 1
  fp16(2^-25) 0x0000 = 0
against Vx's bf16 and f16: 200000 random f32s and 118721 across the fp16 range; 0 bf16 and 0 fp16 differ
(a) 1 + 1000 x 0.001: summed in bf16 1, summed in f32 then rounded to bf16 2
(b) 926 gradients from 1e-9 to 1e-5 in fp16: 342 become 0, worst relative error 1; with loss scale 2^16: 0 become 0, worst relative error 0.00046547203
(c) dynamic loss scaling, 10 steps with one spike: 1 skipped, scale ends at 65536, weight 1 - 9 x 0.001 = 0.99099636
ok
```

- **bf16** keeps f32's 8-bit exponent and cuts the mantissa to 7 bits. It has the same range as
  f32, but only 2 to 3 significant digits.
- **fp16** (IEEE half) has a 5-bit exponent and a 10-bit mantissa. It has more precision, but
  nothing above 65504 and nothing below about 6e-8.

`src/main.vx` converts f32 to both with integer operations on the f32's bits, rounding to nearest
with ties to even. fp16 needs more work: re-biasing the exponent, subnormals, and overflow to
infinity.

## Checks

**Known bit patterns:**
- 0.1 rounds up to `0x3DCD` in bf16; pi rounds down to `0x4049`.
- 65504 is the largest fp16. 65520 is exactly halfway to the next step and rounds to even, which
  is infinity.
- 2^-24 is the smallest fp16 subnormal. 2^-25, halfway to it, rounds to even, which is zero.

**Against Vx's own types:**
- 200,000 random f32 bit patterns, of every kind: normal, subnormal, infinite and NaN.
- 118,721 more spread across fp16's range.
- The fp16 reference is `x as f16`, LLVM's conversion. The bf16 reference is
  `libm_bf16_bits_from_f32` from `core::libm`. That function is bit operations too, since
  `x as bf16` flushes subnormals to zero on CPUs with AVX512-BF16 (the comment in `libm.vx`
  says why).
- NaNs only have to stay NaNs, since payloads differ between converters.
- Every fp16 value also survives the trip to f32 and back.

**(a) A sum kept in bf16 stops growing.** Next to 1, the bf16 step is 2^-7 (about 0.0078), so
1 + 0.001 rounds back to 1. A thousand additions leave 1. Summing in f32 and rounding once gives
2. This is why optimizers keep f32 master weights and accumulate in f32.

**(b) Small gradients vanish in fp16.**
- Of 926 gradients from 1e-9 to 1e-5, 342 become 0 in fp16, and the subnormal ones keep only a
  few bits.
- Multiplying the loss, and so every gradient, by 2^16 first moves them all into fp16's normal
  range. Dividing by 2^16 again in f32 is exact. Then none are lost, and the worst error is under
  2^-11, half of fp16's relative step.

**(c) Dynamic loss scaling.**
- The scale starts at 2^15. It halves on overflow, and the step is skipped. After 3 good steps in
  a row it doubles.
- In 10 SGD steps with a spike at step 4, exactly that step is skipped, the scale ends at 2^16,
  and the weight takes the other nine updates.

Each of these changes makes it fail:
- bf16 truncating instead of rounding;
- fp16 rounding ties away from zero;
- the subnormal shift off by one;
- decoding fp16 subnormals with the wrong scale;
- never skipping an overflowed step;
- unscaling the gradients in fp16 instead of f32;
- letting exponent 31 through as a normal number.

`grad` is a keyword in Vx, so a variable cannot be called `grad`. The program says `gradient`.

Needs Vx newer than v0.0.3 (built from `main`); see the [top-level README](../README.md).
