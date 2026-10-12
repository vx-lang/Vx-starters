# hellinger

Re-runs the interval certificate in the proof of the Hellinger contraction conjecture for
Boolean functions with any output bias, which implies the Courtade-Kumar "most informative
function" bound. The proof is "Hellinger contraction with arbitrary Boolean output bias" in
[openai/math](https://github.com/openai/math)
(`preprints/Hellinger-contraction-with-arbitrary-Boolean-output-bias-September-24-2026`).

```console
$ ./build.sh
ok: pass (798493 boxes)
ok: 6282 points, value and three derivatives each
```

`cert.vx` is a port of the paper's `verification/interval_certificate.cpp` (Apache License
2.0): the arithmetic, the functions and the boxes. `src/main.vx` runs the search. For seven
groups of constants and three modes, it shows that certain functions of three variables are
positive on boxes. A box is proved by interval arithmetic, or by the value at its center and
bounds on the derivatives; otherwise it is split into eight, down to a depth limit. Numbers are
multiples of 10^-12 held in `i64`, products are taken in `i128`, and every result is rounded
outward. `src/gradcheck.vx` checks the derivative bounds with `grad` (below).

The port keeps the C++ step for step:

- The C++ template over intervals and "jets" (a value with three derivatives) is a trait, `Ar`,
  with two implementations, and a third in `src/gradcheck.vx` for plain `f64`.
- Products of the constants, which the C++ recomputes in each call, are computed once in
  `params`, with the same operations and so the same values.
- The C++ throws when a number leaves [-10^18, 10^18], and the box counts as not proved. Here
  each interval carries an `ok` flag that every operation passes on.
- Every expression keeps the grouping and order of the C++, since outward rounding makes the
  result depend on them.

`build.sh` checks more than "pass": it compares the number of boxes visited with the C++ (with a
counter added). Equal counts mean the two programs made the same decision on every box.

| | Time |
| --- | --- |
| The paper's C++, `clang -O2` | 24 s |
| The paper's C++, `g++ -O2` | 67 s |
| This port, `vxc` then `clang -O2` | 29 s |
| This port, `vxc -O0` | 113 s |

One run each, single-threaded, on a shared 16-core machine; times vary with its load. The
self-test adds well under a second.

`vxc -O1` and above take more than four minutes to compile this program, because compile time
grows exponentially with how deeply calls nest
([vx-lang/Vx#1589](https://github.com/vx-lang/Vx/issues/1589)). So `build.sh` asks `vxc` for
unoptimized LLVM, and the bundled `clang` optimizes it in 3 s.

Two checks guard against a port that only looks right:

- `build.sh` compares the box count, not just "pass". Dropping the `mu` term from mode 1 still
  prints "pass", with 729133 boxes; so does a derivative test that ignores one direction, with
  765749. Only the count catches these. Changing a constant in `table` or dropping a term from
  the product of two `A` values makes a box fail.
- Before the search, `self_test` checks on 20,000 random intervals, with exact `i128`
  arithmetic, that products, reciprocals, roots, sums and differences enclose the exact values.
  Rounding a bound the wrong way by one unit (10^-12) does not change the box count, so this is
  what catches it: rounding the top of a product down, the bottom of `1 / x` up, or never
  rounding a root up each make the self-test fail.

## Checking the derivatives with `grad`

A box can be proved from its center value and bounds on the derivatives over the box. The jets
compute those bounds. If a derivative formula were wrong, the search could print "pass" for a
proof that is not one: with a wrong partial derivative in the `E` step, it prints
`pass (794885 boxes)`, and only the comparison with the C++ count shows anything is off. A proof
with no reference program would have nothing to compare with.

`src/gradcheck.vx` checks the jets against Vx's automatic differentiation. It implements `Ar`
for `f64`, so the same expressions run in floating point, and differentiates them with `grad`
(Enzyme) in each of the three directions. For 60 random boxes in each group and mode, at
depths 2 to 7, and 3 random points in each box, it checks that the value and each derivative,
times the box's half-width, lie in the jet's intervals, allowing 10^-9 relative for floating
point. That is 6282 points, in about 2 s. Floating point proves nothing here; the point is that
an independent derivative agrees with the one the proof uses.

It catches each of these, at hundreds of points: the product rule with its factors swapped,
the derivative of a square root off by a factor, and the wrong partial derivative of `E` above.

Two things were needed to make `grad` work here:

- Enzyme fails ("Cannot deduce type of phi") when the function it differentiates calls the
  integer interval code, even only to build constants. So the expressions get small integers
  through the trait (`of_int`), and products of constants come precomputed in `Params`.
- A program that uses `grad` runs only under `vxc -a run-jit`
  ([vx-lang/Vx#1594](https://github.com/vx-lang/Vx/issues/1594)), so the cross-check is a second
  program, not part of the built one.

Not done yet:

- Contracts. The useful one is that every interval an operation returns has
  `-10^18 <= lo <= hi <= 10^18`, which is what keeps the `i128` products from overflowing. The
  prover cannot yet prove an `ensures` about a returned struct
  ([vx-lang/Vx#1590](https://github.com/vx-lang/Vx/issues/1590)).
- Running the boxes on a GPU. The search is recursive and stops at the first failure; it would
  have to become a loop over a list of boxes first.
- The paper's other checks (`profile_certificate.py`, the three bias ranges) use SymPy.
