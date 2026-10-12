# interval

Closed intervals with exact rational ends, built on [`rational`](../rational). They are the
tool for proving an inequality over a range by computer: if a computation on intervals says
`f(X) > 0`, then `f(x) > 0` for every `x` in `X`.

```console
$ ./build.sh
ok
```

```rust
import bigint::int;
import rational::rat;
import interval::ival;

// t^2 - t + 1/3 > 0 for every t in [i/16, (i+1)/16].
let t = Interval::fracs(i, 16, i + 1, 16);
let f = t.pow(2).minus(&t).plus(&Interval::fracs(1, 3, 1, 3));
if f.is_pos() { ... }
```

There are `plus`, `minus`, `times`, `over`, `pow`, `abs`, `sqrt(k)`, `hull`, `intersect`, and
tests such as `is_pos`, `contains` and `below`. `pow(2)` is tighter than `times` with itself:
`[-1, 2]^2` is `[0, 4]`.

Interval libraries in other languages usually use floating point and round each result
outward, which needs control of the rounding mode. Vx has none, so the ends here are exact
fractions instead. They grow as a computation goes on; `round_out(k)` widens an interval to
ends that are multiples of 2^-k to keep them small, and `sqrt(k)` rounds the same way.

The tests check that every operation holds the results on points: for 1500 random pairs of
intervals and points `x` in `X` and `y` in `Y`, `x + y` is in `X + Y`, and so on. They also
prove a small inequality over `[0, 1/2]` by splitting it into 8 pieces.

There is no `exp`, `log` or `sin` yet.
