# rational

Exact fractions, built on [`bigint`](../bigint).

```
$ ./build.sh
ok
```

`rat.vx` defines `Rational`, always in lowest terms with a positive denominator:

```
import bigint::int;
import rational::rat;

let h = Rational::frac(1, 2).plus(&Rational::frac(1, 3));     // 5/6
Rational::parse("-6/4").unwrap().print();                     // -3/2
Rational::frac(2, 3).pow(-2).print();                         // 9/4
```

It has `plus`, `minus`, `times`, `over`, `pow`, `recip`, `compare`, `floor` and `ceil`, and
`floor_bits(k)` and `ceil_bits(k)`, the nearest multiples of 2^-k below and above, which
[`interval`](../interval) uses to keep numbers small. `to_f64` is for printing, not proofs.

The tests check fixed values from Python's `fractions.Fraction`, and the field laws, lowest
terms and rounding on 2000 random triples.
