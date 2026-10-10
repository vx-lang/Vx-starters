# bigint

Integers of any size. The first of three libraries for checking the computer-assisted parts of
proofs, where every digit has to be right: `bigint`, then [`rational`](../rational) and
[`interval`](../interval).

```
$ ./build.sh
ok
```

`nat.vx` works on whole numbers stored as a `Vec<u32>` of base-2^32 digits, least significant
first: add, subtract, multiply (schoolbook), shift, and long division (Knuth's algorithm D).
`int.vx` puts a sign on them as `BigInt`:

```
import bigint::int;

let a = BigInt::parse("-123456789012345678901234567890").unwrap();
let b = BigInt::from_i64(987654321);
a.times(&b).print();                 // -121932631124828532112482853211126352690
a.div_floor(&b).print();             // -124999998873437499902
BigInt::from_i64(2).pow(201).isqrt().print();
```

`+ - * / %` work too, but use up their operands (`a + b` moves `a` and `b`); the methods take
references. `/` and `%` round toward zero, as on `i64`; `div_floor`, `mod_floor` and `div_ceil`
round the other ways. There are also `gcd`, `pow`, `isqrt`, `shl`, `shr_floor`, `bits`,
`compare`, `to_i64` and `to_f64`.

The tests check fixed values from Python, then the identities that division, shifts, `gcd`
and `isqrt` must satisfy on 3000 random numbers. The random digits are often 0, 2^31 and
2^32 - 1, so long division takes the rare step where it corrects its guess of a quotient
digit. Breaking that step, or any of several others, makes the tests fail.

Multiplication is schoolbook and division is quadratic, which is fine for numbers of a few
thousand digits. See [`../FRICTION.md`](../FRICTION.md) for the compiler bugs this library
works around.
