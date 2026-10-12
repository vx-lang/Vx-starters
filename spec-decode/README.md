# spec-decode

Speculative decoding: a small model guesses, the large one checks, and the output is still
exactly the large model's.

```
$ ./build.sh
400000 speculative tokens: chi-square against the target 1.4791583333333334 (limit 18.47), against the draft 571845.823145238
guesses kept: 0.5999875, sum of min(p, q) = 0.6000000000000001
k = 1: 1.750684955488835 tokens per target pass, expected 1.7500000000000002
k = 2: 2.3135410882842864 tokens per target pass, expected 2.3125000000000004
k = 3: 2.738300610641036 tokens per target pass, expected 2.7343750000000004
k = 4: 3.0479906121889146 tokens per target pass, expected 3.0507812500000013
k = 5: 3.2713533318067323 tokens per target pass, expected 3.2880859375000018
k = 6: 3.462176291372386 tokens per target pass, expected 3.466064453125002
draft equal to the target: every guess kept; draft with no overlap: none kept
ok
```

A large target model `p` is slow; a small draft model `q` is fast. The draft guesses `k` tokens,
and the target checks all of them in one pass:

- A guess `x` is kept with probability `min(1, p(x) / q(x))`.
- At the first rejection, a replacement is drawn from what is left, `max(0, p - q)` normalized,
  and the remaining guesses are thrown away.
- If all `k` are kept, the target's pass gives one more token.

The result (Leviathan et al. 2023; Chen et al. 2023) is that the tokens are distributed exactly
as if the target had sampled them itself, whatever the draft is. `src/main.vx` checks:

1. **The distribution is the target's.** With a draft that is wrong in several ways (too much
   weight on one token, too little on another, some on a token the target never picks), 400,000
   tokens match the target by chi-square, are nowhere near the draft, and never include the
   token the target never picks.
2. **The acceptance rate** is `sum over x of min(p(x), q(x))`, here 0.6, within 5 standard
   deviations.
3. **The speed-up.** On a toy language of 6 tokens, where each guess is kept with probability
   `a`, a target pass yields on average `(1 - a^(k+1)) / (1 - a)` tokens. It is checked within 2%
   for `k` from 1 to 6. With `a = 0.75`, 6 guesses give about 3.5 tokens per pass of the large
   model.
4. **The two ends.** A draft equal to the target is always kept; a draft with no overlap never
   is.

Each of these changes makes it fail:

- drawing the replacement from `p` instead of `p - q`;
- keeping a guess with probability `q / p`;
- drawing the replacement from the draft;
- no extra token when every guess is kept;
- not normalizing the leftover.

Needs Vx newer than v0.0.3 (built from `main`); see the [top-level README](../README.md).
