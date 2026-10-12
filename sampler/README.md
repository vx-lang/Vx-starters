# sampler

How an LLM picks its next token from the logits: greedy, temperature, top-k, top-p (nucleus) and
min-p sampling, checked by drawing.

```console
$ ./build.sh
temperature 0.01: 0 of 10000 draws were neither of the two tied top tokens
temperature 1000: largest difference from uniform 0.00026513483977655405
temperature 1: chi-square 8.63 (6 degrees of freedom, limit 22.46), largest frequency error 0.0015
temperature 0.7: chi-square 9.04 (6 degrees of freedom, limit 22.46), largest frequency error 0.0012
top-k 3: chi-square 0.18 (2 degrees of freedom, limit 13.82), largest frequency error 0.00043
top-p 0.8: chi-square 0.60 (3 degrees of freedom, limit 16.27), largest frequency error 0.00050
min-p 0.2: chi-square 2.55 (4 degrees of freedom, limit 18.47), largest frequency error 0.0014
ok
```

(The real output prints the numbers in full.)

The model gives one logit per token. A sampler turns the logits into probabilities with a
softmax at some temperature, may cut the list down, and draws one token:

| Setting | What it keeps |
| --- | --- |
| greedy | the most likely token; on a tie, the lower token id |
| temperature `T` | everything, from `softmax(logits / T)`: small `T` sharpens toward greedy, large `T` flattens toward uniform |
| top-k | the `k` most likely tokens |
| top-p | the smallest set of most likely tokens whose total is at least `p` |
| min-p | tokens at least `p` times as likely as the most likely one |

After a cut, the kept probabilities are scaled to add up to 1. A logit of minus infinity is a
masked token, for example one a grammar forbids. Its probability is exactly 0 and it is never
drawn. The test logits have eight tokens: two tie for the top, and one is masked.

`src/main.vx` checks:

1. Greedy picks the largest logit and the lower id on a tie. It never picks a masked token, even
   when all but one token are masked.
2. At temperature 0.01, every one of 10,000 draws is one of the two tied top tokens. At
   temperature 1000, the probabilities are within 0.001 of uniform over the unmasked tokens.
3. Top-k 3, top-p 0.8 and min-p 0.2 keep exactly the tokens worked out by hand. For p from
   0.01 to 0.99, top-p's set reaches p, and would not reach it without its least likely token.
4. For each of five settings, 200,000 draws (from `std::rand`, seeded) match the intended
   probabilities. The chi-square value stays under the level a correct sampler exceeds once in
   1000 runs, and no frequency is off by more than 0.005. No token outside the kept set is drawn.
5. The same seed gives the same tokens, and another seed does not.

Each of these changes makes it fail:
- greedy giving a tie to the higher id;
- a softmax that ignores the temperature;
- ranking tokens from least to most likely;
- a top-p that stops one token short;
- not scaling the kept probabilities back to 1;
- a draw whose running total runs 2% short.

Needs Vx newer than v0.0.3 (built from `main`); see the [top-level README](../README.md).
