# bpe-tokenizer

Byte-pair encoding, the tokenizer behind GPT models: train it, encode with it, decode with it.

```
$ ./build.sh
aaabdaaabac: 3 merges, 5 tokens
trained on ../FRICTION.md: 13975 bytes, 500 merges
first merges: _t _a _` in it he er es _the or nd _f
last merges: _every _i64 _mo _gener _what _while
../FRICTION.md: 4969 tokens, 2.8124371100825116 bytes per token
../README.md (not seen in training): 1438 tokens, 2.1390820584144645 bytes per token
ok
```

(`_` is a space. The numbers change as the two files change.)

Training starts from the 256 byte values as tokens and repeatedly merges the most frequent
adjacent pair into a new token; the list of merges is the tokenizer. Encoding applies the merges
in the order they were learned. Decoding expands each token back into its two halves, down to
bytes. As in GPT-2, text is first cut into words that start at a space or a newline, and no pair
is counted or merged across a cut.

`src/main.vx` checks:

1. Wikipedia's example, `aaabdaaabac`: the merges are `aa`, `ab` and then `(aa)(ab)`, leaving
   five tokens, `XdXac`. A tie between two pairs goes to the one with the smaller token numbers,
   so training gives the same merges every time.
2. Encoding the training file again gives exactly the tokens training ended with.
3. Decoding gives back the exact bytes, for the training file and for a file it never saw.
4. No token spans a word cut.

Each of these changes makes it fail: encoding with the latest merge first instead of the earliest,
decoding a pair's halves in the wrong order, a merge that does not skip the token it absorbed,
breaking ties the other way, and not cutting text into words.

Pair counts are kept in a dense table of `vocabulary x vocabulary` entries (756 x 756 here), since
Vx's `std` has no general hash map yet: `std::hash_map` declares runtime functions for `i32`
keys, with no type around them.

Needs Vx newer than v0.0.3 (built from `main`); see the [top-level README](../README.md).
