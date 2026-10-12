# batch-sched

Continuous batching against static batching, in a small simulator of an LLM server.

```console
$ ./build.sh
by hand, continuous: first tokens at 8 11 15
                     done at 13 13 15
by hand, static:     first tokens at 10 10 15
                     done at 13 12 15
300 requests, 80894 output tokens
continuous: 4752 steps, 3407.5679398969955 tokens/s, median first token 6603 ms, median done 8040 ms
static:     7802 steps, 2250.837462678959 tokens/s, median first token 14330 ms, median done 14962 ms
ok
```

A request is a prompt to read (prefill) and some tokens to write (decode). The server runs steps;
a step of `n` tokens costs a fixed overhead plus a cost per token.

- **Static batching** takes the requests that are waiting, prefills them together, and decodes
  until the longest one is done. A request that finishes early leaves its place empty, and new
  requests wait for the whole batch.
- **Continuous batching** (Orca, vLLM) decides again at every step. Each running request that is
  decoding gets one token. The rest of the step's token budget goes to prefill, so a long prompt
  is spread over several steps (chunked prefill). A finished request's place goes to the next
  waiting one at once.

Both admit a request only if the KV cache has room for its prompt and all of its output, so
nothing is ever evicted.

## What is checked

**A case worked out by hand.** A budget of 8 tokens per step, 2 places, and a step costs one unit
per token. A arrives at 0 with a 4-token prompt and 3 tokens to write; B at 0 with 6 and 2; C at
1 with 2 and 1.

Continuous:

| Step | Starts | Tokens | Ends | What happens |
| --- | --- | --- | --- | --- |
| 1 | 0 | 8 | 8 | A's prompt (4), first 4 of B's; A's first token |
| 2 | 8 | 3 | 11 | A decodes, the rest of B's prompt (2); B's first token |
| 3 | 11 | 2 | 13 | A and B decode; both done |
| 4 | 13 | 2 | 15 | C's prompt; its only token |

Static: A and B prefill together (10 tokens, done at 10), then decode at 12 (B done) and 13
(A done), then C runs alone until 15. The program must give exactly these times.

**A random workload.** 300 requests, one every 0 to 40 ms, with prompts of 32 to 1024 tokens and
outputs of 8 to 512. Steps take up to 512 tokens and 32 requests, the cache holds 16384 tokens, and
a step costs 4 ms plus 20 µs per token. For both policies:

- every request gets exactly its tokens, its first token after it arrives, and its finish after
  that;
- no continuous step goes over the token budget;
- no step goes over the places or the cache;
- requests are admitted once each, in arrival order.

On this workload continuous batching must have the higher throughput and the lower median time to
first token. It does, by about 1.5x and 2x. The server is overloaded on purpose, so the waits are
seconds long.

Each of these changes makes it fail:

- admitting without checking the cache;
- not writing the first token when a prefill finishes in the middle of a step;
- a prefill chunk that ignores the step budget;
- recording static first tokens before their step ends;
- admitting requests that have not arrived yet;
- finishing a request one token early.

Needs Vx newer than v0.0.3 (built from `main`); see the [top-level README](../README.md).
