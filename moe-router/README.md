# moe-router

Routing tokens to experts in a mixture-of-experts layer: top-k gating, expert capacity, dropped
tokens and the load-balancing loss.

```
$ ./build.sh
by hand: capacity 3, loads 3 3 1, 1 dropped (token 3's choice of expert 0), weights 0.7310585786300048 and 0.26894142136999516
512 tokens, 8 experts, top 2, capacity factor 1: capacity 128, 33 of 1024 assignments dropped, balance loss 1.003089347576322
balance loss: flat router 1, random 1.003089347576322, favouring one expert 1.6540235783166546
dropped by capacity factor: 512 250 218 122 0 (factors 0.5 1 1.25 2 8)
dispatch, experts, combine: 991 rows sent, largest error 0.0000000000000008881784197001252
ok
```

**Routing.** The router scores each token against each expert, and a softmax turns the scores
into probabilities. Each token goes to its top `k` experts (`k = 2`), weighted by those
probabilities renormalized to sum to 1.

**Capacity.** An expert takes at most `ceil(capacity_factor x tokens x k / experts)` tokens per
batch, so that its buffer has a fixed size. Assignments beyond that are dropped, in token order,
as in GShard and the Switch Transformer.

**Balance.** Training adds a load-balancing loss, `experts x sum_i f_i P_i`, where `f_i` is the
share of assignments expert `i` got and `P_i` its mean router probability. It is 1 for an even
spread and grows as routing becomes lopsided.

## What is checked

1. **A batch worked out by hand.** 4 tokens, 3 experts, capacity factor 1, so capacity
   `ceil(8 / 3) = 3`. Every token's first choice is expert 0. Tokens 0 to 2 pick expert 1 second,
   and token 3 picks expert 2. Expert 0 is full after token 2, so token 3's first choice is
   dropped:
   - the loads are 3, 3 and 1, with one assignment dropped;
   - every pair of weights is `1 / (1 + e^-1)` and its complement, since the two scores differ
     by 1.
2. **A random batch.** 512 tokens and 8 experts at capacity factor 1:
   - no expert is over capacity;
   - kept plus dropped is `512 x 2`;
   - each token's two weights sum to 1, the first is the larger, and the two experts differ.
3. **The balance loss** is exactly 1 for a router that scores every expert the same. It is just
   above 1 for random scores, and higher (1.65) when one expert is favoured. Raising the
   capacity factor never drops more, and at 8 it drops nothing.
4. **Dispatch and combine.** Expert `i` multiplies its input by `i + 1`. Each token's features are
   sent to its kept experts, as an all-to-all would. The returned rows are summed with the
   weights. The result must equal the token times `sum of weight x (i + 1)` over its kept
   experts.

Each of these changes makes it fail:

- letting an expert take one past capacity;
- not renormalizing the top-k weights;
- rounding capacity down;
- top-k picking the same expert twice;
- counting only kept assignments in the balance loss;
- dispatching dropped assignments too.

Needs Vx newer than v0.0.3 (built from `main`); see the [top-level README](../README.md).
