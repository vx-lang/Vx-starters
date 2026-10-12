# paged-kv

A paged KV cache, the memory manager inside vLLM-style inference servers, with its index
arithmetic proved by contracts when the program compiles.

```console
$ ./build.sh
Warning[W1009]: Unused function parameter 'blocks'
random workload: 20000 steps, 15223 appends (422 refused: no free block, or the table full), 286 forks, 534 finishes
prompt of 40 tokens and 4 samples of 45: 7 blocks shared, against 15 copied
ok
```

An inference server keeps the attention keys and values of every running sequence. One
contiguous buffer per sequence wastes memory, since nobody knows how long a sequence will get. A
paged cache cuts memory into fixed blocks (16 token slots here) and gives each sequence a block
table, mapping its i-th block to a physical one, as page tables map virtual memory.

- **Fork** (parallel sampling, beam search) copies a sequence's block table, not its data, and
  adds a reference to each block.
- **Append** to a block that another sequence also holds copies it first (copy-on-write).
- **Finish** drops the references; a block with none left goes back on the free list.

A slot holds one number (a token) instead of a key and a value vector, which is enough to check
where everything goes.

## What is checked

**Contracts, proved by the compiler.** `slot(b, off, blocks)` requires `0 <= b < blocks` and
`0 <= off < 16`, and ensures the index is inside the pool. `advance` keeps the fill of a block
between 1 and 16, `release` never takes a reference count below zero, and `top` keeps a
free-list pop inside the list. The prover also checks every call: a call to `slot` compiles only
where the bounds are already known. Numbers read from the tables are unknown to the prover, so
`at` checks them once and panics if they are out of range; without those checks the program does
not compile.

**A random workload against a plain model.** 20,000 random starts, appends, forks and finishes
on a pool of 40 blocks, compared after every step with a model that copies everything. Every
sequence must read back its tokens, and the pool's books must balance: each block is free or
referenced, never both; the reference counts add up to the block-table entries; nothing is lost
or free twice.

**Prefix sharing.** A 40-token prompt fills two blocks and half of a third. Forked into four
samples of 5 more tokens each, the cache uses 7 blocks: the 3 of the prompt, and one copy of the
half block per sample. Four separate copies would use 15.

Each of these changes makes it fail. Three stop the compile: a `slot` that adds one, a missing
offset check before `advance`, and a free-list pop that allows an empty list. Four fail the
run: a fork that does not add references, appending to a shared block without copying it, a
block freed while still held, and a copy that misses the first slot.

## What the prover could not do

The prover works in linear integer arithmetic. It does not reason about `/` or `%`, and it knows
nothing about a variable after `let mut` or an assignment
([vx-lang/Vx#1353](https://github.com/vx-lang/Vx/issues/1353),
[vx-lang/Vx#1356](https://github.com/vx-lang/Vx/issues/1356)). So the proved functions are small
and straight-line. A sequence's length is kept as (blocks, fill of the last block) rather than as
a token count that would need dividing by 16. The warning in the output is
[vx-lang/Vx#1647](https://github.com/vx-lang/Vx/issues/1647): `blocks` appears only in `slot`'s
contract.

Needs Vx newer than v0.0.3 (built from `main`); see the [top-level README](../README.md).
